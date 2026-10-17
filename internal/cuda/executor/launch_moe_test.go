package executor

import (
	"encoding/binary"
	"testing"

	"overgo/internal/cuda/device"
	"overgo/internal/cuda/driver"
	"overgo/internal/tensor"
	"overgo/internal/tensor/dtype"
)

// TestMoELaunchesCoverTokenGroups probes a routed-expert node whose scratch
// holds three tokens: eight tokens run as groups of three, three and two, each
// a route, a hidden and a down launch over the group's tokens, with the
// scratch reused and the token-major operands advanced by the group's start.
func TestMoELaunchesCoverTokenGroups(t *testing.T) {
	const (
		hidden, experts, topK, tokens = 64, 16, 16, 8
		// Sixteen slots of this many rows put three tokens under the limit.
		intermediate = 163840
		group        = 3
		routeBytes   = (2*topK + 1) * 4
	)
	builder := tensor.NewBuilder()
	input := builder.Input("input", dtype.F32, tensor.MustShape(hidden, tokens))
	router := builder.Input("router", dtype.F32, tensor.MustShape(hidden, experts))
	gate := builder.Input("gate", dtype.BF16, tensor.MustShape(hidden, intermediate, experts))
	up := builder.Input("up", dtype.F32, tensor.MustShape(hidden, intermediate, experts))
	down := builder.Input("down", dtype.BF16, tensor.MustShape(intermediate, hidden, experts))
	selected := builder.Input("selected", dtype.F32, tensor.MustShape(topK, tokens))
	node := builder.MoEWithOptions(input, router, up, down, tensor.MoEOptions{
		Gate: gate, SelectedExperts: selected, TopK: topK, Scale: 1,
		Routing: tensor.MoERoutingSoftmax, Activation: tensor.MoEActivationSiLU, ExpertIndexDivisor: 1,
	})
	if err := builder.Err(); err != nil {
		t.Fatal(err)
	}
	// Slot 0 is the output; slots 1.. are the node's inputs in order.
	const output, inputBase, routerInputBase, selectedBase, scores = 1 << 40, 2 << 40, 3 << 40, 8 << 40, 9 << 40
	frame := launchPointerFrame{
		values: []driver.DevicePtr{output, inputBase, routerInputBase, 4 << 40, 5 << 40, 6 << 40, 7 << 40, selectedBase},
		slots:  []int{0, 1, 2, 3, 4, 5, 6, 7},
	}
	var functions functionSet
	for _, id := range []kernelFunctionID{kernelMoeRouteF32, kernelMoeHiddenF32, kernelMoeDownF32} {
		functions[id] = boundKernel{id: id, argumentCount: kernelFunctionArgumentCounts[id]}
	}
	tokenBytes := uint64(routeBytes + topK*intermediate*4)
	blas := &blasState{scores: scores, scoreBytes: group * tokenBytes}
	trace := &device.LaunchTrace{Probe: true}
	if err := launchMoE(&device.State{Trace: trace}, functions, blas, nil, node, node.Attrs, frame); err != nil {
		t.Fatal(err)
	}
	// A record is four header words, then each argument: eight bytes a
	// pointer, four a scalar.
	const (
		headerBytes = 4 * 8
		routeRecord = headerBytes + 6*8 + 5*4
		hiddenBytes = headerBytes + 7*8 + 12*4
		downRecord  = headerBytes + 6*8 + 8*4
	)
	raw := trace.Data()
	if want := 3 * (routeRecord + hiddenBytes + downRecord); len(raw) != want {
		t.Fatalf("launch bytes %d, want three groups of three launches (%d)", len(raw), want)
	}
	word, scalar := binary.LittleEndian.Uint64, binary.LittleEndian.Uint32
	start := uint64(0)
	for _, count := range []uint64{group, group, tokens - 2*group} {
		route, hiddenLaunch, downLaunch := raw[:routeRecord], raw[routeRecord:routeRecord+hiddenBytes], raw[routeRecord+hiddenBytes:routeRecord+hiddenBytes+downRecord]
		raw = raw[routeRecord+hiddenBytes+downRecord:]
		for _, check := range []struct {
			name      string
			got, want uint64
		}{
			{"route kernel", word(route), uint64(kernelMoeRouteF32)},
			{"route grid", word(route[8:]), count | 1<<32 | 1<<48},
			{"route block", word(route[16:]), uint64(moeRouteThreads) | 1<<32 | 1<<48},
			{"route logits", word(route[24:]), experts * 4},
			{"route router input", word(route[32:]), routerInputBase + start*hidden*4},
			{"route selection", word(route[64:]), selectedBase + start*topK*4},
			{"route state", word(route[72:]), scores},
			{"route tokens", uint64(scalar(route[84:])), count},
			{"hidden kernel", word(hiddenLaunch), uint64(kernelMoeHiddenF32)},
			{"hidden grid", word(hiddenLaunch[8:]), (count*topK*intermediate+255)/256 | 1<<32 | 1<<48},
			{"hidden input", word(hiddenLaunch[32:]), inputBase + start*hidden*4},
			{"hidden route state", word(hiddenLaunch[40:]), scores},
			{"hidden activations", word(hiddenLaunch[80:]), scores + group*routeBytes},
			{"hidden gate storage", uint64(scalar(hiddenLaunch[104:])), uint64(dtype.BF16)},
			{"hidden up storage", uint64(scalar(hiddenLaunch[108:])), uint64(dtype.F32)},
			{"hidden count", uint64(scalar(hiddenLaunch[132:])), count * topK * intermediate},
			{"down kernel", word(downLaunch), uint64(kernelMoeDownF32)},
			{"down grid", word(downLaunch[8:]), (count*hidden+255)/256 | 1<<32 | 1<<48},
			{"down route state", word(downLaunch[32:]), scores},
			{"down activations", word(downLaunch[40:]), scores + group*routeBytes},
			{"down output", word(downLaunch[72:]), output + start*hidden*4},
			{"down storage", uint64(scalar(downLaunch[100:])), uint64(dtype.BF16)},
			{"down count", uint64(scalar(downLaunch[108:])), count * hidden},
		} {
			if check.got != check.want {
				t.Fatalf("tokens %d..%d: %s = %#x, want %#x", start, start+count, check.name, check.got, check.want)
			}
		}
		start += count
	}
	blas.scoreBytes--
	if err := launchMoE(&device.State{Trace: trace}, functions, blas, nil, node, node.Attrs, frame); err == nil {
		t.Fatal("a scratch one byte short was accepted")
	}
}

// TestCompileReservesMoEScratch: a routed-expert node reserves one launch
// group's scratch in the score staging -- per token its route words and its
// routed activations -- bounded for a long prompt, whole for one wide token,
// and the same whatever the expert banks' storage.
func TestCompileReservesMoEScratch(t *testing.T) {
	compile := func(tokens, intermediate uint64, topK uint32, bankType dtype.Type, fused bool) *CompiledGraph {
		t.Helper()
		const hidden, experts = 1280, 64
		builder := tensor.NewBuilder()
		input := builder.Input("input", dtype.F32, tensor.MustShape(hidden, tokens))
		router := builder.Input("router", dtype.F32, tensor.MustShape(hidden, experts))
		options := tensor.MoEOptions{
			TopK: topK, Scale: 1, Routing: tensor.MoERoutingSoftmax,
			Activation: tensor.MoEActivationSiLU, FusedGateUp: fused, ExpertIndexDivisor: 1,
		}
		upRows := intermediate
		if fused {
			upRows *= 2
		} else {
			options.Gate = builder.Input("gate", bankType, tensor.MustShape(hidden, intermediate, experts))
		}
		up := builder.Input("up", bankType, tensor.MustShape(hidden, upRows, experts))
		down := builder.Input("down", bankType, tensor.MustShape(intermediate, hidden, experts))
		compiled, err := Compile(builder.MoEWithOptions(input, router, up, down, options))
		if err != nil {
			t.Fatal(err)
		}
		if !compiled.needBlas {
			t.Fatal("routed-expert graph did not request the score staging's owner")
		}
		return compiled
	}
	// Unlimited-OCR's routed layer: six slots of 896 rows, 13 route words.
	const ocrTokenBytes = (13 + 6*896) * 4
	for _, bankType := range []dtype.Type{dtype.F32, dtype.BF16} {
		for _, fused := range []bool{false, true} {
			if got := compile(1, 896, 6, bankType, fused).attentionScoreBytes; got != ocrTokenBytes {
				t.Fatalf("%s decode scratch (fused %t) = %d, want %d", bankType, fused, got, ocrTokenBytes)
			}
		}
	}
	group := moeScratchLimitBytes / ocrTokenBytes
	if got := compile(group+1, 896, 6, dtype.BF16, false).attentionScoreBytes; got != group*ocrTokenBytes {
		t.Fatalf("prefill scratch = %d, want %d tokens of %d", got, group, ocrTokenBytes)
	}
	// Sixteen slots of limit/4 rows: one token alone exceeds the limit.
	const wideRows = moeScratchLimitBytes / 4
	wideTokenBytes := (uint64(tensor.MaxMoETopK)*(wideRows+moeSlotRouteWords) + moeTokenRouteWords) * 4
	if got := compile(3, wideRows, tensor.MaxMoETopK, dtype.F32, false).attentionScoreBytes; got != wideTokenBytes {
		t.Fatalf("wide-token scratch = %d, want one token of %d", got, wideTokenBytes)
	}
}
