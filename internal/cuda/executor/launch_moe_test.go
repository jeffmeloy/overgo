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
// a router-logits, a route, a hidden and a down launch over the group's tokens, with the
// scratch reused and the token-major operands advanced by the group's start.
func TestMoELaunchesCoverTokenGroups(t *testing.T) {
	const (
		hidden, experts, topK, tokens = 64, 16, 16, 8
		// Sixteen slots of this many rows put three tokens under the limit.
		intermediate = 163839
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
	for _, id := range []kernelFunctionID{kernelMoeRouterLogitsF32, kernelMoeRouteF32, kernelMoeHiddenF32, kernelMoeDownF32, kernelMoeHiddenDotF32, kernelMoeDownDotF32} {
		functions[id] = boundKernel{id: id, argumentCount: kernelFunctionArgumentCounts[id]}
	}
	tokenBytes := uint64(routeBytes + topK*intermediate*4 + experts*4)
	blas := &blasState{scores: scores, scoreBytes: group * tokenBytes}
	trace := &device.LaunchTrace{Probe: true}
	if err := launchMoE(&device.State{Trace: trace}, functions, blas, nil, node, node.Attrs, frame); err != nil {
		t.Fatal(err)
	}
	// A record is four header words, then each argument: eight bytes a
	// pointer, four a scalar.
	const (
		headerBytes  = 4 * 8
		logitsRecord = headerBytes + 4*8 + 3*4
		routeRecord  = headerBytes + 5*8 + 5*4
		hiddenBytes  = headerBytes + 7*8 + 13*4
		downRecord   = headerBytes + 6*8 + 9*4
		// A group's activations lead its scratch and its route words follow.
		routeOffset = group * topK * intermediate * 4
	)
	raw := trace.Data()
	if want := 3 * (logitsRecord + routeRecord + hiddenBytes + downRecord); len(raw) != want {
		t.Fatalf("launch bytes %d, want three groups of four launches (%d)", len(raw), want)
	}
	word, scalar := binary.LittleEndian.Uint64, binary.LittleEndian.Uint32
	start := uint64(0)
	for _, count := range []uint64{group, group, tokens - 2*group} {
		logits := raw[:logitsRecord]
		raw = raw[logitsRecord:]
		route, hiddenLaunch, downLaunch := raw[:routeRecord], raw[routeRecord:routeRecord+hiddenBytes], raw[routeRecord+hiddenBytes:routeRecord+hiddenBytes+downRecord]
		raw = raw[routeRecord+hiddenBytes+downRecord:]
		for _, check := range []struct {
			name      string
			got, want uint64
		}{
			{"route kernel", word(route), uint64(kernelMoeRouteF32)},
			{"route grid", word(route[8:]), count | 1<<32 | 1<<48},
			{"route block", word(route[16:]), uint64(moeThreads) | 1<<32 | 1<<48},
			{"route logits", word(route[24:]), experts * 4},
			{"logits kernel", word(logits), uint64(kernelMoeRouterLogitsF32)},
			{"logits grid", word(logits[8:]), (experts+7)/8 | count<<32 | 1<<48},
			{"logits router input", word(logits[32:]), routerInputBase + start*hidden*4},
			{"logits output", word(logits[56:]), scores + group*routeBytes + group*topK*intermediate*4},
			{"logits tokens", uint64(scalar(logits[68:])), count},
			{"route logits input", word(route[32:]), scores + group*routeBytes + group*topK*intermediate*4},
			{"route selection", word(route[48:]), selectedBase + start*topK*4},
			{"route state", word(route[56:]), scores + routeOffset},
			{"route marks no residency", word(route[64:]), 0},
			{"route tokens", uint64(scalar(route[72:])), count},
			{"hidden kernel", word(hiddenLaunch), uint64(kernelMoeHiddenF32)},
			{"hidden grid", word(hiddenLaunch[8:]), count*topK*((intermediate+255)/256) | 1<<32 | 1<<48},
			{"hidden shared bytes", word(hiddenLaunch[24:]), 256 * 9 * 4 * 2},
			{"hidden input", word(hiddenLaunch[32:]), inputBase + start*hidden*4},
			{"hidden route state", word(hiddenLaunch[40:]), scores + routeOffset},
			{"hidden activations", word(hiddenLaunch[80:]), scores},
			{"hidden gate storage", uint64(scalar(hiddenLaunch[104:])), uint64(dtype.BF16)},
			{"hidden up storage", uint64(scalar(hiddenLaunch[108:])), uint64(dtype.F32)},
			{"hidden indexed", uint64(scalar(hiddenLaunch[132:])), 0},
			{"hidden count", uint64(scalar(hiddenLaunch[136:])), count * topK * intermediate},
			{"down kernel", word(downLaunch), uint64(kernelMoeDownF32)},
			{"down grid", word(downLaunch[8:]), (count*hidden+255)/256 | 1<<32 | 1<<48},
			{"down route state", word(downLaunch[32:]), scores + routeOffset},
			{"down activations", word(downLaunch[40:]), scores},
			{"down output", word(downLaunch[72:]), output + start*hidden*4},
			{"down storage", uint64(scalar(downLaunch[100:])), uint64(dtype.BF16)},
			{"down indexed", uint64(scalar(downLaunch[108:])), 0},
			{"down count", uint64(scalar(downLaunch[112:])), count * hidden},
		} {
			if check.got != check.want {
				t.Fatalf("tokens %d..%d: %s = %#x, want %#x", start, start+count, check.name, check.got, check.want)
			}
		}
		start += count
	}
	trace.Reset(true)
	attributes := node.Attrs.(tensor.MoEAttributes)
	attributes.Compute = tensor.MulMatComputeNativeTensorCore
	node.Attrs = attributes
	node.Shape = tensor.MustShape(hidden, 1)
	if err := launchMoE(&device.State{Trace: trace}, functions, blas, nil, node, node.Attrs, frame); err != nil {
		t.Fatal(err)
	}
	const nativeHiddenBlocks = (topK*intermediate*32 + 255) / 256
	native := trace.Data()[logitsRecord+routeRecord:]
	if kernel, grid := word(native), word(native[8:]); kernel != uint64(kernelMoeHiddenDotF32) || grid != nativeHiddenBlocks|1<<32|1<<48 {
		t.Fatalf("native expert kernel %d grid %#x, want kernel %d with one warp per dot product", kernel, grid, kernelMoeHiddenDotF32)
	}
	if kernel, grid := word(native[hiddenBytes:]), word(native[hiddenBytes+8:]); kernel != uint64(kernelMoeDownDotF32) || grid != (hidden*32+255)/256|1<<32|1<<48 {
		t.Fatalf("native down kernel %d grid %#x, want kernel %d with one warp per output channel", kernel, grid, kernelMoeDownDotF32)
	}
	attributes.Compute = tensor.MulMatComputeExact
	node.Attrs = attributes
	node.Shape = tensor.MustShape(hidden, tokens)
	blas.scoreBytes--
	if err := launchMoE(&device.State{Trace: trace}, functions, blas, nil, node, node.Attrs, frame); err == nil {
		t.Fatal("a scratch one byte short was accepted")
	}
}

// TestCompileReservesMoEScratch: a routed-expert node reserves one launch
// group's scratch in the score staging -- per token its route words, its
// routed activations and its router logits -- bounded for a long prompt, whole for one wide token,
// and the same whatever the expert banks' storage.
func TestCompileReservesMoEScratch(t *testing.T) {
	compile := func(tokens, intermediate uint64, topK uint32, bankType dtype.Type, fused bool, compute ...tensor.MulMatCompute) *CompiledGraph {
		t.Helper()
		const hidden, experts = 1280, 64
		builder := tensor.NewBuilder()
		if len(compute) > 0 {
			builder.SetMulMatCompute(compute[0])
		}
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
	// Unlimited-OCR's routed layer: six slots of 896 rows, 13 route words and 64 router logits.
	const ocrTokenBytes = (13 + 6*896 + 64) * 4
	for _, bankType := range []dtype.Type{dtype.F32, dtype.BF16} {
		for _, fused := range []bool{false, true} {
			compiled := compile(1, 896, 6, bankType, fused)
			if got := compiled.attentionScoreBytes; got != ocrTokenBytes {
				t.Fatalf("%s decode scratch (fused %t) = %d, want %d", bankType, fused, got, ocrTokenBytes)
			}
		}
	}
	group := moeScratchLimitBytes / ocrTokenBytes
	compiled := compile(group+1, 896, 6, dtype.BF16, false)
	if got := compiled.attentionScoreBytes; got != group*ocrTokenBytes {
		t.Fatalf("prefill scratch = %d, want %d tokens of %d", got, group, ocrTokenBytes)
	}
	// Sixteen slots of limit/4 rows: one token alone exceeds the limit.
	const wideRows = moeScratchLimitBytes / 4
	wideTokenBytes := (uint64(tensor.MaxMoETopK)*(wideRows+moeSlotRouteWords) + moeTokenRouteWords + 64) * 4
	if got := compile(3, wideRows, tensor.MaxMoETopK, dtype.F32, false).attentionScoreBytes; got != wideTokenBytes {
		t.Fatalf("wide-token scratch = %d, want one token of %d", got, wideTokenBytes)
	}
	for _, fused := range []bool{false, true} {
		native := compile(64, 896, 6, dtype.BF16, fused, tensor.MulMatComputeNativeTensorCore)
		const slotBytes = 64 * ocrTokenBytes
		if native.attentionScoreBytes != slotBytes || native.matmulStagingBytes == 0 || native.matmulStagingBytes > quantStagedChunkBytes {
			t.Fatalf("native scratch: scores=%d staging=%d", native.attentionScoreBytes, native.matmulStagingBytes)
		}
		bounded := compile(1<<20, 896, 6, dtype.BF16, fused, tensor.MulMatComputeNativeTensorCore)
		if bounded.attentionScoreBytes > quantStagedChunkBytes || bounded.matmulStagingBytes > quantStagedChunkBytes {
			t.Fatal("long prompt scratch is unbounded")
		}
		lowReuse := compile(1, 896, 6, dtype.BF16, fused, tensor.MulMatComputeNativeTensorCore)
		if lowReuse.matmulStagingBytes != 0 || lowReuse.attentionScoreBytes != ocrTokenBytes {
			t.Fatal("decode reserves unused matrix staging")
		}
	}
	if huge := compile(64, wideRows, 16, dtype.F32, false, tensor.MulMatComputeNativeTensorCore); huge.matmulStagingBytes != 0 {
		t.Fatal("oversized expert matrix was admitted")
	}
}
