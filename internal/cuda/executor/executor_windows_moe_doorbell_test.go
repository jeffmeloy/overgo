//go:build windows

package executor

import (
	"context"
	"fmt"
	"math"
	"sync/atomic"
	"testing"

	"overgo/internal/cuda/device"
	"overgo/internal/tensor"
	"overgo/internal/tensor/dtype"
	"overgo/internal/tensor/reference"
)

// TestDecodeHostExpertsRingTheDoorbell holds a captured decode step's experts
// in host memory, computed on the CPU while the card computes the rest
// (moeDoorbell), to the host reference: Flash-Next's IQ4_XS gate and up with
// IQ4_NL down, apart and fused, for a decode token, a verify span of three
// that reads shared experts once and one of eight that does not, the card
// holding expert 0 of each bank. The step's first run names its experts on
// host and the second hands them over, reading its output back: the CPU's
// outputs stay within its activations' rounding of the reference, as the
// prompt's do.
func TestDecodeHostExpertsRingTheDoorbell(t *testing.T) {
	// Serial: device fixtures share the one card.
	const (
		// The card keeps expert 0 of each bank beside the tables of a pointer an expert.
		pointerBytes, slotAlignment = 8, 256
		// hostShareLargest and hostShareMean bound the largest and the mean difference from the reference as
		// shares of the largest value, as TestPromptHostExpertsComputeOnTheCPU's do.
		hostShareLargest, hostShareMean = 0.02, 0.005
	)
	for _, layout := range []struct {
		name   string
		fused  bool
		tokens uint64
	}{
		{"token", false, 1}, {"fused-token", true, 1}, {"span", false, 3}, {"wide-span", false, 8},
	} {
		t.Run(layout.name, func(t *testing.T) {
			banks := [3]dtype.Type{dtype.IQ4XS, dtype.IQ4XS, dtype.IQ4NL}
			c := moePinnedCase{
				name: "doorbell", compute: tensor.MulMatComputeNativeTensorCore,
				hidden: 512, routerHidden: 512, intermediate: 128, experts: 48, divisor: 1, topK: 4,
				routing: tensor.MoERoutingSoftmax, activation: tensor.MoEActivationSiLU, normalize: true, scale: 1.25,
				banks: banks, fused: layout.fused, tokens: layout.tokens, expertScale: true,
			}
			shapes := []tensor.Shape{tensor.MustShape(c.hidden, c.intermediate), tensor.MustShape(c.hidden, c.intermediate), tensor.MustShape(c.intermediate, c.hidden)}
			if layout.fused {
				shapes = []tensor.Shape{tensor.MustShape(c.hidden, 2*c.intermediate), shapes[2]}
			}
			limit := uint64(len(shapes)) * c.experts * pointerBytes
			for index, shape := range shapes {
				expert, err := shape.Bytes(banks[3-len(shapes)+index])
				if err != nil {
					t.Fatal(err)
				}
				limit += (expert + slotAlignment - 1) / slotAlignment * slotAlignment
			}
			c.reference = true
			_, want := c.run(t, newFixtureExecutor(t), dtype.F32)
			cuda := newFixtureExecutor(t)
			c.offload, c.limit, c.reference = true, limit, false
			output, host, device, _ := c.prepare(t, cuda, dtype.F32)
			compiled, inputs, err := compileDeviceFeeds([]*tensor.Tensor{output}, device)
			if err != nil {
				t.Fatal(err)
			}
			var got []float32
			for run := range 2 {
				// The ringing run reads its output back too, whose pinned memory it takes before its launch: taken
				// after, it waited for the card, which waited for the ring the worker had not reached.
				if run == 1 {
					if err := compiled.ReadBack(output); err != nil {
						t.Fatal(err)
					}
				}
				retained, err := cuda.ExecuteRetainedCompiled(t.Context(), compiled, host, inputs, nil, nil)
				if err != nil {
					t.Fatal(err)
				}
				value, copyErr := retained.CopyToHost(t.Context(), output)
				if err := retained.Release(t.Context()); err != nil {
					t.Fatal(err)
				}
				if copyErr != nil {
					t.Fatal(copyErr)
				}
				got = value.Data
			}
			if len(compiled.doorbellRings) != 1 {
				t.Fatalf("the step rang for %d routed nodes, want 1", len(compiled.doorbellRings))
			}
			if atomic.LoadUint32(&cuda.resources.blas.doorbell.words[doorbellWork]) == 0 {
				t.Fatal("no ring handed the CPU an expert on host")
			}
			difference, at := maxAbsDifference(t, got, want)
			var scale, mean float64
			for index := range want {
				scale = max(scale, math.Abs(float64(want[index])))
				mean += math.Abs(float64(got[index] - want[index]))
			}
			if mean /= float64(len(want)); difference > hostShareLargest*scale || mean > hostShareMean*scale {
				t.Fatalf("largest difference from the reference %g at %d (value %g), mean %g, largest value %g", difference, at, want[at], mean, scale)
			}
		})
	}
}

// TestDecodeHostExpertsComputeEveryRank holds a captured decode step at Flash-Next's top 10 to the full layer, as
// Strata's measured configuration computes it (STRATA_ROUTE_TAIL_SKIP=0): token t's slot s routes expert (t + 3s) mod
// 48, unnormalized, the card holding expert 0 of each bank, and every slot of an expert on host, its lowest ranks
// too, goes to the CPU and is added.
func TestDecodeHostExpertsComputeEveryRank(t *testing.T) {
	const (
		pointerBytes, slotAlignment     = 8, 256
		hostShareLargest, hostShareMean = 0.02, 0.005
		// topK is Flash-Next's routed experts a token.
		topK = 10
	)
	for _, tokens := range []uint64{1, 3} {
		t.Run(fmt.Sprintf("tokens-%d", tokens), func(t *testing.T) {
			banks := [3]dtype.Type{dtype.IQ4XS, dtype.IQ4XS, dtype.IQ4NL}
			c := moePinnedCase{
				name: "doorbell-ranks", compute: tensor.MulMatComputeNativeTensorCore,
				hidden: 512, routerHidden: 512, intermediate: 128, experts: 48, divisor: 1, topK: topK,
				routing: tensor.MoERoutingSoftmax, activation: tensor.MoEActivationSiLU, scale: 1.25,
				banks: banks, tokens: tokens, selected: true, reference: true,
			}
			_, want := c.run(t, newFixtureExecutor(t), dtype.F32)
			limit := 3 * c.experts * pointerBytes
			for index, shape := range []tensor.Shape{tensor.MustShape(c.hidden, c.intermediate), tensor.MustShape(c.hidden, c.intermediate), tensor.MustShape(c.intermediate, c.hidden)} {
				expert, err := shape.Bytes(banks[index])
				if err != nil {
					t.Fatal(err)
				}
				limit += (expert + slotAlignment - 1) / slotAlignment * slotAlignment
			}
			cuda := newFixtureExecutor(t)
			c.offload, c.limit, c.reference = true, limit, false
			output, host, cardFeeds, _ := c.prepare(t, cuda, dtype.F32)
			compiled, inputs, err := compileDeviceFeeds([]*tensor.Tensor{output}, cardFeeds)
			if err != nil {
				t.Fatal(err)
			}
			var got []float32
			for run := range 2 {
				if run == 1 {
					if err := compiled.ReadBack(output); err != nil {
						t.Fatal(err)
					}
				}
				retained, err := cuda.ExecuteRetainedCompiled(t.Context(), compiled, host, inputs, nil, nil)
				if err != nil {
					t.Fatal(err)
				}
				value, copyErr := retained.CopyToHost(t.Context(), output)
				if err := retained.Release(t.Context()); err != nil {
					t.Fatal(err)
				}
				if copyErr != nil {
					t.Fatal(copyErr)
				}
				got = value.Data
			}
			doorbell := &cuda.resources.blas.doorbell
			if atomic.LoadUint32(&doorbell.words[doorbellWork]) == 0 {
				t.Fatal("no ring handed the CPU an expert on host")
			}
			for slot := range uint32(tokens) * topK {
				hosted := (uint64(slot/topK)+3*uint64(slot%topK))%c.experts != 0
				if flagged := doorbell.ids[slot]&doorbellHostSlot != 0; flagged != hosted {
					t.Fatalf("token %d's slot %d handed to the CPU %t, want %t", slot/topK, slot%topK, flagged, hosted)
				}
			}
			difference, at := maxAbsDifference(t, got, want)
			var scale, mean float64
			for index := range want {
				scale = max(scale, math.Abs(float64(want[index])))
				mean += math.Abs(float64(got[index] - want[index]))
			}
			if mean /= float64(len(want)); difference > hostShareLargest*scale || mean > hostShareMean*scale {
				t.Fatalf("largest difference from the full layer %g at %d (value %g), mean %g, largest value %g", difference, at, want[at], mean, scale)
			}
		})
	}
}

// countedBanks counts the advances of the expert banks it wraps.
type countedBanks struct {
	ExpertBanks
	advances int
}

func (b *countedBanks) AdvanceExperts(ctx context.Context, state *device.State, prompt bool) error {
	b.advances++
	return b.ExpertBanks.AdvanceExperts(ctx, state, prompt)
}

// TestRunsOfNoExpertApplyTheLastRoutes holds the residency's advances to the
// runs between them, as a speculative window's verify and draft take turns:
// an expert run advances before its launch unless a run that reads no routed
// expert has applied the last expert run's routes while its graph ran, and
// such a run applies them once.
func TestRunsOfNoExpertApplyTheLastRoutes(t *testing.T) {
	c := moePinnedCase{
		name: "advance", compute: tensor.MulMatComputeNativeTensorCore,
		hidden: 512, routerHidden: 512, intermediate: 128, experts: 48, divisor: 1, topK: 4,
		routing: tensor.MoERoutingSoftmax, activation: tensor.MoEActivationSiLU, normalize: true, scale: 1,
		banks: [3]dtype.Type{dtype.IQ4XS, dtype.IQ4XS, dtype.IQ4NL}, tokens: 1, offload: true,
	}
	cuda := newFixtureExecutor(t)
	output, host, device, _ := c.prepare(t, cuda, dtype.F32)
	banks := &countedBanks{ExpertBanks: cuda.resources.experts}
	if err := cuda.SetExpertBanks(banks); err != nil {
		t.Fatal(err)
	}
	compiled, inputs, err := compileDeviceFeeds([]*tensor.Tensor{output}, device)
	if err != nil {
		t.Fatal(err)
	}
	builder := tensor.NewBuilder()
	input := builder.Input("input", dtype.F32, tensor.MustShape(4))
	plain := builder.Scale(input, 2)
	for step, run := range []struct {
		experts  bool
		advances int
	}{{true, 1}, {true, 2}, {false, 3}, {true, 3}, {false, 4}, {false, 4}, {true, 4}} {
		if run.experts {
			_, err = cuda.ExecuteCompiled(t.Context(), compiled, host, inputs)
		} else {
			_, err = cuda.executeWithDeviceFeeds(t.Context(), []*tensor.Tensor{plain},
				map[*tensor.Tensor]reference.Value{input: {Shape: input.Shape, Data: []float32{1, 2, 3, 4}}}, nil)
		}
		if err != nil {
			t.Fatal(err)
		}
		if banks.advances != run.advances {
			t.Fatalf("run %d (experts %t): %d advances, want %d", step, run.experts, banks.advances, run.advances)
		}
	}
}
