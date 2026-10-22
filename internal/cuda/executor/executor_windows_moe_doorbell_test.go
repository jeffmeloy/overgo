//go:build windows

package executor

import (
	"math"
	"sync/atomic"
	"testing"

	"overgo/internal/tensor"
	"overgo/internal/tensor/dtype"
)

// TestDecodeHostExpertsRingTheDoorbell holds a captured decode step's experts
// in host memory, computed on the CPU while the card computes the rest
// (moeDoorbell), to the host reference: Flash-Next's IQ4_XS gate and up with
// IQ4_NL down, apart and fused, for a decode token, a verify span of three
// that reads shared experts once and one of eight that does not, the card
// holding expert 0 of each bank. The step's first run names its experts on
// host and the second hands them over: the CPU's outputs stay within its
// activations' rounding of the reference, as the prompt's do.
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
			for range 2 {
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
