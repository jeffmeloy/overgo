//go:build windows

package executor

import (
	"encoding/binary"
	"math"
	"testing"

	"overgo/internal/cuda/device"
	"overgo/internal/tensor"
	"overgo/internal/tensor/dtype"
)

// TestPromptHostExpertsComputeOnTheCPU holds a prompt's experts in host memory
// that few routed rows reach, computed on the CPU where they lie, to the host
// reference: 48 experts (16 of IQ3_S) of each of Flash-Next's layers' formats, IQ3_S gate
// and up with IQ4_NL down and IQ4_XS with Q8_0, and of IQ4_XS with IQ4_NL
// apart and fused, 64 tokens' 128 routed rows, under three an
// expert, the card holding expert 0. The CPU's outputs are placed in the
// slots by moe_host_slots_f32 and stay within its activations' rounding of
// the reference.
func TestPromptHostExpertsComputeOnTheCPU(t *testing.T) {
	// Serial: device fixtures share the one card.
	const (
		// The card keeps expert 0 of each bank beside the tables of a pointer an expert.
		pointerBytes, slotAlignment = 8, 256
		// hostShareLargest and hostShareMean bound the largest and the mean difference from the reference as
		// shares of the largest value.
		hostShareLargest, hostShareMean = 0.02, 0.005
	)
	for _, layout := range []struct {
		name                          string
		banks                         [3]dtype.Type
		fused                         bool
		hidden, intermediate, experts uint64
	}{
		{"iq4_xs-iq4_nl", [3]dtype.Type{dtype.IQ4XS, dtype.IQ4XS, dtype.IQ4NL}, false, 512, 128, 48},
		{"fused-iq4_xs-iq4_nl", [3]dtype.Type{dtype.IQ4XS, dtype.IQ4XS, dtype.IQ4NL}, true, 512, 128, 48},
		// Quantizing IQ3_S weights takes seconds a million, so its layer is the smallest.
		{"iq3_s-iq4_nl", [3]dtype.Type{dtype.IQ3S, dtype.IQ3S, dtype.IQ4NL}, false, 256, 64, 16},
		{"iq4_xs-q8_0", [3]dtype.Type{dtype.IQ4XS, dtype.IQ4XS, dtype.Q8_0}, false, 512, 128, 48},
	} {
		banks, fused := layout.banks, layout.fused
		t.Run(layout.name, func(t *testing.T) {
			c := moePinnedCase{
				name: "host-cpu", compute: tensor.MulMatComputeNativeTensorCore,
				hidden: layout.hidden, routerHidden: layout.hidden, intermediate: layout.intermediate, experts: layout.experts, divisor: 1, topK: 2,
				routing: tensor.MoERoutingSoftmax, activation: tensor.MoEActivationSiLU, normalize: true, scale: 1.25,
				banks: banks, fused: fused, tokens: 64,
			}
			shapes := []tensor.Shape{tensor.MustShape(c.hidden, c.intermediate), tensor.MustShape(c.hidden, c.intermediate), tensor.MustShape(c.intermediate, c.hidden)}
			if fused {
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
			trace := &device.LaunchTrace{}
			if err := cuda.worker.Do(t.Context(), func(state *device.State) error { state.Trace = trace; return nil }); err != nil {
				t.Fatal(err)
			}
			c.offload, c.limit, c.reference = true, limit, false
			got, _ := c.run(t, cuda, dtype.F32)
			difference, at := maxAbsDifference(t, got, want)
			var scale, mean float64
			for index := range want {
				scale = max(scale, math.Abs(float64(want[index])))
				mean += math.Abs(float64(got[index] - want[index]))
			}
			// The CPU quantizes each token's activations as Q8_K and each expert's intermediate ones as Q8_0, as
			// llama.cpp's CPU backend does: 0.2% of the largest value on average and 1.3% at most here, where a
			// row placed at another slot or taken by another expert misses by the value itself.
			if mean /= float64(len(want)); difference > hostShareLargest*scale || mean > hostShareMean*scale {
				t.Fatalf("largest difference from the reference %g at %d (value %g), mean %g, largest value %g", difference, at, want[at], mean, scale)
			}
			placed := false
			for offset := 0; offset+8 <= len(trace.Data()); offset += 4 {
				placed = placed || binary.LittleEndian.Uint64(trace.Data()[offset:]) == uint64(kernelMoeHostSlotsF32)
			}
			if !placed {
				t.Fatal("no CPU outputs were placed in the slots")
			}
		})
	}
}
