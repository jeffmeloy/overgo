// Causal 1-D convolution primitives (mimi/SEANet codec stacks). Channel-major
// [c][T] layouts, PyTorch weight order, f64 accumulation stored f32. Output
// channels are independent and each keeps its serial accumulation order, so
// the parallel channel split is bit-identical to the serial loop.
package hostmath

import "math"

// CausalConv1dSuffix excludes outputs before skip while retaining their input history.
// Input and output are channel-major and weights use [cOut][cIn][k].
// Left padding is k-stride zeros and T must be divisible by stride.
// A zero skip decodes the complete input and bias may be nil.
func CausalConv1dSuffix(x []float32, cIn, T int, w, bias []float32, cOut, k, stride, skip int) []float32 {
	leftPad := k - stride
	outT := T/stride - skip
	out := make([]float32, cOut*outT)
	// outT*cIn*k MACs per output channel.
	parallelRangeCost(cOut, outT*cIn*k, macF64, func(coLo, coHi int) {
		causalConv1dChannels(out, x, w, bias, cIn, T, outT, k, stride, leftPad-skip*stride, coLo, coHi)
	})
	return out
}

// causalConv1dChannels: output channels [coLo,coHi) of CausalConv1dSuffix — the
// serial kernel the dispatch calibration times.
func causalConv1dChannels(out, x, w, bias []float32, cIn, T, outT, k, stride, leftPad, coLo, coHi int) {
	for co := coLo; co < coHi; co++ {
		for ot := range outT {
			start := ot*stride - leftPad
			var acc float64
			for ci := range cIn {
				xRow := x[ci*T:]
				wRow := w[(co*cIn+ci)*k:]
				for j := range k {
					ti := start + j
					if ti < 0 || ti >= T {
						continue // zero padding
					}
					acc += float64(xRow[ti]) * float64(wRow[j])
				}
			}
			if bias != nil {
				acc += float64(bias[co])
			}
			out[co*outT+ot] = float32(acc)
		}
	}
}

// ConvTranspose1dSuffix excludes outputs before skip and trims the trailing overlap.
// Accumulation keeps ascending input channel, time and kernel order.
// Input and output are channel-major and weights use [cIn][cOut/groups][k].
// A zero skip decodes the complete input and bias may be nil.
func ConvTranspose1dSuffix(x []float32, cIn, T int, w, bias []float32, cOut, k, stride, groups, skip int) []float32 {
	outT := T*stride - skip
	cInPerG := cIn / groups
	cOutPerG := cOut / groups
	out := make([]float32, cOut*outT)
	// cInPerG*T*k MACs per output channel; the outT accumulate-copy is
	// lower-order (outT = T*stride <= T*k).
	parallelRangeCost(cOut, cInPerG*T*k, macF64, func(coLo, coHi int) {
		acc := make([]float64, outT)
		for co := coLo; co < coHi; co++ {
			clear(acc)
			g := co / cOutPerG
			cog := co % cOutPerG
			for cig := range cInPerG {
				ci := g*cInPerG + cig
				xRow := x[ci*T:]
				wRow := w[(ci*cOutPerG+cog)*k:]
				for t := range T {
					xv := float64(xRow[t])
					if xv == 0 {
						continue
					}
					base := t*stride - skip
					for j := max(0, -base); j < min(k, outT-base); j++ {
						acc[base+j] += xv * float64(wRow[j])
					}
				}
			}
			var b float64
			if bias != nil {
				b = float64(bias[co])
			}
			for t := range outT {
				out[co*outT+t] = float32(acc[t] + b)
			}
		}
	})
	return out
}

// ELUInPlace: ELU with alpha=1: x -> x if x>0 else exp(x)-1.
func ELUInPlace(x []float32) {
	for i, v := range x {
		if v < 0 {
			x[i] = float32(math.Exp(float64(v)) - 1)
		}
	}
}
