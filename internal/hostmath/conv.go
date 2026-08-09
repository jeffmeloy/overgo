// Causal 1-D convolution primitives (mimi/SEANet codec stacks). Channel-major
// [c][T] layouts, PyTorch weight order, f64 accumulation stored f32. Output
// channels are independent and each keeps its serial accumulation order, so
// the parallel channel split is bit-identical to the serial loop.
package hostmath

import "math"

// CausalConv1d: fully causal 1-D convolution. x is [cIn][T] channel-major;
// w is the PyTorch Conv1d layout [cOut][cIn][k]; bias may be nil. Left pad
// is k-stride zeros (the reference streaming invariant: T divisible by
// stride). Returns [cOut][T/stride].
func CausalConv1d(x []float32, cIn, T int, w, bias []float32, cOut, k, stride int) []float32 {
	leftPad := k - stride
	outT := T / stride
	out := make([]float32, cOut*outT)
	parallelRange(cOut, func(coLo, coHi int) {
		for co := coLo; co < coHi; co++ {
			for ot := 0; ot < outT; ot++ {
				start := ot*stride - leftPad
				var acc float64
				for ci := 0; ci < cIn; ci++ {
					xRow := x[ci*T:]
					wRow := w[(co*cIn+ci)*k:]
					for j := 0; j < k; j++ {
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
	})
	return out
}

// ConvTranspose1dTrim: 1-D transposed convolution, trimming the trailing
// k-stride samples (the reference streaming carry). x is [cIn][T]
// channel-major; w is the PyTorch ConvTranspose1d layout [cIn][cOut/groups][k];
// bias may be nil. Returns [cOut][T*stride]. For a fixed co the contributions
// land in ascending (ci, t, j) order — the serial addition sequence.
func ConvTranspose1dTrim(x []float32, cIn, T int, w, bias []float32, cOut, k, stride, groups int) []float32 {
	fullT := (T-1)*stride + k
	outT := T * stride
	cInPerG := cIn / groups
	cOutPerG := cOut / groups
	out := make([]float32, cOut*outT)
	parallelRange(cOut, func(coLo, coHi int) {
		acc := make([]float64, fullT)
		for co := coLo; co < coHi; co++ {
			clear(acc)
			g := co / cOutPerG
			cog := co % cOutPerG
			for cig := 0; cig < cInPerG; cig++ {
				ci := g*cInPerG + cig
				xRow := x[ci*T:]
				wRow := w[(ci*cOutPerG+cog)*k:]
				for t := 0; t < T; t++ {
					xv := float64(xRow[t])
					if xv == 0 {
						continue
					}
					base := t * stride
					for j := 0; j < k; j++ {
						acc[base+j] += xv * float64(wRow[j])
					}
				}
			}
			var b float64
			if bias != nil {
				b = float64(bias[co])
			}
			for t := 0; t < outT; t++ {
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
