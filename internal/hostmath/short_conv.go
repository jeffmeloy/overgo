package hostmath

import "math"

// ShortConvForward is the qwen3.5 GDN-mix short convolution: a depthwise causal
// 1-D conv (stride 1, left-pad k-1) followed by SiLU -- SiLU(SSMConv(x)). x and
// the output are [channels, T] channel-major; w is [channels, k] (one kernel per
// channel); bias may be nil. This is the conv the recurrence's q/k/v pass through.
func ShortConvForward(x []float32, channels, T int, w, bias []float32, k int) []float32 {
	pre := shortConvPre(x, channels, T, w, bias, k)
	for i, v := range pre {
		pre[i] = float32(float64(v) / (1 + math.Exp(-float64(v))))
	}
	return pre
}

// shortConvPre is the pre-activation depthwise causal conv (no SiLU).
func shortConvPre(x []float32, channels, T int, w, bias []float32, k int) []float32 {
	out := make([]float32, channels*T)
	for c := range channels {
		xRow := x[c*T:]
		wRow := w[c*k:]
		for t := range T {
			var acc float64
			for j := range k {
				ti := t - (k - 1) + j
				if ti < 0 || ti >= T {
					continue
				}
				acc += float64(xRow[ti]) * float64(wRow[j])
			}
			if bias != nil {
				acc += float64(bias[c])
			}
			out[c*T+t] = float32(acc)
		}
	}
	return out
}

// ShortConvBackward is the VJP of ShortConvForward: given the output cotangent
// dY [channels, T], returns dX, dW (per-channel kernel grads [channels, k]) and
// dBias ([channels], nil when bias is nil). Backprops through SiLU then the
// depthwise causal conv (each channel independent).
func ShortConvBackward(x, dY []float32, channels, T int, w, bias []float32, k int) (dX, dW, dBias []float32) {
	pre := shortConvPre(x, channels, T, w, bias, k)
	dConv := make([]float64, channels*T)
	for i := range pre {
		v := float64(pre[i])
		s := 1 / (1 + math.Exp(-v))
		dConv[i] = float64(dY[i]) * s * (1 + v*(1-s)) // silu'(v)
	}
	dX = make([]float32, channels*T)
	dW = make([]float32, channels*k)
	if bias != nil {
		dBias = make([]float32, channels)
	}
	for c := range channels {
		xRow := x[c*T:]
		wRow := w[c*k:]
		dxRow := make([]float64, T)
		dwRow := make([]float64, k)
		var db float64
		for t := range T {
			dc := dConv[c*T+t]
			db += dc
			for j := range k {
				ti := t - (k - 1) + j
				if ti < 0 || ti >= T {
					continue
				}
				dxRow[ti] += dc * float64(wRow[j])
				dwRow[j] += dc * float64(xRow[ti])
			}
		}
		for t := range T {
			dX[c*T+t] = float32(dxRow[t])
		}
		for j := range k {
			dW[c*k+j] = float32(dwRow[j])
		}
		if bias != nil {
			dBias[c] = float32(db)
		}
	}
	return dX, dW, dBias
}
