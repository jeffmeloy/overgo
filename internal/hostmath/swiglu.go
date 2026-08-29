package hostmath

import "math"

// Clamped SwiGLU (DeepSeek-V4 section 4.2.3 training-stability form).
//
// Ordinary SwiGLU is silu(a) * b for a fused [2*hidden] projection split in
// half. The clamped form bounds each branch before the product:
//
//	a <- clamp(a, -SwiGLUClampLinear, +SwiGLUClampLinear)   (linear branch)
//	b <- min(b, SwiGLUClampGate)                            (gate branch, upper only)
//
// The asymmetry is not an oversight: the linear branch passes through silu and
// is bounded on both sides, while the gate branch multiplies and only its
// POSITIVE tail can make the product explode. Reproducing the asymmetry exactly
// is required for checkpoint parity, so it is expressed as two named constants
// rather than one. Neutral name on purpose: this is a published FFN variant.
const (
	// SwiGLUClampLinear bounds the silu branch symmetrically.
	SwiGLUClampLinear = 10.0
	// SwiGLUClampGate bounds the multiplicative branch from ABOVE only.
	SwiGLUClampGate = 10.0
)

// SwiGLUClampedInto writes silu(clamp(a)) * min(b, gate) into out.
//
// fused holds rows x (2*hidden) values: the first hidden columns of each row
// are the linear branch, the second hidden columns the gate branch -- the
// layout produced by a single [2*hidden, d] projection split with chunk(2).
// out holds rows x hidden values and may not alias fused.
func SwiGLUClampedInto(out, fused []float32, rows, hidden int) {
	for r := range rows {
		a := fused[r*2*hidden : r*2*hidden+hidden]
		b := fused[r*2*hidden+hidden : (r+1)*2*hidden]
		o := out[r*hidden : (r+1)*hidden]
		for i := range hidden {
			av := float64(a[i])
			if av > SwiGLUClampLinear {
				av = SwiGLUClampLinear
			} else if av < -SwiGLUClampLinear {
				av = -SwiGLUClampLinear
			}
			bv := float64(b[i])
			bv = min(bv, SwiGLUClampGate)
			// silu(x) = x*sigmoid(x), written so the clamped range never
			// evaluates exp on a large positive argument.
			o[i] = float32(av / (1.0 + math.Exp(-av)) * bv)
		}
	}
}
