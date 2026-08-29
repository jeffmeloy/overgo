package hostmath

import "math"

// L2NormForward matches the l2_norm_f32 kernel: per row (length width), divide by
// the L2 norm floored at eps -- y = x / max(sqrt(sum(x^2)), eps). x is [rows,width]
// row-major. qwen3.5 L2-normalizes q and k per head (width=stateWidth) before the
// gated-delta recurrence.
func L2NormForward(x []float32, rows, width int, eps float64) []float32 {
	out := make([]float32, len(x))
	for r := range rows {
		row := x[r*width : r*width+width]
		var ss float64
		for _, v := range row {
			ss += float64(v) * float64(v)
		}
		inv := 1.0 / max(math.Sqrt(ss), eps)
		for c, v := range row {
			out[r*width+c] = float32(float64(v) * inv)
		}
	}
	return out
}

// L2NormBackward is the VJP of L2NormForward. With inv = 1/max(sqrt(sum(x^2)),eps):
// in the unclamped region (norm>eps) dX = inv*dY - inv^3 * x * (dY·x); in the
// clamped region inv is constant so dX = inv*dY.
func L2NormBackward(x, dY []float32, rows, width int, eps float64) []float32 {
	dX := make([]float32, len(x))
	for r := range rows {
		row := x[r*width : r*width+width]
		var ss float64
		for _, v := range row {
			ss += float64(v) * float64(v)
		}
		norm := math.Sqrt(ss)
		inv := 1.0 / max(norm, eps)
		clamped := norm <= eps
		var dot float64
		if !clamped {
			for c, v := range row {
				dot += float64(dY[r*width+c]) * float64(v)
			}
		}
		inv3 := inv * inv * inv
		for c, v := range row {
			g := inv * float64(dY[r*width+c])
			if !clamped {
				g -= inv3 * float64(v) * dot
			}
			dX[r*width+c] = float32(g)
		}
	}
	return dX
}
