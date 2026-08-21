// Adaptive shift/scale modulation (AdaLN chunks): y = x*(1+scale) + shift
// with the [d] shift/scale rows broadcast over tokens, and its VJP. The
// caller owns the chunk table layout (which [d] slice of a modulation vector
// each call consumes) and any residual-gate arithmetic around it.
package hostmath

// AdaptiveShiftScale: dst = x*(1+scale) + shift per row; shift/scale are [d]
// vectors broadcast over rows. dst may alias x.
func AdaptiveShiftScale(dst, x, shift, scale []float32, rows, d int) {
	for r := 0; r < rows; r++ {
		xr := x[r*d : (r+1)*d]
		dr := dst[r*d : (r+1)*d]
		for i := 0; i < d; i++ {
			dr[i] = xr[i]*(1+scale[i]) + shift[i]
		}
	}
}

// AdaptiveShiftScaleBackward: VJP of AdaptiveShiftScale. Writes
// dx = dy*(1+scale) (set), and accumulates the broadcast reductions
// dShift += sum_rows dy and dScale += sum_rows dy*x (either may be nil to
// skip). x is the modulation INPUT (the normalized rows). dx may alias dy.
func AdaptiveShiftScaleBackward(dx, dShift, dScale, x, scale, dy []float32, rows, d int) {
	for r := 0; r < rows; r++ {
		xr := x[r*d : (r+1)*d]
		dyr := dy[r*d : (r+1)*d]
		for i := 0; i < d; i++ {
			g := dyr[i]
			if dShift != nil {
				dShift[i] += g
			}
			if dScale != nil {
				dScale[i] += g * xr[i]
			}
			if dx != nil {
				dx[r*d+i] = g * (1 + scale[i])
			}
		}
	}
}
