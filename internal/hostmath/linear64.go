package hostmath

// f64-accumulate linears for bf16-disciplined reference paths (weights and
// activations are bf16-rounded f32; the dot itself accumulates in f64, which
// is the reference oracle's contract). Storage variants share one kernel
// shape; the bf16 variant expands uint16 bf16 weights inline (exact).

import "math"

// LinearF64: dst[rows,out] = x[rows,in] * w[out,in]^T (+bias), f64 accumulate.
func LinearF64(dst, x, w, bias []float32, rows, inDim, outDim int) {
	if rows == 1 {
		parallelRangeCost(outDim, inDim, macF64, func(oLo, oHi int) {
			linearF64Cols(dst, x, w, bias, oLo, oHi, inDim)
		})
		return
	}
	parallelRangeCost(rows, inDim*outDim, macF64, func(rLo, rHi int) {
		for r := rLo; r < rHi; r++ {
			linearF64Cols(dst[r*outDim:(r+1)*outDim], x[r*inDim:(r+1)*inDim], w, bias, 0, outDim, inDim)
		}
	})
}

func linearF64Cols(dst, x, w, bias []float32, oLo, oHi, inDim int) {
	for o := oLo; o < oHi; o++ {
		wr := w[o*inDim : (o+1)*inDim]
		var acc float64
		for i, v := range wr {
			acc += float64(x[i]) * float64(v)
		}
		if bias != nil {
			acc += float64(bias[o])
		}
		dst[o] = float32(acc)
	}
}

// LinearBF16F64: LinearF64 with row-major bf16 (uint16) weight storage.
func LinearBF16F64(dst, x []float32, w []uint16, bias []float32, rows, inDim, outDim int) {
	if rows == 1 {
		parallelRangeCost(outDim, inDim, macF64, func(oLo, oHi int) {
			linearBF16F64Cols(dst, x, w, bias, oLo, oHi, inDim)
		})
		return
	}
	parallelRangeCost(rows, inDim*outDim, macF64, func(rLo, rHi int) {
		for r := rLo; r < rHi; r++ {
			linearBF16F64Cols(dst[r*outDim:(r+1)*outDim], x[r*inDim:(r+1)*inDim], w, bias, 0, outDim, inDim)
		}
	})
}

func linearBF16F64Cols(dst, x []float32, w []uint16, bias []float32, oLo, oHi, inDim int) {
	for o := oLo; o < oHi; o++ {
		wr := w[o*inDim : (o+1)*inDim]
		var acc float64
		for i, v := range wr {
			acc += float64(x[i]) * float64(math.Float32frombits(uint32(v)<<16))
		}
		if bias != nil {
			acc += float64(bias[o])
		}
		dst[o] = float32(acc)
	}
}
