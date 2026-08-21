package hostmath

// FP64 accumulation; FP32/BF16 storage.

import (
	"fmt"
	"math"
)

// LinearF64: dst[rows,out] = x[rows,in] * w[out,in]^T (+bias), f64 accumulate.
func LinearF64(dst, x, w, bias []float32, rows, inDim, outDim int) {
	linearF64(dst, x, w, bias, rows, inDim, outDim, false)
}

func linearF64(dst, x, w, bias []float32, rows, inDim, outDim int, biasFirst bool) {
	if rows == 1 {
		parallelRangeCost(outDim, inDim, macF64, func(oLo, oHi int) {
			linearF64Cols(dst, x, w, bias, oLo, oHi, inDim, biasFirst)
		})
		return
	}
	parallelRangeCost(rows, inDim*outDim, macF64, func(rLo, rHi int) {
		for r := rLo; r < rHi; r++ {
			linearF64Cols(dst[r*outDim:(r+1)*outDim], x[r*inDim:(r+1)*inDim], w, bias, 0, outDim, inDim, biasFirst)
		}
	})
}

// LinearF64New allocates the LinearF64 destination.
func LinearF64New(x, w, bias []float32, rows, inDim, outDim int) []float32 {
	dst := make([]float32, rows*outDim)
	LinearF64(dst, x, w, bias, rows, inDim, outDim)
	return dst
}

// LinearF64BiasFirstNew rounds once after bias-first FP64 accumulation.
func LinearF64BiasFirstNew(x, w, bias []float32, rows, inDim, outDim int) []float32 {
	dst := make([]float32, rows*outDim)
	linearF64(dst, x, w, bias, rows, inDim, outDim, true)
	return dst
}

// LinearF64BiasFirstStrided writes one projection into strided output rows.
func LinearF64BiasFirstStrided(dst, x, w, bias []float32, rows, inDim, outDim, dstStride, dstOffset int) {
	parallelRangeCost(rows, inDim*outDim, macF64, func(rLo, rHi int) {
		for row := rLo; row < rHi; row++ {
			start := row*dstStride + dstOffset
			linearF64Cols(dst[start:start+outDim], x[row*inDim:(row+1)*inDim], w, bias, 0, outDim, inDim, true)
		}
	})
}

func linearF64Cols(dst, x, w, bias []float32, oLo, oHi, inDim int, biasFirst bool) {
	for o := oLo; o < oHi; o++ {
		wr := w[o*inDim : (o+1)*inDim]
		var acc float64
		if biasFirst && bias != nil {
			acc = float64(bias[o])
		}
		for i, v := range wr {
			acc += float64(x[i]) * float64(v)
		}
		if !biasFirst && bias != nil {
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

// ChannelMixF64Into: channel-major pointwise linear; FP64 accumulation.
func ChannelMixF64Into(dst, x, w, bias []float32, inChannels, outChannels, positions int) error {
	if len(x) != inChannels*positions || len(dst) != outChannels*positions {
		return fmt.Errorf("hostmath channel mix: bad lengths dst=%d x=%d", len(dst), len(x))
	}
	if len(w) != outChannels*inChannels {
		return fmt.Errorf("hostmath channel mix: weight len=%d want %d", len(w), outChannels*inChannels)
	}
	if bias != nil && len(bias) != outChannels {
		return fmt.Errorf("hostmath channel mix: bias len=%d want %d", len(bias), outChannels)
	}
	ParallelRangeF64(outChannels, inChannels*positions, func(lo, hi int) {
		for outChannel := lo; outChannel < hi; outChannel++ {
			for position := 0; position < positions; position++ {
				acc := 0.0
				if bias != nil {
					acc = float64(bias[outChannel])
				}
				for inChannel := 0; inChannel < inChannels; inChannel++ {
					acc += float64(x[inChannel*positions+position]) * float64(w[outChannel*inChannels+inChannel])
				}
				dst[outChannel*positions+position] = float32(acc)
			}
		}
	})
	return nil
}
