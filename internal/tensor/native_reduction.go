package tensor

import (
	"fmt"
	"math"
	"math/bits"
)

// Native reduction constants follow Torch 2.12 Reduce.cuh and ReduceSumProdKernel.cu.
// Reconsider vector widths and launch limits together when changing source policy.
const (
	nativeReductionF32Vector = 4
	// NativeReductionBF16Vector is Torch 2.12 sum's BF16 input-vector width.
	// Reconsider it with ReduceSumProdKernel.cu's input_vec_size contract.
	NativeReductionBF16Vector    = 8
	nativeReductionWarp          = 32
	nativeReductionThreads       = 512
	nativeReductionMinimumValues = 16
	nativeReductionMaximumValues = 256
)

// NativeReductionGeometry derives source reduction lanes and cooperating rows.
// It refuses unaligned input and reductions requiring multiple CUDA blocks.
func NativeReductionGeometry(shape Shape, bf16 bool) (uint32, uint32, error) {
	width := shape.Dims[0]
	rows := uint64(1)
	for i := 1; i < int(shape.Rank); i++ {
		if shape.Dims[i] == 0 || rows > math.MaxUint32/shape.Dims[i] {
			return 0, 0, fmt.Errorf("native reduction row count exceeds positive CUDA limits")
		}
		rows *= shape.Dims[i]
	}
	vector := uint64(nativeReductionF32Vector)
	if bf16 {
		vector = NativeReductionBF16Vector
	}
	if width < nativeReductionWarp*nativeReductionF32Vector || width%vector != 0 || width > math.MaxUint32 {
		return 0, 0, fmt.Errorf("native reduction width %d requires CUDA-sized aligned vectorized rows of at least %d values", width, nativeReductionWarp*nativeReductionF32Vector)
	}
	power := func(v uint64) uint32 { return 1 << (bits.Len64(v) - 1) }
	x := min(power(width/vector), uint32(nativeReductionWarp))
	y := min(power(rows), uint32(nativeReductionThreads)/x)
	x = min(power(width/vector), uint32(nativeReductionThreads)/y)
	if 1+(width-1)/uint64(x) < min(uint64(y)*nativeReductionMinimumValues, uint64(nativeReductionMaximumValues)) {
		y = 1
	}
	if 1+(width-1)/uint64(x*y) >= nativeReductionMaximumValues {
		return 0, 0, fmt.Errorf("native reduction width %d requires unsupported cross-block reduction", width)
	}
	return x, y, nil
}
