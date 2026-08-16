package executor

import (
	"errors"
	"math"

	"overgo/internal/tensor"
	"overgo/internal/tensor/dtype"
)

// ReserveExactWeightStaging raises this graph's shared mul_mat staging
// reservation to the full F32 footprint of every native-dtype weight it
// stages, so each multi-token F16/BF16/FP8 mul_mat upconverts in one piece
// and runs one SGEMM instead of the bounded row-tiled loop.
//
// The bounded tiling (nativeWeightStagingLimitBytes) exists for model-scale
// weights whose full F32 image cannot be reserved; splitting the SGEMM row
// dimension changes cuBLAS kernel selection and therefore the accumulated
// rounding of every staged product. Graphs whose committed oracles were
// recorded against single-call staging numerics call this after Compile to
// keep that evidence bit-reproducible; graphs without such evidence keep the
// bounded default.
func (c *CompiledGraph) ReserveExactWeightStaging() error {
	if c == nil {
		return errors.New("CUDA compiled graph is unavailable")
	}
	for _, node := range c.order {
		if node.Op != tensor.OpMulMat {
			continue
		}
		weightType := node.Inputs[0].Type
		if weightType != dtype.F16 && weightType != dtype.BF16 && weightType != dtype.F8E4M3 {
			continue
		}
		if node.Inputs[1].Shape.Dims[1] == 1 {
			// Single-token decode reads the native weight directly; no staging.
			continue
		}
		if attributes, ok := node.Attrs.(tensor.MulMatAttributes); ok &&
			attributes.Compute == tensor.MulMatComputeBF16TensorCore {
			// Tensor-core mul_mat stages the BF16 activation, not the weight,
			// and is already reserved at full extent by Compile.
			continue
		}
		inner, rows := node.Inputs[0].Shape.Dims[0], node.Inputs[0].Shape.Dims[1]
		if inner == 0 || rows > math.MaxUint64/inner || inner*rows > math.MaxUint64/4 {
			return errors.New("exact weight staging size overflows")
		}
		c.matmulStagingBytes = max(c.matmulStagingBytes, inner*rows*4)
	}
	return nil
}
