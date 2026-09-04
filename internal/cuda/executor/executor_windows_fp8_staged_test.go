//go:build windows

package executor

import (
	"encoding/binary"
	"math"
	"testing"

	"overgo/internal/tensor"
	"overgo/internal/tensor/dtype"
)

// TestExecutorFp8StagedMulMatMatchesReference: past the column floor
// under the native policy, native fp8 weights prefill through f16
// staging with the row scale folded and the tensor-core GEMM, and hold
// the measured half-staged bound against the exact e4m3-times-scale F32
// reference.
func TestExecutorFp8StagedMulMatMatchesReference(t *testing.T) {
	const (
		inner   = 256
		rows    = 19
		columns = 96
	)
	leftShape := tensor.MustShape(inner, rows)
	scales := make([]float32, rows)
	dequantized := make([]float32, rows*inner)
	storage := make([]byte, rows*inner+rows*4)
	for row := range rows {
		scales[row] = 0.125 * float32(1+row%7)
		for column := range inner {
			b := fp8PatternByte(row*inner + column)
			storage[row*inner+column] = b
			dequantized[row*inner+column] = dtype.F8E4M3ToFloat32(b) * scales[row]
		}
		binary.LittleEndian.PutUint32(storage[rows*inner+row*4:], math.Float32bits(scales[row]))
	}
	rightValue := patternedValue(tensor.MustShape(inner, columns), 7, 0.05, 0.02)
	checkResidentBinaryGraphWithPolicy(
		t, dtype.F8E4M3, leftShape, storage, dequantized, rightValue,
		func(builder *tensor.Builder, left, right *tensor.Tensor) *tensor.Tensor {
			return builder.MulMat(left, right)
		},
		accuracyHalfStaged,
		func(builder *tensor.Builder) { builder.SetMulMatCompute(tensor.MulMatComputeNativeTensorCore) },
	)
}
