package executor

import (
	"testing"

	"overgo/internal/quant"
	"overgo/internal/tensor"
	"overgo/internal/tensor/dtype"
)

// TestExecutorBF16MulMatMatchesReference: native-BF16 residency serves matmul
// weights at 2 bytes through the SAME native-dtype mechanism as F16. Decode
// (right rows == 1) reads BF16 directly via the custom warp kernel; prefill
// (right rows > 1) upconverts the BF16 weight to F32 (bits<<16, the exact
// resident F32-copy dequant) and runs the identical SGEMM. Both must match the
// BF16-dequantized F32 reference; the greedy selection must be exact.
func TestExecutorBF16MulMatMatchesReference(t *testing.T) {
	leftShape := tensor.MustShape(64, 19)
	leftValue := patternedValue(leftShape, 11, 0.03, -0.1)
	storage, err := quant.Quantize(dtype.BF16, leftValue.Data)
	if err != nil {
		t.Fatal(err)
	}
	dequantized, err := quant.Dequantize(dtype.BF16, storage, uint64(len(leftValue.Data)))
	if err != nil {
		t.Fatal(err)
	}
	checkResidentMulMat(t, dtype.BF16, leftShape, storage, dequantized)
}
