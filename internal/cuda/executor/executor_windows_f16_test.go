package executor

import (
	"testing"

	"overgo/internal/quant"
	"overgo/internal/tensor"
	"overgo/internal/tensor/dtype"
)

// TestExecutorF16MulMatMatchesReference: native-F16 residency serves matmul
// weights at 2 bytes. Decode (right rows == 1) reads F16 directly via the custom
// warp kernel; prefill (right rows > 1) upconverts the F16 weight to F32 and runs
// SGEMM. Both must match the F16-dequantized F32 reference; the greedy selection
// must be exact.
func TestExecutorF16MulMatMatchesReference(t *testing.T) {
	leftShape := tensor.MustShape(64, 19)
	leftValue := patternedValue(leftShape, 11, 0.03, -0.1)
	storage, err := quant.Quantize(dtype.F16, leftValue.Data)
	if err != nil {
		t.Fatal(err)
	}
	dequantized, err := quant.Dequantize(dtype.F16, storage, uint64(len(leftValue.Data)))
	if err != nil {
		t.Fatal(err)
	}
	checkResidentMulMat(t, dtype.F16, leftShape, storage, dequantized)
}
