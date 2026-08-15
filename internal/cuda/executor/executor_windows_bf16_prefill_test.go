package executor

import (
	"context"
	"testing"

	"overgo/internal/cuda/driver"
	cudatest "overgo/internal/cuda/testutil"
	"overgo/internal/quant"
	"overgo/internal/tensor"
	"overgo/internal/tensor/dtype"
	"overgo/internal/tensor/reference"
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

func TestExecutorBF16TensorCoreMulMatMatchesRoundedReference(t *testing.T) {
	cudatest.Require(t)
	leftShape, rightShape := tensor.MustShape(64, 37), tensor.MustShape(64, 29)
	leftValue := patternedValue(leftShape, 11, 0.03, -0.1)
	rightValue := patternedValue(rightShape, 7, 0.05, 0.02)
	leftStorage, err := quant.Quantize(dtype.BF16, leftValue.Data)
	if err != nil {
		t.Fatal(err)
	}
	leftRounded, err := quant.Dequantize(dtype.BF16, leftStorage, uint64(len(leftValue.Data)))
	if err != nil {
		t.Fatal(err)
	}
	rightStorage, err := quant.Quantize(dtype.BF16, rightValue.Data)
	if err != nil {
		t.Fatal(err)
	}
	rightRounded, err := quant.Dequantize(dtype.BF16, rightStorage, uint64(len(rightValue.Data)))
	if err != nil {
		t.Fatal(err)
	}

	referenceBuilder := tensor.NewBuilder()
	referenceLeft := referenceBuilder.Input("left", dtype.F32, leftShape)
	referenceRight := referenceBuilder.Input("right", dtype.F32, rightShape)
	referenceOutput := referenceBuilder.MulMat(referenceLeft, referenceRight)
	want, err := reference.Execute(
		[]*tensor.Tensor{referenceOutput},
		map[*tensor.Tensor]reference.Value{
			referenceLeft:  {Shape: leftShape, Data: leftRounded},
			referenceRight: {Shape: rightShape, Data: rightRounded},
		},
	)
	if err != nil {
		t.Fatal(err)
	}

	builder := tensor.NewBuilder()
	left := builder.Input("left", dtype.BF16, leftShape)
	right := builder.Input("right", dtype.F32, rightShape)
	output := builder.MulMatWithCompute(left, right, tensor.MulMatComputeBF16TensorCore)
	worker := newFixtureWorker(t)
	pointer := copyFixtureDeviceBytes(t, worker, leftStorage)
	cuda := newFixtureExecutorWithWorker(t, worker)
	got, err := cuda.executeWithDeviceFeeds(
		context.Background(), []*tensor.Tensor{output},
		map[*tensor.Tensor]reference.Value{right: rightValue},
		map[*tensor.Tensor]driver.DevicePtr{left: pointer},
	)
	if err != nil {
		t.Fatal(err)
	}
	compare(t, got[output].Data, want[referenceOutput].Data, accuracyProjection)
}
