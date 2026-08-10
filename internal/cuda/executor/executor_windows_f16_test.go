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

// TestExecutorF16MulMatMatchesReference: native-F16 residency serves matmul
// weights at 2 bytes. Decode (right rows == 1) reads F16 directly via the custom
// warp kernel; prefill (right rows > 1) upconverts the F16 weight to F32 and runs
// SGEMM. Both must match the F16-dequantized F32 reference; the greedy selection
// must be exact.
func TestExecutorF16MulMatMatchesReference(t *testing.T) {
	cudatest.Require(t)
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
	for _, testCase := range []struct {
		name      string
		rightRows uint64
	}{
		{"decode", 1},
		{"prefill", 3},
	} {
		t.Run(testCase.name, func(t *testing.T) {
			rightShape := tensor.MustShape(64, testCase.rightRows)
			rightValue := patternedValue(rightShape, 7, 0.05, 0.02)

			referenceBuilder := tensor.NewBuilder()
			referenceLeft := referenceBuilder.Input("left", dtype.F32, leftShape)
			referenceRight := referenceBuilder.Input("right", dtype.F32, rightShape)
			referenceOutput := referenceBuilder.MulMat(referenceLeft, referenceRight)
			want, err := reference.Execute([]*tensor.Tensor{referenceOutput}, map[*tensor.Tensor]reference.Value{
				referenceLeft: {Shape: leftShape, Data: dequantized}, referenceRight: rightValue,
			})
			if err != nil {
				t.Fatal(err)
			}

			builder := tensor.NewBuilder()
			left := builder.Input("left", dtype.F16, leftShape)
			right := builder.Input("right", dtype.F32, rightShape)
			output := builder.MulMat(left, right)
			worker := newFixtureWorker(t)
			pointer := copyFixtureDeviceBytes(t, worker, storage)
			cuda := newFixtureExecutorWithWorker(t, worker)
			got, err := cuda.ExecuteWithDeviceFeeds(context.Background(), []*tensor.Tensor{output},
				map[*tensor.Tensor]reference.Value{right: rightValue}, map[*tensor.Tensor]driver.DevicePtr{left: pointer})
			if err != nil {
				t.Fatal(err)
			}
			compare(t, got[output].Data, want[referenceOutput].Data, accuracyQuantized)
		})
	}

	// Greedy selection over the decode projection must be exact.
	rightShape := tensor.MustShape(64, 1)
	rightValue := patternedValue(rightShape, 7, 0.05, 0.02)
	referenceBuilder := tensor.NewBuilder()
	referenceLeft := referenceBuilder.Input("left", dtype.F32, leftShape)
	referenceRight := referenceBuilder.Input("right", dtype.F32, rightShape)
	referenceOutput := referenceBuilder.MulMat(referenceLeft, referenceRight)
	referenceSelection := referenceBuilder.TopK(referenceOutput, 1)
	wantSelection, err := reference.Execute([]*tensor.Tensor{referenceSelection}, map[*tensor.Tensor]reference.Value{
		referenceLeft: {Shape: leftShape, Data: dequantized}, referenceRight: rightValue,
	})
	if err != nil {
		t.Fatal(err)
	}
	builder := tensor.NewBuilder()
	left := builder.Input("left", dtype.F16, leftShape)
	right := builder.Input("right", dtype.F32, rightShape)
	selection := builder.TopK(builder.MulMat(left, right), 1)
	worker := newFixtureWorker(t)
	pointer := copyFixtureDeviceBytes(t, worker, storage)
	cuda := newFixtureExecutorWithWorker(t, worker)
	gotSelection, err := cuda.ExecuteWithDeviceFeeds(context.Background(), []*tensor.Tensor{selection},
		map[*tensor.Tensor]reference.Value{right: rightValue}, map[*tensor.Tensor]driver.DevicePtr{left: pointer})
	if err != nil {
		t.Fatal(err)
	}
	compare(t, gotSelection[selection].Data, wantSelection[referenceSelection].Data, accuracyExact)
}
