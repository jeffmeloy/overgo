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

// The native tensor-core policy multiplies an F16 weight in F16 with the
// activation rounded to F16 and F32 accumulation: the result matches the
// F32 reference computed over the same rounded operands.
func TestExecutorF16NativeTensorCoreMulMatMatchesRoundedReference(t *testing.T) {
	cudatest.Require(t)
	for _, weightType := range []dtype.Type{dtype.F16, dtype.BF16} {
		t.Run(weightType.String(), func(t *testing.T) {
			leftShape, rightShape := tensor.MustShape(64, 37), tensor.MustShape(64, 29)
			leftValue := patternedValue(leftShape, 11, 0.03, -0.1)
			rightValue := patternedValue(rightShape, 7, 0.05, 0.02)
			leftStorage, err := quant.Quantize(weightType, leftValue.Data)
			if err != nil {
				t.Fatal(err)
			}
			leftRounded, err := quant.Dequantize(weightType, leftStorage, uint64(len(leftValue.Data)))
			if err != nil {
				t.Fatal(err)
			}
			rightStorage, err := quant.Quantize(weightType, rightValue.Data)
			if err != nil {
				t.Fatal(err)
			}
			rightRounded, err := quant.Dequantize(weightType, rightStorage, uint64(len(rightValue.Data)))
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
			builder.SetMulMatCompute(tensor.MulMatComputeNativeTensorCore)
			left := builder.Input("left", weightType, leftShape)
			right := builder.Input("right", dtype.F32, rightShape)
			output := builder.MulMat(left, right)
			// An F32 x F32 mul_mat under the same policy stays exact.
			scores := builder.MulMat(right, right)
			if attributes, ok := scores.Attrs.(tensor.MulMatAttributes); ok && attributes.Compute != tensor.MulMatComputeExact {
				t.Fatalf("F32 mul_mat took the native tensor-core policy: %+v", attributes)
			}
			worker := newFixtureWorker(t)
			pointer := copyFixtureDeviceBytes(t, worker, leftStorage)
			cuda := newFixtureExecutorWithWorker(t, worker)
			got, err := cuda.executeWithDeviceFeeds(
				context.WithoutCancel(t.Context()), []*tensor.Tensor{output},
				map[*tensor.Tensor]reference.Value{right: rightValue},
				map[*tensor.Tensor]driver.DevicePtr{left: pointer},
			)
			if err != nil {
				t.Fatal(err)
			}
			compare(t, got[output].Data, want[referenceOutput].Data, accuracyProjection)
		})
	}
}
