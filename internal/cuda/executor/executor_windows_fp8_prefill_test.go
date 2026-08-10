package executor

import (
	"context"
	"encoding/binary"
	"math"
	"testing"

	"overgo/internal/cuda/driver"
	cudatest "overgo/internal/cuda/testutil"
	"overgo/internal/tensor"
	"overgo/internal/tensor/dtype"
	"overgo/internal/tensor/reference"
)

// fp8PatternByte: deterministic finite e4m3 byte with broad coverage (both
// signs, mantissas 0..7, exponents 0..9 -- includes subnormals, excludes the
// sole NaN code 0x7f/0xff and the largest exponents so magnitudes stay moderate).
func fp8PatternByte(index int) byte {
	value := (index*61 + 17) & 0xff
	exponent := (value >> 3) & 0x0f
	if exponent > 9 {
		exponent %= 10
	}
	return byte((value & 0x80) | (exponent << 3) | (value & 0x07))
}

// TestExecutorFP8MulMatMatchesReference: native-FP8(E4M3) residency serves matmul
// weights at 1 byte + a per-output-row F32 scale carried in the same resident
// buffer ([rows*inner e4m3 | rows*4 scale]), through the SAME native-dtype
// mechanism as F16/BF16. Decode (right rows == 1) reads the e4m3 quads directly
// and folds scale[row] once; prefill (right rows > 1) upconverts the weight to
// F32 with the per-row scale folded in and runs the identical SGEMM. Both must
// match the exact e4m3-dequant-times-scale F32 reference; greedy argmax exact.
func TestExecutorFP8MulMatMatchesReference(t *testing.T) {
	cudatest.Require(t)
	const inner, rows = 64, 19 // inner % 4 == 0 for the packed-quad decode
	leftShape := tensor.MustShape(inner, rows)

	weightBytes := make([]byte, rows*inner)
	scales := make([]float32, rows)
	dequantized := make([]float32, rows*inner)
	for row := 0; row < rows; row++ {
		scales[row] = 0.125 * float32(1+row%7) // positive, varied per output row
		for column := 0; column < inner; column++ {
			b := fp8PatternByte(row*inner + column)
			weightBytes[row*inner+column] = b
			dequantized[row*inner+column] = dtype.F8E4M3ToFloat32(b) * scales[row]
		}
	}
	// Combined resident buffer: all e4m3 bytes, then the per-row F32 scales.
	storage := make([]byte, rows*inner+rows*4)
	copy(storage, weightBytes)
	for row := 0; row < rows; row++ {
		binary.LittleEndian.PutUint32(storage[rows*inner+row*4:], math.Float32bits(scales[row]))
	}

	for _, testCase := range []struct {
		name      string
		rightRows uint64
	}{
		{"decode", 1},
		{"prefill", 3},
	} {
		t.Run(testCase.name, func(t *testing.T) {
			rightShape := tensor.MustShape(inner, testCase.rightRows)
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
			left := builder.Input("left", dtype.F8E4M3, leftShape)
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
	rightShape := tensor.MustShape(inner, 1)
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
	left := builder.Input("left", dtype.F8E4M3, leftShape)
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
