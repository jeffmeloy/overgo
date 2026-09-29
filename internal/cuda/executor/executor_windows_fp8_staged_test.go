//go:build windows

package executor

import (
	"context"
	"encoding/binary"
	"math"
	"testing"

	"overgo/internal/cuda/driver"
	"overgo/internal/tensor"
	"overgo/internal/tensor/dtype"
	"overgo/internal/tensor/reference"
)

const (
	// accuracyFP8Accumulation bounds the FP8 tensor-core product against
	// the float64 product of the same e4m3 operands and scales, as a
	// fraction of the output's absolute product mass sum|w*q|: measured
	// 2.7e-5 over a 512-long inner dimension, the Ada FP8 MMA
	// accumulating with less than F32 precision.
	accuracyFP8Accumulation = 1e-4
	// fp8E4M3Max is the largest finite e4m3 magnitude, where each
	// activation column's absolute maximum lands.
	fp8E4M3Max = 448
	// e4m3RelativeRounding: round-to-nearest over three mantissa bits
	// moves a normal value by at most a sixteenth of itself.
	e4m3RelativeRounding = 1.0 / 16
	// e4m3SubnormalRounding: half the 2^-9 subnormal step, in units of
	// the column scale.
	e4m3SubnormalRounding = 1.0 / 1024
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

// TestExecutorFp8TensorCoreMulMatMatchesReference: in an aligned shape
// past the column floor under the native policy, native fp8 weights
// multiply the activation quantized to e4m3 with one scale per token
// column on the FP8 tensor cores. Against the float64 product of the
// same e4m3 operands and scales (the host encoder is the device
// quantizer's oracle) each output errs by at most
// accuracyFP8Accumulation of its absolute product mass; against the
// float64 product with the unquantized F32 activation, by at most the
// activation's e4m3 rounding bound plus that accumulation slack.
func TestExecutorFp8TensorCoreMulMatMatchesReference(t *testing.T) {
	const (
		inner   = 512
		rows    = 64
		columns = 96
	)
	scales := make([]float64, rows)
	weights := make([]float64, rows*inner)
	storage := make([]byte, rows*inner+rows*4)
	for row := range rows {
		scale := 0.125 * float32(1+row%7)
		scales[row] = float64(scale)
		for column := range inner {
			b := fp8PatternByte(row*inner + column)
			storage[row*inner+column] = b
			weights[row*inner+column] = float64(dtype.F8E4M3ToFloat32(b))
		}
		binary.LittleEndian.PutUint32(storage[rows*inner+row*4:], math.Float32bits(scale))
	}
	activation := make([]float32, inner*columns)
	quantized := make([]float64, inner*columns)
	// rounding: the e4m3 rounding bound of each activation element.
	rounding := make([]float64, inner*columns)
	for column := range columns {
		values := activation[column*inner : (column+1)*inner]
		for index := range values {
			values[index] = float32(1+column%5) * (0.05*float32((index*7+column*3)%19-9) + 0.003*float32(index%11))
		}
		var amax float32
		for _, value := range values {
			amax = max(amax, float32(math.Abs(float64(value))))
		}
		scale := amax / fp8E4M3Max
		for index, value := range values {
			quantized[column*inner+index] = float64(dtype.F8E4M3ToFloat32(dtype.Float32ToF8E4M3(value/scale))) * float64(scale)
			rounding[column*inner+index] = max(
				math.Abs(float64(value))*e4m3RelativeRounding, float64(scale)*e4m3SubnormalRounding,
			)
		}
	}
	wantQuantized := make([]float64, rows*columns)
	wantExact := make([]float64, rows*columns)
	quantizedMass := make([]float64, rows*columns)
	roundingBound := make([]float64, rows*columns)
	for column := range columns {
		for row := range rows {
			at := column*rows + row
			for index := range inner {
				weight := weights[row*inner+index] * scales[row]
				wantQuantized[at] += weight * quantized[column*inner+index]
				wantExact[at] += weight * float64(activation[column*inner+index])
				quantizedMass[at] += math.Abs(weight * quantized[column*inner+index])
				roundingBound[at] += math.Abs(weight) * rounding[column*inner+index]
			}
		}
	}

	builder := tensor.NewBuilder()
	builder.SetMulMatCompute(tensor.MulMatComputeNativeTensorCore)
	left := builder.Input("left", dtype.F8E4M3, tensor.MustShape(inner, rows))
	right := builder.Input("right", dtype.F32, tensor.MustShape(inner, columns))
	output := builder.MulMat(left, right)
	if !fp8TensorCoreMulMat(output) {
		t.Fatal("fixture shape does not take the FP8 tensor-core path")
	}
	rightValue, err := reference.NewValue(right.Shape, activation)
	if err != nil {
		t.Fatal(err)
	}
	worker := newFixtureWorker(t)
	pointer := copyFixtureDeviceBytes(t, worker, storage)
	cuda := newFixtureExecutorWithWorker(t, worker)
	got, err := cuda.executeWithDeviceFeeds(
		context.WithoutCancel(t.Context()),
		[]*tensor.Tensor{output},
		map[*tensor.Tensor]reference.Value{right: rightValue},
		map[*tensor.Tensor]driver.DevicePtr{left: pointer},
	)
	if err != nil {
		t.Fatal(err)
	}
	var magnitude, quantizedError, exactError, accumulationRatio, roundingRatio float64
	for at, value := range got[output].Data {
		quantizedDifference := math.Abs(float64(value) - wantQuantized[at])
		exactDifference := math.Abs(float64(value) - wantExact[at])
		slack := accuracyFP8Accumulation * quantizedMass[at]
		if quantizedDifference > slack || exactDifference > roundingBound[at]+slack {
			t.Fatalf(
				"value[%d] = %v, want %v with e4m3 activation and %v exact: differences %g and %g exceed %g and %g",
				at, value, wantQuantized[at], wantExact[at], quantizedDifference, exactDifference, slack, roundingBound[at]+slack,
			)
		}
		magnitude = max(magnitude, math.Abs(wantExact[at]))
		quantizedError, exactError = max(quantizedError, quantizedDifference), max(exactError, exactDifference)
		accumulationRatio = max(accumulationRatio, quantizedDifference/quantizedMass[at])
		roundingRatio = max(roundingRatio, exactDifference/roundingBound[at])
	}
	t.Logf(
		"max |output| %.4g; max error vs e4m3-activation reference %.3g (%.3g of product mass), vs F32-activation reference %.3g (%.3g of the e4m3 rounding bound)",
		magnitude, quantizedError, accumulationRatio, exactError, roundingRatio,
	)
}
