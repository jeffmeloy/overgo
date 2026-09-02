//go:build windows

package executor

import (
	"testing"

	"overgo/internal/quant"
	"overgo/internal/tensor"
	"overgo/internal/tensor/dtype"
)

// TestExecutorSpanColumns pins the span kernel contract: up to eight
// input columns ride the q8-input integer kernels in one launch, one
// weight read serving every column, at the documented int8-activation
// tolerance; per-column greedy selection stays exact; and a ninth column
// opens a second span launch on the same path rather than falling back
// to the per-column float kernel.
func TestExecutorSpanColumns(t *testing.T) {
	for _, storageType := range []dtype.Type{dtype.Q8_0, dtype.Q4K, dtype.Q5K, dtype.Q6K} {
		t.Run(storageType.String(), func(t *testing.T) {
			leftShape := tensor.MustShape(512, 19)
			leftValue := patternedValue(leftShape, 11, 0.03, -0.1)
			storage, err := quant.Quantize(storageType, leftValue.Data)
			if err != nil {
				t.Fatal(err)
			}
			dequantized, err := quant.Dequantize(storageType, storage, uint64(len(leftValue.Data)))
			if err != nil {
				t.Fatal(err)
			}
			const inputQuantizedDecodeTolerance = 8e-2
			for _, testCase := range []residentProjectionCase{
				{name: "span2", rightRows: 2, tolerance: inputQuantizedDecodeTolerance},
				{name: "span5", rightRows: 5, tolerance: inputQuantizedDecodeTolerance},
				{name: "span8", rightRows: 8, tolerance: inputQuantizedDecodeTolerance},
				{name: "span8-greedy", rightRows: 8, selectTopK: true, tolerance: accuracyExact},
				{name: "span9", rightRows: 9, tolerance: inputQuantizedDecodeTolerance},
			} {
				t.Run(testCase.name, func(t *testing.T) {
					rightShape := tensor.MustShape(leftShape.Slice()[0], testCase.rightRows)
					rightValue := patternedValue(rightShape, 7, 0.05, 0.02)
					checkResidentBinaryGraph(
						t, storageType, leftShape, storage, dequantized, rightValue,
						func(builder *tensor.Builder, left, right *tensor.Tensor) *tensor.Tensor {
							output := builder.MulMat(left, right)
							if testCase.selectTopK {
								return builder.TopK(output, 1)
							}
							return output
						},
						testCase.tolerance,
					)
				})
			}
		})
	}
}
