package reference

import (
	"math"
	"overgo/internal/tensor"
	"overgo/internal/tensor/dtype"
)

func nativeL2Norm(shape tensor.Shape, input Value, epsilon float32) (Value, error) {
	x, y, err := tensor.NativeReductionGeometry(shape, true)
	if err != nil {
		return Value{}, err
	}
	width, threads := int(shape.Dims[0]), int(x*y)
	output := make([]float32, len(input.Data))
	partial := make([]float32, threads)
	for base := 0; base < len(input.Data); base += width {
		for lane := range threads {
			var sums [tensor.NativeReductionBF16Vector]float32
			for col := lane * len(sums); col < width; col += threads * len(sums) {
				for part := range sums {
					v := input.Data[base+col+part]
					sums[part] = float32(sums[part] + dtype.RoundBF16(float32(v*v)))
				}
			}
			partial[lane] = sums[0]
			for _, v := range sums[1:] {
				partial[lane] = float32(partial[lane] + v)
			}
		}
		for offset := int(x) / tensor.PairedExtent; offset > 0; offset /= tensor.PairedExtent {
			for lane := range threads {
				if lane%int(x) < offset {
					partial[lane] = float32(partial[lane] + partial[lane+offset])
				}
			}
		}
		for offset := int(y) / tensor.PairedExtent; offset > 0; offset /= tensor.PairedExtent {
			for row := range offset {
				lane := row * int(x)
				partial[lane] = float32(partial[lane] + partial[lane+offset*int(x)])
			}
		}
		shifted := dtype.RoundBF16(dtype.RoundBF16(partial[0]) + epsilon)
		inverse := dtype.RoundBF16(float32(1 / math.Sqrt(float64(shifted))))
		for col := range width {
			output[base+col] = dtype.RoundBF16(float32(input.Data[base+col] * inverse))
		}
	}
	return Value{Shape: shape, Data: output}, nil
}
