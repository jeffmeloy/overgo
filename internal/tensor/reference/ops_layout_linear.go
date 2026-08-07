package reference

import (
	"errors"
	"fmt"
	"math"
	"slices"

	"overgo/internal/tensor"
)

func transpose2D(shape tensor.Shape, input Value) (Value, error) {
	width := int(input.Shape.Dims[0])
	rows := int(input.Shape.Dims[1])
	output := make([]float32, len(input.Data))
	for row := range rows {
		for column := range width {
			output[row+rows*column] = input.Data[column+width*row]
		}
	}
	return Value{Shape: shape, Data: output}, nil
}

func groupSlice(
	shape tensor.Shape,
	input Value,
	attributes tensor.GroupSliceAttributes,
) (Value, error) {
	elements, err := shape.Elements()
	if err != nil || elements > uint64(math.MaxInt) {
		return Value{}, errors.New("invalid GroupSlice output size")
	}
	width := int(attributes.Width)
	groups := int(attributes.Groups)
	stride := int(attributes.Stride)
	offset := int(attributes.Offset)
	inputWidth := int(input.Shape.Dims[0])
	output := make([]float32, int(elements))
	for index := range output {
		column := index % width
		remainder := index / width
		group := remainder % groups
		outer := remainder / groups
		output[index] = input.Data[outer*inputWidth+offset+group*stride+column]
	}
	return Value{Shape: shape, Data: output}, nil
}

func flatSlice(
	shape tensor.Shape,
	input Value,
	attributes tensor.FlatSliceAttributes,
) (Value, error) {
	elements, err := shape.Elements()
	if err != nil || elements > uint64(math.MaxInt) ||
		attributes.Offset > uint64(len(input.Data)) ||
		elements > uint64(len(input.Data))-attributes.Offset {
		return Value{}, errors.New("invalid FlatSlice storage range")
	}
	start := int(attributes.Offset)
	return Value{
		Shape: shape,
		Data:  slices.Clone(input.Data[start : start+int(elements)]),
	}, nil
}

func softmax(shape tensor.Shape, input Value) (Value, error) {
	width := int(shape.Dims[0])
	if width == 0 || len(input.Data)%width != 0 {
		return Value{}, errors.New("invalid softmax row width")
	}
	output := make([]float32, len(input.Data))
	for row := 0; row < len(input.Data); row += width {
		maximum := input.Data[row]
		for _, value := range input.Data[row+1 : row+width] {
			if value > maximum {
				maximum = value
			}
		}
		var sum float64
		for column, value := range input.Data[row : row+width] {
			exponent := math.Exp(float64(value - maximum))
			output[row+column] = float32(exponent)
			sum += exponent
		}
		for column := range width {
			output[row+column] /= float32(sum)
		}
	}
	return Value{Shape: shape, Data: output}, nil
}

func mulMat(shape tensor.Shape, left, right Value) (Value, error) {
	k := int(left.Shape.Dims[0])
	m := int(left.Shape.Dims[1])
	n := int(right.Shape.Dims[1])
	if int(right.Shape.Dims[0]) != k {
		return Value{}, errors.New("mul_mat inner dimensions differ")
	}
	output := make([]float32, m*n)
	for row := 0; row < n; row++ {
		for column := 0; column < m; column++ {
			var sum float64
			for inner := 0; inner < k; inner++ {
				sum += float64(left.Data[column*k+inner]) * float64(right.Data[row*k+inner])
			}
			output[row*m+column] = float32(sum)
		}
	}
	return Value{Shape: shape, Data: output}, nil
}

func groupedMulMat(shape tensor.Shape, left, right Value) (Value, error) {
	k := int(left.Shape.Dims[0])
	m := int(left.Shape.Dims[1])
	groups := int(left.Shape.Dims[2])
	n := int(right.Shape.Dims[2])
	if int(right.Shape.Dims[0]) != k || int(right.Shape.Dims[1]) != groups {
		return Value{}, errors.New("grouped_mul_mat dimensions differ")
	}
	output := make([]float32, m*groups*n)
	for token := range n {
		for group := range groups {
			for row := range m {
				var sum float64
				for inner := range k {
					leftIndex := (group*m+row)*k + inner
					rightIndex := (token*groups+group)*k + inner
					sum += float64(left.Data[leftIndex]) * float64(right.Data[rightIndex])
				}
				output[(token*groups+group)*m+row] = float32(sum)
			}
		}
	}
	return Value{Shape: shape, Data: output}, nil
}

func getRows(shape tensor.Shape, table Value, rows []uint32) (Value, error) {
	width := int(table.Shape.Dims[0])
	if width <= 0 || len(table.Data)%width != 0 {
		return Value{}, errors.New("invalid get_rows table width")
	}
	output := make([]float32, width*len(rows))
	for outputRow, tableRow := range rows {
		if uint64(tableRow) >= table.Shape.Dims[1] {
			return Value{}, fmt.Errorf("get_rows row %d exceeds table", tableRow)
		}
		copy(
			output[outputRow*width:(outputRow+1)*width],
			table.Data[int(tableRow)*width:(int(tableRow)+1)*width],
		)
	}
	return Value{Shape: shape, Data: output}, nil
}
