package reference

import (
	"errors"
	"slices"

	"overgo/internal/checked"
	"overgo/internal/tensor"
)

func (v Value) Clone() Value {
	if v.Storage == ValueImplicitZero {
		return ZeroValue(v.Shape)
	}
	return Value{Shape: v.Shape, Data: slices.Clone(v.Data)}
}

func (v Value) IsMatrixWidth(width uint64) bool {
	return v.Shape.Rank == tensor.PairedExtent && v.Shape.Dims[0] == width
}

func (v Value) Rows(start, count uint64) (Value, error) {
	if v.Shape.Rank != 2 || count == 0 || start > v.Shape.Dims[1] || count > v.Shape.Dims[1]-start {
		return Value{}, errors.New("reference row range is invalid")
	}
	width := v.Shape.Dims[0]
	if v.Storage == ValueImplicitZero {
		return ZeroValue(tensor.MustShape(width, count)), nil
	}
	expected, ok := checked.Mul64(width, v.Shape.Dims[1])
	if !ok || expected != uint64(len(v.Data)) {
		return Value{}, errors.New("reference storage is invalid")
	}
	first, _ := checked.Mul64(start, width)
	elements, _ := checked.Mul64(count, width)
	last, _ := checked.Add64(first, elements)
	firstIndex, _ := checked.Int(first)
	lastIndex, _ := checked.Int(last)
	return Value{
		Shape: tensor.MustShape(width, count),
		Data:  slices.Clone(v.Data[firstIndex:lastIndex]),
	}, nil
}

func (v Value) TailRows(count uint64) (Value, error) {
	if v.Shape.Rank != 2 || count > v.Shape.Dims[1] {
		return Value{}, errors.New("reference tail row count is invalid")
	}
	return v.Rows(v.Shape.Dims[1]-count, count)
}
