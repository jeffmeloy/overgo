package tensor

import (
	"errors"
	"fmt"
	"math"

	"llamacpp2go/internal/tensor/dtype"
)

const MaxDimensions = 4

// Shape uses ggml dimension order: dimension 0 is the contiguous row width.
type Shape struct {
	Dims [MaxDimensions]uint64
	Rank uint8
}

func NewShape(dimensions ...uint64) (Shape, error) {
	if len(dimensions) == 0 || len(dimensions) > MaxDimensions {
		return Shape{}, fmt.Errorf("tensor rank must be in [1,%d]", MaxDimensions)
	}
	shape := Shape{Dims: [MaxDimensions]uint64{1, 1, 1, 1}, Rank: uint8(len(dimensions))}
	for index, dimension := range dimensions {
		if dimension == 0 {
			return Shape{}, fmt.Errorf("dimension %d is zero", index)
		}
		shape.Dims[index] = dimension
	}
	if _, err := shape.Elements(); err != nil {
		return Shape{}, err
	}
	return shape, nil
}

func MustShape(dimensions ...uint64) Shape {
	shape, err := NewShape(dimensions...)
	if err != nil {
		panic(err)
	}
	return shape
}

func (s Shape) Elements() (uint64, error) {
	if s.Rank == 0 || s.Rank > MaxDimensions {
		return 0, errors.New("invalid tensor rank")
	}
	var elements uint64 = 1
	for index := range MaxDimensions {
		dimension := s.Dims[index]
		if dimension == 0 {
			return 0, fmt.Errorf("dimension %d is zero", index)
		}
		if elements > math.MaxUint64/dimension {
			return 0, errors.New("tensor element count overflows uint64")
		}
		elements *= dimension
	}
	return elements, nil
}

func (s Shape) Bytes(dataType dtype.Type) (uint64, error) {
	traits, ok := dataType.Traits()
	if !ok {
		return 0, fmt.Errorf("unknown tensor type %d", dataType)
	}
	if s.Dims[0]%traits.BlockSize != 0 {
		return 0, fmt.Errorf(
			"row width %d is not divisible by %s block size %d",
			s.Dims[0],
			traits.Name,
			traits.BlockSize,
		)
	}
	elements, err := s.Elements()
	if err != nil {
		return 0, err
	}
	blocks := elements / traits.BlockSize
	if blocks > math.MaxUint64/traits.TypeSize {
		return 0, errors.New("tensor byte size overflows uint64")
	}
	return blocks * traits.TypeSize, nil
}

func (s Shape) Equal(other Shape) bool {
	return s.Rank == other.Rank && s.Dims == other.Dims
}

func (s Shape) Slice() []uint64 {
	result := make([]uint64, s.Rank)
	copy(result, s.Dims[:s.Rank])
	return result
}
