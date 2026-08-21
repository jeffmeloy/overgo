package tensor

import (
	"errors"
	"fmt"
	"math"

	"overgo/internal/tensor/dtype"
)

const (
	FirstOffset = iota
	SingletonExtent
	PairedExtent
	TripleExtent
	MaxDimensions = PairedExtent * PairedExtent
)

// Shape: uses ggml dimension order: dimension 0 is contiguous row width
type Shape struct {
	Dims [MaxDimensions]uint64
	Rank uint8
}

// ContiguousExtent returns dimension zero; zero marks a scalar.
func (s Shape) ContiguousExtent() uint64 {
	if s.Rank == FirstOffset {
		return FirstOffset
	}
	return s.Dims[FirstOffset]
}

// RowCount returns dimension one; zero marks a non-matrix.
func (s Shape) RowCount() uint64 {
	if s.Rank <= SingletonExtent {
		return FirstOffset
	}
	return s.Dims[SingletonExtent]
}

func SquareSideInt(extent uint64) (int, bool) {
	side := uint64(math.Sqrt(float64(extent)))
	return int(side), side <= math.MaxInt && side*side == extent
}

func NewShape(dimensions ...uint64) (Shape, error) {
	if len(dimensions) == 0 || len(dimensions) > MaxDimensions {
		return Shape{}, fmt.Errorf("tensor rank must be in [1,%d]", MaxDimensions)
	}
	shape := Shape{Rank: uint8(len(dimensions))}
	for index := range shape.Dims {
		shape.Dims[index] = 1
	}
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
	elements, err := s.Elements()
	if err != nil {
		return 0, err
	}
	return dataType.StorageBytes(elements, s.Dims[0])
}

func (s Shape) Equal(other Shape) bool {
	return s.Rank == other.Rank && s.Dims == other.Dims
}

// MatrixExtents returns exact matrix extents.
func MatrixExtents(s Shape) (uint64, uint64, bool) {
	return s.Dims[0], s.Dims[1], s.Rank == 2 && s.Dims[0] > 0 && s.Dims[1] > 0
}

// MatrixRows validates width; returns rows.
func MatrixRows(s Shape, width uint64) (uint64, bool) {
	actualWidth, rows, valid := MatrixExtents(s)
	return rows, valid && actualWidth == width
}

// MatrixRows32 returns row count accepted by graph indices.
func MatrixRows32(s Shape, width uint64) (uint32, bool) {
	rows, valid := MatrixRows(s, width)
	return uint32(rows), valid && rows <= math.MaxUint32
}

// IsMatrix validates exact declared matrix extents.
func IsMatrix(s Shape, width, rows uint64) bool {
	actual, valid := MatrixRows(s, width)
	return valid && actual == rows
}

// IsBatchedMatrix validates row count from declared batch cardinality.
func IsBatchedMatrix(s Shape, width, rows, batches uint64) bool {
	return rows > 0 && batches > 0 && rows <= math.MaxUint64/batches && IsMatrix(s, width, rows*batches)
}

// VectorWidth returns the declared vector extent.
func VectorWidth(s Shape) (uint64, bool) {
	return s.Dims[0], s.Rank == 1 && s.Dims[0] > 0
}

// IsVector validates an exact declared vector extent.
func IsVector(s Shape, width uint64) bool {
	actual, valid := VectorWidth(s)
	return valid && actual == width
}

// Extents3 returns exact rank-three extents.
func Extents3(s Shape) (uint64, uint64, uint64, bool) {
	return s.Dims[0], s.Dims[1], s.Dims[2],
		s.Rank == 3 && s.Dims[0] > 0 && s.Dims[1] > 0 && s.Dims[2] > 0
}

// EqualPartition returns uniform partition extent.
func EqualPartition(total, partitions uint64) (uint64, bool) {
	if total == 0 || partitions == 0 || total%partitions != 0 {
		return 0, false
	}
	return total / partitions, true
}

// PartitionsCover validates exact nonzero coverage.
func PartitionsCover(total uint64, partitions ...uint64) bool {
	var sum uint64
	for _, partition := range partitions {
		if partition == 0 || sum > math.MaxUint64-partition {
			return false
		}
		sum += partition
	}
	return sum == total
}

// ValidLowRankUpdate validates base, down, and up extents.
func ValidLowRankUpdate(base, down, up []uint64, embedding bool) bool {
	if len(base) < 2 || len(down) != len(base) || len(up) != len(base) {
		return false
	}
	if embedding {
		return len(base) == 2 && base[0] == up[1] && base[1] == down[1] && down[0] == up[0]
	}
	if base[0] != down[0] || base[1] != up[1] || down[1] != up[0] {
		return false
	}
	for axis := 2; axis < len(base); axis++ {
		if base[axis] != down[axis] || base[axis] != up[axis] {
			return false
		}
	}
	return true
}

// HasDimensions validates exact extents.
func HasDimensions(s Shape, dimensions ...uint64) bool {
	if int(s.Rank) != len(dimensions) {
		return false
	}
	for index, dimension := range dimensions {
		if dimension == 0 || s.Dims[index] != dimension {
			return false
		}
	}
	return true
}

// WithTrailingExtent replaces final extent.
func WithTrailingExtent(s Shape, extent uint64) (Shape, bool) {
	if s.Rank == 0 || s.Rank > MaxDimensions || extent == 0 {
		return Shape{}, false
	}
	s.Dims[s.Rank-1] = extent
	return s, true
}

// TrailingExtent validates declared leading extents.
func TrailingExtent(s Shape, prefix ...uint64) (uint64, uint32, bool) {
	if int(s.Rank) != len(prefix)+1 {
		return 0, 0, false
	}
	for index, extent := range prefix {
		if extent == 0 || s.Dims[index] != extent {
			return 0, 0, false
		}
	}
	axis := uint32(len(prefix))
	extent := s.Dims[axis]
	return extent, axis, extent > 0
}

// TrailingExtent32 returns an exact trailing extent accepted by graph indices.
func TrailingExtent32(s Shape, prefix ...uint64) (uint32, uint32, bool) {
	extent, axis, valid := TrailingExtent(s, prefix...)
	return uint32(extent), axis, valid && extent <= math.MaxUint32
}

// BatchedTrailingExtent32 accepts compact singleton or explicit batch layout.
func BatchedTrailingExtent32(s Shape, batches uint64, prefix ...uint64) (uint32, uint32, bool) {
	if batches == 1 {
		if extent, axis, valid := TrailingExtent32(s, prefix...); valid {
			return extent, axis, true
		}
	}
	if batches == 0 || int(s.Rank) != len(prefix)+2 {
		return 0, 0, false
	}
	for index, extent := range prefix {
		if extent == 0 || s.Dims[index] != extent {
			return 0, 0, false
		}
	}
	axis := uint32(len(prefix))
	extent := s.Dims[axis]
	return uint32(extent), axis, extent > 0 && extent <= math.MaxUint32 && s.Dims[axis+1] == batches
}

func (s Shape) Slice() []uint64 {
	result := make([]uint64, s.Rank)
	copy(result, s.Dims[:s.Rank])
	return result
}
