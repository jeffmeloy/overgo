package safetensors

import (
	"encoding/binary"
	"fmt"
	"math"

	"overgo/internal/checked"
)

// Dimension returns one host-representable tensor extent from a source.
func Dimension(source *Source, name string, axis int) (int, error) {
	if source == nil {
		return 0, fmt.Errorf("safetensors: tensor source is unavailable")
	}
	tensor, ok := source.Tensors[name]
	if !ok {
		return 0, fmt.Errorf("safetensors: missing tensor %s", name)
	}
	if axis < 0 || axis >= len(tensor.Shape) {
		return 0, fmt.Errorf("safetensors: tensor %s has no axis %d", name, axis)
	}
	extent, ok := checked.Int(tensor.Shape[axis])
	if !ok {
		return 0, fmt.Errorf("safetensors: tensor %s axis %d exceeds host range", name, axis)
	}
	return extent, nil
}

// HostShape converts an exact-rank tensor shape to host integer extents.
func HostShape(tensor Tensor, rank int) ([]int, error) {
	if rank <= 0 || len(tensor.Shape) != rank {
		return nil, fmt.Errorf("safetensors: tensor %s rank %d, want %d", tensor.Name, len(tensor.Shape), rank)
	}
	shape := make([]int, rank)
	for axis, extent := range tensor.Shape {
		value, ok := checked.Int(extent)
		if !ok || value <= 0 {
			return nil, fmt.Errorf("safetensors: tensor %s axis %d is invalid", tensor.Name, axis)
		}
		shape[axis] = value
	}
	return shape, nil
}

// MatrixShape returns validated two-dimensional tensor geometry.
func MatrixShape(tensor Tensor) (uint64, uint64, error) {
	if len(tensor.Shape) != 2 || tensor.Shape[0] == 0 || tensor.Shape[1] == 0 {
		return 0, 0, fmt.Errorf("safetensors: tensor %s is not a non-empty matrix", tensor.Name)
	}
	return tensor.Shape[0], tensor.Shape[1], nil
}

// ReadTensorRowsF32 opens a repository and gathers selected rows from a
// row-contiguous BF16 or F32 matrix into caller-independent F32 storage.
func ReadTensorRowsF32(directory, name string, rowWidth int, rows []int) ([]float32, error) {
	source, err := OpenSource(directory)
	if err != nil {
		return nil, err
	}
	defer source.Close()
	tensor, ok := source.Tensors[name]
	if !ok {
		return nil, fmt.Errorf("safetensors: missing tensor %s", name)
	}
	_, width, shapeErr := MatrixShape(tensor)
	if shapeErr != nil || rowWidth <= 0 || width != uint64(rowWidth) {
		return nil, fmt.Errorf("safetensors: tensor %s shape %v, want [*,%d]", name, tensor.Shape, rowWidth)
	}
	elementBytes := 0
	switch tensor.DType {
	case "BF16":
		elementBytes = 2
	case "F32":
		elementBytes = 4
	default:
		return nil, fmt.Errorf("safetensors: tensor %s dtype %s cannot produce F32 rows", name, tensor.DType)
	}
	rowBytes := rowWidth * elementBytes
	output := make([]float32, len(rows)*rowWidth)
	buffer := make([]byte, rowBytes)
	for index, row := range rows {
		if row < 0 || uint64(row) >= tensor.Shape[0] {
			return nil, fmt.Errorf("safetensors: tensor %s row %d is out of range", name, row)
		}
		if _, err := tensor.ReadAt(buffer, int64(row)*int64(rowBytes)); err != nil {
			return nil, fmt.Errorf("safetensors: tensor %s row %d: %w", name, row, err)
		}
		destination := output[index*rowWidth : (index+1)*rowWidth]
		for column := range destination {
			if elementBytes == 2 {
				destination[column] = math.Float32frombits(uint32(binary.LittleEndian.Uint16(buffer[column*2:])) << 16)
			} else {
				destination[column] = math.Float32frombits(binary.LittleEndian.Uint32(buffer[column*4:]))
			}
		}
	}
	return output, nil
}

// TensorRowsF64 gathers selected rows from a row-contiguous BF16 or F32 matrix.
func TensorRowsF64(tensor Tensor, rowWidth int, rows []int) ([]float64, error) {
	_, width, shapeErr := MatrixShape(tensor)
	if shapeErr != nil || rowWidth <= 0 || width != uint64(rowWidth) {
		return nil, fmt.Errorf("safetensors: tensor %s shape %v, want [*,%d]", tensor.Name, tensor.Shape, rowWidth)
	}
	elementBytes := 0
	switch tensor.DType {
	case "BF16":
		elementBytes = 2
	case "F32":
		elementBytes = 4
	default:
		return nil, fmt.Errorf("safetensors: tensor %s dtype %s cannot produce F64 rows", tensor.Name, tensor.DType)
	}
	rowBytes := rowWidth * elementBytes
	output := make([]float64, len(rows)*rowWidth)
	buffer := make([]byte, rowBytes)
	for index, row := range rows {
		if row < 0 || uint64(row) >= tensor.Shape[0] {
			return nil, fmt.Errorf("safetensors: tensor %s row %d is out of range", tensor.Name, row)
		}
		if _, err := tensor.ReadAt(buffer, int64(row)*int64(rowBytes)); err != nil {
			return nil, fmt.Errorf("safetensors: tensor %s row %d: %w", tensor.Name, row, err)
		}
		destination := output[index*rowWidth : (index+1)*rowWidth]
		for column := range destination {
			if elementBytes == 2 {
				destination[column] = float64(math.Float32frombits(uint32(binary.LittleEndian.Uint16(buffer[column*2:])) << 16))
			} else {
				destination[column] = float64(math.Float32frombits(binary.LittleEndian.Uint32(buffer[column*4:])))
			}
		}
	}
	return output, nil
}
