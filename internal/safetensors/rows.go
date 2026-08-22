package safetensors

import (
	"fmt"

	"overgo/internal/binaryschema"
	"overgo/internal/checked"
	"overgo/internal/extent"
	"overgo/internal/tensor/dtype"
)

// Dimension returns one host-representable tensor extent from a source.
func Dimension(source *Source, name string, axis int) (int, error) {
	if source == nil {
		return extent.FirstOffset, fmt.Errorf("safetensors: tensor source is unavailable")
	}
	tensor, ok := source.Tensors[name]
	if !ok {
		return extent.FirstOffset, fmt.Errorf("safetensors: missing tensor %s", name)
	}
	if axis < extent.FirstOffset || axis >= len(tensor.Shape) {
		return extent.FirstOffset, fmt.Errorf("safetensors: tensor %s has no axis %d", name, axis)
	}
	dimension, ok := checked.Int(tensor.Shape[axis])
	if !ok {
		return extent.FirstOffset, fmt.Errorf("safetensors: tensor %s axis %d exceeds host range", name, axis)
	}
	return dimension, nil
}

// HostShape converts an exact-rank tensor shape to host integer extents.
func HostShape(tensor Tensor, rank int) ([]int, error) {
	if rank <= extent.FirstOffset || len(tensor.Shape) != rank {
		return nil, fmt.Errorf("safetensors: tensor %s rank %d, want %d", tensor.Name, len(tensor.Shape), rank)
	}
	shape := make([]int, rank)
	for axis, dimension := range tensor.Shape {
		value, ok := checked.Int(dimension)
		if !ok || value <= extent.FirstOffset {
			return nil, fmt.Errorf("safetensors: tensor %s axis %d is invalid", tensor.Name, axis)
		}
		shape[axis] = value
	}
	return shape, nil
}

// MatrixShape returns validated two-dimensional tensor geometry.
func MatrixShape(tensor Tensor) (uint64, uint64, error) {
	if len(tensor.Shape) != extent.PairedExtent || tensor.Shape[extent.FirstOffset] == extent.FirstOffset || tensor.Shape[extent.SingletonExtent] == extent.FirstOffset {
		return extent.FirstOffset, extent.FirstOffset, fmt.Errorf("safetensors: tensor %s is not a non-empty matrix", tensor.Name)
	}
	return tensor.Shape[extent.FirstOffset], tensor.Shape[extent.SingletonExtent], nil
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
	if shapeErr != nil || rowWidth <= extent.FirstOffset || width != uint64(rowWidth) {
		return nil, fmt.Errorf("safetensors: tensor %s shape %v, want [*,%d]", name, tensor.Shape, rowWidth)
	}
	elementBytes := extent.FirstOffset
	switch tensor.DType {
	case "BF16":
		elementBytes = binaryschema.Uint16Bytes
	case "F32":
		elementBytes = binaryschema.Uint32Bytes
	default:
		return nil, fmt.Errorf("safetensors: tensor %s dtype %s cannot produce F32 rows", name, tensor.DType)
	}
	rowBytes := rowWidth * elementBytes
	output := make([]float32, len(rows)*rowWidth)
	buffer := make([]byte, rowBytes)
	for index, row := range rows {
		if row < extent.FirstOffset || uint64(row) >= tensor.Shape[extent.FirstOffset] {
			return nil, fmt.Errorf("safetensors: tensor %s row %d is out of range", name, row)
		}
		if _, err := tensor.ReadAt(buffer, int64(row)*int64(rowBytes)); err != nil {
			return nil, fmt.Errorf("safetensors: tensor %s row %d: %w", name, row, err)
		}
		destination := output[index*rowWidth : (index+extent.SingletonExtent)*rowWidth]
		for column := range destination {
			if elementBytes == binaryschema.Uint16Bytes {
				destination[column] = dtype.BF16ToFloat32(binaryschema.LittleEndian.Uint16(buffer[column*binaryschema.Uint16Bytes:]))
			} else {
				destination[column] = binaryschema.LittleEndian.Float32(buffer[column*binaryschema.Uint32Bytes:])
			}
		}
	}
	return output, nil
}

// TensorRowsF64 gathers selected rows from a row-contiguous BF16 or F32 matrix.
func TensorRowsF64(tensor Tensor, rowWidth int, rows []int) ([]float64, error) {
	_, width, shapeErr := MatrixShape(tensor)
	if shapeErr != nil || rowWidth <= extent.FirstOffset || width != uint64(rowWidth) {
		return nil, fmt.Errorf("safetensors: tensor %s shape %v, want [*,%d]", tensor.Name, tensor.Shape, rowWidth)
	}
	elementBytes := extent.FirstOffset
	switch tensor.DType {
	case "BF16":
		elementBytes = binaryschema.Uint16Bytes
	case "F32":
		elementBytes = binaryschema.Uint32Bytes
	default:
		return nil, fmt.Errorf("safetensors: tensor %s dtype %s cannot produce F64 rows", tensor.Name, tensor.DType)
	}
	rowBytes := rowWidth * elementBytes
	output := make([]float64, len(rows)*rowWidth)
	buffer := make([]byte, rowBytes)
	for index, row := range rows {
		if row < extent.FirstOffset || uint64(row) >= tensor.Shape[extent.FirstOffset] {
			return nil, fmt.Errorf("safetensors: tensor %s row %d is out of range", tensor.Name, row)
		}
		if _, err := tensor.ReadAt(buffer, int64(row)*int64(rowBytes)); err != nil {
			return nil, fmt.Errorf("safetensors: tensor %s row %d: %w", tensor.Name, row, err)
		}
		destination := output[index*rowWidth : (index+extent.SingletonExtent)*rowWidth]
		for column := range destination {
			if elementBytes == binaryschema.Uint16Bytes {
				destination[column] = float64(dtype.BF16ToFloat32(binaryschema.LittleEndian.Uint16(buffer[column*binaryschema.Uint16Bytes:])))
			} else {
				destination[column] = float64(binaryschema.LittleEndian.Float32(buffer[column*binaryschema.Uint32Bytes:]))
			}
		}
	}
	return output, nil
}
