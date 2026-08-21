package model

import (
	"context"
	"errors"
	"fmt"
	"math"

	"overgo/internal/gguf"
	"overgo/internal/quant"
	"overgo/internal/tensor"
	"overgo/internal/tensor/reference"
)

// LoadHostTensor: reads and dequantizes one GGUF tensor
func LoadHostTensor(ctx context.Context, file *gguf.File, info gguf.TensorInfo) (reference.Value, error) {
	if file == nil {
		return reference.Value{}, errors.New("host tensor: GGUF file is nil")
	}
	if err := ctx.Err(); err != nil {
		return reference.Value{}, err
	}
	shape, err := tensor.NewShape(info.Extents()...)
	if err != nil {
		return reference.Value{}, fmt.Errorf("host tensor %q shape: %w", info.Name, err)
	}
	elements, err := shape.Elements()
	if err != nil {
		return reference.Value{}, fmt.Errorf("host tensor %q elements: %w", info.Name, err)
	}
	if info.Size > uint64(math.MaxInt) {
		return reference.Value{}, fmt.Errorf("host tensor %q storage exceeds addressable memory", info.Name)
	}
	storage := make([]byte, int(info.Size))
	if err := file.ReadTensorData(info, storage); err != nil {
		return reference.Value{}, fmt.Errorf("read host tensor %q: %w", info.Name, err)
	}
	if err := ctx.Err(); err != nil {
		return reference.Value{}, err
	}
	data, err := quant.Dequantize(info.Type, storage, elements)
	if err != nil {
		return reference.Value{}, fmt.Errorf("dequantize host tensor %q: %w", info.Name, err)
	}
	return reference.Value{Shape: shape, Data: data}, nil
}

// LoadHostRows dequantizes selected table entries.
func LoadHostRows(
	ctx context.Context,
	file *gguf.File,
	info gguf.TensorInfo,
	indices []uint32,
) (reference.Value, error) {
	if file == nil {
		return reference.Value{}, errors.New("host rows: GGUF file is nil")
	}
	if len(indices) == tensor.FirstOffset {
		return reference.Value{}, errors.New("host rows: row list is empty")
	}
	layout, err := info.RowLayout()
	if err != nil {
		return reference.Value{}, fmt.Errorf("host rows: %w", err)
	}
	if layout.BytesPerRow > uint64(math.MaxInt) || layout.ElementsPerRow > uint64(math.MaxInt) ||
		uint64(len(indices)) > math.MaxUint64/layout.ElementsPerRow ||
		uint64(len(indices))*layout.ElementsPerRow > uint64(math.MaxInt) {
		return reference.Value{}, errors.New("host rows: result exceeds addressable memory")
	}
	storage := make([]byte, int(layout.BytesPerRow))
	output := make([]float32, int(layout.ElementsPerRow)*len(indices))
	for outputIndex, index := range indices {
		if err := ctx.Err(); err != nil {
			return reference.Value{}, err
		}
		if uint64(index) >= layout.Count {
			return reference.Value{}, fmt.Errorf("host rows: row %d exceeds tensor row count %d", index, layout.Count)
		}
		offset := uint64(index) * layout.BytesPerRow
		if err := file.ReadTensorRange(info, offset, storage); err != nil {
			return reference.Value{}, fmt.Errorf("read host row %d from %q: %w", index, info.Name, err)
		}
		first := outputIndex * int(layout.ElementsPerRow)
		last := first + int(layout.ElementsPerRow)
		if err := quant.DequantizeInto(info.Type, storage, output[first:last]); err != nil {
			return reference.Value{}, fmt.Errorf("dequantize host row %d from %q: %w", index, info.Name, err)
		}
	}
	shape, err := tensor.NewShape(layout.ElementsPerRow, uint64(len(indices)))
	if err != nil {
		return reference.Value{}, err
	}
	return reference.Value{Shape: shape, Data: output}, nil
}

// DotRows: streams rank-2 output table and computes one logit per row
func DotRows(
	ctx context.Context,
	file *gguf.File,
	info gguf.TensorInfo,
	vector []float32,
) ([]float32, error) {
	if file == nil {
		return nil, errors.New("dot rows: GGUF file is nil")
	}
	layout, err := info.RowLayout()
	if err != nil {
		return nil, fmt.Errorf("dot rows: %w", err)
	}
	if uint64(len(vector)) != layout.ElementsPerRow {
		return nil, fmt.Errorf("dot rows: vector has %d values, need %d", len(vector), layout.ElementsPerRow)
	}
	if layout.Count > math.MaxUint32 || layout.Count > uint64(math.MaxInt) {
		return nil, errors.New("dot rows: row count is invalid")
	}
	capacity := min(uint64(defaultWeightChunkSize)/layout.BytesPerRow, layout.Count)
	capacity = max(capacity, uint64(tensor.SingletonExtent))
	storageBytes := capacity * layout.BytesPerRow
	if storageBytes > uint64(math.MaxInt) || capacity > math.MaxUint64/layout.ElementsPerRow ||
		capacity*layout.ElementsPerRow > uint64(math.MaxInt) {
		return nil, errors.New("dot rows: chunk exceeds addressable memory")
	}
	storage := make([]byte, int(storageBytes))
	values := make([]float32, int(capacity*layout.ElementsPerRow))

	scores := make([]float32, int(layout.Count))
	for first := uint64(tensor.FirstOffset); first < layout.Count; {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		count := capacity
		if remaining := layout.Count - first; count > remaining {
			count = remaining
		}
		chunk := storage[:count*layout.BytesPerRow]
		if err := file.ReadTensorRange(info, first*layout.BytesPerRow, chunk); err != nil {
			return nil, fmt.Errorf("read output rows at %d: %w", first, err)
		}
		chunkValues := values[:count*layout.ElementsPerRow]
		if err := quant.DequantizeInto(info.Type, chunk, chunkValues); err != nil {
			return nil, fmt.Errorf("dequantize output rows at %d: %w", first, err)
		}
		for row := uint64(tensor.FirstOffset); row < count; row++ {
			var dot float64
			offset := int(row * layout.ElementsPerRow)
			for column, input := range vector {
				dot += float64(chunkValues[offset+column]) * float64(input)
			}
			scores[first+row] = float32(dot)
		}
		first += count
	}
	return scores, nil
}

// LoadHostLayer materializes one releasable layer.
func LoadHostLayer(
	ctx context.Context,
	file *gguf.File,
	info LayerWeights,
) (HostLayer, error) {
	var result HostLayer
	if err := loadHostLayerGraphFields(ctx, file, &info, &result); err != nil {
		return HostLayer{}, err
	}
	return result, nil
}

// GraphInputs builds zero-copy graph feeds.
func (layer *HostLayer) GraphInputs(
	builder *tensor.Builder,
	prefix string,
) (LayerGraphWeights, map[*tensor.Tensor]reference.Value, error) {
	if layer == nil {
		return LayerGraphWeights{}, nil, errors.New("host layer is nil")
	}
	if builder == nil {
		return LayerGraphWeights{}, nil, errors.New("host layer graph builder is nil")
	}
	feeds := make(map[*tensor.Tensor]reference.Value)
	result := LayerGraphWeights{}
	bindHostLayerGraphFields(builder, prefix, layer, &result, feeds)
	if err := builder.Err(); err != nil {
		return LayerGraphWeights{}, nil, err
	}
	return result, feeds, nil
}
