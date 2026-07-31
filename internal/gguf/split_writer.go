package gguf

import (
	"errors"
	"fmt"
	"io"
	"math"
)

// SplitOptions: controls tensor partitioning for WriteSplit
type SplitOptions struct {
	MaxTensors           int
	MaxBytes             uint64
	NoTensorsInFirstFile bool
}

// SplitWriter: opens one output shard; WriteSplit closes every successfully
// opened writer before opening next shard
type SplitWriter func(index, count uint16) (io.WriteCloser, error)

// WriteSplit: streams parsed logical model into canonical GGUF shards
func (f *File) WriteSplit(open SplitWriter, options SplitOptions) error {
	if f == nil {
		return errors.New("GGUF file is nil")
	}
	if open == nil {
		return errors.New("GGUF split writer is nil")
	}
	partitions, err := f.planSplits(options)
	if err != nil {
		return err
	}
	count := uint16(len(partitions))
	baseMetadata := make([]Metadata, 0, len(f.Metadata))
	for _, item := range f.Metadata {
		switch item.Key {
		case "split.no", "split.count", "split.tensors.count":
			continue
		default:
			baseMetadata = append(baseMetadata, item)
		}
	}
	for index, partition := range partitions {
		metadataCapacity := 3
		if index == 0 {
			metadataCapacity += len(baseMetadata)
		}
		metadata := make([]Metadata, 0, metadataCapacity)
		if index == 0 {
			metadata = append(metadata, baseMetadata...)
		}
		metadata = append(metadata,
			Metadata{
				Key: "split.no",
				Value: Value{
					Type: ValueTypeUint16,
					Data: uint16(index),
				},
			},
			Metadata{
				Key: "split.count",
				Value: Value{
					Type: ValueTypeUint16,
					Data: count,
				},
			},
			Metadata{
				Key: "split.tensors.count",
				Value: Value{
					Type: ValueTypeInt32,
					Data: int32(len(f.Tensors)),
				},
			},
		)
		tensors := make([]TensorData, len(partition))
		for tensorIndex, info := range partition {
			shape := make([]uint64, info.Dimensions)
			copy(shape, info.Shape[:info.Dimensions])
			tensors[tensorIndex] = TensorData{
				Name:  info.Name,
				Shape: shape,
				Type:  info.Type,
				Data:  &tensorRangeReader{file: f, tensor: info},
			}
		}
		destination, openErr := open(uint16(index), count)
		if openErr != nil {
			return fmt.Errorf("open GGUF split %d of %d: %w", index+1, count, openErr)
		}
		if destination == nil {
			return fmt.Errorf("open GGUF split %d of %d returned a nil writer", index+1, count)
		}
		writeErr := Write(
			destination,
			metadata,
			tensors,
			WriteOptions{Version: f.Version, Alignment: f.Alignment},
		)
		closeErr := destination.Close()
		if writeErr != nil {
			return fmt.Errorf("write GGUF split %d of %d: %w", index+1, count, writeErr)
		}
		if closeErr != nil {
			return fmt.Errorf("close GGUF split %d of %d: %w", index+1, count, closeErr)
		}
	}
	return nil
}

func (f *File) planSplits(options SplitOptions) ([][]TensorInfo, error) {
	if len(f.Tensors) > math.MaxInt32 {
		return nil, fmt.Errorf("GGUF tensor count %d exceeds split int32 metadata", len(f.Tensors))
	}
	maxTensors := options.MaxTensors
	if maxTensors <= 0 {
		maxTensors = 128
	}
	partitions := make([][]TensorInfo, 0)
	if options.NoTensorsInFirstFile {
		partitions = append(partitions, nil)
	}
	current := make([]TensorInfo, 0, min(maxTensors, len(f.Tensors)))
	var currentBytes uint64
	flush := func() {
		if len(current) == 0 {
			return
		}
		partitions = append(partitions, current)
		current = make([]TensorInfo, 0, min(maxTensors, len(f.Tensors)))
		currentBytes = 0
	}
	for _, tensor := range f.Tensors {
		paddedSize, overflow := alignUp(tensor.Size, f.Alignment)
		if overflow {
			return nil, fmt.Errorf("tensor %q padded size overflows uint64", tensor.Name)
		}
		exceedsBytes := false
		if options.MaxBytes != 0 && len(current) != 0 {
			exceedsBytes = currentBytes > options.MaxBytes ||
				paddedSize > options.MaxBytes-currentBytes
		}
		if len(current) >= maxTensors || exceedsBytes {
			flush()
		}
		current = append(current, tensor)
		if currentBytes > math.MaxUint64-paddedSize {
			return nil, errors.New("GGUF split tensor bytes overflow uint64")
		}
		currentBytes += paddedSize
	}
	flush()
	if len(partitions) == 0 {
		partitions = append(partitions, nil)
	}
	if len(partitions) > math.MaxUint16 {
		return nil, fmt.Errorf("GGUF split count %d exceeds uint16", len(partitions))
	}
	return partitions, nil
}
