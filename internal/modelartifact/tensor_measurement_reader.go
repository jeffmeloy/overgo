package modelartifact

import (
	"encoding/binary"
	"errors"
	"fmt"
	"math"
	"slices"
	"strings"

	"overgo/internal/gguf"
	"overgo/internal/quant"
	"overgo/internal/safetensors"
	"overgo/internal/tensor/dtype"
)

func MeasureGGUF(
	inventory TensorInventoryDocument,
	file *gguf.File,
	policy MeasurementPolicy,
) (TensorMeasurementDocument, error) {
	if file == nil || inventory.Format != TensorFormatGGUF || len(file.Tensors) != len(inventory.Tensors) {
		return TensorMeasurementDocument{}, errors.New("model artifact: incompatible GGUF measurement source")
	}
	if err := inventory.ValidateIdentity(); err != nil {
		return TensorMeasurementDocument{}, err
	}
	measurements := make([]TensorMeasurement, 0, len(file.Tensors))
	var readBytes uint64
	for _, tensor := range file.Tensors {
		traits, ok := tensor.Type.Traits()
		if !ok {
			return TensorMeasurementDocument{}, fmt.Errorf("model artifact: tensor %q has unsupported storage", tensor.Name)
		}
		elements, err := ggufTensorElements(tensor)
		if err != nil || elements%traits.BlockSize != 0 {
			return TensorMeasurementDocument{}, fmt.Errorf("model artifact: tensor %q has invalid block geometry", tensor.Name)
		}
		fact, ok := inventory.Tensor(tensor.Name)
		if !ok || fact.Storage != strings.ToLower(tensor.Type.String()) || fact.Bytes != tensor.Size ||
			!slices.Equal(fact.Shape, tensor.Shape[:tensor.Dimensions]) {
			return TensorMeasurementDocument{}, fmt.Errorf("model artifact: tensor %q differs from inventory", tensor.Name)
		}
		blocks := elements / traits.BlockSize
		blockSamples := min(blocks, policy.MaxSamplesPerTensor/traits.BlockSize)
		if blockSamples == 0 || blockSamples > math.MaxInt || blockSamples > math.MaxUint64/traits.TypeSize {
			return TensorMeasurementDocument{}, errors.New("model artifact: invalid GGUF measurement policy")
		}
		bytesNeeded := blockSamples * traits.TypeSize
		if readBytes > policy.MaxReadBytes || bytesNeeded > policy.MaxReadBytes-readBytes {
			return TensorMeasurementDocument{}, errors.New("model artifact: GGUF measurement exceeds read budget")
		}
		samples := make([]float64, 0, blockSamples*traits.BlockSize)
		storage := make([]byte, traits.TypeSize)
		for sample := uint64(0); sample < blockSamples; sample++ {
			block := evenlySpacedIndex(sample, blockSamples, blocks)
			if err := file.ReadTensorRange(tensor, block*traits.TypeSize, storage); err != nil {
				return TensorMeasurementDocument{}, err
			}
			values, err := quant.Dequantize(tensor.Type, storage, traits.BlockSize)
			if err != nil {
				return TensorMeasurementDocument{}, err
			}
			for _, value := range values {
				samples = append(samples, float64(value))
			}
		}
		measurement, err := measurementFromSamples(tensor.Name, elements, samples)
		if err != nil {
			return TensorMeasurementDocument{}, err
		}
		measurements = append(measurements, measurement)
		readBytes += bytesNeeded
	}
	return NewTensorMeasurementDocument(inventory.ID, policy, readBytes, measurements)
}

func MeasureSafetensors(
	inventory TensorInventoryDocument,
	source *safetensors.Source,
	policy MeasurementPolicy,
) (TensorMeasurementDocument, error) {
	if source == nil || inventory.Format != TensorFormatSafetensors || len(source.Tensors) != len(inventory.Tensors) {
		return TensorMeasurementDocument{}, errors.New("model artifact: incompatible Safetensors measurement source")
	}
	if err := inventory.ValidateIdentity(); err != nil {
		return TensorMeasurementDocument{}, err
	}
	measurements := make([]TensorMeasurement, 0, len(source.Tensors))
	var readBytes uint64
	for _, name := range source.Names() {
		tensor := source.Tensors[name]
		width, ok := safetensors.DTypeBytes(tensor.DType)
		if !ok || width > math.MaxInt {
			return TensorMeasurementDocument{}, fmt.Errorf("model artifact: tensor %q has unsupported storage", name)
		}
		elements := tensor.Elements()
		fact, ok := inventory.Tensor(name)
		if !ok || fact.Storage != strings.ToLower(tensor.DType) || fact.Bytes != uint64(tensor.Size()) ||
			!slices.Equal(fact.Shape, tensor.Shape) {
			return TensorMeasurementDocument{}, fmt.Errorf("model artifact: tensor %q differs from inventory", name)
		}
		sampleCount := min(elements, policy.MaxSamplesPerTensor)
		bytesNeeded := sampleCount * width
		if sampleCount == 0 || readBytes > policy.MaxReadBytes || bytesNeeded > policy.MaxReadBytes-readBytes {
			return TensorMeasurementDocument{}, errors.New("model artifact: Safetensors measurement exceeds read budget")
		}
		storage := make([]byte, width)
		samples := make([]float64, sampleCount)
		for sample := uint64(0); sample < sampleCount; sample++ {
			element := evenlySpacedIndex(sample, sampleCount, elements)
			if _, err := tensor.ReadAt(storage, int64(element*width)); err != nil {
				return TensorMeasurementDocument{}, err
			}
			value, err := decodeSafetensorScalar(tensor.DType, storage)
			if err != nil {
				return TensorMeasurementDocument{}, fmt.Errorf("model artifact: tensor %q: %w", name, err)
			}
			samples[sample] = value
		}
		measurement, err := measurementFromSamples(name, elements, samples)
		if err != nil {
			return TensorMeasurementDocument{}, err
		}
		measurements = append(measurements, measurement)
		readBytes += bytesNeeded
	}
	return NewTensorMeasurementDocument(inventory.ID, policy, readBytes, measurements)
}

func decodeSafetensorScalar(dataType string, data []byte) (float64, error) {
	switch strings.ToUpper(dataType) {
	case "BOOL", "U8":
		return float64(data[0]), nil
	case "I8":
		return float64(int8(data[0])), nil
	case "U16":
		return float64(binary.LittleEndian.Uint16(data)), nil
	case "I16":
		return float64(int16(binary.LittleEndian.Uint16(data))), nil
	case "F16":
		return float64(dtype.Float16ToFloat32(binary.LittleEndian.Uint16(data))), nil
	case "BF16":
		return float64(dtype.BF16ToFloat32(binary.LittleEndian.Uint16(data))), nil
	case "U32":
		return float64(binary.LittleEndian.Uint32(data)), nil
	case "I32":
		return float64(int32(binary.LittleEndian.Uint32(data))), nil
	case "F32":
		return float64(math.Float32frombits(binary.LittleEndian.Uint32(data))), nil
	case "U64":
		return float64(binary.LittleEndian.Uint64(data)), nil
	case "I64":
		return float64(int64(binary.LittleEndian.Uint64(data))), nil
	case "F64":
		return math.Float64frombits(binary.LittleEndian.Uint64(data)), nil
	default:
		return 0, fmt.Errorf("unsupported measurement dtype %q", dataType)
	}
}

func ggufTensorElements(tensor gguf.TensorInfo) (uint64, error) {
	elements := uint64(1)
	for _, dimension := range tensor.Shape[:tensor.Dimensions] {
		if dimension != 0 && elements > math.MaxUint64/dimension {
			return 0, errors.New("tensor shape overflows")
		}
		elements *= dimension
	}
	return elements, nil
}

func evenlySpacedIndex(sample, sampleCount, population uint64) uint64 {
	quotient, remainder := population/sampleCount, population%sampleCount
	return sample*quotient + sample*remainder/sampleCount
}
