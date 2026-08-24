package modelartifact

import (
	"encoding/binary"
	"errors"
	"fmt"
	"math"
	"slices"
	"strings"

	"os"
	"path/filepath"

	"overgo/internal/checked"
	"overgo/internal/gguf"
	"overgo/internal/quant"
	"overgo/internal/safetensors"
	"overgo/internal/tensor/dtype"
	"overgo/internal/tensorstats"
)

// MeasureAtLocation characterizes a model at a recorded location, dispatching on
// the inventory's format. Callers address a model by its OvergoDB identity (the
// inventory) and location, never by file type: GGUF opens the file, safetensors
// opens its directory. The opened source is closed before returning.
func MeasureAtLocation(
	inventory TensorInventoryDocument, location string, policy MeasurementPolicy,
) (TensorMeasurementDocument, error) {
	switch inventory.Format {
	case TensorFormatGGUF:
		file, err := gguf.Open(location)
		if err != nil {
			return TensorMeasurementDocument{}, err
		}
		defer file.Close()
		return MeasureGGUF(inventory, file, policy)
	case TensorFormatSafetensors:
		directory := location
		if info, err := os.Stat(location); err == nil && !info.IsDir() {
			directory = filepath.Dir(location)
		}
		source, err := safetensors.OpenSource(directory)
		if err != nil {
			return TensorMeasurementDocument{}, err
		}
		defer source.Close()
		return MeasureSafetensors(inventory, source, policy)
	default:
		return TensorMeasurementDocument{}, fmt.Errorf("model artifact: unsupported inventory format %q", inventory.Format)
	}
}

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
	meter := readMeter{limit: policy.MaxReadBytes, format: "GGUF"}
	for _, tensor := range file.Tensors {
		fact, ok := inventory.Tensor(tensor.Name)
		if !ok || fact.Storage != strings.ToLower(tensor.Type.String()) || fact.Bytes != tensor.Size ||
			!slices.Equal(fact.Shape, tensor.Shape[:tensor.Dimensions]) {
			return TensorMeasurementDocument{}, fmt.Errorf("model artifact: tensor %q differs from inventory", tensor.Name)
		}
		samples, elements, bytesNeeded, err := sampleGGUFTensorValues(file, tensor, policy.MaxSamplesPerTensor)
		if err != nil {
			return TensorMeasurementDocument{}, err
		}
		if err := meter.charge(bytesNeeded); err != nil {
			return TensorMeasurementDocument{}, err
		}
		measurement, err := measurementFromSamples(tensor.Name, elements, samples)
		if err != nil {
			return TensorMeasurementDocument{}, err
		}
		dims := tensor.Shape[:tensor.Dimensions]
		if err := collectMatrixSpectrum(&measurement, &meter, dims, policy.SpectralMaxDim,
			tensor.Size,
			func() ([]float64, error) { return fullGGUFTensorValues(file, tensor) },
		); err != nil {
			return TensorMeasurementDocument{}, err
		}
		measurements = append(measurements, measurement)
	}
	return newTensorMeasurementDocument(inventory.ID, policy, meter.used, measurements)
}

// readMeter is the single owner of a measurement run's read budget: every
// byte that leaves storage -- sampled or spectral, any format -- is charged
// before it is read, so recorded ReadBytes can never understate the reads.
type readMeter struct {
	used   uint64
	limit  uint64
	format string
}

func (m *readMeter) charge(bytes uint64) error {
	if m.used > m.limit || bytes > m.limit-m.used {
		return fmt.Errorf("model artifact: %s measurement exceeds read budget", m.format)
	}
	m.used += bytes
	return nil
}

// collectMatrixSpectrum: budgeted normalized spectrum measurement.
func collectMatrixSpectrum(
	measurement *TensorMeasurement,
	meter *readMeter,
	extents []uint64,
	maxExtent uint64,
	fullBytes uint64,
	read func() ([]float64, error),
) error {
	if !checked.Nonzero(maxExtent) {
		return nil // spectral not attempted: status stays empty by contract
	}
	if !tensorstats.IsMatrix(extents) {
		measurement.SpectralStatus = SpectralNotApplicable
		return nil
	}
	if extents[0] > maxExtent || extents[1] > maxExtent {
		measurement.SpectralStatus = SpectralDeferred
		return nil
	}
	if err := meter.charge(fullBytes); err != nil {
		return err
	}
	data, err := read()
	if err != nil {
		return err
	}
	value, ok := tensorstats.EffectiveRankOf(data, int(extents[1]), int(extents[0]))
	if !ok {
		measurement.SpectralStatus = SpectralNotApplicable
		return nil
	}
	measurement.EffectiveRank = value
	measurement.SpectralStatus = SpectralComputed
	return nil
}

// fullGGUFTensorValues reads and dequantizes every element of a GGUF tensor in
// storage order. It is sampleGGUFTensorValues at full coverage: with the sample
// count equal to the block count, evenlySpacedIndex is the identity, so the read
// is sequential. Callers must bound the tensor size (see SpectralMaxDim).
func fullGGUFTensorValues(file *gguf.File, tensor gguf.TensorInfo) ([]float64, error) {
	values, _, _, err := sampleGGUFTensorValues(file, tensor, math.MaxUint64)
	return values, err
}

// sampleGGUFTensorValues evenly samples up to maxSamples stored values from one
// GGUF tensor, dequantizing each sampled block to float64. It returns the
// samples, the tensor's total element count, and the bytes read. The sampling
// is deterministic (evenly spaced blocks) so repeated reads agree.
func sampleGGUFTensorValues(file *gguf.File, tensor gguf.TensorInfo, maxSamples uint64) ([]float64, uint64, uint64, error) {
	traits, ok := tensor.Type.Traits()
	if !ok {
		return nil, 0, 0, fmt.Errorf("model artifact: tensor %q has unsupported storage", tensor.Name)
	}
	elements, err := tensor.ElementCount()
	blocks, aligned := traits.BlockCount(elements)
	if err != nil || !aligned {
		return nil, 0, 0, fmt.Errorf("model artifact: tensor %q has invalid block geometry", tensor.Name)
	}
	blockSamples := min(blocks, maxSamples/traits.BlockSize)
	sampleElements, elementOK := checked.Mul64(blockSamples, traits.BlockSize)
	capacity, capacityOK := checked.Int(sampleElements)
	bytesRead, bytesOK := checked.Bytes(blockSamples, traits.TypeSize)
	if !checked.Nonzero(blockSamples) || !elementOK || !capacityOK || !bytesOK {
		return nil, 0, 0, errors.New("model artifact: invalid GGUF measurement policy")
	}
	samples := make([]float64, 0, capacity)
	storage := make([]byte, traits.TypeSize)
	values := make([]float32, traits.BlockSize)
	for sample := uint64(0); sample < blockSamples; sample++ {
		block := evenlySpacedIndex(sample, blockSamples, blocks)
		if err := file.ReadTensorRange(tensor, block*traits.TypeSize, storage); err != nil {
			return nil, 0, 0, err
		}
		if err := quant.DequantizeInto(tensor.Type, storage, values); err != nil {
			return nil, 0, 0, err
		}
		for _, value := range values {
			samples = append(samples, float64(value))
		}
	}
	return samples, elements, bytesRead, nil
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
	meter := readMeter{limit: policy.MaxReadBytes, format: "Safetensors"}
	for _, name := range source.Names() {
		tensor := source.Tensors[name]
		width, ok := safetensors.DTypeBytes(tensor.DType)
		if !ok {
			return TensorMeasurementDocument{}, fmt.Errorf("model artifact: tensor %q has unsupported storage", name)
		}
		elements := tensor.Elements()
		fact, ok := inventory.Tensor(name)
		if !ok || fact.Storage != strings.ToLower(tensor.DType) || fact.Bytes != uint64(tensor.Size()) ||
			!slices.Equal(fact.Shape, tensor.Shape) {
			return TensorMeasurementDocument{}, fmt.Errorf("model artifact: tensor %q differs from inventory", name)
		}
		sampleCount := min(elements, policy.MaxSamplesPerTensor)
		if !checked.Nonzero(sampleCount) {
			return TensorMeasurementDocument{}, errors.New("model artifact: invalid Safetensors measurement policy")
		}
		if err := meter.charge(sampleCount * width); err != nil {
			return TensorMeasurementDocument{}, err
		}
		readValues := func(count uint64) ([]float64, error) {
			storage := make([]byte, width)
			values := make([]float64, count)
			for sample := uint64(0); sample < count; sample++ {
				element := evenlySpacedIndex(sample, count, elements)
				if _, err := tensor.ReadAt(storage, int64(element*width)); err != nil {
					return nil, err
				}
				value, err := decodeSafetensorScalar(tensor.DType, storage)
				if err != nil {
					return nil, fmt.Errorf("model artifact: tensor %q: %w", name, err)
				}
				values[sample] = value
			}
			return values, nil
		}
		samples, err := readValues(sampleCount)
		if err != nil {
			return TensorMeasurementDocument{}, err
		}
		measurement, err := measurementFromSamples(name, elements, samples)
		if err != nil {
			return TensorMeasurementDocument{}, err
		}
		if err := collectMatrixSpectrum(&measurement, &meter, tensor.Shape, policy.SpectralMaxDim,
			elements*width,
			func() ([]float64, error) { return readValues(elements) },
		); err != nil {
			return TensorMeasurementDocument{}, err
		}
		measurements = append(measurements, measurement)
	}
	return newTensorMeasurementDocument(inventory.ID, policy, meter.used, measurements)
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

func evenlySpacedIndex(sample, sampleCount, population uint64) uint64 {
	quotient, remainder := population/sampleCount, population%sampleCount
	return sample*quotient + sample*remainder/sampleCount
}
