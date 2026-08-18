package gguf

import (
	"errors"
	"fmt"
	"io"
	"math"
	"strings"

	"overgo/internal/quant"
	"overgo/internal/tensor/dtype"
)

const quantizationVersion = 2

// QuantizeOptions: model quantization controls.
type QuantizeOptions struct {
	WriteOptions
	ShouldQuantize       func(TensorInfo) bool
	Importance           map[string][]float32
	ImportanceFile       string
	ImportanceDatasets   []string
	ImportanceChunkCount uint32
}

// QuantizeReport: conversion totals.
type QuantizeReport struct {
	Converted   int
	Preserved   int
	InputBytes  uint64
	OutputBytes uint64
}

// QuantizeTo: bounded model quantization.
func (f *File) QuantizeTo(
	destination io.Writer,
	target DType,
	options QuantizeOptions,
) (QuantizeReport, error) {
	if f == nil {
		return QuantizeReport{}, errors.New("GGUF file is nil")
	}
	if destination == nil {
		return QuantizeReport{}, errors.New("GGUF destination is nil")
	}
	if !quant.CanQuantize(target) {
		return QuantizeReport{}, fmt.Errorf(
			"quantization target %s is not implemented",
			target,
		)
	}
	targetTraits, _ := target.Traits()
	writeOptions := options.WriteOptions
	if writeOptions.Version == 0 {
		writeOptions.Version = f.Version
	}
	if writeOptions.Alignment == 0 {
		writeOptions.Alignment = f.Alignment
	}

	metadata, err := quantizedMetadata(f.Metadata, target, options)
	if err != nil {
		return QuantizeReport{}, err
	}
	tensors := make([]TensorData, len(f.Tensors))
	var report QuantizeReport
	for index, tensor := range f.Tensors {
		shape := make([]uint64, tensor.Dimensions)
		copy(shape, tensor.Shape[:tensor.Dimensions])
		outputType := tensor.Type
		reader := io.Reader(&tensorRangeReader{file: f, tensor: tensor})
		selected := defaultQuantizeTensor(tensor, targetTraits.BlockSize)
		if options.ShouldQuantize != nil {
			selected = options.ShouldQuantize(tensor)
		}
		converted := false
		if selected && tensor.Type != target {
			var importance []float32
			if quant.RequiresImportance(target) {
				importance = options.Importance[tensor.Name]
				if len(importance) == 0 {
					if optionalImportanceTensor(tensor.Name) {
						selected = false
					} else {
						return report, fmt.Errorf("tensor %q requires importance weights for %s", tensor.Name, target)
					}
				}
			}
			if selected && tensor.Shape[0]%targetTraits.BlockSize != 0 {
				return report, fmt.Errorf(
					"tensor %q row size %d is not divisible by %s block size %d",
					tensor.Name,
					tensor.Shape[0],
					targetTraits.Name,
					targetTraits.BlockSize,
				)
			}
			if selected {
				reader, err = newQuantizingReader(f, tensor, target, importance)
				if err != nil {
					return report, err
				}
				outputType = target
				converted = true
			}
		}
		if converted {
			report.Converted++
		} else {
			report.Preserved++
		}
		outputSize, err := tensorStorageSize(tensor, outputType)
		if err != nil {
			return report, err
		}
		if report.InputBytes > math.MaxUint64-tensor.Size ||
			report.OutputBytes > math.MaxUint64-outputSize {
			return report, errors.New("quantization report byte count overflows uint64")
		}
		report.InputBytes += tensor.Size
		report.OutputBytes += outputSize
		tensors[index] = TensorData{
			Name:  tensor.Name,
			Shape: shape,
			Type:  outputType,
			Data:  reader,
		}
	}
	if err := Write(destination, metadata, tensors, writeOptions); err != nil {
		return report, err
	}
	return report, nil
}

func optionalImportanceTensor(name string) bool {
	return strings.HasSuffix(name, "token_embd.weight") || strings.HasSuffix(name, "output.weight")
}

func defaultQuantizeTensor(tensor TensorInfo, targetBlockSize uint64) bool {
	if tensor.Dimensions < 2 || tensor.Shape[0]%targetBlockSize != 0 {
		return false
	}
	traits, ok := tensor.Type.Traits()
	if !ok {
		return false
	}
	switch tensor.Type {
	case dtype.F32, dtype.F16, dtype.BF16:
		return true
	default:
		return traits.Quantized
	}
}

func quantizedMetadata(metadata []Metadata, target DType, options QuantizeOptions) ([]Metadata, error) {
	fileType, ok := quantizedFileType(target)
	if !ok {
		return nil, fmt.Errorf("no GGUF file type mapping for %s", target)
	}
	result := make([]Metadata, 0, len(metadata)+6)
	fileTypeFound := false
	versionFound := false
	for _, item := range metadata {
		if isSplitMetadataKey(item.Key) {
			continue
		}
		switch item.Key {
		case "quantize.imatrix.file", "quantize.imatrix.dataset",
			"quantize.imatrix.entries_count", "quantize.imatrix.chunks_count":
			continue
		case "general.file_type":
			item.Value = Value{Type: ValueTypeUint32, Data: fileType}
			fileTypeFound = true
		case "general.quantization_version":
			item.Value = Value{
				Type: ValueTypeUint32,
				Data: uint32(quantizationVersion),
			}
			versionFound = true
		}
		result = append(result, item)
	}
	if !fileTypeFound {
		result = append(result, Metadata{
			Key:   "general.file_type",
			Value: Value{Type: ValueTypeUint32, Data: fileType},
		})
	}
	if !versionFound {
		result = append(result, Metadata{
			Key: "general.quantization_version",
			Value: Value{
				Type: ValueTypeUint32,
				Data: uint32(quantizationVersion),
			},
		})
	}
	if options.ImportanceFile != "" {
		result = append(result, Metadata{
			Key:   "quantize.imatrix.file",
			Value: Value{Type: ValueTypeString, Data: options.ImportanceFile},
		})
		if len(options.ImportanceDatasets) != 0 {
			result = append(result, Metadata{
				Key:   "quantize.imatrix.dataset",
				Value: Value{Type: ValueTypeString, Data: options.ImportanceDatasets[0]},
			})
		}
		result = append(result, Metadata{
			Key: "quantize.imatrix.entries_count",
			Value: Value{
				Type: ValueTypeInt64,
				Data: int64(len(options.Importance)),
			},
		})
		if options.ImportanceChunkCount != 0 {
			result = append(result, Metadata{
				Key: "quantize.imatrix.chunks_count",
				Value: Value{
					Type: ValueTypeInt64,
					Data: int64(options.ImportanceChunkCount),
				},
			})
		}
	}
	return result, nil
}

func quantizedFileType(target DType) (uint32, bool) {
	switch target {
	case dtype.F32:
		return fileTypeAllF32, true
	case dtype.F16:
		return fileTypeMostlyF16, true
	case dtype.Q4_0:
		return fileTypeMostlyQ4_0, true
	case dtype.Q4_1:
		return fileTypeMostlyQ4_1, true
	case dtype.Q8_0:
		return fileTypeMostlyQ8_0, true
	case dtype.Q5_0:
		return fileTypeMostlyQ5_0, true
	case dtype.Q5_1:
		return fileTypeMostlyQ5_1, true
	case dtype.Q2K:
		return fileTypeMostlyQ2K, true
	case dtype.Q3K:
		return fileTypeMostlyQ3K, true
	case dtype.Q4K:
		return fileTypeMostlyQ4K, true
	case dtype.Q5K:
		return fileTypeMostlyQ5K, true
	case dtype.Q6K:
		return fileTypeMostlyQ6K, true
	case dtype.BF16:
		return fileTypeMostlyBF16, true
	case dtype.IQ4NL:
		return fileTypeMostlyIQ4NL, true
	case dtype.IQ2S:
		return fileTypeMostlyIQ2S, true
	case dtype.IQ2XXS:
		return fileTypeMostlyIQ2XXS, true
	case dtype.IQ2XS:
		return fileTypeMostlyIQ2XS, true
	case dtype.IQ1S:
		return fileTypeMostlyIQ1S, true
	case dtype.IQ1M:
		return fileTypeMostlyIQ1M, true
	case dtype.IQ3XXS:
		return fileTypeMostlyIQ3XXS, true
	case dtype.IQ3S:
		return fileTypeMostlyIQ3S, true
	case dtype.IQ4XS:
		return fileTypeMostlyIQ4XS, true
	case dtype.TQ1_0:
		return fileTypeMostlyTQ1_0, true
	case dtype.TQ2_0:
		return fileTypeMostlyTQ2_0, true
	case dtype.MXFP4:
		return fileTypeMostlyMXFP4, true
	case dtype.NVFP4:
		return fileTypeMostlyNVFP4, true
	case dtype.Q1_0:
		return fileTypeMostlyQ1_0, true
	case dtype.Q2_0:
		return fileTypeMostlyQ2_0, true
	default:
		return 0, false
	}
}

func tensorStorageSize(tensor TensorInfo, dataType DType) (uint64, error) {
	traits, ok := dataType.Traits()
	if !ok {
		return 0, fmt.Errorf("tensor %q has unknown type %d", tensor.Name, dataType)
	}
	elements, err := tensor.ElementCount()
	if err != nil {
		return 0, err
	}
	if elements%traits.BlockSize != 0 {
		return 0, fmt.Errorf(
			"tensor %q element count %d is not divisible by %s block size %d",
			tensor.Name,
			elements,
			traits.Name,
			traits.BlockSize,
		)
	}
	blocks := elements / traits.BlockSize
	if blocks > math.MaxUint64/traits.TypeSize {
		return 0, fmt.Errorf("tensor %q byte size overflows uint64", tensor.Name)
	}
	return blocks * traits.TypeSize, nil
}

type quantizingReader struct {
	file              *File
	tensor            TensorInfo
	target            DType
	sourceTraits      TypeTraits
	targetTraits      TypeTraits
	elementsRemaining uint64
	sourceOffset      uint64
	chunkElements     uint64
	totalElements     uint64
	rowWidth          uint64
	rowsPerGroup      uint64
	importance        []float32
	buffer            []byte
	bufferOffset      int
}

func newQuantizingReader(
	file *File,
	tensor TensorInfo,
	target DType,
	importance []float32,
) (*quantizingReader, error) {
	sourceTraits, ok := tensor.Type.Traits()
	if !ok {
		return nil, fmt.Errorf("tensor %q has unknown type %d", tensor.Name, tensor.Type)
	}
	targetTraits, ok := target.Traits()
	if !ok {
		return nil, fmt.Errorf("unknown quantization target %d", target)
	}
	elements, err := tensor.ElementCount()
	if err != nil {
		return nil, err
	}
	baseElements, overflow := leastCommonMultiple(
		sourceTraits.BlockSize,
		targetTraits.BlockSize,
	)
	if overflow || baseElements == 0 {
		return nil, fmt.Errorf("tensor %q conversion block size overflows", tensor.Name)
	}
	if elements%baseElements != 0 {
		return nil, fmt.Errorf(
			"tensor %q element count %d is not divisible by conversion block size %d",
			tensor.Name,
			elements,
			baseElements,
		)
	}
	rowWidth := tensor.Shape[0]
	if rowWidth == 0 || rowWidth%baseElements != 0 {
		return nil, fmt.Errorf("tensor %q row width is not conversion-block aligned", tensor.Name)
	}
	const targetChunkElements = uint64(256 << 10)
	chunkRows := targetChunkElements / rowWidth
	if chunkRows == 0 {
		chunkRows = 1
	}
	chunkElements := chunkRows * rowWidth
	rowsPerGroup := uint64(1)
	if tensor.Dimensions > 1 {
		rowsPerGroup = tensor.Shape[1]
	}
	groups := uint64(1)
	if tensor.Dimensions > 2 {
		groups = tensor.Shape[2]
	}
	if tensor.Dimensions > 3 && tensor.Shape[3] != 1 && len(importance) != 0 {
		return nil, fmt.Errorf("tensor %q rank-4 importance mapping is unsupported", tensor.Name)
	}
	if len(importance) != 0 && (groups != 0 && rowWidth > math.MaxUint64/groups) {
		return nil, fmt.Errorf("tensor %q importance count overflows uint64", tensor.Name)
	}
	if len(importance) != 0 && uint64(len(importance)) != rowWidth*groups {
		return nil, fmt.Errorf(
			"tensor %q importance count %d differs from expected %d",
			tensor.Name, len(importance), rowWidth*groups,
		)
	}
	return &quantizingReader{
		file:              file,
		tensor:            tensor,
		target:            target,
		sourceTraits:      sourceTraits,
		targetTraits:      targetTraits,
		elementsRemaining: elements,
		chunkElements:     chunkElements,
		totalElements:     elements,
		rowWidth:          rowWidth,
		rowsPerGroup:      rowsPerGroup,
		importance:        importance,
	}, nil
}

func (r *quantizingReader) Read(destination []byte) (int, error) {
	if len(destination) == 0 {
		return 0, nil
	}
	if r.bufferOffset == len(r.buffer) {
		if r.elementsRemaining == 0 {
			return 0, io.EOF
		}
		if err := r.fill(); err != nil {
			return 0, err
		}
	}
	count := copy(destination, r.buffer[r.bufferOffset:])
	r.bufferOffset += count
	return count, nil
}

func (r *quantizingReader) fill() error {
	elements := min(r.elementsRemaining, r.chunkElements)
	sourceBlocks := elements / r.sourceTraits.BlockSize
	if sourceBlocks > uint64(math.MaxInt)/r.sourceTraits.TypeSize {
		return errors.New("quantization source chunk exceeds addressable memory")
	}
	sourceSize := sourceBlocks * r.sourceTraits.TypeSize
	source := make([]byte, int(sourceSize))
	if err := r.file.ReadTensorRange(r.tensor, r.sourceOffset, source); err != nil {
		return err
	}
	values, err := quant.Dequantize(r.tensor.Type, source, elements)
	if err != nil {
		return fmt.Errorf("dequantize tensor %q: %w", r.tensor.Name, err)
	}
	if len(r.importance) != 0 {
		weights := make([]float32, len(values))
		firstElement := r.totalElements - r.elementsRemaining
		firstRow := firstElement / r.rowWidth
		for index := range values {
			row := firstRow + uint64(index)/r.rowWidth
			column := uint64(index) % r.rowWidth
			group := row / r.rowsPerGroup
			weights[index] = r.importance[group*r.rowWidth+column]
		}
		r.buffer, err = quant.QuantizeWeighted(r.target, values, weights)
	} else {
		r.buffer, err = quant.Quantize(r.target, values)
	}
	if err != nil {
		return fmt.Errorf("quantize tensor %q: %w", r.tensor.Name, err)
	}
	r.bufferOffset = 0
	r.sourceOffset += sourceSize
	r.elementsRemaining -= elements
	return nil
}

func leastCommonMultiple(left, right uint64) (uint64, bool) {
	divisor := greatestCommonDivisor(left, right)
	reduced := left / divisor
	if reduced > math.MaxUint64/right {
		return 0, true
	}
	return reduced * right, false
}

func greatestCommonDivisor(left, right uint64) uint64 {
	for right != 0 {
		left, right = right, left%right
	}
	return left
}
