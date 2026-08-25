package gguf

import (
	"errors"
	"fmt"
	"io"
	"math"
	"strings"

	"overgo/internal/extent"
	"overgo/internal/quant"
	"overgo/internal/tensor/dtype"
)

const (
	quantizationVersion       = extent.PairedExtent
	quantizationChunkElements = uint64(256 << 10)
)

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
			_, rowAligned := targetTraits.BlockCount(tensor.Shape[0])
			if selected && !rowAligned {
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
	_, rowAligned := dtype.BlockCount(tensor.Shape[0], targetBlockSize)
	if tensor.Dimensions < 2 || !rowAligned {
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
			item.Value = Value{Type: ValueTypeUint32, Data: uint32(fileType)}
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
			Value: Value{Type: ValueTypeUint32, Data: uint32(fileType)},
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

type fileTypeBinding struct {
	fileType FileType
	valid    bool
}

var quantizedFileTypes = [dtype.Count]fileTypeBinding{
	dtype.F32:    {fileTypeAllF32, true},
	dtype.F16:    {fileTypeMostlyF16, true},
	dtype.Q4_0:   {fileTypeMostlyQ4_0, true},
	dtype.Q4_1:   {fileTypeMostlyQ4_1, true},
	dtype.Q5_0:   {fileTypeMostlyQ5_0, true},
	dtype.Q5_1:   {fileTypeMostlyQ5_1, true},
	dtype.Q8_0:   {fileTypeMostlyQ8_0, true},
	dtype.Q2K:    {fileTypeMostlyQ2K, true},
	dtype.Q3K:    {fileTypeMostlyQ3K, true},
	dtype.Q4K:    {fileTypeMostlyQ4K, true},
	dtype.Q5K:    {fileTypeMostlyQ5K, true},
	dtype.Q6K:    {fileTypeMostlyQ6K, true},
	dtype.IQ2XXS: {fileTypeMostlyIQ2XXS, true},
	dtype.IQ2XS:  {fileTypeMostlyIQ2XS, true},
	dtype.IQ3XXS: {fileTypeMostlyIQ3XXS, true},
	dtype.IQ1S:   {fileTypeMostlyIQ1S, true},
	dtype.IQ4NL:  {fileTypeMostlyIQ4NL, true},
	dtype.IQ3S:   {fileTypeMostlyIQ3S, true},
	dtype.IQ2S:   {fileTypeMostlyIQ2S, true},
	dtype.IQ4XS:  {fileTypeMostlyIQ4XS, true},
	dtype.IQ1M:   {fileTypeMostlyIQ1M, true},
	dtype.BF16:   {fileTypeMostlyBF16, true},
	dtype.TQ1_0:  {fileTypeMostlyTQ1_0, true},
	dtype.TQ2_0:  {fileTypeMostlyTQ2_0, true},
	dtype.MXFP4:  {fileTypeMostlyMXFP4, true},
	dtype.NVFP4:  {fileTypeMostlyNVFP4, true},
	dtype.Q1_0:   {fileTypeMostlyQ1_0, true},
	dtype.Q2_0:   {fileTypeMostlyQ2_0, true},
}

func quantizedFileType(target DType) (FileType, bool) {
	if target >= dtype.Count {
		return 0, false
	}
	binding := quantizedFileTypes[target]
	return binding.fileType, binding.valid
}

func tensorStorageSize(tensor TensorInfo, dataType DType) (uint64, error) {
	elements, err := tensor.ElementCount()
	if err != nil {
		return 0, err
	}
	bytes, err := dataType.StorageBytes(elements, tensor.Shape[0])
	if err != nil {
		return 0, fmt.Errorf("tensor %q: %w", tensor.Name, err)
	}
	return bytes, nil
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
	if overflow || baseElements == extent.FirstOffset {
		return nil, fmt.Errorf("tensor %q conversion block size overflows", tensor.Name)
	}
	if elements%baseElements != extent.FirstOffset {
		return nil, fmt.Errorf(
			"tensor %q element count %d is not divisible by conversion block size %d",
			tensor.Name,
			elements,
			baseElements,
		)
	}
	rowWidth := tensor.Shape[extent.FirstOffset]
	if rowWidth == extent.FirstOffset || rowWidth%baseElements != extent.FirstOffset {
		return nil, fmt.Errorf("tensor %q row width is not conversion-block aligned", tensor.Name)
	}
	chunkRows := quantizationChunkElements / rowWidth
	if chunkRows == extent.FirstOffset {
		chunkRows = extent.SingletonExtent
	}
	chunkElements := chunkRows * rowWidth
	rowsPerGroup := uint64(extent.SingletonExtent)
	if tensor.Dimensions > extent.SingletonExtent {
		rowsPerGroup = tensor.Shape[extent.SingletonExtent]
	}
	groups := uint64(extent.SingletonExtent)
	if tensor.Dimensions > extent.PairedExtent {
		groups = tensor.Shape[extent.PairedExtent]
	}
	if tensor.Dimensions > extent.TripleExtent && tensor.Shape[extent.TripleExtent] != extent.SingletonExtent && len(importance) != extent.FirstOffset {
		return nil, fmt.Errorf("tensor %q rank-4 importance mapping is unsupported", tensor.Name)
	}
	if len(importance) != extent.FirstOffset && (groups != extent.FirstOffset && rowWidth > math.MaxUint64/groups) {
		return nil, fmt.Errorf("tensor %q importance count overflows uint64", tensor.Name)
	}
	if len(importance) != extent.FirstOffset && uint64(len(importance)) != rowWidth*groups {
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
	if len(destination) == extent.FirstOffset {
		return extent.FirstOffset, nil
	}
	if r.bufferOffset == len(r.buffer) {
		if r.elementsRemaining == extent.FirstOffset {
			return extent.FirstOffset, io.EOF
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
	if len(r.importance) != extent.FirstOffset {
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
