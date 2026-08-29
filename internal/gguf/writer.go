package gguf

import (
	"cmp"
	"errors"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"
	"slices"

	"overgo/internal/binaryschema"
	"overgo/internal/checked"
)

const (
	paddingBufferBytes = 4096
	maxCopyChunkBytes  = 1 << 30
)

// WriteOptions: controls canonical GGUF serialization
type WriteOptions struct {
	Version   uint32
	Alignment uint64
}

// TensorData: describes one tensor and its streamed storage bytes; Data must
// provide at least exact byte size implied by Shape and Type
type TensorData struct {
	Name  string
	Shape []uint64
	Type  DType
	Data  io.Reader
}

type preparedTensor struct {
	input  TensorData
	info   TensorInfo
	offset uint64
}

// WriteFileExclusive: create, stream, sync; remove partial output on failure.
func WriteFileExclusive(path string, metadata []Metadata, tensors []TensorData, options WriteOptions) error {
	absolute, err := filepath.Abs(path)
	if err != nil {
		return err
	}
	file, err := os.OpenFile(absolute, os.O_WRONLY|os.O_CREATE|os.O_EXCL, 0o644)
	if err != nil {
		return fmt.Errorf("GGUF create %s: %w", absolute, err)
	}
	succeeded := false
	defer func() {
		_ = file.Close()
		if !succeeded {
			_ = os.Remove(absolute)
		}
	}()
	if err := Write(file, metadata, tensors, options); err != nil {
		return fmt.Errorf("GGUF write %s: %w", absolute, err)
	}
	if err := file.Sync(); err != nil {
		return err
	}
	if err := file.Close(); err != nil {
		return err
	}
	succeeded = true
	return nil
}

// Write: serializes one canonical GGUF file without buffering tensor payloads
// Metadata and tensors retain caller order; tensor offsets are generated from
// their physical sizes and selected alignment
func Write(
	destination io.Writer,
	metadata []Metadata,
	tensors []TensorData,
	options WriteOptions,
) error {
	if destination == nil {
		return errors.New("GGUF destination is nil")
	}
	version := options.Version
	version = cmp.Or(version, CurrentVersion)
	if version < 2 || version > CurrentVersion {
		return fmt.Errorf("unsupported GGUF version %d", version)
	}
	alignment := options.Alignment
	alignment = cmp.Or(alignment, DefaultAlignment)
	if !isPowerOfTwo(alignment) {
		return fmt.Errorf("alignment %d is not a power of two", alignment)
	}
	if alignment > math.MaxUint32 {
		return fmt.Errorf("alignment %d exceeds GGUF uint32 metadata", alignment)
	}

	normalizedMetadata, err := prepareMetadata(metadata, alignment)
	if err != nil {
		return err
	}
	preparedTensors, err := prepareTensors(tensors, alignment)
	if err != nil {
		return err
	}

	writer := &countingWriter{destination: destination, schema: binaryschema.LittleEndian}
	if err := writer.write([]byte(Magic)); err != nil {
		return err
	}
	if err := writer.uint32(version); err != nil {
		return err
	}
	if err := writer.uint64(uint64(len(preparedTensors))); err != nil {
		return err
	}
	if err := writer.uint64(uint64(len(normalizedMetadata))); err != nil {
		return err
	}
	for _, item := range normalizedMetadata {
		if err := writer.string(item.Key); err != nil {
			return fmt.Errorf("write metadata key %q: %w", item.Key, err)
		}
		if err := writer.value(item.Value); err != nil {
			return fmt.Errorf("write metadata %q: %w", item.Key, err)
		}
	}
	for _, item := range preparedTensors {
		if err := writer.string(item.info.Name); err != nil {
			return fmt.Errorf("write tensor %q name: %w", item.info.Name, err)
		}
		if err := writer.uint32(item.info.Dimensions); err != nil {
			return fmt.Errorf("write tensor %q dimensions: %w", item.info.Name, err)
		}
		for axis := uint32(0); axis < item.info.Dimensions; axis++ {
			if err := writer.uint64(item.info.Shape[axis]); err != nil {
				return fmt.Errorf("write tensor %q shape: %w", item.info.Name, err)
			}
		}
		if err := writer.uint32(uint32(item.info.Type)); err != nil {
			return fmt.Errorf("write tensor %q type: %w", item.info.Name, err)
		}
		if err := writer.uint64(item.offset); err != nil {
			return fmt.Errorf("write tensor %q offset: %w", item.info.Name, err)
		}
	}
	if len(preparedTensors) == 0 {
		return nil
	}
	if err := writer.padTo(alignment); err != nil {
		return err
	}
	for _, item := range preparedTensors {
		if err := writer.copyExact(item.input.Data, item.info.Size); err != nil {
			return fmt.Errorf("write tensor %q data: %w", item.info.Name, err)
		}
		if err := writer.padTo(alignment); err != nil {
			return fmt.Errorf("write tensor %q padding: %w", item.info.Name, err)
		}
	}
	return nil
}

// WriteTo: emits parsed logical model as one canonical GGUF file; When
// source: split, tensor payloads stream from their original shards and split
// bookkeeping metadata is removed from single-file result
func (f *File) WriteTo(destination io.Writer, options WriteOptions) error {
	if f == nil {
		return errors.New("GGUF file is nil")
	}
	if options.Version == 0 {
		options.Version = f.Version
	}
	if options.Alignment == 0 {
		options.Alignment = f.Alignment
	}
	metadata := make([]Metadata, 0, len(f.Metadata))
	for _, item := range f.Metadata {
		if isSplitMetadataKey(item.Key) {
			continue
		}
		metadata = append(metadata, item)
	}
	tensors := make([]TensorData, len(f.Tensors))
	for index, info := range f.Tensors {
		shape := make([]uint64, info.Dimensions)
		copy(shape, info.Shape[:info.Dimensions])
		tensors[index] = TensorData{
			Name:  info.Name,
			Shape: shape,
			Type:  info.Type,
			Data:  &tensorRangeReader{file: f, tensor: info},
		}
	}
	return Write(destination, metadata, tensors, options)
}

type tensorRangeReader struct {
	file   *File
	tensor TensorInfo
	offset uint64
}

func (r *tensorRangeReader) Read(destination []byte) (int, error) {
	if r.offset >= r.tensor.Size {
		return 0, io.EOF
	}
	remaining := r.tensor.Size - r.offset
	if uint64(len(destination)) > remaining {
		destination = destination[:remaining]
	}
	if len(destination) == 0 {
		return 0, nil
	}
	if err := r.file.ReadTensorRange(r.tensor, r.offset, destination); err != nil {
		return 0, err
	}
	r.offset += uint64(len(destination))
	return len(destination), nil
}

func prepareMetadata(metadata []Metadata, alignment uint64) ([]Metadata, error) {
	result := slices.Clone(metadata)
	seen := make(map[string]int, len(result)+1)
	alignmentFound := false
	for index, item := range result {
		if item.Key == "" {
			return nil, fmt.Errorf("metadata key %d is empty", index)
		}
		if previous, duplicate := seen[item.Key]; duplicate {
			return nil, fmt.Errorf(
				"duplicate metadata key %q at indexes %d and %d",
				item.Key,
				previous,
				index,
			)
		}
		seen[item.Key] = index
		if err := validateValue(item.Value); err != nil {
			return nil, fmt.Errorf("metadata %q: %w", item.Key, err)
		}
		if item.Key == "general.alignment" {
			value, ok := item.Value.Data.(uint32)
			if item.Value.Type != ValueTypeUint32 || !ok {
				return nil, errors.New(`metadata "general.alignment" must be uint32`)
			}
			if uint64(value) != alignment {
				return nil, fmt.Errorf(
					`metadata "general.alignment" is %d, write alignment is %d`,
					value,
					alignment,
				)
			}
			alignmentFound = true
		}
	}
	if alignment != DefaultAlignment && !alignmentFound {
		result = append(result, Metadata{
			Key: "general.alignment",
			Value: Value{
				Type: ValueTypeUint32,
				Data: uint32(alignment),
			},
		})
	}
	return result, nil
}

func prepareTensors(tensors []TensorData, alignment uint64) ([]preparedTensor, error) {
	result := make([]preparedTensor, len(tensors))
	seen := make(map[string]int, len(tensors))
	var offset uint64
	for index, input := range tensors {
		if input.Name == "" {
			return nil, fmt.Errorf("tensor name %d is empty", index)
		}
		if len(input.Name) >= MaxTensorName {
			return nil, fmt.Errorf("tensor name %q is too long", input.Name)
		}
		if previous, duplicate := seen[input.Name]; duplicate {
			return nil, fmt.Errorf(
				"duplicate tensor name %q at indexes %d and %d",
				input.Name,
				previous,
				index,
			)
		}
		seen[input.Name] = index
		info, err := NewTensorInfo(input.Name, input.Type, input.Shape)
		if err != nil {
			return nil, err
		}
		info.Offset = offset
		if input.Data == nil {
			return nil, fmt.Errorf("tensor %q data is nil", input.Name)
		}
		paddedSize, ok := checked.Align(info.Size, alignment)
		if !ok || offset > math.MaxUint64-paddedSize {
			return nil, fmt.Errorf("tensor %q data offset overflows uint64", input.Name)
		}
		result[index] = preparedTensor{input: input, info: info, offset: offset}
		offset += paddedSize
	}
	return result, nil
}

func validateValue(value Value) error {
	if value.Type >= valueTypeCount {
		return fmt.Errorf("invalid GGUF value type %d", value.Type)
	}
	if value.Type != ValueTypeArray {
		return validateScalar(value.Type, value.Data)
	}
	if value.ArrayType >= valueTypeCount || value.ArrayType == ValueTypeArray {
		return fmt.Errorf("invalid GGUF array element type %d", value.ArrayType)
	}
	switch value.ArrayType {
	case ValueTypeUint8:
		_, ok := value.Data.([]uint8)
		return requireValueType(ok, "[]uint8")
	case ValueTypeInt8:
		_, ok := value.Data.([]int8)
		return requireValueType(ok, "[]int8")
	case ValueTypeUint16:
		_, ok := value.Data.([]uint16)
		return requireValueType(ok, "[]uint16")
	case ValueTypeInt16:
		_, ok := value.Data.([]int16)
		return requireValueType(ok, "[]int16")
	case ValueTypeUint32:
		_, ok := value.Data.([]uint32)
		return requireValueType(ok, "[]uint32")
	case ValueTypeInt32:
		_, ok := value.Data.([]int32)
		return requireValueType(ok, "[]int32")
	case ValueTypeFloat32:
		_, ok := value.Data.([]float32)
		return requireValueType(ok, "[]float32")
	case ValueTypeBool:
		_, ok := value.Data.([]bool)
		return requireValueType(ok, "[]bool")
	case ValueTypeString:
		_, ok := value.Data.([]string)
		return requireValueType(ok, "[]string")
	case ValueTypeUint64:
		_, ok := value.Data.([]uint64)
		return requireValueType(ok, "[]uint64")
	case ValueTypeInt64:
		_, ok := value.Data.([]int64)
		return requireValueType(ok, "[]int64")
	case ValueTypeFloat64:
		_, ok := value.Data.([]float64)
		return requireValueType(ok, "[]float64")
	default:
		return fmt.Errorf("unsupported GGUF array element type %s", value.ArrayType)
	}
}

func validateScalar(valueType ValueType, data any) error {
	var ok bool
	switch valueType {
	case ValueTypeUint8:
		_, ok = data.(uint8)
	case ValueTypeInt8:
		_, ok = data.(int8)
	case ValueTypeUint16:
		_, ok = data.(uint16)
	case ValueTypeInt16:
		_, ok = data.(int16)
	case ValueTypeUint32:
		_, ok = data.(uint32)
	case ValueTypeInt32:
		_, ok = data.(int32)
	case ValueTypeFloat32:
		_, ok = data.(float32)
	case ValueTypeBool:
		_, ok = data.(bool)
	case ValueTypeString:
		_, ok = data.(string)
	case ValueTypeUint64:
		_, ok = data.(uint64)
	case ValueTypeInt64:
		_, ok = data.(int64)
	case ValueTypeFloat64:
		_, ok = data.(float64)
	default:
		return fmt.Errorf("unsupported GGUF scalar type %s", valueType)
	}
	return requireValueType(ok, valueType.String())
}

func requireValueType(ok bool, name string) error {
	if !ok {
		return fmt.Errorf("value data does not have type %s", name)
	}
	return nil
}

type countingWriter struct {
	destination io.Writer
	offset      uint64
	buffer      [binaryschema.Uint64Bytes]byte
	schema      binaryschema.Fixed
}

func (w *countingWriter) write(data []byte) error {
	for len(data) > 0 {
		count, err := w.destination.Write(data)
		if count < 0 || count > len(data) {
			return errors.New("GGUF writer returned an invalid byte count")
		}
		if uint64(count) > math.MaxUint64-w.offset {
			return errors.New("GGUF output offset overflows uint64")
		}
		w.offset += uint64(count)
		data = data[count:]
		if err != nil {
			return err
		}
		if count == 0 {
			return io.ErrShortWrite
		}
	}
	return nil
}

func (w *countingWriter) uint8(value uint8) error {
	w.buffer[0] = value
	return w.write(w.buffer[:1])
}

func (w *countingWriter) uint16(value uint16) error {
	w.schema.PutUint16(w.buffer[:binaryschema.Uint16Bytes], value)
	return w.write(w.buffer[:binaryschema.Uint16Bytes])
}

func (w *countingWriter) uint32(value uint32) error {
	w.schema.PutUint32(w.buffer[:binaryschema.Uint32Bytes], value)
	return w.write(w.buffer[:binaryschema.Uint32Bytes])
}

func (w *countingWriter) uint64(value uint64) error {
	w.schema.PutUint64(w.buffer[:binaryschema.Uint64Bytes], value)
	return w.write(w.buffer[:binaryschema.Uint64Bytes])
}

func (w *countingWriter) string(value string) error {
	if err := w.uint64(uint64(len(value))); err != nil {
		return err
	}
	return w.write([]byte(value))
}

func (w *countingWriter) value(value Value) error {
	if err := w.uint32(uint32(value.Type)); err != nil {
		return err
	}
	if value.Type != ValueTypeArray {
		return w.scalar(value.Type, value.Data)
	}
	if err := w.uint32(uint32(value.ArrayType)); err != nil {
		return err
	}
	if err := w.uint64(uint64(value.Count())); err != nil {
		return err
	}
	return w.array(value.ArrayType, value.Data)
}

func (w *countingWriter) scalar(valueType ValueType, data any) error {
	switch valueType {
	case ValueTypeUint8:
		return w.uint8(data.(uint8))
	case ValueTypeInt8:
		return w.uint8(uint8(data.(int8)))
	case ValueTypeUint16:
		return w.uint16(data.(uint16))
	case ValueTypeInt16:
		return w.uint16(uint16(data.(int16)))
	case ValueTypeUint32:
		return w.uint32(data.(uint32))
	case ValueTypeInt32:
		return w.uint32(uint32(data.(int32)))
	case ValueTypeFloat32:
		return w.uint32(math.Float32bits(data.(float32)))
	case ValueTypeBool:
		if data.(bool) {
			return w.uint8(1)
		}
		return w.uint8(0)
	case ValueTypeString:
		return w.string(data.(string))
	case ValueTypeUint64:
		return w.uint64(data.(uint64))
	case ValueTypeInt64:
		return w.uint64(uint64(data.(int64)))
	case ValueTypeFloat64:
		return w.uint64(math.Float64bits(data.(float64)))
	default:
		return fmt.Errorf("unsupported GGUF scalar type %s", valueType)
	}
}

func (w *countingWriter) array(valueType ValueType, data any) error {
	switch valueType {
	case ValueTypeUint8:
		return w.write(data.([]uint8))
	case ValueTypeInt8:
		for _, value := range data.([]int8) {
			if err := w.uint8(uint8(value)); err != nil {
				return err
			}
		}
	case ValueTypeUint16:
		for _, value := range data.([]uint16) {
			if err := w.uint16(value); err != nil {
				return err
			}
		}
	case ValueTypeInt16:
		for _, value := range data.([]int16) {
			if err := w.uint16(uint16(value)); err != nil {
				return err
			}
		}
	case ValueTypeUint32:
		for _, value := range data.([]uint32) {
			if err := w.uint32(value); err != nil {
				return err
			}
		}
	case ValueTypeInt32:
		for _, value := range data.([]int32) {
			if err := w.uint32(uint32(value)); err != nil {
				return err
			}
		}
	case ValueTypeFloat32:
		for _, value := range data.([]float32) {
			if err := w.uint32(math.Float32bits(value)); err != nil {
				return err
			}
		}
	case ValueTypeBool:
		for _, value := range data.([]bool) {
			if err := w.scalar(ValueTypeBool, value); err != nil {
				return err
			}
		}
	case ValueTypeString:
		for _, value := range data.([]string) {
			if err := w.string(value); err != nil {
				return err
			}
		}
	case ValueTypeUint64:
		for _, value := range data.([]uint64) {
			if err := w.uint64(value); err != nil {
				return err
			}
		}
	case ValueTypeInt64:
		for _, value := range data.([]int64) {
			if err := w.uint64(uint64(value)); err != nil {
				return err
			}
		}
	case ValueTypeFloat64:
		for _, value := range data.([]float64) {
			if err := w.uint64(math.Float64bits(value)); err != nil {
				return err
			}
		}
	default:
		return fmt.Errorf("unsupported GGUF array type %s", valueType)
	}
	return nil
}

func (w *countingWriter) padTo(alignment uint64) error {
	aligned, ok := checked.Align(w.offset, alignment)
	if !ok {
		return errors.New("GGUF output alignment overflows uint64")
	}
	padding := aligned - w.offset
	zeros := make([]byte, min(padding, paddingBufferBytes))
	for padding > 0 {
		count := min(padding, uint64(len(zeros)))
		if err := w.write(zeros[:count]); err != nil {
			return err
		}
		padding -= count
	}
	return nil
}

func (w *countingWriter) copyExact(source io.Reader, size uint64) error {
	if source == nil {
		return errors.New("tensor data source is nil")
	}
	remaining := size
	for remaining > 0 {
		chunk := min(remaining, uint64(maxCopyChunkBytes))
		written, err := io.CopyN(w.destination, source, int64(chunk))
		if written > 0 {
			if uint64(written) > math.MaxUint64-w.offset {
				return errors.New("GGUF output offset overflows uint64")
			}
			w.offset += uint64(written)
			remaining -= uint64(written)
		}
		if err != nil {
			return err
		}
		if uint64(written) != chunk {
			return io.ErrUnexpectedEOF
		}
	}
	return nil
}
