package gguf

import (
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"math"
	"os"
	"strings"
)

// Options: bounds allocations made from file-controlled counts
type Options struct {
	MaxStringBytes   uint64
	MaxArrayElements uint64
	MaxMetadata      uint64
	MaxTensors       uint64
	MaxAlignment     uint64
	MaxSplitFiles    uint64
}

func DefaultOptions() Options {
	return Options{
		MaxStringBytes:   256 << 20,
		MaxArrayElements: 16 << 20,
		MaxMetadata:      1 << 20,
		MaxTensors:       1 << 20,
		MaxAlignment:     1 << 20,
		MaxSplitFiles:    1024,
	}
}

// File: parsed GGUF file; Tensor data remains in backing ReaderAt
type File struct {
	Version    uint32
	Alignment  uint64
	Metadata   []Metadata
	Tensors    []TensorInfo
	DataOffset uint64
	DataSize   uint64
	SplitCount uint16

	source          io.ReaderAt
	size            uint64
	closers         []io.Closer
	metadataByKey   map[string]int
	tensorByName    map[string]int
	tensorLocations map[string]tensorLocation
}

type tensorLocation struct {
	source     io.ReaderAt
	size       uint64
	dataOffset uint64
	tensor     TensorInfo
}

// Open: parses GGUF file and automatically loads every locally named split
func Open(path string) (*File, error) {
	return OpenWithOptions(path, DefaultOptions())
}

// OpenWithOptions: parses GGUF file with explicit allocation and split-count
// limits; Split models must be opened through their first
// <prefix>-00001-of-XXXXX.gguf file
func OpenWithOptions(path string, options Options) (*File, error) {
	options = normalizeOptions(options)
	file, err := openPath(path, options)
	if err != nil {
		return nil, err
	}
	succeeded := false
	defer func() {
		if !succeeded {
			_ = file.Close()
		}
	}()

	splitCount, configured, err := metadataUint16(file, "split.count")
	if err != nil {
		return nil, err
	}
	if !configured || splitCount <= 1 {
		file.SplitCount = 1
		succeeded = true
		return file, nil
	}
	if uint64(splitCount) > options.MaxSplitFiles {
		return nil, fmt.Errorf(
			"GGUF split count %d exceeds limit %d",
			splitCount,
			options.MaxSplitFiles,
		)
	}
	splitIndex, ok, err := metadataUint16(file, "split.no")
	if err != nil {
		return nil, err
	}
	if !ok {
		return nil, errors.New(`split GGUF is missing uint16 metadata "split.no"`)
	}
	if splitIndex != 0 {
		return nil, fmt.Errorf(
			"split GGUF index is %d; model must be opened through its first split",
			splitIndex,
		)
	}
	totalTensors, ok, err := metadataTensorCount(file)
	if err != nil {
		return nil, err
	}
	if !ok {
		return nil, errors.New(`split GGUF is missing int32 metadata "split.tensors.count"`)
	}
	if uint64(totalTensors) > options.MaxTensors {
		return nil, fmt.Errorf(
			"split tensor count %d exceeds limit %d",
			totalTensors,
			options.MaxTensors,
		)
	}
	prefix, err := splitPathPrefix(path, splitIndex, splitCount)
	if err != nil {
		return nil, err
	}
	file.SplitCount = splitCount
	for index := uint16(1); index < splitCount; index++ {
		partPath := formatSplitPath(prefix, index, splitCount)
		part, openErr := openPath(partPath, options)
		if openErr != nil {
			return nil, fmt.Errorf("open GGUF split %d %q: %w", index, partPath, openErr)
		}
		file.closers = append(file.closers, part.closers...)
		part.closers = nil
		if err := file.appendSplit(part, index, splitCount, totalTensors); err != nil {
			return nil, fmt.Errorf("GGUF split %d %q: %w", index, partPath, err)
		}
	}
	if len(file.Tensors) != int(totalTensors) {
		return nil, fmt.Errorf(
			"split GGUF declares %d tensors but %d were loaded",
			totalTensors,
			len(file.Tensors),
		)
	}
	succeeded = true
	return file, nil
}

func openPath(path string, options Options) (*File, error) {
	handle, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	stat, err := handle.Stat()
	if err != nil {
		_ = handle.Close()
		return nil, err
	}
	if stat.Size() < 0 {
		_ = handle.Close()
		return nil, errors.New("GGUF file has a negative size")
	}
	file, err := Parse(handle, uint64(stat.Size()), options)
	if err != nil {
		_ = handle.Close()
		return nil, err
	}
	file.closers = []io.Closer{handle}
	return file, nil
}

// Parse: parses GGUF directory from random-access source
func Parse(source io.ReaderAt, size uint64, options Options) (*File, error) {
	if source == nil {
		return nil, errors.New("GGUF source is nil")
	}
	options = normalizeOptions(options)
	cursor := &cursor{source: source, size: size, options: options}

	magic, err := cursor.bytes(4)
	if err != nil {
		return nil, fmt.Errorf("read GGUF magic: %w", err)
	}
	if string(magic) != Magic {
		return nil, fmt.Errorf("invalid GGUF magic %q", string(magic))
	}
	version, err := cursor.uint32()
	if err != nil {
		return nil, fmt.Errorf("read GGUF version: %w", err)
	}
	if version < 2 || version > CurrentVersion {
		return nil, fmt.Errorf("unsupported GGUF version %d", version)
	}
	tensorCount, err := cursor.uint64()
	if err != nil {
		return nil, fmt.Errorf("read tensor count: %w", err)
	}
	metadataCount, err := cursor.uint64()
	if err != nil {
		return nil, fmt.Errorf("read metadata count: %w", err)
	}
	if tensorCount > options.MaxTensors {
		return nil, fmt.Errorf("tensor count %d exceeds limit %d", tensorCount, options.MaxTensors)
	}
	if metadataCount > options.MaxMetadata {
		return nil, fmt.Errorf("metadata count %d exceeds limit %d", metadataCount, options.MaxMetadata)
	}
	if tensorCount > uint64(maxInt()) || metadataCount > uint64(maxInt()) {
		return nil, errors.New("GGUF directory count exceeds addressable memory")
	}

	file := &File{
		Version:       version,
		Alignment:     DefaultAlignment,
		Metadata:      make([]Metadata, 0, int(metadataCount)),
		Tensors:       make([]TensorInfo, 0, int(tensorCount)),
		source:        source,
		size:          size,
		metadataByKey: make(map[string]int, int(metadataCount)),
		tensorByName:  make(map[string]int, int(tensorCount)),
	}

	for i := uint64(0); i < metadataCount; i++ {
		key, readErr := cursor.string()
		if readErr != nil {
			return nil, fmt.Errorf("read metadata key %d: %w", i, readErr)
		}
		if key == "" {
			return nil, fmt.Errorf("metadata key %d is empty", i)
		}
		if previous, exists := file.metadataByKey[key]; exists {
			return nil, fmt.Errorf("duplicate metadata key %q at indexes %d and %d", key, previous, i)
		}
		value, readErr := cursor.value()
		if readErr != nil {
			return nil, fmt.Errorf("read metadata %q: %w", key, readErr)
		}
		file.metadataByKey[key] = len(file.Metadata)
		file.Metadata = append(file.Metadata, Metadata{Key: key, Value: value})
	}

	if index, ok := file.metadataByKey["general.alignment"]; ok {
		value := file.Metadata[index].Value
		alignment, valid := value.Data.(uint32)
		if value.Type != ValueTypeUint32 || !valid {
			return nil, errors.New(`metadata "general.alignment" must be uint32`)
		}
		file.Alignment = uint64(alignment)
	}
	if !isPowerOfTwo(file.Alignment) {
		return nil, fmt.Errorf("alignment %d is not a power of two", file.Alignment)
	}
	if file.Alignment > options.MaxAlignment {
		return nil, fmt.Errorf("alignment %d exceeds limit %d", file.Alignment, options.MaxAlignment)
	}

	var expectedOffset uint64
	for i := uint64(0); i < tensorCount; i++ {
		tensor, readErr := cursor.tensor()
		if readErr != nil {
			return nil, fmt.Errorf("read tensor %d: %w", i, readErr)
		}
		if previous, exists := file.tensorByName[tensor.Name]; exists {
			return nil, fmt.Errorf("duplicate tensor name %q at indexes %d and %d", tensor.Name, previous, i)
		}
		if tensor.Offset != expectedOffset {
			return nil, fmt.Errorf("tensor %q offset is %d, expected %d", tensor.Name, tensor.Offset, expectedOffset)
		}
		paddedSize, overflow := alignUp(tensor.Size, file.Alignment)
		if overflow || expectedOffset > math.MaxUint64-paddedSize {
			return nil, fmt.Errorf("tensor %q data size overflows uint64", tensor.Name)
		}
		expectedOffset += paddedSize
		file.tensorByName[tensor.Name] = len(file.Tensors)
		file.Tensors = append(file.Tensors, tensor)
	}

	dataOffset := cursor.offset
	if tensorCount > 0 {
		var overflow bool
		dataOffset, overflow = alignUp(dataOffset, file.Alignment)
		if overflow {
			return nil, errors.New("GGUF data offset overflows uint64")
		}
	}
	if dataOffset > size || expectedOffset > size-dataOffset {
		return nil, fmt.Errorf("tensor data at %d with length %d exceeds file size %d", dataOffset, expectedOffset, size)
	}
	file.DataOffset = dataOffset
	file.DataSize = expectedOffset
	file.SplitCount = 1
	file.tensorLocations = make(map[string]tensorLocation, len(file.Tensors))
	for _, tensor := range file.Tensors {
		file.tensorLocations[tensor.Name] = tensorLocation{
			source:     source,
			size:       size,
			dataOffset: dataOffset,
			tensor:     tensor,
		}
	}
	return file, nil
}

func normalizeOptions(options Options) Options {
	defaults := DefaultOptions()
	if options.MaxStringBytes == 0 {
		options.MaxStringBytes = defaults.MaxStringBytes
	}
	if options.MaxArrayElements == 0 {
		options.MaxArrayElements = defaults.MaxArrayElements
	}
	if options.MaxMetadata == 0 {
		options.MaxMetadata = defaults.MaxMetadata
	}
	if options.MaxTensors == 0 {
		options.MaxTensors = defaults.MaxTensors
	}
	if options.MaxAlignment == 0 {
		options.MaxAlignment = defaults.MaxAlignment
	}
	if options.MaxSplitFiles == 0 {
		options.MaxSplitFiles = defaults.MaxSplitFiles
	}
	return options
}

// Close closes file opened by Open; Files parsed from caller-owned
// ReaderAt do not require closing
func (f *File) Close() error {
	if f == nil || len(f.closers) == 0 {
		return nil
	}
	closers := f.closers
	f.closers = nil
	failures := make([]error, 0)
	for _, closer := range closers {
		if err := closer.Close(); err != nil {
			failures = append(failures, err)
		}
	}
	return errors.Join(failures...)
}

// MetadataValue finds metadata value by key
func (f *File) MetadataValue(key string) (Value, bool) {
	index, ok := f.metadataByKey[key]
	if !ok {
		return Value{}, false
	}
	return f.Metadata[index].Value, true
}

// Tensor finds tensor descriptor by name
func (f *File) Tensor(name string) (TensorInfo, bool) {
	index, ok := f.tensorByName[name]
	if !ok {
		return TensorInfo{}, false
	}
	return f.Tensors[index], true
}

// ReadTensorData: reads entire tensor into caller-provided buffer
func (f *File) ReadTensorData(tensor TensorInfo, destination []byte) error {
	if uint64(len(destination)) != tensor.Size {
		return fmt.Errorf("tensor %q destination has %d bytes, need %d", tensor.Name, len(destination), tensor.Size)
	}
	return f.ReadTensorRange(tensor, 0, destination)
}

// ReadTensorRange: reads bounded byte range of tensor; permits large
// weights to be streamed to device without allocating full host copy
func (f *File) ReadTensorRange(tensor TensorInfo, tensorOffset uint64, destination []byte) error {
	if tensorOffset > tensor.Size || uint64(len(destination)) > tensor.Size-tensorOffset {
		return fmt.Errorf(
			"tensor %q range [%d,%d) exceeds tensor size %d",
			tensor.Name,
			tensorOffset,
			tensorOffset+uint64(len(destination)),
			tensor.Size,
		)
	}
	source := f.source
	size := f.size
	dataOffset := f.DataOffset
	if location, ok := f.tensorLocations[tensor.Name]; ok {
		if tensor != location.tensor {
			return fmt.Errorf("tensor %q descriptor does not match the GGUF directory", tensor.Name)
		}
		source = location.source
		size = location.size
		dataOffset = location.dataOffset
	}
	offset := dataOffset + tensor.Offset
	if offset < dataOffset || offset > math.MaxUint64-tensorOffset {
		return fmt.Errorf("tensor %q offset overflows uint64", tensor.Name)
	}
	offset += tensorOffset
	if offset > size || uint64(len(destination)) > size-offset {
		return fmt.Errorf("tensor %q range is outside the GGUF file", tensor.Name)
	}
	if offset > math.MaxInt64 {
		return fmt.Errorf("tensor %q offset exceeds ReaderAt range", tensor.Name)
	}
	n, err := source.ReadAt(destination, int64(offset))
	if err != nil && !errors.Is(err, io.EOF) {
		return err
	}
	if n != len(destination) {
		return io.ErrUnexpectedEOF
	}
	return nil
}

func (f *File) appendSplit(
	part *File,
	index, count uint16,
	totalTensors uint32,
) error {
	if part.Version != f.Version {
		return fmt.Errorf("version is %d, expected %d", part.Version, f.Version)
	}
	partCount, ok, err := metadataUint16(part, "split.count")
	if err != nil {
		return err
	}
	if !ok || partCount != count {
		return fmt.Errorf(`metadata "split.count" is %d, expected %d`, partCount, count)
	}
	partIndex, ok, err := metadataUint16(part, "split.no")
	if err != nil {
		return err
	}
	if !ok || partIndex != index {
		return fmt.Errorf(`metadata "split.no" is %d, expected %d`, partIndex, index)
	}
	partTotal, ok, err := metadataTensorCount(part)
	if err != nil {
		return err
	}
	if !ok || partTotal != totalTensors {
		return fmt.Errorf(
			`metadata "split.tensors.count" is %d, expected %d`,
			partTotal,
			totalTensors,
		)
	}
	if f.DataSize > math.MaxUint64-part.DataSize {
		return errors.New("combined split tensor data size overflows uint64")
	}
	f.DataSize += part.DataSize
	for _, original := range part.Tensors {
		if previous, exists := f.tensorByName[original.Name]; exists {
			return fmt.Errorf(
				"duplicate tensor name %q at combined indexes %d and %d",
				original.Name,
				previous,
				len(f.Tensors),
			)
		}
		tensor := original
		tensor.Shard = index
		location, exists := part.tensorLocations[original.Name]
		if !exists {
			return fmt.Errorf("tensor %q has no shard location", original.Name)
		}
		location.tensor = tensor
		f.tensorByName[tensor.Name] = len(f.Tensors)
		f.tensorLocations[tensor.Name] = location
		f.Tensors = append(f.Tensors, tensor)
	}
	return nil
}

func metadataUint16(file *File, key string) (uint16, bool, error) {
	value, ok := file.MetadataValue(key)
	if !ok {
		return 0, false, nil
	}
	result, valid := value.Data.(uint16)
	if value.Type != ValueTypeUint16 || !valid {
		return 0, true, fmt.Errorf(`metadata %q must be uint16`, key)
	}
	return result, true, nil
}

func metadataTensorCount(file *File) (uint32, bool, error) {
	const key = "split.tensors.count"
	value, ok := file.MetadataValue(key)
	if !ok {
		return 0, false, nil
	}
	result, valid := value.Data.(int32)
	if value.Type != ValueTypeInt32 || !valid || result < 0 {
		return 0, true, fmt.Errorf(`metadata %q must be a non-negative int32`, key)
	}
	return uint32(result), true, nil
}

func splitPathPrefix(path string, index, count uint16) (string, error) {
	suffix := fmt.Sprintf("-%05d-of-%05d.gguf", int(index)+1, count)
	if !strings.HasSuffix(path, suffix) {
		return "", fmt.Errorf(
			"split GGUF path %q does not end in %q",
			path,
			suffix,
		)
	}
	prefix := strings.TrimSuffix(path, suffix)
	if prefix == "" {
		return "", fmt.Errorf("split GGUF path %q has an empty prefix", path)
	}
	return prefix, nil
}

func formatSplitPath(prefix string, index, count uint16) string {
	return fmt.Sprintf("%s-%05d-of-%05d.gguf", prefix, int(index)+1, count)
}

type cursor struct {
	source  io.ReaderAt
	size    uint64
	offset  uint64
	options Options
}

func (c *cursor) bytes(count uint64) ([]byte, error) {
	if count > c.size-c.offset {
		return nil, io.ErrUnexpectedEOF
	}
	if count > uint64(maxInt()) {
		return nil, errors.New("requested read exceeds addressable memory")
	}
	buffer := make([]byte, int(count))
	n, err := c.source.ReadAt(buffer, int64(c.offset))
	c.offset += uint64(n)
	if err != nil && !errors.Is(err, io.EOF) {
		return nil, err
	}
	if n != len(buffer) {
		return nil, io.ErrUnexpectedEOF
	}
	return buffer, nil
}

func (c *cursor) uint8() (uint8, error) {
	data, err := c.bytes(1)
	if err != nil {
		return 0, err
	}
	return data[0], nil
}

func (c *cursor) uint16() (uint16, error) {
	data, err := c.bytes(2)
	if err != nil {
		return 0, err
	}
	return binary.LittleEndian.Uint16(data), nil
}

func (c *cursor) uint32() (uint32, error) {
	data, err := c.bytes(4)
	if err != nil {
		return 0, err
	}
	return binary.LittleEndian.Uint32(data), nil
}

func (c *cursor) uint64() (uint64, error) {
	data, err := c.bytes(8)
	if err != nil {
		return 0, err
	}
	return binary.LittleEndian.Uint64(data), nil
}

func (c *cursor) string() (string, error) {
	length, err := c.uint64()
	if err != nil {
		return "", err
	}
	if length > c.options.MaxStringBytes {
		return "", fmt.Errorf("string length %d exceeds limit %d", length, c.options.MaxStringBytes)
	}
	data, err := c.bytes(length)
	if err != nil {
		return "", err
	}
	return string(data), nil
}

func (c *cursor) value() (Value, error) {
	rawType, err := c.uint32()
	if err != nil {
		return Value{}, err
	}
	valueType := ValueType(rawType)
	if valueType >= valueTypeCount {
		return Value{}, fmt.Errorf("invalid GGUF value type %d", rawType)
	}
	if valueType != ValueTypeArray {
		data, readErr := c.scalar(valueType)
		return Value{Type: valueType, Data: data}, readErr
	}

	rawArrayType, err := c.uint32()
	if err != nil {
		return Value{}, err
	}
	arrayType := ValueType(rawArrayType)
	if arrayType >= valueTypeCount || arrayType == ValueTypeArray {
		return Value{}, fmt.Errorf("invalid GGUF array element type %d", rawArrayType)
	}
	count, err := c.uint64()
	if err != nil {
		return Value{}, err
	}
	if count > c.options.MaxArrayElements {
		return Value{}, fmt.Errorf("array length %d exceeds limit %d", count, c.options.MaxArrayElements)
	}
	if count > uint64(maxInt()) {
		return Value{}, errors.New("array length exceeds addressable memory")
	}
	data, err := c.array(arrayType, int(count))
	if err != nil {
		return Value{}, err
	}
	return Value{Type: ValueTypeArray, ArrayType: arrayType, Data: data}, nil
}

func (c *cursor) scalar(valueType ValueType) (any, error) {
	switch valueType {
	case ValueTypeUint8:
		return c.uint8()
	case ValueTypeInt8:
		value, err := c.uint8()
		return int8(value), err
	case ValueTypeUint16:
		return c.uint16()
	case ValueTypeInt16:
		value, err := c.uint16()
		return int16(value), err
	case ValueTypeUint32:
		return c.uint32()
	case ValueTypeInt32:
		value, err := c.uint32()
		return int32(value), err
	case ValueTypeFloat32:
		value, err := c.uint32()
		return math.Float32frombits(value), err
	case ValueTypeBool:
		value, err := c.uint8()
		return value != 0, err
	case ValueTypeString:
		return c.string()
	case ValueTypeUint64:
		return c.uint64()
	case ValueTypeInt64:
		value, err := c.uint64()
		return int64(value), err
	case ValueTypeFloat64:
		value, err := c.uint64()
		return math.Float64frombits(value), err
	default:
		return nil, fmt.Errorf("unsupported scalar type %s", valueType)
	}
}

func (c *cursor) array(valueType ValueType, count int) (any, error) {
	width := fixedWidth(valueType)
	if width != 0 {
		bytesNeeded := uint64(count) * width
		if uint64(count) != 0 && bytesNeeded/uint64(count) != width {
			return nil, errors.New("array byte size overflows uint64")
		}
		if bytesNeeded > c.size-c.offset {
			return nil, io.ErrUnexpectedEOF
		}
	}
	switch valueType {
	case ValueTypeUint8:
		return c.bytes(uint64(count))
	case ValueTypeInt8:
		raw, err := c.bytes(uint64(count))
		values := make([]int8, len(raw))
		for i := range raw {
			values[i] = int8(raw[i])
		}
		return values, err
	case ValueTypeUint16:
		values := make([]uint16, count)
		return values, readArray(values, c.uint16)
	case ValueTypeInt16:
		values := make([]int16, count)
		err := readArrayConvert(values, c.uint16, func(v uint16) int16 { return int16(v) })
		return values, err
	case ValueTypeUint32:
		values := make([]uint32, count)
		return values, readArray(values, c.uint32)
	case ValueTypeInt32:
		values := make([]int32, count)
		err := readArrayConvert(values, c.uint32, func(v uint32) int32 { return int32(v) })
		return values, err
	case ValueTypeFloat32:
		values := make([]float32, count)
		err := readArrayConvert(values, c.uint32, math.Float32frombits)
		return values, err
	case ValueTypeBool:
		raw, err := c.bytes(uint64(count))
		values := make([]bool, len(raw))
		for i := range raw {
			values[i] = raw[i] != 0
		}
		return values, err
	case ValueTypeString:
		if uint64(count) > math.MaxUint64/8 || uint64(count)*8 > c.size-c.offset {
			return nil, io.ErrUnexpectedEOF
		}
		values := make([]string, count)
		for i := range values {
			value, err := c.string()
			if err != nil {
				return nil, err
			}
			values[i] = value
		}
		return values, nil
	case ValueTypeUint64:
		values := make([]uint64, count)
		return values, readArray(values, c.uint64)
	case ValueTypeInt64:
		values := make([]int64, count)
		err := readArrayConvert(values, c.uint64, func(v uint64) int64 { return int64(v) })
		return values, err
	case ValueTypeFloat64:
		values := make([]float64, count)
		err := readArrayConvert(values, c.uint64, math.Float64frombits)
		return values, err
	default:
		return nil, fmt.Errorf("unsupported array type %s", valueType)
	}
}

func (c *cursor) tensor() (TensorInfo, error) {
	name, err := c.string()
	if err != nil {
		return TensorInfo{}, err
	}
	if name == "" {
		return TensorInfo{}, errors.New("tensor name is empty")
	}
	if len(name) >= MaxTensorName {
		return TensorInfo{}, fmt.Errorf("tensor name %q is too long", name)
	}
	dimensions, err := c.uint32()
	if err != nil {
		return TensorInfo{}, err
	}
	if dimensions == 0 || dimensions > MaxDimensions {
		return TensorInfo{}, fmt.Errorf("tensor %q has invalid dimension count %d", name, dimensions)
	}
	tensor := TensorInfo{Name: name, Dimensions: dimensions, Shape: [MaxDimensions]uint64{1, 1, 1, 1}}
	var elements uint64 = 1
	for i := uint32(0); i < dimensions; i++ {
		dimension, readErr := c.uint64()
		if readErr != nil {
			return TensorInfo{}, readErr
		}
		if dimension == 0 || dimension > math.MaxInt64 {
			return TensorInfo{}, fmt.Errorf("tensor %q dimension %d is invalid: %d", name, i, dimension)
		}
		if elements > math.MaxUint64/dimension {
			return TensorInfo{}, fmt.Errorf("tensor %q element count overflows uint64", name)
		}
		elements *= dimension
		tensor.Shape[i] = dimension
	}
	rawType, err := c.uint32()
	if err != nil {
		return TensorInfo{}, err
	}
	tensor.Type = DType(rawType)
	traits, ok := tensor.Type.Traits()
	if !ok || traits.BlockSize == 0 || traits.TypeSize == 0 {
		return TensorInfo{}, fmt.Errorf("tensor %q has unsupported type %d", name, rawType)
	}
	if tensor.Shape[0]%traits.BlockSize != 0 {
		return TensorInfo{}, fmt.Errorf(
			"tensor %q row size %d is not divisible by %s block size %d",
			name,
			tensor.Shape[0],
			traits.Name,
			traits.BlockSize,
		)
	}
	blocks := elements / traits.BlockSize
	if blocks > math.MaxUint64/traits.TypeSize {
		return TensorInfo{}, fmt.Errorf("tensor %q byte size overflows uint64", name)
	}
	tensor.Size = blocks * traits.TypeSize
	tensor.Offset, err = c.uint64()
	if err != nil {
		return TensorInfo{}, err
	}
	return tensor, nil
}

func fixedWidth(valueType ValueType) uint64 {
	switch valueType {
	case ValueTypeUint8, ValueTypeInt8, ValueTypeBool:
		return 1
	case ValueTypeUint16, ValueTypeInt16:
		return 2
	case ValueTypeUint32, ValueTypeInt32, ValueTypeFloat32:
		return 4
	case ValueTypeUint64, ValueTypeInt64, ValueTypeFloat64:
		return 8
	default:
		return 0
	}
}

func readArray[T any](values []T, read func() (T, error)) error {
	for i := range values {
		value, err := read()
		if err != nil {
			return err
		}
		values[i] = value
	}
	return nil
}

func readArrayConvert[T, U any](values []T, read func() (U, error), convert func(U) T) error {
	for i := range values {
		value, err := read()
		if err != nil {
			return err
		}
		values[i] = convert(value)
	}
	return nil
}

func alignUp(value, alignment uint64) (uint64, bool) {
	if alignment == 0 {
		return 0, true
	}
	mask := alignment - 1
	if value > math.MaxUint64-mask {
		return 0, true
	}
	return (value + mask) &^ mask, false
}

func isPowerOfTwo(value uint64) bool {
	return value != 0 && value&(value-1) == 0
}

func maxInt() int {
	return int(^uint(0) >> 1)
}
