package gguf

import (
	"fmt"
	"math"

	"overgo/internal/tensor/dtype"
)

const (
	Magic            = "GGUF"
	CurrentVersion   = 3
	DefaultAlignment = 32
	MaxDimensions    = 4
	MaxTensorName    = 64
)

// ValueType: GGUF metadata value type
type ValueType uint32

const (
	ValueTypeUint8 ValueType = iota
	ValueTypeInt8
	ValueTypeUint16
	ValueTypeInt16
	ValueTypeUint32
	ValueTypeInt32
	ValueTypeFloat32
	ValueTypeBool
	ValueTypeString
	ValueTypeArray
	ValueTypeUint64
	ValueTypeInt64
	ValueTypeFloat64
	valueTypeCount
)

var valueTypeNames = [...]string{
	"uint8",
	"int8",
	"uint16",
	"int16",
	"uint32",
	"int32",
	"float32",
	"bool",
	"string",
	"array",
	"uint64",
	"int64",
	"float64",
}

func (t ValueType) String() string {
	if t >= valueTypeCount {
		return fmt.Sprintf("value_type_%d", t)
	}
	return valueTypeNames[t]
}

type DType = dtype.Type
type TypeTraits = dtype.Traits

const (
	DTypeF32    = dtype.F32
	DTypeF16    = dtype.F16
	DTypeQ4_0   = dtype.Q4_0
	DTypeQ4_1   = dtype.Q4_1
	DTypeQ5_0   = dtype.Q5_0
	DTypeQ5_1   = dtype.Q5_1
	DTypeQ8_0   = dtype.Q8_0
	DTypeQ8_1   = dtype.Q8_1
	DTypeQ2K    = dtype.Q2K
	DTypeQ3K    = dtype.Q3K
	DTypeQ4K    = dtype.Q4K
	DTypeQ5K    = dtype.Q5K
	DTypeQ6K    = dtype.Q6K
	DTypeQ8K    = dtype.Q8K
	DTypeIQ2XXS = dtype.IQ2XXS
	DTypeIQ2XS  = dtype.IQ2XS
	DTypeIQ3XXS = dtype.IQ3XXS
	DTypeIQ1S   = dtype.IQ1S
	DTypeIQ4NL  = dtype.IQ4NL
	DTypeIQ3S   = dtype.IQ3S
	DTypeIQ2S   = dtype.IQ2S
	DTypeIQ4XS  = dtype.IQ4XS
	DTypeI8     = dtype.I8
	DTypeI16    = dtype.I16
	DTypeI32    = dtype.I32
	DTypeI64    = dtype.I64
	DTypeF64    = dtype.F64
	DTypeIQ1M   = dtype.IQ1M
	DTypeBF16   = dtype.BF16
	DTypeTQ1_0  = dtype.TQ1_0
	DTypeTQ2_0  = dtype.TQ2_0
	DTypeMXFP4  = dtype.MXFP4
	DTypeNVFP4  = dtype.NVFP4
	DTypeQ1_0   = dtype.Q1_0
	DTypeQ2_0   = dtype.Q2_0
	DTypeF8E4M3 = dtype.F8E4M3
	DTypeCount  = dtype.Count
)

// Value: contains one GGUF metadata value; ArrayType is set only when Type is
// ValueTypeArray
type Value struct {
	Type      ValueType
	ArrayType ValueType
	Data      any
}

func (v Value) Count() int {
	switch value := v.Data.(type) {
	case []uint8:
		return len(value)
	case []int8:
		return len(value)
	case []uint16:
		return len(value)
	case []int16:
		return len(value)
	case []uint32:
		return len(value)
	case []int32:
		return len(value)
	case []float32:
		return len(value)
	case []bool:
		return len(value)
	case []string:
		return len(value)
	case []uint64:
		return len(value)
	case []int64:
		return len(value)
	case []float64:
		return len(value)
	default:
		return 1
	}
}

// Metadata: named GGUF metadata value
type Metadata struct {
	Key   string
	Value Value
}

const (
	splitNumberKey      = "split.no"
	splitCountKey       = "split.count"
	splitTensorCountKey = "split.tensors.count"
)

func isSplitMetadataKey(key string) bool {
	return key == splitNumberKey || key == splitCountKey || key == splitTensorCountKey
}

// StringMetadata constructs string metadata.
func StringMetadata(key, value string) Metadata {
	return Metadata{Key: key, Value: Value{Type: ValueTypeString, Data: value}}
}

// Uint32Metadata constructs uint32 metadata.
func Uint32Metadata(key string, value uint32) Metadata {
	return Metadata{Key: key, Value: Value{Type: ValueTypeUint32, Data: value}}
}

// Float32Metadata constructs float32 metadata.
func Float32Metadata(key string, value float32) Metadata {
	return Metadata{Key: key, Value: Value{Type: ValueTypeFloat32, Data: value}}
}

// BoolMetadata constructs bool metadata.
func BoolMetadata(key string, value bool) Metadata {
	return Metadata{Key: key, Value: Value{Type: ValueTypeBool, Data: value}}
}

// ArrayMetadata constructs typed GGUF array metadata.
func ArrayMetadata(key string, elementType ValueType, data any) Metadata {
	return Metadata{Key: key, Value: Value{Type: ValueTypeArray, ArrayType: elementType, Data: data}}
}

// TensorInfo: describes one tensor without loading its data
type TensorInfo struct {
	Name       string
	Dimensions uint32
	Shape      [MaxDimensions]uint64
	Type       DType
	Shard      uint16
	Offset     uint64
	Size       uint64
}

func (tensor TensorInfo) Extents() []uint64 {
	extents := make([]uint64, tensor.Dimensions)
	copy(extents, tensor.Shape[:tensor.Dimensions])
	return extents
}

// MatrixRows returns rank-two row count; zero marks absence.
func (tensor *TensorInfo) MatrixRows() uint64 {
	if tensor == nil || tensor.Dimensions != 2 {
		return 0
	}
	return tensor.Shape[1]
}

// TensorRowLayout: validated rank-two storage geometry.
type TensorRowLayout struct {
	ElementsPerRow uint64
	Count          uint64
	BytesPerRow    uint64
}

// RowLayout validates rank-two quantized storage.
func (tensor TensorInfo) RowLayout() (TensorRowLayout, error) {
	if tensor.Dimensions != 2 {
		return TensorRowLayout{}, fmt.Errorf("tensor %q must have rank 2", tensor.Name)
	}
	traits, ok := tensor.Type.Traits()
	width, count := tensor.Shape[0], tensor.Shape[1]
	if !ok || width == 0 || count == 0 || width%traits.BlockSize != 0 {
		return TensorRowLayout{}, fmt.Errorf("tensor %q has unsupported row layout", tensor.Name)
	}
	blocks := width / traits.BlockSize
	if blocks > math.MaxUint64/traits.TypeSize {
		return TensorRowLayout{}, fmt.Errorf("tensor %q row storage overflows", tensor.Name)
	}
	bytes := blocks * traits.TypeSize
	if count > math.MaxUint64/bytes || count*bytes != tensor.Size {
		return TensorRowLayout{}, fmt.Errorf("tensor %q storage is inconsistent with its shape", tensor.Name)
	}
	return TensorRowLayout{ElementsPerRow: width, Count: count, BytesPerRow: bytes}, nil
}

// NewTensorInfo validates and compiles one logical tensor descriptor.
func NewTensorInfo(name string, storage DType, shape []uint64) (TensorInfo, error) {
	if name == "" {
		return TensorInfo{}, fmt.Errorf("tensor name is empty")
	}
	if len(name) >= MaxTensorName {
		return TensorInfo{}, fmt.Errorf("tensor name %q is too long", name)
	}
	if len(shape) == 0 || len(shape) > MaxDimensions {
		return TensorInfo{}, fmt.Errorf("tensor %q has invalid dimension count %d", name, len(shape))
	}
	info := TensorInfo{
		Name: name, Dimensions: uint32(len(shape)), Shape: [MaxDimensions]uint64{1, 1, 1, 1}, Type: storage,
	}
	for axis, dimension := range shape {
		if dimension == 0 || dimension > math.MaxInt64 {
			return TensorInfo{}, fmt.Errorf("tensor %q dimension %d is invalid: %d", name, axis, dimension)
		}
		info.Shape[axis] = dimension
	}
	elements, err := info.ElementCount()
	if err != nil {
		return TensorInfo{}, err
	}
	info.Size, err = storage.StorageBytes(elements, info.Shape[0])
	if err != nil {
		return TensorInfo{}, fmt.Errorf("tensor %q: %w", name, err)
	}
	return info, nil
}

// ElementCount: validated shape product.
func (tensor TensorInfo) ElementCount() (uint64, error) {
	if tensor.Dimensions == 0 || tensor.Dimensions > MaxDimensions {
		return 0, fmt.Errorf("tensor %q has invalid dimension count %d", tensor.Name, tensor.Dimensions)
	}
	elements := uint64(1)
	for _, dimension := range tensor.Shape[:tensor.Dimensions] {
		if dimension == 0 || elements > math.MaxUint64/dimension {
			return 0, fmt.Errorf("tensor %q element count overflows uint64", tensor.Name)
		}
		elements *= dimension
	}
	return elements, nil
}

// ReverseShape converts source-major dimensions to GGML order.
func ReverseShape(shape []uint64) []uint64 {
	result := make([]uint64, len(shape))
	for index, dimension := range shape {
		result[len(shape)-1-index] = dimension
	}
	return result
}
