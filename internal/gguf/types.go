package gguf

import (
	"fmt"

	"llamacpp2go/internal/tensor/dtype"
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
