package testutil

import (
	"bytes"
	"encoding/binary"

	"overgo/internal/gguf"
)

// GGUFScalar builds one scalar metadata entry for a hermetic GGUF fixture.
func GGUFScalar(key string, valueType gguf.ValueType, data any) gguf.Metadata {
	return gguf.Metadata{Key: key, Value: gguf.Value{Type: valueType, Data: data}}
}

// GGUFArray builds one array metadata entry for a hermetic GGUF fixture.
func GGUFArray(key string, valueType gguf.ValueType, data any) gguf.Metadata {
	return gguf.Metadata{Key: key, Value: gguf.Value{
		Type: gguf.ValueTypeArray, ArrayType: valueType, Data: data,
	}}
}

// GGUFTensorF32 builds one deterministic F32 tensor for a hermetic GGUF
// fixture: vectors near one, matrices as small signed values, both varied
// by the seed so distinct tensors never coincide.
func GGUFTensorF32(name string, shape []uint64, seed int) gguf.TensorData {
	elements := uint64(1)
	for _, dimension := range shape {
		elements *= dimension
	}
	values := make([]float32, elements)
	for index := range values {
		if len(shape) == 1 {
			values[index] = 1 + float32((index+seed)%3)*0.01
		} else {
			values[index] = float32((index*17+seed*13)%29-14) * 0.01
		}
	}
	var storage bytes.Buffer
	_ = binary.Write(&storage, binary.LittleEndian, values)
	return gguf.TensorData{Name: name, Shape: shape, Type: gguf.DTypeF32, Data: bytes.NewReader(storage.Bytes())}
}
