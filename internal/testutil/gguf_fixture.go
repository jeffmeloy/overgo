package testutil

import (
	"bytes"
	"encoding/binary"

	"overgo/internal/gguf"
)

// ModelBytes is a GGUF model's bytes on disk: every shard its first file declares, where the first of a split model's
// shards holds only its metadata (Flash-Next's 10.9 MB of 93.7 GB).
func ModelBytes(location string) (int64, error) {
	file, err := gguf.Open(location)
	if err != nil {
		return 0, err
	}
	var total int64
	for _, shard := range file.Shards() {
		total += int64(shard.Size)
	}
	return total, file.Close()
}

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
