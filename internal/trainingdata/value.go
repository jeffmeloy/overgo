package trainingdata

import (
	"encoding/binary"
	"errors"
	"math"
)

const EncodingFloat32LE = "float32-le"

// EncodeFloat32 stores one processor-owned tensor payload.
func EncodeFloat32(values []float32) []byte {
	data := make([]byte, len(values)*4)
	for index, value := range values {
		binary.LittleEndian.PutUint32(data[index*4:], math.Float32bits(value))
	}
	return data
}

// Float32 decodes one processor-owned little-endian tensor payload.
func Float32(value Value) ([]float32, error) {
	if value.Encoding != EncodingFloat32LE || len(value.Data) == 0 || len(value.Data)%4 != 0 {
		return nil, errors.New("training data: invalid float32 payload")
	}
	result := make([]float32, len(value.Data)/4)
	for index := range result {
		result[index] = math.Float32frombits(binary.LittleEndian.Uint32(value.Data[index*4:]))
	}
	return result, nil
}
