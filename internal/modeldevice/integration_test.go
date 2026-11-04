package modeldevice

import (
	"bytes"
	"encoding/binary"
	"testing"

	"overgo/internal/gguf"
	"overgo/internal/tensor/dtype"
	"overgo/internal/testskip"
)

func requireIntegration(t *testing.T) {
	t.Helper()
	testskip.Short(t)
}

// hostTensorFixture writes a one-tensor GGUF file, four F32 values named
// weight, the same fixture the model inventory tests build for themselves.
func hostTensorFixture(t *testing.T) []byte {
	t.Helper()
	var payload, buffer bytes.Buffer
	for value := float32(1); value <= 4; value++ {
		if err := binary.Write(&payload, binary.LittleEndian, value); err != nil {
			t.Fatal(err)
		}
	}
	if err := gguf.Write(&buffer, nil, []gguf.TensorData{{Name: "weight", Shape: []uint64{4}, Type: dtype.F32, Data: &payload}}, gguf.WriteOptions{}); err != nil {
		t.Fatal(err)
	}
	return buffer.Bytes()
}
