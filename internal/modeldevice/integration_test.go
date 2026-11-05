package modeldevice

import (
	"bytes"
	"encoding/binary"
	"testing"

	"overgo/internal/gguf"
	"overgo/internal/tensor/dtype"
	"overgo/internal/testcheck"
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
		testcheck.NoError(t, binary.Write(&payload, binary.LittleEndian, value))
	}
	testcheck.NoError(t, gguf.Write(&buffer, nil, []gguf.TensorData{{Name: "weight", Shape: []uint64{4}, Type: dtype.F32, Data: &payload}}, gguf.WriteOptions{}))
	return buffer.Bytes()
}
