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
	if testing.Short() {
		t.Skip(testskip.ShortIntegration)
	}
}

// hostTensorFixture writes a one-tensor GGUF file, four F32 values named
// weight, the same fixture the model inventory tests build for themselves.
func hostTensorFixture(t *testing.T) []byte {
	t.Helper()
	var buffer bytes.Buffer
	write := func(value any) {
		if err := binary.Write(&buffer, binary.LittleEndian, value); err != nil {
			t.Fatal(err)
		}
	}
	writeString := func(value string) {
		write(uint64(len(value)))
		_, _ = buffer.WriteString(value)
	}
	_, _ = buffer.WriteString(gguf.Magic)
	write(uint32(gguf.CurrentVersion))
	write(uint64(1))
	write(uint64(0))
	writeString("weight")
	write(uint32(1))
	write(uint64(4))
	write(uint32(dtype.F32))
	write(uint64(0))
	for buffer.Len()%gguf.DefaultAlignment != 0 {
		_ = buffer.WriteByte(0)
	}
	for value := float32(1); value <= 4; value++ {
		write(value)
	}
	_, _ = buffer.Write(make([]byte, 16))
	return buffer.Bytes()
}
