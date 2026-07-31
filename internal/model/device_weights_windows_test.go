//go:build windows

package model

import (
	"bytes"
	"context"
	"encoding/binary"
	"os"
	"testing"

	"llamacpp2go/internal/cuda/device"
	"llamacpp2go/internal/gguf"
	"llamacpp2go/internal/tensor/dtype"
)

func TestDeviceWeightsIntegration(t *testing.T) {
	if os.Getenv("LLAMACPP2GO_CUDA_TEST") == "" {
		t.Skip("set LLAMACPP2GO_CUDA_TEST=1 to run CUDA integration tests")
	}
	data := weightFixture(t)
	file, err := gguf.Parse(bytes.NewReader(data), uint64(len(data)), gguf.DefaultOptions())
	if err != nil {
		t.Fatal(err)
	}
	worker, err := device.New(0)
	if err != nil {
		t.Fatal(err)
	}
	defer worker.Close()
	weights, err := NewDeviceWeights(worker)
	if err != nil {
		t.Fatal(err)
	}
	if err := weights.Load(context.Background(), file, file.Tensors); err != nil {
		t.Fatal(err)
	}
	if weights.Count() != 1 {
		t.Fatalf("weight count = %d, want 1", weights.Count())
	}
	var downloaded = make([]byte, 16)
	err = weights.Do(context.Background(), func(state *device.State, tensors map[string]DeviceTensor) error {
		return state.Driver.MemcpyDtoH(downloaded, tensors["weight"].Pointer)
	})
	if err != nil {
		t.Fatal(err)
	}
	for index := 0; index < 4; index++ {
		value := binary.LittleEndian.Uint32(downloaded[index*4:])
		if value != uint32(index+1) {
			t.Fatalf("downloaded value %d = %d, want %d", index, value, index+1)
		}
	}
	if err := weights.Close(); err != nil {
		t.Fatal(err)
	}
}

func weightFixture(t *testing.T) []byte {
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
	for value := uint32(1); value <= 4; value++ {
		write(value)
	}
	_, _ = buffer.Write(make([]byte, 16))
	return buffer.Bytes()
}
