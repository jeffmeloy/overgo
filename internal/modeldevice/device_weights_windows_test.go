//go:build windows

package modeldevice

import (
	"bytes"
	"encoding/binary"
	cudatest "overgo/internal/cuda/testutil"
	"overgo/internal/testcheck"
	"testing"

	"overgo/internal/cuda/device"
	"overgo/internal/gguf"
	"overgo/internal/tensor/dtype"
)

const modelTestDeviceOrdinal = 0

func TestDeviceWeightsIntegration(t *testing.T) {
	requireIntegration(t)
	cudatest.Require(t)
	var payload, encoded bytes.Buffer
	for value := uint32(1); value <= 4; value++ {
		testcheck.NoError(t, binary.Write(&payload, binary.LittleEndian, value))
	}
	testcheck.NoError(t, gguf.Write(&encoded, nil, []gguf.TensorData{{Name: "weight", Shape: []uint64{4}, Type: dtype.F32, Data: &payload}}, gguf.WriteOptions{}))
	data := encoded.Bytes()
	file, err := gguf.Parse(bytes.NewReader(data), uint64(len(data)), gguf.DefaultOptions())
	testcheck.NoError(t, err)
	worker, err := device.New(modelTestDeviceOrdinal)
	testcheck.NoError(t, err)
	defer worker.Close()
	weights, err := NewDeviceWeights(worker)
	testcheck.NoError(t, err)
	loader := NewLoader(file)
	loader.Raw(weights, file.Tensors)
	testcheck.NoError(t, loader.Load(t.Context(), nil))
	if _, loaded := weights.Lookup("weight"); !loaded {
		t.Fatal("weight was not loaded")
	}
	var downloaded = make([]byte, 16)
	err = weights.Do(t.Context(), func(state *device.State, tensors map[string]DeviceTensor) error {
		return state.Driver.MemcpyDtoH(downloaded, tensors["weight"].Pointer)
	})
	testcheck.NoError(t, err)
	for index := range 4 {
		value := binary.LittleEndian.Uint32(downloaded[index*4:])
		if value != uint32(index+1) {
			t.Fatalf("downloaded value %d = %d, want %d", index, value, index+1)
		}
	}
	testcheck.NoError(t, weights.Close())
}
