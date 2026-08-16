//go:build windows

package model

import (
	"bytes"
	"context"
	cudatest "overgo/internal/cuda/testutil"
	"testing"

	"overgo/internal/cuda/device"
	"overgo/internal/cuda/executor"
	"overgo/internal/gguf"
	"overgo/internal/tensor"
)

func TestDeviceF32WeightsFeedExecutor(t *testing.T) {
	requireIntegration(t)
	cudatest.Require(t)
	data := hostTensorFixture(t)
	file, err := gguf.Parse(bytes.NewReader(data), uint64(len(data)), gguf.DefaultOptions())
	if err != nil {
		t.Fatal(err)
	}
	worker, err := device.New(0)
	if err != nil {
		t.Fatal(err)
	}
	defer worker.Close()
	weights, err := NewDeviceF32Weights(worker)
	if err != nil {
		t.Fatal(err)
	}
	defer weights.Close()
	if err := weights.Load(context.Background(), file, file.Tensors); err != nil {
		t.Fatal(err)
	}
	builder := tensor.NewBuilder()
	input, pointer, err := weights.Input(builder, "weight")
	if err != nil {
		t.Fatal(err)
	}
	output := builder.Scale(input, 2)
	cuda, err := executor.NewWithWorker(worker)
	if err != nil {
		t.Fatal(err)
	}
	defer cuda.Close()
	compiled, err := executor.Compile(output)
	if err != nil {
		t.Fatal(err)
	}
	inputs := compiled.NewDeviceInputs()
	if err := inputs.Set(input, pointer); err != nil {
		t.Fatal(err)
	}
	results, err := cuda.ExecuteCompiled(context.Background(), compiled, nil, inputs)
	if err != nil {
		t.Fatal(err)
	}
	want := []float32{2, 4, 6, 8}
	for index := range want {
		if results[output].Data[index] != want[index] {
			t.Fatalf("output[%d] = %v, want %v", index, results[output].Data[index], want[index])
		}
	}
}
