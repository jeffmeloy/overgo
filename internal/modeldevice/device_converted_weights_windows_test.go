//go:build windows

package modeldevice

import (
	"bytes"
	"context"
	"slices"
	"testing"

	"overgo/internal/cuda/device"
	"overgo/internal/cuda/executor"
	cudatest "overgo/internal/cuda/testutil"
	"overgo/internal/gguf"
	"overgo/internal/model"
	"overgo/internal/tensor"
	"overgo/internal/tensor/dtype"
	"overgo/internal/tensor/reference"
)

func TestDeviceConvertedWeightsFeedExecutor(t *testing.T) {
	requireIntegration(t)
	cudatest.Require(t)
	data := hostTensorFixture(t)
	file, err := gguf.Parse(bytes.NewReader(data), uint64(len(data)), gguf.DefaultOptions())
	if err != nil {
		t.Fatal(err)
	}
	info := file.Tensors[tensor.FirstOffset]
	source, err := model.LoadHostTensor(t.Context(), file, info)
	if err != nil {
		t.Fatal(err)
	}
	worker, err := device.New(modelTestDeviceOrdinal)
	if err != nil {
		t.Fatal(err)
	}
	defer worker.Close()
	weights, err := NewDeviceConvertedWeights(worker, dtype.F32)
	if err != nil {
		t.Fatal(err)
	}
	defer weights.Close()
	if err := weights.Load(t.Context(), file, file.Tensors); err != nil {
		t.Fatal(err)
	}
	builder := tensor.NewBuilder()
	input, pointer, err := weights.Input(builder, info.Name)
	if err != nil {
		t.Fatal(err)
	}
	output := builder.Add(input, input)
	want, err := reference.Execute([]*tensor.Tensor{output}, map[*tensor.Tensor]reference.Value{
		input: {Shape: source.Shape, Data: source.Data},
	})
	if err != nil {
		t.Fatal(err)
	}
	cuda, err := executor.NewWithWorker(worker)
	if err != nil {
		t.Fatal(err)
	}
	defer cuda.Close()
	indexed, err := executor.CompileIndexed(output)
	if err != nil {
		t.Fatal(err)
	}
	if err := indexed.Inputs.Set(input, pointer); err != nil {
		t.Fatal(err)
	}
	results, err := cuda.ExecuteCompiled(context.WithoutCancel(t.Context()), indexed.Graph, nil, indexed.Inputs)
	if err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(results[output].Data, want[output].Data) {
		t.Fatalf("output = %v, want %v", results[output].Data, want[output].Data)
	}
}
