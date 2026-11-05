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
	"overgo/internal/testcheck"
)

func TestDeviceConvertedWeightsFeedExecutor(t *testing.T) {
	requireIntegration(t)
	cudatest.Require(t)
	data := hostTensorFixture(t)
	file, err := gguf.Parse(bytes.NewReader(data), uint64(len(data)), gguf.DefaultOptions())
	testcheck.NoError(t, err)
	info := file.Tensors[tensor.FirstOffset]
	source, err := model.LoadHostTensor(t.Context(), file, info)
	testcheck.NoError(t, err)
	worker, err := device.New(modelTestDeviceOrdinal)
	testcheck.NoError(t, err)
	defer worker.Close()
	weights, err := NewDeviceConvertedWeights(worker, dtype.F32)
	testcheck.NoError(t, err)
	defer weights.Close()
	loader := NewLoader(file)
	loader.Converted(weights, file.Tensors)
	testcheck.NoError(t, loader.Load(t.Context(), nil))
	builder := tensor.NewBuilder()
	input, pointer, err := weights.Input(builder, info.Name)
	testcheck.NoError(t, err)
	output := builder.Add(input, input)
	want, err := reference.Execute([]*tensor.Tensor{output}, map[*tensor.Tensor]reference.Value{
		input: {Shape: source.Shape, Data: source.Data},
	})
	testcheck.NoError(t, err)
	cuda, err := executor.NewWithWorker(worker)
	testcheck.NoError(t, err)
	defer cuda.Close()
	indexed, err := executor.CompileIndexed(output)
	testcheck.NoError(t, err)
	testcheck.NoError(t, indexed.Inputs.Set(input, pointer))
	results, err := cuda.ExecuteCompiled(context.WithoutCancel(t.Context()), indexed.Graph, nil, indexed.Inputs)
	testcheck.NoError(t, err)
	if !slices.Equal(results[output].Data, want[output].Data) {
		t.Fatalf("output = %v, want %v", results[output].Data, want[output].Data)
	}
}
