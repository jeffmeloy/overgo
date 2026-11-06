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
	file := testcheck.Must(gguf.Parse(bytes.NewReader(data), uint64(len(data)), gguf.DefaultOptions()))(t)
	info := file.Tensors[tensor.FirstOffset]
	source := testcheck.Must(model.LoadHostTensor(t.Context(), file, info))(t)
	worker := testcheck.Must(device.New(modelTestDeviceOrdinal))(t)
	defer worker.Close()
	weights := testcheck.Must(NewDeviceConvertedWeights(worker, dtype.F32))(t)
	defer weights.Close()
	loader := NewLoader(file)
	loader.Converted(weights, file.Tensors)
	testcheck.NoError(t, loader.Load(t.Context(), nil))
	builder := tensor.NewBuilder()
	input, pointer, err := weights.Input(builder, info.Name)
	testcheck.NoError(t, err)
	output := builder.Add(input, input)
	want := testcheck.Must(reference.Execute([]*tensor.Tensor{output}, map[*tensor.Tensor]reference.Value{
		input: {Shape: source.Shape, Data: source.Data},
	}))(t)
	cuda := testcheck.Must(executor.NewWithWorker(worker))(t)
	defer cuda.Close()
	indexed := testcheck.Must(executor.CompileIndexed(output))(t)
	testcheck.NoError(t, indexed.Inputs.Set(input, pointer))
	results := testcheck.Must(cuda.ExecuteCompiled(context.WithoutCancel(t.Context()), indexed.Graph, nil, indexed.Inputs))(t)
	if !slices.Equal(results[output].Data, want[output].Data) {
		t.Fatalf("output = %v, want %v", results[output].Data, want[output].Data)
	}
}
