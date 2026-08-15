//go:build windows

package graphruntime

import (
	"context"
	"testing"

	cudatest "overgo/internal/cuda/testutil"
	"overgo/internal/tensor"
	"overgo/internal/tensor/dtype"
	"overgo/internal/tensor/reference"
)

func TestResidentGraphReusesStaticInputs(t *testing.T) {
	cudatest.Require(t)
	builder := tensor.NewBuilder()
	dynamic := builder.Input("dynamic", dtype.F32, tensor.MustShape(2))
	static := builder.Input("static", dtype.F32, tensor.MustShape(2))
	output := builder.Add(dynamic, static)
	staticValue := reference.Value{Shape: static.Shape, Data: []float32{3, 4}}
	runtime, err := NewResidentGraph(t.Context(), 0, map[*tensor.Tensor]reference.Value{static: staticValue}, output)
	if err != nil {
		t.Fatal(err)
	}
	defer runtime.Close(context.Background())
	for _, values := range [][]float32{{1, 2}, {5, 6}} {
		result, err := runtime.Execute(t.Context(), map[*tensor.Tensor]reference.Value{
			dynamic: {Shape: dynamic.Shape, Data: values},
		})
		if err != nil {
			t.Fatal(err)
		}
		if got := result[output].Data; got[0] != values[0]+3 || got[1] != values[1]+4 {
			t.Fatalf("resident output %v", got)
		}
	}
}
