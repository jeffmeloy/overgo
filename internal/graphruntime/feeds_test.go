package graphruntime

import (
	"testing"

	"overgo/internal/tensor"
	"overgo/internal/tensor/dtype"
	"overgo/internal/tensor/reference"
)

func TestHostFeedsExecute(t *testing.T) {
	builder := tensor.NewBuilder()
	feeds := NewFeeds()
	value := reference.Value{Shape: tensor.MustShape(1), Data: []float32{7}}
	node := feeds.Input(builder, "input", value)
	results, err := feeds.Execute(t.Context(), []*tensor.Tensor{node}, nil)
	if err != nil || results[node].Data[0] != 7 {
		t.Fatalf("result = %v, error = %v", results[node].Data, err)
	}
}

func TestFeedsReuseCompiledOutputGraph(t *testing.T) {
	builder := tensor.NewBuilder()
	input := builder.Input("input", dtype.F32, tensor.MustShape(1))
	first, second := builder.Scale(input, 2), builder.Scale(input, 3)
	feeds := NewFeeds()
	compiled, err := feeds.compile([]*tensor.Tensor{first})
	if err != nil {
		t.Fatal(err)
	}
	reused, err := feeds.compile([]*tensor.Tensor{first})
	if err != nil || reused != compiled {
		t.Fatalf("compiled graph reuse = %t, error = %v", reused == compiled, err)
	}
	replaced, err := feeds.compile([]*tensor.Tensor{second})
	if err != nil || replaced == compiled {
		t.Fatalf("compiled graph replacement = %t, error = %v", replaced != compiled, err)
	}
}
