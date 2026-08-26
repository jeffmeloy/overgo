package graphruntime

import (
	"context"
	"testing"

	"overgo/internal/cuda/driver"
	"overgo/internal/tensor"
	"overgo/internal/tensor/dtype"
	"overgo/internal/tensor/reference"
)

func TestHostFeedsExecute(t *testing.T) {
	builder := tensor.NewBuilder()
	feeds := NewFeeds(context.Background(), nil)
	value := reference.Value{Shape: tensor.MustShape(1), Data: []float32{7}}
	node := feeds.Input(builder, "input", value)
	results, err := feeds.Execute(node)
	if err != nil || results[node].Data[0] != 7 {
		t.Fatalf("result = %v, error = %v", results[node].Data, err)
	}
}

func TestFeedsReuseCompiledOutputGraph(t *testing.T) {
	builder := tensor.NewBuilder()
	input := builder.Input("input", dtype.F32, tensor.MustShape(1))
	first, second := builder.Scale(input, 2), builder.Scale(input, 3)
	feeds := NewFeeds(context.Background(), nil)
	feeds.SetDevice(input, driver.DevicePtr(1))
	compiled, err := feeds.compile([]*tensor.Tensor{first})
	if err != nil {
		t.Fatal(err)
	}
	slot, ok := compiled.Graph.InputSlot(input)
	if !ok || compiled.Inputs.Pointers[slot] != driver.DevicePtr(1) {
		t.Fatalf("device binding slot=%d found=%t", slot, ok)
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
