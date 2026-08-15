package graphruntime

import (
	"testing"

	"overgo/internal/tensor"
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
