package graphruntime

import (
	"testing"

	"llamacpp2go/internal/tensor"
	"llamacpp2go/internal/tensor/reference"
)

func TestHostFeedsExecute(t *testing.T) {
	builder := tensor.NewBuilder()
	feeds := NewFeeds()
	value := reference.Value{Shape: tensor.MustShape(1), Data: []float32{7}}
	node := feeds.Input(builder, "input", value)
	results, err := feeds.Execute(
		[]*tensor.Tensor{node}, false,
		func(outputs []*tensor.Tensor, host map[*tensor.Tensor]reference.Value) (map[*tensor.Tensor]reference.Value, error) {
			return reference.Execute(outputs, host)
		}, nil,
	)
	if err != nil || results[node].Data[0] != 7 {
		t.Fatalf("result = %v, error = %v", results[node].Data, err)
	}
}
