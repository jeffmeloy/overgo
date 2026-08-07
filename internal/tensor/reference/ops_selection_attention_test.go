package reference

import (
	"testing"

	"llamacpp2go/internal/tensor"
)

func TestSelectTopKPairs(t *testing.T) {
	output := make([]float32, 4)
	selectTopKPairs([]float32{0, 1, 1, 3, 2, 2}, output, 2)
	if output[0] != 1 || output[1] != 3 || output[2] != 2 || output[3] != 2 {
		t.Fatalf("top-K pairs = %v", output)
	}
}

func TestTopKPartialMerge(t *testing.T) {
	input := Value{Shape: tensor.MustShape(5, 1), Data: []float32{1, 5, 2, 4, 3}}
	attributes := tensor.TopKAttributes{K: 2, Chunk: 3}
	partialShape := tensor.MustShape(2, 2, 2, 1)
	partials, err := topKPartials(partialShape, input, attributes)
	if err != nil {
		t.Fatal(err)
	}
	if partials.Data[0] < 0 {
		t.Fatalf("partials = %v", partials.Data)
	}
	merged, err := topKPairs(tensor.MustShape(2, 2, 1), partials, attributes)
	if err != nil {
		t.Fatal(err)
	}
	if merged.Data[0] != 1 || merged.Data[2] != 3 {
		t.Fatalf("merged = %v", merged.Data)
	}
}

func TestTopKPartialMergeMultiChunk(t *testing.T) {
	const width = 4099
	data := make([]float32, width*2)
	for index := range data {
		data[index] = -0.2 + 0.013*float32(((index*7+17*3)%19)-9)
	}
	attributes := tensor.TopKAttributes{K: 40, Chunk: 1024}
	partials, err := topKPartials(
		tensor.MustShape(2, 40, 5, 2),
		Value{Shape: tensor.MustShape(width, 2), Data: data}, attributes,
	)
	if err != nil {
		t.Fatal(err)
	}
	if partials.Data[0] < 0 {
		t.Fatalf("partial head = %v", partials.Data[:8])
	}
	merged, err := topKPairs(tensor.MustShape(2, 40, 2), partials, attributes)
	if err != nil {
		t.Fatal(err)
	}
	if merged.Data[0] < 0 {
		t.Fatalf("merged head = %v", merged.Data[:8])
	}
}
