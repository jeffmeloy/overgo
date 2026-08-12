package reference

import (
	"math"
	"testing"

	"overgo/internal/tensor"
)

func TestCacheAppendPreservesCapacityAndWritesLogicalOffset(t *testing.T) {
	shape := tensor.MustShape(2, 1, 4)
	result, err := cacheAppend(
		shape,
		Value{Shape: tensor.MustShape(2, 1, 2), Data: []float32{1, 2, 3, 4}},
		Value{Shape: tensor.MustShape(2, 1, 1), Data: []float32{5, 6}},
		tensor.CacheAppendAttributes{Axis: 2, Offset: 2},
	)
	if err != nil {
		t.Fatal(err)
	}
	want := []float32{1, 2, 3, 4, 5, 6, 0, 0}
	for index := range want {
		if result.Data[index] != want[index] {
			t.Fatalf("cache append = %v, want %v", result.Data, want)
		}
	}
}

func TestAttentionUsesLogicalTokensWithinCapacity(t *testing.T) {
	query := Value{Shape: tensor.MustShape(1, 1, 1), Data: []float32{1}}
	compactKey := Value{Shape: tensor.MustShape(1, 1, 2), Data: []float32{1, 2}}
	compactValue := Value{Shape: tensor.MustShape(1, 1, 2), Data: []float32{10, 20}}
	capacityKey := Value{Shape: tensor.MustShape(1, 1, 4), Data: []float32{1, 2, 100, 100}}
	capacityValue := Value{Shape: tensor.MustShape(1, 1, 4), Data: []float32{10, 20, -100, -100}}
	shape := tensor.MustShape(1, 1, 1)
	compact, err := attention(
		shape, query, compactKey, compactValue, nil, nil, nil, nil,
		tensor.AttentionAttributes{Scale: 1, Causal: true, QueryStart: 1},
	)
	if err != nil {
		t.Fatal(err)
	}
	capacity, err := attention(
		shape, query, capacityKey, capacityValue, nil, nil, nil, nil,
		tensor.AttentionAttributes{
			Scale: 1, Causal: true, QueryStart: 1, KeyValueTokens: 2,
		},
	)
	if err != nil {
		t.Fatal(err)
	}
	if delta := math.Abs(float64(compact.Data[0] - capacity.Data[0])); delta > 1e-6 {
		t.Fatalf("logical/capacity attention = %v/%v", compact.Data, capacity.Data)
	}
}

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
