package optimizer

import (
	"math"
	"slices"
	"testing"
)

func TestTensorPackRestoreIsAtomic(t *testing.T) {
	for _, pack := range []*TensorPack{nil, {}} {
		if err := pack.RestoreWeights(nil); err == nil {
			t.Fatal("uninitialized pack admitted restoration")
		}
	}
	weights := map[string][]float32{"a": {1, 2}, "b": {3, 4}}
	pack, err := NewTensorPack(weights, MatrixGeometry(map[string][2]int{"a": {2, 1}, "b": {1, 2}}))
	if err != nil {
		t.Fatal(err)
	}
	for name, replacement := range map[string]map[string][]float32{
		"missing":  {"a": {5, 6}},
		"extra":    {"a": {5, 6}, "b": {7, 8}, "c": {}},
		"name":     {"a": {5, 6}, "c": {7, 8}},
		"shape":    {"a": {5, 6}, "b": {7}},
		"nan":      {"a": {5, 6}, "b": {7, float32(math.NaN())}},
		"infinity": {"a": {5, 6}, "b": {7, float32(math.Inf(1))}},
	} {
		t.Run(name, func(t *testing.T) {
			if err := pack.RestoreWeights(replacement); err == nil {
				t.Fatal("invalid replacement admitted")
			}
			if !slices.Equal(pack.weights, []float32{1, 2, 3, 4}) || !slices.Equal(weights["a"], []float32{1, 2}) || !slices.Equal(weights["b"], []float32{3, 4}) {
				t.Fatal("rejected restoration changed weights")
			}
		})
	}
	gradients := pack.BindMapViews(weights)
	gradients["a"][0] = 9
	// Cross-aliased ranges must swap, not overwrite a source before it is read.
	if err := pack.RestoreWeights(map[string][]float32{"a": weights["b"], "b": weights["a"]}); err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(pack.weights, []float32{3, 4, 1, 2}) || !slices.Equal(weights["a"], []float32{3, 4}) || !slices.Equal(weights["b"], []float32{1, 2}) || !slices.Equal(pack.gradients, make([]float32, 4)) {
		t.Fatal("restoration lost aliased input or retained stale gradients")
	}
}
