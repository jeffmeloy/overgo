package inference

import (
	"maps"
	"slices"
	"testing"

	"overgo/internal/gguf"
	"overgo/internal/model"
	"overgo/internal/tensor/dtype"
)

// TestNativeExpertBankTensors: BF16 routed-expert banks stay at file width a
// whole layer at a time; a layer with an adapted, F32, quantized or rank-2
// bank keeps the load path it had.
func TestNativeExpertBankTensors(t *testing.T) {
	bank := func(name string, storage dtype.Type, rank uint32) *gguf.TensorInfo {
		return &gguf.TensorInfo{Name: name, Type: storage, Dimensions: rank}
	}
	layer := func(prefix string, gate, up, down dtype.Type) model.LayerWeights {
		return model.LayerWeights{
			FeedForwardGateExperts: bank(prefix+".gate", gate, expertBankRank),
			FeedForwardUpExperts:   bank(prefix+".up", up, expertBankRank),
			FeedForwardDownExperts: bank(prefix+".down", down, expertBankRank),
		}
	}
	weights := model.Weights{Layers: []model.LayerWeights{
		{},
		layer("native", dtype.BF16, dtype.BF16, dtype.BF16),
		{
			FeedForwardGateUpExperts: bank("fused.gate_up", dtype.BF16, expertBankRank),
			FeedForwardDownExperts:   bank("fused.down", dtype.BF16, expertBankRank),
		},
		layer("adapted", dtype.BF16, dtype.BF16, dtype.BF16),
		layer("widened", dtype.F32, dtype.F32, dtype.F32),
		layer("quantized", dtype.Q8_0, dtype.Q8_0, dtype.Q8_0),
		layer("mixed", dtype.BF16, dtype.BF16, dtype.F32),
		layer("half", dtype.F16, dtype.F16, dtype.F16),
		{
			FeedForwardUpExperts:   bank("flat.up", dtype.BF16, expertBankRank-1),
			FeedForwardDownExperts: bank("flat.down", dtype.BF16, expertBankRank-1),
		},
	}}
	got := nativeExpertBankTensors(weights, map[string]struct{}{"adapted.up": {}})
	want := []string{"fused.down", "fused.gate_up", "native.down", "native.gate", "native.up"}
	if names := slices.Sorted(maps.Keys(got)); !slices.Equal(names, want) {
		t.Fatalf("native expert banks = %v, want %v", names, want)
	}
}
