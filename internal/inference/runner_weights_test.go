package inference

import (
	"maps"
	"slices"
	"testing"

	"overgo/internal/gguf"
	"overgo/internal/model"
	"overgo/internal/tensor/dtype"
)

// TestNativeExpertBankTensors covers native storage and complete-layer eligibility.
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
	want := []string{"fused.down", "fused.gate_up", "mixed.down", "mixed.gate", "mixed.up", "native.down", "native.gate", "native.up", "quantized.down", "quantized.gate", "quantized.up", "widened.down", "widened.gate", "widened.up"}
	if names := slices.Sorted(maps.Keys(got)); !slices.Equal(names, want) {
		t.Fatalf("native expert banks = %v, want %v", names, want)
	}
	expert, err := gguf.NewTensorInfo("experts", dtype.BF16, []uint64{2, 2, 4})
	if err != nil {
		t.Fatal(err)
	}
	dense, err := gguf.NewTensorInfo("dense", dtype.F32, []uint64{2})
	if err != nil {
		t.Fatal(err)
	}
	converted, err := gguf.NewTensorInfo("converted", dtype.BF16, []uint64{2})
	if err != nil {
		t.Fatal(err)
	}
	for _, tc := range []struct {
		available uint64
		backed    int
	}{{48, 0}, {47, 1}} {
		resident, backed, err := splitExpertResidency([]gguf.TensorInfo{expert, dense}, []gguf.TensorInfo{converted}, map[string]struct{}{"experts": {}}, tc.available)
		if err != nil || len(backed) != tc.backed || len(resident) != 2-tc.backed {
			t.Fatalf("available %d: resident=%v backed=%v err=%v", tc.available, resident, backed, err)
		}
		if len(backed) != 0 && (backed[0] != expert || resident[0] != dense) {
			t.Fatal("placement changed the file descriptors")
		}
	}
	if _, _, err := splitExpertResidency([]gguf.TensorInfo{{Size: ^uint64(0)}, dense}, nil, nil, 0); err == nil {
		t.Fatal("overflowing weight extent accepted")
	}
}
