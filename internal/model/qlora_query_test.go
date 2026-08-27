package model

import (
	"testing"

	"overgo/internal/gguf"
)

// TestQLoRAQueryNormBindsOwnTensor pins the fix for the QLoRA pointer-aliasing
// hazard (finding evidence:sha256:3aa4a494): AttentionQNorm must dereference
// the attn_q_a_norm tensor info and AttentionQB the attn_q_b info, through
// distinct pointees. Under the historical shared-variable binding, both
// pointers aliased one variable and AttentionQNorm's pointee held attn_q_b's
// info -- this test fails against that binding.
func TestQLoRAQueryNormBindsOwnTensor(t *testing.T) {
	spec := Spec{
		CommonSpec:    CommonSpec{EmbeddingLength: 8},
		AttentionSpec: AttentionSpec{QLoRARank: 3},
	}
	const queryLength = 12
	items := []gguf.TensorInfo{
		{Name: "blk.0.attn_q_a.weight", Dimensions: 2, Shape: [4]uint64{8, 3}},
		{Name: "blk.0.attn_q_a_norm.weight", Dimensions: 1, Shape: [4]uint64{3}},
		{Name: "blk.0.attn_q_b.weight", Dimensions: 2, Shape: [4]uint64{3, queryLength}},
	}
	catalog := weightCatalog{tensors: map[string]int{}, items: items}
	for index, item := range items {
		catalog.tensors[item.Name] = index
	}
	var layer LayerWeights
	if err := bindTensorProgram(catalog, "blk.0.", qLoRAQueryBindings(spec, queryLength, &layer)); err != nil {
		t.Fatal(err)
	}
	if layer.AttentionQ == nil || layer.AttentionQ.Name != "blk.0.attn_q_a.weight" {
		t.Fatalf("AttentionQ binds %+v", layer.AttentionQ)
	}
	if layer.AttentionQNorm == nil || layer.AttentionQNorm.Name != "blk.0.attn_q_a_norm.weight" ||
		layer.AttentionQNorm.Dimensions != 1 || layer.AttentionQNorm.Shape[0] != 3 {
		t.Fatalf("AttentionQNorm binds %+v, want the [QLoRARank] norm tensor", layer.AttentionQNorm)
	}
	if layer.AttentionQB == nil || layer.AttentionQB.Name != "blk.0.attn_q_b.weight" {
		t.Fatalf("AttentionQB binds %+v", layer.AttentionQB)
	}
	if layer.AttentionQNorm == layer.AttentionQB {
		t.Fatal("AttentionQNorm and AttentionQB alias one pointee (the historical bug)")
	}
	missing := weightCatalog{tensors: map[string]int{}, items: nil}
	if err := bindTensorProgram(missing, "blk.0.", qLoRAQueryBindings(spec, queryLength, &layer)); err == nil {
		t.Fatal("missing tensors accepted")
	}
}
