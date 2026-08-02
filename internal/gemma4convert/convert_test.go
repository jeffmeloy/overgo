package gemma4convert

import "testing"

func TestModelTensorName(t *testing.T) {
	for source, want := range map[string]string{
		"model.language_model.embed_tokens.weight":                         "token_embd.weight",
		"model.language_model.norm.weight":                                 "output_norm.weight",
		"model.language_model.layers.17.input_layernorm.weight":            "blk.17.attn_norm.weight",
		"model.language_model.layers.17.self_attn.q_proj.weight":           "blk.17.attn_q.weight",
		"model.language_model.layers.17.mlp.down_proj.weight":              "blk.17.ffn_down.weight",
		"model.language_model.layers.17.post_feedforward_layernorm.weight": "blk.17.post_ffw_norm.weight",
		"model.language_model.layers.17.layer_scalar":                      "blk.17.layer_output_scale.weight",
	} {
		got, ok := modelTensorName(source)
		if !ok || got != want {
			t.Fatalf("mapping %q = %q, %t; want %q", source, got, ok, want)
		}
	}
	if _, ok := modelTensorName("model.embed_vision.patch_dense.weight"); ok {
		t.Fatal("projector tensor mapped into language model")
	}
}

func TestReverseShape(t *testing.T) {
	actual := reverseShape([]uint64{2, 3, 5})
	want := []uint64{5, 3, 2}
	for index := range want {
		if actual[index] != want[index] {
			t.Fatalf("shape = %v, want %v", actual, want)
		}
	}
}
