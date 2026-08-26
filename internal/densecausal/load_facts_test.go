package densecausal

import "testing"

func TestResolvedModelFactsRejectMissingConfig(t *testing.T) {
	for _, config := range []artifactConfig{
		{},
		{LanguageConfig: &artifactConfig{}},
	} {
		if _, _, err := config.resolvedDecoder(); err == nil {
			t.Fatalf("missing decoder facts accepted: %+v", config)
		}
	}
}

func TestDecoderTensorSelection(t *testing.T) {
	for name, want := range map[string]bool{
		"model.embed_tokens.weight":                  true,
		"model.layers.0.self_attn.q_proj.weight":     true,
		"model.norm.weight":                          true,
		"lm_head.weight":                             true,
		"vision_model.layers.0.self_attn.qkv.weight": false,
		"multi_modal_projector.linear_1.weight":      false,
	} {
		if got := decoderTensor(name); got != want {
			t.Fatalf("decoderTensor(%q) = %t, want %t", name, got, want)
		}
	}
}
