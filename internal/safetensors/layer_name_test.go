package safetensors

import "testing"

// TestALayerNameIsReadByItsParts: the prefix, a decimal layer, a dot and something after it, and nothing else reads as one.
func TestALayerNameIsReadByItsParts(t *testing.T) {
	t.Parallel()
	const prefix = "model.layers."
	if layer, rest, ok := LayerName("model.layers.12.self_attn.q_proj.weight", prefix); !ok || layer != "12" || rest != "self_attn.q_proj.weight" {
		t.Fatalf("read layer %q and rest %q (%t)", layer, rest, ok)
	}
	for _, name := range []string{
		"", "model.layers.", "model.layers.12", "model.layers.12.", "model.layers..weight", "model.layers.x.weight", "model.layers.1x.weight",
		"model.layers.-1.weight", "other.model.layers.1.weight", "model.norm.weight",
	} {
		if layer, rest, ok := LayerName(name, prefix); ok {
			t.Errorf("%q read as layer %q, rest %q", name, layer, rest)
		}
	}
}
