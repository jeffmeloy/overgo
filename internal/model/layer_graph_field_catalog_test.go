package model

import "testing"

func TestLayerGraphFieldsCoverMirroredTensorFields(t *testing.T) {
	fields := make(map[string]layerGraphField, len(layerGraphFields))
	for _, field := range layerGraphFields {
		if _, exists := fields[field.name]; exists {
			t.Fatalf("duplicate field %q", field.name)
		}
		fields[field.name] = field
	}
	for _, name := range []string{
		"AttentionQNorm", "FeedForwardRouter", "SSMQueryConv", "SSMKeyConv",
		"SSMValueConv", "SSMForgetA", "SSMOutputGateB", "TimeMixW1", "ShortConvKernel",
	} {
		field, ok := fields[name]
		if !ok {
			t.Errorf("missing mirrored field %q", name)
			continue
		}
		if field.inputName == "" {
			t.Errorf("field %q has no graph input name", name)
		}
	}
	for _, name := range []string{"EmbeddingSkip", "PerLayerInput", "AttentionBlockIDs"} {
		if _, ok := fields[name]; ok {
			t.Errorf("runtime-only field %q entered weight catalog", name)
		}
	}
}

func TestGraphInputName(t *testing.T) {
	for input, want := range map[string]string{
		"AttentionQNorm": "attention_q_norm",
		"SSMQueryConv":   "ssm_query_conv",
		"TimeMixW1":      "time_mix_w1",
	} {
		if got := graphInputName(input); got != want {
			t.Errorf("graphInputName(%q) = %q, want %q", input, got, want)
		}
	}
}
