package model

import (
	"bytes"
	"context"
	"testing"

	"llamacpp2go/internal/gguf"
)

func TestLayerGraphFieldsCoverMirroredTensorFields(t *testing.T) {
	fields := make(map[string]layerGraphField, len(layerGraphFields))
	for _, field := range layerGraphFields {
		if _, exists := fields[field.name]; exists {
			t.Fatalf("duplicate field %q", field.name)
		}
		fields[field.name] = field
	}
	for _, name := range []string{
		"AttentionQ", "AttentionQNorm", "FeedForwardRouter", "SSMQueryConv", "SSMKeyConv",
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
	if fields["AttentionQ"].optional {
		t.Error("required attention Q classified as optional")
	}
	if !fields["AttentionQNorm"].optional {
		t.Error("optional attention Q norm classified as required")
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

func TestLoadHostLayerGraphFields(t *testing.T) {
	data := hostTensorFixture(t)
	file, err := gguf.Parse(bytes.NewReader(data), uint64(len(data)), gguf.DefaultOptions())
	if err != nil {
		t.Fatal(err)
	}
	info := file.Tensors[0]
	layer := LayerWeights{
		AttentionQ:     info,
		AttentionQNorm: &info, FeedForwardRouterBias: &info,
		SSMQueryConv: &info, TimeMixW1: &info, ShortConvKernel: &info,
	}
	host := HostLayer{}
	if err := loadHostLayerGraphFields(context.Background(), file, &layer, &host); err != nil {
		t.Fatal(err)
	}
	if host.AttentionQ.Shape.Rank == 0 || len(host.AttentionQ.Data) != 4 {
		t.Fatalf("required attention Q was not loaded: %+v", host.AttentionQ)
	}
	for name, value := range map[string]any{
		"attention Q norm": host.AttentionQNorm, "router bias": host.FeedForwardRouterBias,
		"Kimi query convolution": host.SSMQueryConv, "RWKV mix": host.TimeMixW1,
		"short convolution": host.ShortConvKernel,
	} {
		if value == nil {
			t.Errorf("%s was not loaded", name)
		}
	}
}
