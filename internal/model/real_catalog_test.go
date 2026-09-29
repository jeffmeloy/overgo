package model_test

import (
	"testing"

	"overgo/internal/gguf"
	"overgo/internal/model"
	"overgo/internal/modeltest"
	"overgo/internal/testskip"
)

// realCatalog reads a registered GGUF's spec and weight catalog the way
// serving reads them, without loading tensor data.
func realCatalog(t *testing.T, path string) (model.Spec, model.Weights) {
	t.Helper()
	file, err := gguf.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer file.Close()
	spec, err := model.ReadSpec(file)
	if err != nil {
		t.Fatal(err)
	}
	weights, err := model.ReadWeights(file, spec)
	if err != nil {
		t.Fatal(err)
	}
	return spec, weights
}

// TestReadRealQwen35Catalog holds the validated Qwen3.5 fixture's 32-block
// hybrid catalog: recurrent blocks with a full-attention block every fourth.
func TestReadRealQwen35Catalog(t *testing.T) {
	testskip.Short(t)
	spec, weights := realCatalog(t, modeltest.Directory(t, "Qwen3.5-9B-Q8_0.gguf"))
	if spec.Architecture != "qwen35" || len(weights.Layers) != 32 ||
		!weights.Layers[0].Recurrent || weights.Layers[3].Recurrent {
		t.Fatalf("unexpected real Qwen3.5 catalog: spec=%+v layers=%d", spec, len(weights.Layers))
	}
}

// TestReadRealKimiLinearCatalog holds any registered Kimi Linear GGUF to a
// complete catalog with both recurrent and attention blocks.
func TestReadRealKimiLinearCatalog(t *testing.T) {
	testskip.Short(t)
	spec, weights := realCatalog(t, modeltest.GGUF(t, "kimi-linear"))
	if spec.Architecture != "kimi-linear" || len(weights.Layers) != int(spec.BlockCount) {
		t.Fatalf("unexpected real Kimi Linear catalog: spec=%+v layers=%d", spec, len(weights.Layers))
	}
	var recurrent, attention bool
	for _, layer := range weights.Layers {
		recurrent = recurrent || layer.Recurrent
		attention = attention || !layer.Recurrent
	}
	if !recurrent || !attention {
		t.Fatal("real Kimi Linear catalog lacks hybrid layers")
	}
}

// TestReadRealGemma3Catalog holds the validated Gemma 3 fixture to its strict
// 48-block catalog with post-attention and post-feed-forward norms.
func TestReadRealGemma3Catalog(t *testing.T) {
	testskip.Short(t)
	spec, weights := realCatalog(t, modeltest.Directory(t, "gemma-3-12b-it-IQ4_XS.gguf"))
	if spec.Architecture != "gemma3" || len(weights.Layers) != 48 ||
		weights.Layers[0].AttentionPostNorm == nil ||
		weights.Layers[0].FeedForwardPostNorm == nil {
		t.Fatalf("unexpected real Gemma 3 catalog: spec=%+v layers=%d", spec, len(weights.Layers))
	}
}

// TestReadRealUMT5Catalog holds the validated UMT5-XXL encoder fixture to its
// 24-block catalog with relative attention bias.
func TestReadRealUMT5Catalog(t *testing.T) {
	testskip.Short(t)
	spec, weights := realCatalog(t, modeltest.Directory(t, "umt5-xxl-encoder-Q5_K_S.gguf"))
	if spec.Architecture != "t5encoder" || len(weights.Layers) != 24 ||
		weights.Layers[0].AttentionRelativeBias == nil {
		t.Fatalf("unexpected real UMT5 catalog: spec=%+v layers=%d", spec, len(weights.Layers))
	}
}
