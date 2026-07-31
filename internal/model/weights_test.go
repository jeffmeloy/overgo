package model

import (
	"fmt"
	"os"
	"strings"
	"testing"

	"llamacpp2go/internal/gguf"
	"llamacpp2go/internal/tensor/dtype"
)

func TestReadWeightsQwen3(t *testing.T) {
	spec := Spec{
		Architecture:      "qwen3",
		BlockCount:        1,
		EmbeddingLength:   8,
		FeedForwardLength: 16,
		HeadCount:         2,
		HeadCountKV:       1,
		KeyLength:         4,
		ValueLength:       4,
		VocabularySize:    32,
	}
	file := &gguf.File{Tensors: []gguf.TensorInfo{
		tensorInfo("token_embd.weight", 8, 32),
		tensorInfo("output_norm.weight", 8),
		tensorInfo("blk.0.attn_norm.weight", 8),
		tensorInfo("blk.0.attn_q.weight", 8, 8),
		tensorInfo("blk.0.attn_k.weight", 8, 4),
		tensorInfo("blk.0.attn_v.weight", 8, 4),
		tensorInfo("blk.0.attn_output.weight", 8, 8),
		tensorInfo("blk.0.attn_q_norm.weight", 4),
		tensorInfo("blk.0.attn_k_norm.weight", 4),
		tensorInfo("blk.0.rope_freqs.weight", 2),
		tensorInfo("blk.0.ffn_norm.weight", 8),
		tensorInfo("blk.0.ffn_gate.weight", 8, 16),
		tensorInfo("blk.0.ffn_up.weight", 8, 16),
		tensorInfo("blk.0.ffn_down.weight", 16, 8),
	}}
	weights, err := ReadWeights(file, spec)
	if err != nil {
		t.Fatal(err)
	}
	if len(weights.Layers) != 1 ||
		weights.Layers[0].AttentionQNorm == nil ||
		weights.Layers[0].RopeFactors == nil {
		t.Fatalf("unexpected weights: %+v", weights)
	}
}

func TestReadWeightsQwen2WithOutputBias(t *testing.T) {
	spec := Spec{
		Architecture:      "qwen2",
		BlockCount:        1,
		EmbeddingLength:   8,
		FeedForwardLength: 16,
		HeadCount:         2,
		HeadCountKV:       1,
		KeyLength:         4,
		ValueLength:       4,
		VocabularySize:    32,
	}
	file := &gguf.File{Tensors: []gguf.TensorInfo{
		tensorInfo("token_embd.weight", 8, 32),
		tensorInfo("output_norm.weight", 8),
		tensorInfo("output.weight", 8, 32),
		tensorInfo("output.bias", 32),
		tensorInfo("blk.0.attn_norm.weight", 8),
		tensorInfo("blk.0.attn_q.weight", 8, 8),
		tensorInfo("blk.0.attn_q.bias", 8),
		tensorInfo("blk.0.attn_k.weight", 8, 4),
		tensorInfo("blk.0.attn_k.bias", 4),
		tensorInfo("blk.0.attn_v.weight", 8, 4),
		tensorInfo("blk.0.attn_v.bias", 4),
		tensorInfo("blk.0.attn_output.weight", 8, 8),
		tensorInfo("blk.0.ffn_norm.weight", 8),
		tensorInfo("blk.0.ffn_gate.weight", 8, 16),
		tensorInfo("blk.0.ffn_up.weight", 8, 16),
		tensorInfo("blk.0.ffn_down.weight", 16, 8),
	}}
	weights, err := ReadWeights(file, spec)
	if err != nil {
		t.Fatal(err)
	}
	layer := weights.Layers[0]
	if weights.Output == nil ||
		weights.OutputBias == nil ||
		layer.AttentionQBias == nil ||
		layer.AttentionKBias == nil ||
		layer.AttentionVBias == nil {
		t.Fatalf("unexpected Qwen 2 weights: %+v", weights)
	}
}

func TestReadWeightsInternLM2EXAONEAndXVERSE(t *testing.T) {
	for _, architecture := range []string{"internlm2", "exaone", "xverse"} {
		t.Run(architecture, func(t *testing.T) {
			spec := Spec{
				Architecture:      architecture,
				BlockCount:        1,
				EmbeddingLength:   8,
				FeedForwardLength: 16,
				HeadCount:         2,
				HeadCountKV:       1,
				KeyLength:         4,
				ValueLength:       4,
				VocabularySize:    32,
			}
			tensors := []gguf.TensorInfo{
				tensorInfo("token_embd.weight", 8, 32),
				tensorInfo("output_norm.weight", 8),
				tensorInfo("blk.0.attn_norm.weight", 8),
				tensorInfo("blk.0.attn_q.weight", 8, 8),
				tensorInfo("blk.0.attn_k.weight", 8, 4),
				tensorInfo("blk.0.attn_v.weight", 8, 4),
				tensorInfo("blk.0.attn_output.weight", 8, 8),
				tensorInfo("blk.0.ffn_norm.weight", 8),
				tensorInfo("blk.0.ffn_gate.weight", 8, 16),
				tensorInfo("blk.0.ffn_up.weight", 8, 16),
				tensorInfo("blk.0.ffn_down.weight", 16, 8),
			}
			if architecture == "internlm2" || architecture == "xverse" {
				tensors = append(tensors, tensorInfo("output.weight", 8, 32))
			} else {
				tensors = append(tensors, tensorInfo("blk.0.rope_freqs.weight", 2))
			}
			weights, err := ReadWeights(&gguf.File{Tensors: tensors}, spec)
			if err != nil {
				t.Fatal(err)
			}
			if (architecture == "internlm2" || architecture == "xverse") &&
				weights.Output == nil {
				t.Fatalf("%s output weight was not cataloged", architecture)
			}
			if architecture == "exaone" && weights.Layers[0].RopeFactors == nil {
				t.Fatal("EXAONE RoPE factors were not cataloged")
			}
		})
	}
}

func TestReadWeightsInternLM2AndXVERSERequireOutput(t *testing.T) {
	for _, architecture := range []string{"internlm2", "xverse"} {
		spec := Spec{
			Architecture: architecture, EmbeddingLength: 8, VocabularySize: 32,
		}
		_, err := ReadWeights(&gguf.File{Tensors: []gguf.TensorInfo{
			tensorInfo("token_embd.weight", 8, 32),
			tensorInfo("output_norm.weight", 8),
		}}, spec)
		if err == nil || !strings.Contains(err.Error(), "output.weight") {
			t.Fatalf("%s error = %v, want required output weight", architecture, err)
		}
	}
}

func TestReadWeightsOLMo2PostNormalizedBlock(t *testing.T) {
	spec := Spec{
		Architecture:      "olmo2",
		BlockCount:        1,
		EmbeddingLength:   8,
		FeedForwardLength: 16,
		HeadCount:         2,
		HeadCountKV:       1,
		KeyLength:         4,
		ValueLength:       4,
		VocabularySize:    32,
	}
	file := &gguf.File{Tensors: []gguf.TensorInfo{
		tensorInfo("token_embd.weight", 8, 32),
		tensorInfo("output_norm.weight", 8),
		tensorInfo("output.weight", 8, 32),
		tensorInfo("blk.0.attn_q.weight", 8, 8),
		tensorInfo("blk.0.attn_k.weight", 8, 4),
		tensorInfo("blk.0.attn_v.weight", 8, 4),
		tensorInfo("blk.0.attn_output.weight", 8, 8),
		tensorInfo("blk.0.attn_q_norm.weight", 8),
		tensorInfo("blk.0.attn_k_norm.weight", 4),
		tensorInfo("blk.0.post_attention_norm.weight", 8),
		tensorInfo("blk.0.ffn_gate.weight", 8, 16),
		tensorInfo("blk.0.ffn_up.weight", 8, 16),
		tensorInfo("blk.0.ffn_down.weight", 16, 8),
		tensorInfo("blk.0.post_ffw_norm.weight", 8),
	}}
	weights, err := ReadWeights(file, spec)
	if err != nil {
		t.Fatal(err)
	}
	layer := weights.Layers[0]
	if layer.AttentionNorm.Name != "" ||
		layer.FeedForwardNorm.Name != "" ||
		layer.AttentionQNorm == nil ||
		layer.AttentionQNorm.Shape[0] != 8 ||
		layer.AttentionKNorm == nil ||
		layer.AttentionKNorm.Shape[0] != 4 ||
		layer.AttentionPostNorm == nil ||
		layer.FeedForwardPostNorm == nil {
		t.Fatalf("unexpected OLMo2 weights: %+v", layer)
	}
}

func TestReadWeightsSmolLM3(t *testing.T) {
	spec := Spec{
		Architecture:      "smollm3",
		BlockCount:        1,
		EmbeddingLength:   8,
		FeedForwardLength: 16,
		HeadCount:         2,
		HeadCountKV:       1,
		KeyLength:         4,
		ValueLength:       4,
		VocabularySize:    32,
	}
	file := &gguf.File{Tensors: []gguf.TensorInfo{
		tensorInfo("token_embd.weight", 8, 32),
		tensorInfo("output_norm.weight", 8),
		tensorInfo("blk.0.attn_norm.weight", 8),
		tensorInfo("blk.0.attn_q.weight", 8, 8),
		tensorInfo("blk.0.attn_k.weight", 8, 4),
		tensorInfo("blk.0.attn_v.weight", 8, 4),
		tensorInfo("blk.0.attn_output.weight", 8, 8),
		tensorInfo("blk.0.ffn_norm.weight", 8),
		tensorInfo("blk.0.ffn_gate.weight", 8, 16),
		tensorInfo("blk.0.ffn_up.weight", 8, 16),
		tensorInfo("blk.0.ffn_down.weight", 16, 8),
	}}
	weights, err := ReadWeights(file, spec)
	if err != nil {
		t.Fatal(err)
	}
	if weights.Output != nil || len(weights.Layers) != 1 {
		t.Fatalf("unexpected tied-output SmolLM3 weights: %+v", weights)
	}
}

func TestReadWeightsMiniCPM(t *testing.T) {
	spec := Spec{
		Architecture:      "minicpm",
		BlockCount:        1,
		EmbeddingLength:   8,
		FeedForwardLength: 16,
		HeadCount:         2,
		HeadCountKV:       1,
		KeyLength:         4,
		ValueLength:       4,
		VocabularySize:    32,
	}
	file := &gguf.File{Tensors: []gguf.TensorInfo{
		tensorInfo("token_embd.weight", 8, 32),
		tensorInfo("output_norm.weight", 8),
		tensorInfo("blk.0.attn_norm.weight", 8),
		tensorInfo("blk.0.attn_q.weight", 8, 8),
		tensorInfo("blk.0.attn_k.weight", 8, 4),
		tensorInfo("blk.0.attn_v.weight", 8, 4),
		tensorInfo("blk.0.attn_output.weight", 8, 8),
		tensorInfo("blk.0.attn_output.bias", 8),
		tensorInfo("blk.0.ffn_norm.weight", 8),
		tensorInfo("blk.0.ffn_gate.weight", 8, 16),
		tensorInfo("blk.0.ffn_up.weight", 8, 16),
		tensorInfo("blk.0.ffn_down.weight", 16, 8),
		tensorInfo("blk.0.ffn_gate.bias", 16),
		tensorInfo("blk.0.ffn_up.bias", 16),
		tensorInfo("blk.0.ffn_down.bias", 8),
	}}
	weights, err := ReadWeights(file, spec)
	if err != nil {
		t.Fatal(err)
	}
	layer := weights.Layers[0]
	if weights.Output != nil ||
		layer.AttentionOutputBias == nil ||
		layer.FeedForwardGateBias == nil ||
		layer.FeedForwardUpBias == nil ||
		layer.FeedForwardDownBias == nil {
		t.Fatalf("unexpected tied-output MiniCPM weights: %+v", weights)
	}
}

func TestReadWeightsGraniteDense(t *testing.T) {
	spec := Spec{
		Architecture:      "granite",
		BlockCount:        1,
		EmbeddingLength:   8,
		FeedForwardLength: 16,
		HeadCount:         2,
		HeadCountKV:       1,
		KeyLength:         4,
		ValueLength:       4,
		VocabularySize:    32,
	}
	file := &gguf.File{Tensors: []gguf.TensorInfo{
		tensorInfo("token_embd.weight", 8, 32),
		tensorInfo("output_norm.weight", 8),
		tensorInfo("blk.0.attn_norm.weight", 8),
		tensorInfo("blk.0.attn_q.weight", 8, 8),
		tensorInfo("blk.0.attn_k.weight", 8, 4),
		tensorInfo("blk.0.attn_v.weight", 8, 4),
		tensorInfo("blk.0.attn_output.weight", 8, 8),
		tensorInfo("blk.0.attn_output.bias", 8),
		tensorInfo("blk.0.ffn_norm.weight", 8),
		tensorInfo("blk.0.ffn_gate.weight", 8, 16),
		tensorInfo("blk.0.ffn_up.weight", 8, 16),
		tensorInfo("blk.0.ffn_down.weight", 16, 8),
		tensorInfo("blk.0.ffn_gate.bias", 16),
		tensorInfo("blk.0.ffn_up.bias", 16),
		tensorInfo("blk.0.ffn_down.bias", 8),
	}}
	weights, err := ReadWeights(file, spec)
	if err != nil {
		t.Fatal(err)
	}
	if weights.Output != nil ||
		weights.Layers[0].AttentionOutputBias == nil ||
		weights.Layers[0].FeedForwardGateBias == nil {
		t.Fatalf("unexpected dense Granite weights: %+v", weights)
	}
}

func TestReadWeightsMaincoder(t *testing.T) {
	spec := Spec{
		Architecture:      "maincoder",
		BlockCount:        1,
		EmbeddingLength:   8,
		FeedForwardLength: 16,
		HeadCount:         2,
		HeadCountKV:       1,
		KeyLength:         4,
		ValueLength:       4,
		VocabularySize:    32,
	}
	file := &gguf.File{Tensors: []gguf.TensorInfo{
		tensorInfo("token_embd.weight", 8, 32),
		tensorInfo("output_norm.weight", 8),
		tensorInfo("blk.0.attn_norm.weight", 8),
		tensorInfo("blk.0.attn_q.weight", 8, 8),
		tensorInfo("blk.0.attn_k.weight", 8, 4),
		tensorInfo("blk.0.attn_v.weight", 8, 4),
		tensorInfo("blk.0.attn_output.weight", 8, 8),
		tensorInfo("blk.0.attn_q_norm.weight", 4),
		tensorInfo("blk.0.attn_k_norm.weight", 4),
		tensorInfo("blk.0.ffn_norm.weight", 8),
		tensorInfo("blk.0.ffn_gate.weight", 8, 16),
		tensorInfo("blk.0.ffn_up.weight", 8, 16),
		tensorInfo("blk.0.ffn_down.weight", 16, 8),
	}}
	weights, err := ReadWeights(file, spec)
	if err != nil {
		t.Fatal(err)
	}
	if weights.Output != nil ||
		weights.Layers[0].AttentionQNorm == nil ||
		weights.Layers[0].AttentionKNorm == nil {
		t.Fatalf("unexpected Maincoder weights: %+v", weights)
	}
}

func TestReadWeightsDenseMistral3(t *testing.T) {
	spec := Spec{
		Architecture:      "mistral3",
		BlockCount:        1,
		EmbeddingLength:   8,
		FeedForwardLength: 16,
		HeadCount:         2,
		HeadCountKV:       1,
		KeyLength:         4,
		ValueLength:       4,
		VocabularySize:    32,
	}
	file := &gguf.File{Tensors: []gguf.TensorInfo{
		tensorInfo("token_embd.weight", 8, 32),
		tensorInfo("output_norm.weight", 8),
		tensorInfo("blk.0.attn_norm.weight", 8),
		tensorInfo("blk.0.attn_q.weight", 8, 8),
		tensorInfo("blk.0.attn_k.weight", 8, 4),
		tensorInfo("blk.0.attn_v.weight", 8, 4),
		tensorInfo("blk.0.attn_output.weight", 8, 8),
		tensorInfo("blk.0.attn_output.bias", 8),
		tensorInfo("blk.0.ffn_norm.weight", 8),
		tensorInfo("blk.0.ffn_gate.weight", 8, 16),
		tensorInfo("blk.0.ffn_up.weight", 8, 16),
		tensorInfo("blk.0.ffn_down.weight", 16, 8),
		tensorInfo("blk.0.ffn_gate.bias", 16),
		tensorInfo("blk.0.ffn_up.bias", 16),
		tensorInfo("blk.0.ffn_down.bias", 8),
	}}
	weights, err := ReadWeights(file, spec)
	if err != nil {
		t.Fatal(err)
	}
	layer := weights.Layers[0]
	if weights.Output != nil ||
		layer.AttentionOutputBias == nil ||
		layer.FeedForwardGateBias == nil ||
		layer.FeedForwardUpBias == nil ||
		layer.FeedForwardDownBias == nil {
		t.Fatalf("unexpected dense Mistral 3 weights: %+v", weights)
	}
}

func TestReadWeightsOrion(t *testing.T) {
	spec := Spec{
		Architecture:      "orion",
		BlockCount:        1,
		EmbeddingLength:   8,
		FeedForwardLength: 16,
		HeadCount:         2,
		HeadCountKV:       1,
		KeyLength:         4,
		ValueLength:       4,
		VocabularySize:    32,
		LayerNormEpsilon:  1e-5,
	}
	file := &gguf.File{Tensors: []gguf.TensorInfo{
		tensorInfo("token_embd.weight", 8, 32),
		tensorInfo("output_norm.weight", 8),
		tensorInfo("output_norm.bias", 8),
		tensorInfo("output.weight", 8, 32),
		tensorInfo("blk.0.attn_norm.weight", 8),
		tensorInfo("blk.0.attn_norm.bias", 8),
		tensorInfo("blk.0.attn_q.weight", 8, 8),
		tensorInfo("blk.0.attn_k.weight", 8, 4),
		tensorInfo("blk.0.attn_v.weight", 8, 4),
		tensorInfo("blk.0.attn_output.weight", 8, 8),
		tensorInfo("blk.0.ffn_norm.weight", 8),
		tensorInfo("blk.0.ffn_norm.bias", 8),
		tensorInfo("blk.0.ffn_gate.weight", 8, 16),
		tensorInfo("blk.0.ffn_up.weight", 8, 16),
		tensorInfo("blk.0.ffn_down.weight", 16, 8),
	}}
	weights, err := ReadWeights(file, spec)
	if err != nil {
		t.Fatal(err)
	}
	layer := weights.Layers[0]
	if weights.Output == nil ||
		weights.OutputNormBias == nil ||
		layer.AttentionNormBias == nil ||
		layer.FeedForwardNormBias == nil {
		t.Fatalf("unexpected Orion weights: %+v", weights)
	}
}

func TestReadWeightsStarCoder2(t *testing.T) {
	spec := Spec{
		Architecture:      "starcoder2",
		BlockCount:        1,
		EmbeddingLength:   8,
		FeedForwardLength: 16,
		HeadCount:         2,
		HeadCountKV:       1,
		KeyLength:         4,
		ValueLength:       4,
		VocabularySize:    32,
		LayerNormEpsilon:  1e-5,
	}
	file := &gguf.File{Tensors: []gguf.TensorInfo{
		tensorInfo("token_embd.weight", 8, 32),
		tensorInfo("output_norm.weight", 8),
		tensorInfo("output_norm.bias", 8),
		tensorInfo("blk.0.attn_norm.weight", 8),
		tensorInfo("blk.0.attn_norm.bias", 8),
		tensorInfo("blk.0.attn_q.weight", 8, 8),
		tensorInfo("blk.0.attn_k.weight", 8, 4),
		tensorInfo("blk.0.attn_v.weight", 8, 4),
		tensorInfo("blk.0.attn_output.weight", 8, 8),
		tensorInfo("blk.0.attn_output.bias", 8),
		tensorInfo("blk.0.ffn_norm.weight", 8),
		tensorInfo("blk.0.ffn_norm.bias", 8),
		tensorInfo("blk.0.ffn_up.weight", 8, 16),
		tensorInfo("blk.0.ffn_up.bias", 16),
		tensorInfo("blk.0.ffn_down.weight", 16, 8),
		tensorInfo("blk.0.ffn_down.bias", 8),
	}}
	weights, err := ReadWeights(file, spec)
	if err != nil {
		t.Fatal(err)
	}
	layer := weights.Layers[0]
	if weights.Output != nil ||
		weights.OutputNormBias == nil ||
		layer.FeedForwardGate.Name != "" ||
		layer.AttentionOutputBias == nil ||
		layer.FeedForwardUpBias == nil ||
		layer.FeedForwardDownBias == nil {
		t.Fatalf("unexpected StarCoder2 weights: %+v", weights)
	}
}

func TestReadWeightsCodeShellFallsBackToOutputEmbedding(t *testing.T) {
	spec := Spec{
		Architecture:      "codeshell",
		BlockCount:        1,
		EmbeddingLength:   8,
		FeedForwardLength: 16,
		HeadCount:         2,
		HeadCountKV:       1,
		KeyLength:         4,
		ValueLength:       4,
		VocabularySize:    32,
		LayerNormEpsilon:  1e-5,
	}
	file := &gguf.File{Tensors: []gguf.TensorInfo{
		tensorInfo("output.weight", 8, 32),
		tensorInfo("output_norm.weight", 8),
		tensorInfo("output_norm.bias", 8),
		tensorInfo("blk.0.attn_norm.weight", 8),
		tensorInfo("blk.0.attn_norm.bias", 8),
		tensorInfo("blk.0.attn_q.weight", 8, 8),
		tensorInfo("blk.0.attn_k.weight", 8, 4),
		tensorInfo("blk.0.attn_v.weight", 8, 4),
		tensorInfo("blk.0.attn_output.weight", 8, 8),
		tensorInfo("blk.0.attn_output.bias", 8),
		tensorInfo("blk.0.ffn_norm.weight", 8),
		tensorInfo("blk.0.ffn_norm.bias", 8),
		tensorInfo("blk.0.ffn_up.weight", 8, 16),
		tensorInfo("blk.0.ffn_up.bias", 16),
		tensorInfo("blk.0.ffn_down.weight", 16, 8),
		tensorInfo("blk.0.ffn_down.bias", 8),
	}}
	weights, err := ReadWeights(file, spec)
	if err != nil {
		t.Fatal(err)
	}
	if weights.TokenEmbedding.Name != "output.weight" ||
		weights.Output == nil ||
		weights.Output.Name != "output.weight" ||
		weights.Layers[0].FeedForwardGate.Name != "" {
		t.Fatalf("unexpected CodeShell weights: %+v", weights)
	}
}

func TestReadWeightsGemma2(t *testing.T) {
	spec := Spec{
		Architecture:      "gemma2",
		BlockCount:        1,
		EmbeddingLength:   8,
		FeedForwardLength: 16,
		HeadCount:         2,
		HeadCountKV:       1,
		KeyLength:         4,
		ValueLength:       4,
		VocabularySize:    32,
	}
	file := &gguf.File{Tensors: []gguf.TensorInfo{
		tensorInfo("token_embd.weight", 8, 32),
		tensorInfo("output_norm.weight", 8),
		tensorInfo("blk.0.attn_norm.weight", 8),
		tensorInfo("blk.0.attn_q.weight", 8, 8),
		tensorInfo("blk.0.attn_k.weight", 8, 4),
		tensorInfo("blk.0.attn_v.weight", 8, 4),
		tensorInfo("blk.0.attn_output.weight", 8, 8),
		tensorInfo("blk.0.post_attention_norm.weight", 8),
		tensorInfo("blk.0.ffn_norm.weight", 8),
		tensorInfo("blk.0.ffn_gate.weight", 8, 16),
		tensorInfo("blk.0.ffn_up.weight", 8, 16),
		tensorInfo("blk.0.ffn_down.weight", 16, 8),
		tensorInfo("blk.0.post_ffw_norm.weight", 8),
	}}
	weights, err := ReadWeights(file, spec)
	if err != nil {
		t.Fatal(err)
	}
	layer := weights.Layers[0]
	if layer.AttentionQNorm != nil ||
		layer.AttentionKNorm != nil ||
		layer.AttentionPostNorm == nil ||
		layer.FeedForwardPostNorm == nil {
		t.Fatalf("unexpected Gemma 2 weights: %+v", layer)
	}
}

func TestReadWeightsGemma(t *testing.T) {
	spec := Spec{
		Architecture:      "gemma",
		BlockCount:        1,
		EmbeddingLength:   8,
		FeedForwardLength: 16,
		HeadCount:         2,
		HeadCountKV:       1,
		KeyLength:         4,
		ValueLength:       4,
		VocabularySize:    32,
	}
	file := &gguf.File{Tensors: []gguf.TensorInfo{
		tensorInfo("token_embd.weight", 8, 32),
		tensorInfo("output_norm.weight", 8),
		tensorInfo("blk.0.attn_norm.weight", 8),
		tensorInfo("blk.0.attn_q.weight", 8, 8),
		tensorInfo("blk.0.attn_k.weight", 8, 4),
		tensorInfo("blk.0.attn_v.weight", 8, 4),
		tensorInfo("blk.0.attn_output.weight", 8, 8),
		tensorInfo("blk.0.ffn_norm.weight", 8),
		tensorInfo("blk.0.ffn_gate.weight", 8, 16),
		tensorInfo("blk.0.ffn_up.weight", 8, 16),
		tensorInfo("blk.0.ffn_down.weight", 16, 8),
	}}
	weights, err := ReadWeights(file, spec)
	if err != nil {
		t.Fatal(err)
	}
	layer := weights.Layers[0]
	if layer.AttentionQNorm != nil ||
		layer.AttentionKNorm != nil ||
		layer.AttentionPostNorm != nil ||
		layer.FeedForwardPostNorm != nil {
		t.Fatalf("unexpected Gemma weights: %+v", layer)
	}
}

func TestReadWeightsRejectsRoPEFactorShape(t *testing.T) {
	spec := Spec{
		Architecture:      "llama",
		BlockCount:        1,
		EmbeddingLength:   8,
		FeedForwardLength: 16,
		HeadCount:         2,
		HeadCountKV:       1,
		KeyLength:         4,
		ValueLength:       4,
		VocabularySize:    32,
	}
	file := &gguf.File{Tensors: []gguf.TensorInfo{
		tensorInfo("token_embd.weight", 8, 32),
		tensorInfo("output_norm.weight", 8),
		tensorInfo("blk.0.attn_norm.weight", 8),
		tensorInfo("blk.0.attn_q.weight", 8, 8),
		tensorInfo("blk.0.attn_k.weight", 8, 4),
		tensorInfo("blk.0.attn_v.weight", 8, 4),
		tensorInfo("blk.0.attn_output.weight", 8, 8),
		tensorInfo("blk.0.rope_freqs.weight", 3),
		tensorInfo("blk.0.ffn_norm.weight", 8),
		tensorInfo("blk.0.ffn_gate.weight", 8, 16),
		tensorInfo("blk.0.ffn_up.weight", 8, 16),
		tensorInfo("blk.0.ffn_down.weight", 16, 8),
	}}
	_, err := ReadWeights(file, spec)
	if err == nil || !strings.Contains(err.Error(), "incompatible shape") {
		t.Fatalf("error = %v, want incompatible RoPE factor shape", err)
	}
}

func TestReadWeightsRejectsShape(t *testing.T) {
	spec := Spec{EmbeddingLength: 8, VocabularySize: 32}
	file := &gguf.File{Tensors: []gguf.TensorInfo{
		tensorInfo("token_embd.weight", 7, 32),
	}}
	_, err := ReadWeights(file, spec)
	if err == nil || !strings.Contains(err.Error(), "dimension 0") {
		t.Fatalf("error = %v", err)
	}
}

func TestReadWeightsQwen35Hybrid(t *testing.T) {
	spec := Spec{
		Architecture:          "qwen35",
		BlockCount:            4,
		EmbeddingLength:       8,
		FeedForwardLength:     16,
		HeadCount:             2,
		HeadCountKV:           1,
		KeyLength:             4,
		ValueLength:           4,
		VocabularySize:        32,
		SSMConvKernel:         3,
		SSMInnerSize:          8,
		SSMStateSize:          2,
		SSMTimeStepRank:       4,
		SSMGroupCount:         2,
		FullAttentionInterval: 4,
	}
	tensors := []gguf.TensorInfo{
		tensorInfo("token_embd.weight", 8, 32),
		tensorInfo("output_norm.weight", 8),
	}
	for block := range uint32(4) {
		prefix := fmt.Sprintf("blk.%d.", block)
		tensors = append(tensors,
			tensorInfo(prefix+"attn_norm.weight", 8),
			tensorInfo(prefix+"post_attention_norm.weight", 8),
			tensorInfo(prefix+"ffn_gate.weight", 8, 16),
			tensorInfo(prefix+"ffn_up.weight", 8, 16),
			tensorInfo(prefix+"ffn_down.weight", 16, 8),
		)
		if block == 3 {
			tensors = append(tensors,
				tensorInfo(prefix+"attn_q.weight", 8, 16),
				tensorInfo(prefix+"attn_k.weight", 8, 4),
				tensorInfo(prefix+"attn_v.weight", 8, 4),
				tensorInfo(prefix+"attn_output.weight", 8, 8),
				tensorInfo(prefix+"attn_q_norm.weight", 4),
				tensorInfo(prefix+"attn_k_norm.weight", 4),
			)
		} else {
			tensors = append(tensors,
				tensorInfo(prefix+"attn_qkv.weight", 8, 16),
				tensorInfo(prefix+"attn_gate.weight", 8, 8),
				tensorInfo(prefix+"ssm_conv1d.weight", 3, 16),
				tensorInfo(prefix+"ssm_dt.bias", 4),
				tensorInfo(prefix+"ssm_a", 4),
				tensorInfo(prefix+"ssm_beta.weight", 8, 4),
				tensorInfo(prefix+"ssm_alpha.weight", 8, 4),
				tensorInfo(prefix+"ssm_norm.weight", 2),
				tensorInfo(prefix+"ssm_out.weight", 8, 8),
			)
		}
	}
	weights, err := ReadWeights(&gguf.File{Tensors: tensors}, spec)
	if err != nil {
		t.Fatal(err)
	}
	if !weights.Layers[0].Recurrent || weights.Layers[3].Recurrent ||
		weights.Layers[0].SSMConv1D == nil || weights.Layers[3].AttentionQNorm == nil {
		t.Fatalf("unexpected Qwen3.5 layer catalog: %+v", weights.Layers)
	}
}

func TestReadRealQwen35Catalog(t *testing.T) {
	path := os.Getenv("LLAMACPP2GO_QWEN35_MODEL")
	if path == "" {
		t.Skip("set LLAMACPP2GO_QWEN35_MODEL to run real Qwen3.5 catalog validation")
	}
	file, err := gguf.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer file.Close()
	spec, err := ReadSpec(file)
	if err != nil {
		t.Fatal(err)
	}
	weights, err := ReadWeights(file, spec)
	if err != nil {
		t.Fatal(err)
	}
	if spec.Architecture != "qwen35" || len(weights.Layers) != 32 ||
		!weights.Layers[0].Recurrent || weights.Layers[3].Recurrent {
		t.Fatalf("unexpected real Qwen3.5 catalog: spec=%+v layers=%d", spec, len(weights.Layers))
	}
}

func TestReadRealGemma3Catalog(t *testing.T) {
	path := os.Getenv("LLAMACPP2GO_GEMMA3_MODEL")
	if path == "" {
		t.Skip("set LLAMACPP2GO_GEMMA3_MODEL to run real Gemma 3 catalog validation")
	}
	file, err := gguf.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer file.Close()
	spec, err := ReadSpec(file)
	if err != nil {
		t.Fatal(err)
	}
	weights, err := ReadWeights(file, spec)
	if err != nil {
		t.Fatal(err)
	}
	if spec.Architecture != "gemma3" || len(weights.Layers) != 48 ||
		weights.Layers[0].AttentionPostNorm == nil ||
		weights.Layers[0].FeedForwardPostNorm == nil {
		t.Fatalf("unexpected real Gemma 3 catalog: spec=%+v layers=%d", spec, len(weights.Layers))
	}
}

func TestReadWeightsT5Encoder(t *testing.T) {
	spec := Spec{
		Architecture:      "t5encoder",
		BlockCount:        1,
		EmbeddingLength:   8,
		FeedForwardLength: 16,
		HeadCount:         2,
		HeadCountKV:       2,
		KeyLength:         4,
		ValueLength:       4,
		VocabularySize:    32,
		RelativeBuckets:   4,
	}
	file := &gguf.File{Tensors: []gguf.TensorInfo{
		tensorInfo("token_embd.weight", 8, 32),
		tensorInfo("enc.output_norm.weight", 8),
		tensorInfo("enc.blk.0.attn_norm.weight", 8),
		tensorInfo("enc.blk.0.attn_q.weight", 8, 8),
		tensorInfo("enc.blk.0.attn_k.weight", 8, 8),
		tensorInfo("enc.blk.0.attn_v.weight", 8, 8),
		tensorInfo("enc.blk.0.attn_o.weight", 8, 8),
		tensorInfo("enc.blk.0.attn_rel_b.weight", 2, 4),
		tensorInfo("enc.blk.0.ffn_norm.weight", 8),
		tensorInfo("enc.blk.0.ffn_gate.weight", 8, 16),
		tensorInfo("enc.blk.0.ffn_up.weight", 8, 16),
		tensorInfo("enc.blk.0.ffn_down.weight", 16, 8),
	}}
	weights, err := ReadWeights(file, spec)
	if err != nil {
		t.Fatal(err)
	}
	if weights.OutputNorm.Name != "enc.output_norm.weight" ||
		len(weights.Layers) != 1 ||
		weights.Layers[0].AttentionRelativeBias == nil {
		t.Fatalf("unexpected T5 encoder weights: %+v", weights)
	}
}

func TestReadRealUMT5Catalog(t *testing.T) {
	path := os.Getenv("LLAMACPP2GO_UMT5_MODEL")
	if path == "" {
		t.Skip("set LLAMACPP2GO_UMT5_MODEL to run real UMT5 catalog validation")
	}
	file, err := gguf.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer file.Close()
	spec, err := ReadSpec(file)
	if err != nil {
		t.Fatal(err)
	}
	weights, err := ReadWeights(file, spec)
	if err != nil {
		t.Fatal(err)
	}
	if spec.Architecture != "t5encoder" || len(weights.Layers) != 24 ||
		weights.Layers[0].AttentionRelativeBias == nil {
		t.Fatalf("unexpected real UMT5 catalog: spec=%+v layers=%d", spec, len(weights.Layers))
	}
}

func TestReadWeightsAcceptsDenseProjectionBiases(t *testing.T) {
	spec := Spec{
		Architecture:      "llama",
		BlockCount:        1,
		EmbeddingLength:   8,
		FeedForwardLength: 16,
		HeadCount:         2,
		HeadCountKV:       1,
		KeyLength:         4,
		ValueLength:       4,
		VocabularySize:    32,
	}
	file := &gguf.File{Tensors: []gguf.TensorInfo{
		tensorInfo("token_embd.weight", 8, 32),
		tensorInfo("output_norm.weight", 8),
		tensorInfo("blk.0.attn_norm.weight", 8),
		tensorInfo("blk.0.attn_q.weight", 8, 8),
		tensorInfo("blk.0.attn_k.weight", 8, 4),
		tensorInfo("blk.0.attn_v.weight", 8, 4),
		tensorInfo("blk.0.attn_output.weight", 8, 8),
		tensorInfo("blk.0.attn_q.bias", 8),
		tensorInfo("blk.0.attn_k.bias", 4),
		tensorInfo("blk.0.attn_v.bias", 4),
		tensorInfo("blk.0.attn_output.bias", 8),
		tensorInfo("blk.0.ffn_norm.weight", 8),
		tensorInfo("blk.0.ffn_gate.weight", 8, 16),
		tensorInfo("blk.0.ffn_up.weight", 8, 16),
		tensorInfo("blk.0.ffn_down.weight", 16, 8),
		tensorInfo("blk.0.ffn_gate.bias", 16),
		tensorInfo("blk.0.ffn_up.bias", 16),
		tensorInfo("blk.0.ffn_down.bias", 8),
	}}
	weights, err := ReadWeights(file, spec)
	if err != nil {
		t.Fatal(err)
	}
	layer := weights.Layers[0]
	if layer.AttentionQBias == nil ||
		layer.AttentionKBias == nil ||
		layer.AttentionVBias == nil ||
		layer.AttentionOutputBias == nil ||
		layer.FeedForwardGateBias == nil ||
		layer.FeedForwardUpBias == nil ||
		layer.FeedForwardDownBias == nil {
		t.Fatalf("projection bias catalog is incomplete: %+v", layer)
	}
}

func tensorInfo(name string, shape ...uint64) gguf.TensorInfo {
	item := gguf.TensorInfo{
		Name:       name,
		Dimensions: uint32(len(shape)),
		Type:       dtype.F32,
		Shape:      [4]uint64{1, 1, 1, 1},
	}
	copy(item.Shape[:], shape)
	return item
}
