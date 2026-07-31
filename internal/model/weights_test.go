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

func TestReadWeightsBaichuan7B(t *testing.T) {
	spec := Spec{
		Architecture:      "baichuan",
		BlockCount:        1,
		EmbeddingLength:   8,
		FeedForwardLength: 16,
		HeadCount:         2,
		HeadCountKV:       2,
		KeyLength:         4,
		ValueLength:       4,
		VocabularySize:    32,
		RMSNormEpsilon:    1e-6,
	}
	file := &gguf.File{Tensors: []gguf.TensorInfo{
		tensorInfo("token_embd.weight", 8, 32),
		tensorInfo("output_norm.weight", 8),
		tensorInfo("output.weight", 8, 32),
		tensorInfo("blk.0.attn_norm.weight", 8),
		tensorInfo("blk.0.attn_q.weight", 8, 8),
		tensorInfo("blk.0.attn_k.weight", 8, 8),
		tensorInfo("blk.0.attn_v.weight", 8, 8),
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
	if weights.Output == nil ||
		weights.Output.Name != "output.weight" ||
		weights.Layers[0].FeedForwardGate.Name == "" {
		t.Fatalf("unexpected Baichuan weights: %+v", weights)
	}
}

func TestReadWeightsArcee(t *testing.T) {
	spec := Spec{
		Architecture:      "arcee",
		BlockCount:        1,
		EmbeddingLength:   8,
		FeedForwardLength: 16,
		HeadCount:         2,
		HeadCountKV:       1,
		KeyLength:         4,
		ValueLength:       4,
		VocabularySize:    32,
		RMSNormEpsilon:    1e-5,
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
		tensorInfo("blk.0.ffn_up.weight", 8, 16),
		tensorInfo("blk.0.ffn_down.weight", 16, 8),
	}}
	weights, err := ReadWeights(file, spec)
	if err != nil {
		t.Fatal(err)
	}
	if weights.Output != nil ||
		weights.Layers[0].FeedForwardGate.Name != "" ||
		weights.Layers[0].FeedForwardUp.Name == "" {
		t.Fatalf("unexpected Arcee weights: %+v", weights)
	}
}

func TestReadWeightsNemotron(t *testing.T) {
	spec := Spec{
		Architecture:      "nemotron",
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
	if weights.Output == nil ||
		weights.OutputNormBias == nil ||
		layer.FeedForwardGate.Name != "" ||
		layer.AttentionNormBias == nil ||
		layer.FeedForwardNormBias == nil ||
		layer.AttentionOutputBias == nil {
		t.Fatalf("unexpected Nemotron weights: %+v", weights)
	}
}

func TestReadWeightsJais2(t *testing.T) {
	spec := Spec{
		Architecture: "jais2", BlockCount: 1, EmbeddingLength: 8,
		FeedForwardLength: 16, HeadCount: 2, HeadCountKV: 2,
		KeyLength: 4, ValueLength: 4, VocabularySize: 32, LayerNormEpsilon: 1e-5,
	}
	file := &gguf.File{Tensors: []gguf.TensorInfo{
		tensorInfo("token_embd.weight", 8, 32),
		tensorInfo("output_norm.weight", 8), tensorInfo("output_norm.bias", 8),
		tensorInfo("blk.0.attn_norm.weight", 8), tensorInfo("blk.0.attn_norm.bias", 8),
		tensorInfo("blk.0.attn_q.weight", 8, 8), tensorInfo("blk.0.attn_q.bias", 8),
		tensorInfo("blk.0.attn_k.weight", 8, 8), tensorInfo("blk.0.attn_k.bias", 8),
		tensorInfo("blk.0.attn_v.weight", 8, 8), tensorInfo("blk.0.attn_v.bias", 8),
		tensorInfo("blk.0.attn_output.weight", 8, 8), tensorInfo("blk.0.attn_output.bias", 8),
		tensorInfo("blk.0.ffn_norm.weight", 8), tensorInfo("blk.0.ffn_norm.bias", 8),
		tensorInfo("blk.0.ffn_up.weight", 8, 16), tensorInfo("blk.0.ffn_up.bias", 16),
		tensorInfo("blk.0.ffn_down.weight", 16, 8), tensorInfo("blk.0.ffn_down.bias", 8),
	}}
	weights, err := ReadWeights(file, spec)
	if err != nil {
		t.Fatal(err)
	}
	layer := weights.Layers[0]
	if weights.Output != nil || layer.FeedForwardGate.Name != "" ||
		layer.AttentionQBias == nil || layer.AttentionKBias == nil ||
		layer.AttentionVBias == nil || layer.AttentionOutputBias == nil ||
		layer.FeedForwardUpBias == nil || layer.FeedForwardDownBias == nil {
		t.Fatalf("unexpected Jais2 weights: %+v", weights)
	}
}

func TestReadWeightsOLMoWithoutNormTensors(t *testing.T) {
	spec := Spec{
		Architecture: "olmo", BlockCount: 1, EmbeddingLength: 8,
		FeedForwardLength: 16, HeadCount: 2, HeadCountKV: 1,
		KeyLength: 4, ValueLength: 4, VocabularySize: 32, LayerNormEpsilon: 1e-5,
	}
	file := &gguf.File{Tensors: []gguf.TensorInfo{
		tensorInfo("token_embd.weight", 8, 32),
		tensorInfo("blk.0.attn_q.weight", 8, 8),
		tensorInfo("blk.0.attn_k.weight", 8, 4),
		tensorInfo("blk.0.attn_v.weight", 8, 4),
		tensorInfo("blk.0.attn_output.weight", 8, 8),
		tensorInfo("blk.0.ffn_gate.weight", 8, 16),
		tensorInfo("blk.0.ffn_up.weight", 8, 16),
		tensorInfo("blk.0.ffn_down.weight", 16, 8),
	}}
	weights, err := ReadWeights(file, spec)
	if err != nil {
		t.Fatal(err)
	}
	if weights.OutputNorm.Name != "" || weights.Output != nil ||
		weights.Layers[0].AttentionNorm.Name != "" ||
		weights.Layers[0].FeedForwardNorm.Name != "" {
		t.Fatalf("unexpected OLMo weights: %+v", weights)
	}
}

func TestReadWeightsSeedOSSMapsPostAttentionNorm(t *testing.T) {
	spec := Spec{
		Architecture: "seed_oss", BlockCount: 1, EmbeddingLength: 8,
		FeedForwardLength: 16, HeadCount: 2, HeadCountKV: 1,
		KeyLength: 4, ValueLength: 4, VocabularySize: 32, RMSNormEpsilon: 1e-5,
	}
	file := &gguf.File{Tensors: []gguf.TensorInfo{
		tensorInfo("token_embd.weight", 8, 32), tensorInfo("output_norm.weight", 8),
		tensorInfo("blk.0.attn_norm.weight", 8),
		tensorInfo("blk.0.attn_q.weight", 8, 8),
		tensorInfo("blk.0.attn_k.weight", 8, 4),
		tensorInfo("blk.0.attn_v.weight", 8, 4),
		tensorInfo("blk.0.attn_output.weight", 8, 8),
		tensorInfo("blk.0.post_attention_norm.weight", 8),
		tensorInfo("blk.0.ffn_gate.weight", 8, 16),
		tensorInfo("blk.0.ffn_up.weight", 8, 16),
		tensorInfo("blk.0.ffn_down.weight", 16, 8),
	}}
	weights, err := ReadWeights(file, spec)
	if err != nil {
		t.Fatal(err)
	}
	if weights.Output != nil ||
		weights.Layers[0].FeedForwardNorm.Name != "blk.0.post_attention_norm.weight" ||
		weights.Layers[0].AttentionPostNorm != nil {
		t.Fatalf("unexpected Seed-OSS weights: %+v", weights)
	}
}

func TestReadWeightsCohere2UsesOneNormPerBlock(t *testing.T) {
	spec := Spec{
		Architecture: "cohere2", BlockCount: 1, EmbeddingLength: 8,
		FeedForwardLength: 16, HeadCount: 2, HeadCountKV: 1,
		KeyLength: 4, ValueLength: 4, RopeDimensionCount: 2,
		VocabularySize: 32, LayerNormEpsilon: 1e-5,
	}
	file := &gguf.File{Tensors: []gguf.TensorInfo{
		tensorInfo("token_embd.weight", 8, 32), tensorInfo("output_norm.weight", 8),
		tensorInfo("output.weight", 8, 32),
		tensorInfo("blk.0.attn_norm.weight", 8),
		tensorInfo("blk.0.attn_q.weight", 8, 8),
		tensorInfo("blk.0.attn_k.weight", 8, 4),
		tensorInfo("blk.0.attn_v.weight", 8, 4),
		tensorInfo("blk.0.attn_output.weight", 8, 8),
		tensorInfo("blk.0.rope_freqs.weight", 1),
		tensorInfo("blk.0.ffn_gate.weight", 8, 16),
		tensorInfo("blk.0.ffn_up.weight", 8, 16),
		tensorInfo("blk.0.ffn_down.weight", 16, 8),
	}}
	weights, err := ReadWeights(file, spec)
	if err != nil {
		t.Fatal(err)
	}
	if weights.Output != nil ||
		weights.Layers[0].AttentionNorm.Name != "blk.0.attn_norm.weight" ||
		weights.Layers[0].FeedForwardNorm.Name != "" ||
		weights.Layers[0].RopeFactors == nil {
		t.Fatalf("unexpected Cohere2 weights: %+v", weights)
	}
}

func TestReadWeightsCommandR64UsesPerHeadQKNorms(t *testing.T) {
	spec := Spec{
		Architecture: "command-r", BlockCount: 64, EmbeddingLength: 8,
		FeedForwardLength: 16, HeadCount: 2, HeadCountKV: 1,
		KeyLength: 4, ValueLength: 4, VocabularySize: 32, LayerNormEpsilon: 1e-5,
	}
	tensors := []gguf.TensorInfo{
		tensorInfo("token_embd.weight", 8, 32), tensorInfo("output_norm.weight", 8),
		tensorInfo("output.weight", 8, 32),
	}
	for block := range uint32(64) {
		prefix := fmt.Sprintf("blk.%d.", block)
		tensors = append(tensors,
			tensorInfo(prefix+"attn_norm.weight", 8),
			tensorInfo(prefix+"attn_q.weight", 8, 8),
			tensorInfo(prefix+"attn_k.weight", 8, 4),
			tensorInfo(prefix+"attn_v.weight", 8, 4),
			tensorInfo(prefix+"attn_output.weight", 8, 8),
			tensorInfo(prefix+"attn_q_norm.weight", 4, 2),
			tensorInfo(prefix+"attn_k_norm.weight", 4, 1),
			tensorInfo(prefix+"ffn_gate.weight", 8, 16),
			tensorInfo(prefix+"ffn_up.weight", 8, 16),
			tensorInfo(prefix+"ffn_down.weight", 16, 8),
		)
	}
	weights, err := ReadWeights(&gguf.File{Tensors: tensors}, spec)
	if err != nil {
		t.Fatal(err)
	}
	if weights.Output != nil || weights.Layers[0].FeedForwardNorm.Name != "" ||
		weights.Layers[0].AttentionQNorm == nil ||
		weights.Layers[0].AttentionQNorm.Dimensions != 2 ||
		weights.Layers[0].AttentionKNorm == nil {
		t.Fatalf("unexpected Command R weights: %+v", weights)
	}
}

func TestReadWeightsPLaMoUsesOneNormPerBlock(t *testing.T) {
	spec := Spec{
		Architecture: "plamo", BlockCount: 1, EmbeddingLength: 8,
		FeedForwardLength: 16, HeadCount: 2, HeadCountKV: 1,
		KeyLength: 4, ValueLength: 4, VocabularySize: 32, RMSNormEpsilon: 1e-5,
	}
	file := &gguf.File{Tensors: []gguf.TensorInfo{
		tensorInfo("token_embd.weight", 8, 32), tensorInfo("output_norm.weight", 8),
		tensorInfo("output.weight", 8, 32), tensorInfo("blk.0.attn_norm.weight", 8),
		tensorInfo("blk.0.attn_q.weight", 8, 8), tensorInfo("blk.0.attn_k.weight", 8, 4),
		tensorInfo("blk.0.attn_v.weight", 8, 4), tensorInfo("blk.0.attn_output.weight", 8, 8),
		tensorInfo("blk.0.ffn_gate.weight", 8, 16), tensorInfo("blk.0.ffn_up.weight", 8, 16),
		tensorInfo("blk.0.ffn_down.weight", 16, 8),
	}}
	weights, err := ReadWeights(file, spec)
	if err != nil {
		t.Fatal(err)
	}
	if weights.Output == nil || weights.Layers[0].FeedForwardNorm.Name != "" {
		t.Fatalf("unexpected PLaMo weights: %+v", weights)
	}
}

func TestReadWeightsStableLMOptionalNormLayouts(t *testing.T) {
	for _, sequential := range []bool{false, true} {
		spec := Spec{
			Architecture: "stablelm", BlockCount: 1, EmbeddingLength: 8,
			FeedForwardLength: 16, HeadCount: 2, HeadCountKV: 1,
			KeyLength: 4, ValueLength: 4, RopeDimensionCount: 2,
			VocabularySize: 32, LayerNormEpsilon: 1e-5,
		}
		tensors := []gguf.TensorInfo{
			tensorInfo("token_embd.weight", 8, 32), tensorInfo("output_norm.weight", 8),
			tensorInfo("output_norm.bias", 8), tensorInfo("output.weight", 8, 32),
			tensorInfo("blk.0.attn_norm.weight", 8), tensorInfo("blk.0.attn_norm.bias", 8),
			tensorInfo("blk.0.attn_q.weight", 8, 8), tensorInfo("blk.0.attn_k.weight", 8, 4),
			tensorInfo("blk.0.attn_v.weight", 8, 4), tensorInfo("blk.0.attn_output.weight", 8, 8),
			tensorInfo("blk.0.ffn_gate.weight", 8, 16), tensorInfo("blk.0.ffn_up.weight", 8, 16),
			tensorInfo("blk.0.ffn_down.weight", 16, 8),
		}
		if sequential {
			tensors = append(tensors,
				tensorInfo("blk.0.ffn_norm.weight", 8),
				tensorInfo("blk.0.ffn_norm.bias", 8),
			)
		} else {
			tensors = append(tensors,
				tensorInfo("blk.0.attn_q_norm.weight", 4, 2),
				tensorInfo("blk.0.attn_k_norm.weight", 4, 1),
			)
		}
		weights, err := ReadWeights(&gguf.File{Tensors: tensors}, spec)
		if err != nil {
			t.Fatal(err)
		}
		layer := weights.Layers[0]
		if sequential != (layer.FeedForwardNorm.Name != "") ||
			sequential == (layer.AttentionQNorm != nil) {
			t.Fatalf("unexpected StableLM layout: %+v", layer)
		}
	}
}

func TestReadWeightsPhi2FusedQKV(t *testing.T) {
	spec := Spec{
		Architecture: "phi2", BlockCount: 1, EmbeddingLength: 8,
		FeedForwardLength: 16, HeadCount: 2, HeadCountKV: 2,
		KeyLength: 4, ValueLength: 4, RopeDimensionCount: 2,
		VocabularySize: 32, LayerNormEpsilon: 1e-5,
	}
	file := &gguf.File{Tensors: []gguf.TensorInfo{
		tensorInfo("token_embd.weight", 8, 32),
		tensorInfo("output_norm.weight", 8), tensorInfo("output_norm.bias", 8),
		tensorInfo("output.weight", 8, 32), tensorInfo("output.bias", 32),
		tensorInfo("blk.0.attn_norm.weight", 8), tensorInfo("blk.0.attn_norm.bias", 8),
		tensorInfo("blk.0.attn_qkv.weight", 8, 24), tensorInfo("blk.0.attn_qkv.bias", 24),
		tensorInfo("blk.0.attn_output.weight", 8, 8), tensorInfo("blk.0.attn_output.bias", 8),
		tensorInfo("blk.0.ffn_up.weight", 8, 16), tensorInfo("blk.0.ffn_up.bias", 16),
		tensorInfo("blk.0.ffn_down.weight", 16, 8), tensorInfo("blk.0.ffn_down.bias", 8),
	}}
	weights, err := ReadWeights(file, spec)
	if err != nil {
		t.Fatal(err)
	}
	layer := weights.Layers[0]
	if weights.Output == nil || weights.OutputBias == nil ||
		layer.AttentionQKV == nil || layer.AttentionQKVBias == nil ||
		layer.AttentionQ.Name != "" || layer.AttentionK.Name != "" || layer.AttentionV.Name != "" ||
		layer.FeedForwardNorm.Name != "" || layer.FeedForwardGate.Name != "" ||
		layer.AttentionOutputBias == nil || layer.FeedForwardUpBias == nil ||
		layer.FeedForwardDownBias == nil {
		t.Fatalf("unexpected Phi-2 weights: %+v", weights)
	}
}

func TestReadWeightsPhi3LongRoPEAndFusedFFN(t *testing.T) {
	spec := Spec{
		Architecture: "phi3", BlockCount: 1, ContextLength: 128,
		OriginalContextLength: 32, EmbeddingLength: 8, FeedForwardLength: 16,
		HeadCount: 2, HeadCountKV: 2, KeyLength: 4, ValueLength: 4,
		RopeDimensionCount: 4, RopeScalingType: "longrope",
		RopeAttentionFactor: 1.1, VocabularySize: 32, RMSNormEpsilon: 1e-5,
	}
	file := &gguf.File{Tensors: []gguf.TensorInfo{
		tensorInfo("token_embd.weight", 8, 32), tensorInfo("output_norm.weight", 8),
		tensorInfo("blk.0.attn_norm.weight", 8),
		tensorInfo("blk.0.attn_qkv.weight", 8, 24), tensorInfo("blk.0.attn_qkv.bias", 24),
		tensorInfo("blk.0.attn_output.weight", 8, 8),
		tensorInfo("blk.0.ffn_norm.weight", 8),
		tensorInfo("blk.0.ffn_up.weight", 8, 32),
		tensorInfo("blk.0.ffn_down.weight", 16, 8),
		tensorInfo("blk.0.rope_factors_long.weight", 2),
		tensorInfo("blk.0.rope_factors_short.weight", 2),
	}}
	weights, err := ReadWeights(file, spec)
	if err != nil {
		t.Fatal(err)
	}
	layer := weights.Layers[0]
	if weights.Output != nil || layer.AttentionQKV == nil || layer.AttentionQKVBias == nil ||
		layer.RopeFactors == nil || layer.RopeFactors.Name != "blk.0.rope_factors_long.weight" ||
		layer.FeedForwardUp.Shape[1] != 32 || layer.FeedForwardGate.Name != "" {
		t.Fatalf("unexpected Phi-3 weights: %+v", weights)
	}
	spec.ContextLength = 16
	shortWeights, err := ReadWeights(file, spec)
	if err != nil {
		t.Fatal(err)
	}
	if shortWeights.Layers[0].RopeFactors == nil ||
		shortWeights.Layers[0].RopeFactors.Name != "blk.0.rope_factors_short.weight" {
		t.Fatalf("Phi-3 short-context factors were not selected: %+v", shortWeights.Layers[0])
	}
}

func TestReadWeightsGPTNeoX(t *testing.T) {
	spec := Spec{
		Architecture: "gptneox", BlockCount: 1, EmbeddingLength: 8,
		FeedForwardLength: 16, HeadCount: 2, HeadCountKV: 2,
		KeyLength: 4, ValueLength: 4, RopeDimensionCount: 2,
		VocabularySize: 32, LayerNormEpsilon: 1e-5,
	}
	file := &gguf.File{Tensors: []gguf.TensorInfo{
		tensorInfo("token_embd.weight", 8, 32),
		tensorInfo("output_norm.weight", 8), tensorInfo("output_norm.bias", 8),
		tensorInfo("output.weight", 8, 32),
		tensorInfo("blk.0.attn_norm.weight", 8), tensorInfo("blk.0.attn_norm.bias", 8),
		tensorInfo("blk.0.attn_qkv.weight", 8, 24), tensorInfo("blk.0.attn_qkv.bias", 24),
		tensorInfo("blk.0.attn_output.weight", 8, 8), tensorInfo("blk.0.attn_output.bias", 8),
		tensorInfo("blk.0.ffn_norm.weight", 8), tensorInfo("blk.0.ffn_norm.bias", 8),
		tensorInfo("blk.0.ffn_up.weight", 8, 16), tensorInfo("blk.0.ffn_up.bias", 16),
		tensorInfo("blk.0.ffn_down.weight", 16, 8), tensorInfo("blk.0.ffn_down.bias", 8),
	}}
	weights, err := ReadWeights(file, spec)
	if err != nil {
		t.Fatal(err)
	}
	layer := weights.Layers[0]
	if weights.Output == nil || layer.AttentionQKV == nil || layer.AttentionQKVBias == nil ||
		layer.AttentionQ.Name != "" || layer.FeedForwardNorm.Name == "" ||
		layer.FeedForwardNormBias == nil || layer.AttentionOutputBias == nil ||
		layer.FeedForwardUpBias == nil || layer.FeedForwardDownBias == nil ||
		layer.FeedForwardGate.Name != "" {
		t.Fatalf("unexpected GPT-NeoX weights: %+v", weights)
	}
}

func TestReadWeightsFalcon40B(t *testing.T) {
	spec := Spec{
		Architecture: "falcon", BlockCount: 1, EmbeddingLength: 8,
		FeedForwardLength: 16, HeadCount: 2, HeadCountKV: 1,
		KeyLength: 4, ValueLength: 4, RopeDimensionCount: 4,
		VocabularySize: 32, LayerNormEpsilon: 1e-5,
	}
	file := &gguf.File{Tensors: []gguf.TensorInfo{
		tensorInfo("token_embd.weight", 8, 32),
		tensorInfo("output_norm.weight", 8), tensorInfo("output_norm.bias", 8),
		tensorInfo("blk.0.attn_norm.weight", 8), tensorInfo("blk.0.attn_norm.bias", 8),
		tensorInfo("blk.0.attn_norm_2.weight", 8), tensorInfo("blk.0.attn_norm_2.bias", 8),
		tensorInfo("blk.0.attn_qkv.weight", 8, 16),
		tensorInfo("blk.0.attn_output.weight", 8, 8),
		tensorInfo("blk.0.ffn_up.weight", 8, 16),
		tensorInfo("blk.0.ffn_down.weight", 16, 8),
	}}
	weights, err := ReadWeights(file, spec)
	if err != nil {
		t.Fatal(err)
	}
	layer := weights.Layers[0]
	if weights.Output != nil || layer.AttentionQKV == nil || layer.AttentionQKVBias != nil ||
		layer.AttentionNorm2 == nil || layer.AttentionNorm2Bias == nil ||
		layer.FeedForwardNorm.Name != "" || layer.FeedForwardGate.Name != "" ||
		layer.AttentionOutputBias != nil || layer.FeedForwardUpBias != nil ||
		layer.FeedForwardDownBias != nil {
		t.Fatalf("unexpected Falcon weights: %+v", weights)
	}
}

func TestReadWeightsBitNetSubNormsAndScales(t *testing.T) {
	spec := Spec{
		Architecture: "bitnet", BlockCount: 1, EmbeddingLength: 8,
		FeedForwardLength: 16, HeadCount: 2, HeadCountKV: 1,
		KeyLength: 4, ValueLength: 4, VocabularySize: 32, RMSNormEpsilon: 1e-5,
	}
	tensors := []gguf.TensorInfo{
		tensorInfo("token_embd.weight", 8, 32), tensorInfo("output_norm.weight", 8),
		tensorInfo("blk.0.attn_norm.weight", 8), tensorInfo("blk.0.attn_sub_norm.weight", 8),
		tensorInfo("blk.0.attn_q.weight", 8, 8), tensorInfo("blk.0.attn_k.weight", 8, 4),
		tensorInfo("blk.0.attn_v.weight", 8, 4), tensorInfo("blk.0.attn_output.weight", 8, 8),
		tensorInfo("blk.0.ffn_norm.weight", 8), tensorInfo("blk.0.ffn_sub_norm.weight", 16),
		tensorInfo("blk.0.ffn_gate.weight", 8, 16), tensorInfo("blk.0.ffn_up.weight", 8, 16),
		tensorInfo("blk.0.ffn_down.weight", 16, 8),
	}
	for _, name := range []string{
		"attn_q.scale", "attn_k.scale", "attn_v.scale", "attn_output.scale",
		"ffn_gate.scale", "ffn_up.scale", "ffn_down.scale",
	} {
		tensors = append(tensors, tensorInfo("blk.0."+name, 1))
	}
	weights, err := ReadWeights(&gguf.File{Tensors: tensors}, spec)
	if err != nil {
		t.Fatal(err)
	}
	layer := weights.Layers[0]
	if weights.Output != nil || layer.AttentionSubNorm == nil || layer.FeedForwardSubNorm == nil ||
		layer.AttentionQScale == nil || layer.AttentionKScale == nil ||
		layer.AttentionVScale == nil || layer.AttentionOutputScale == nil ||
		layer.FeedForwardGateScale == nil || layer.FeedForwardUpScale == nil ||
		layer.FeedForwardDownScale == nil {
		t.Fatalf("unexpected BitNet weights: %+v", weights)
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
