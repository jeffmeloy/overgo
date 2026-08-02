package model

import (
	"fmt"

	"llamacpp2go/internal/gguf"

	"strings"

	"testing"
)

func TestReadWeightsGLM4NextNBlock(t *testing.T) {
	spec := Spec{CommonSpec: CommonSpec{Architecture: "glm4", BlockCount: 1,
		NextNPredictLayers: 1, EmbeddingLength: 8, FeedForwardLength: 12,
		VocabularySize: 32, RMSNormEpsilon: 1e-5}, AttentionSpec: AttentionSpec{
		HeadCount: 2, HeadCountKV: 1, KeyLength: 4, ValueLength: 4,
		RopeDimensionCount: 4, RopeFrequencyBase: 10000},
	}
	tensors := []gguf.TensorInfo{
		tensorInfo("token_embd.weight", 8, 32), tensorInfo("output_norm.weight", 8),
		tensorInfo("output.weight", 8, 32),
	}
	for block := 0; block < 2; block++ {
		prefix := fmt.Sprintf("blk.%d.", block)
		tensors = append(tensors,
			tensorInfo(prefix+"attn_norm.weight", 8),
			tensorInfo(prefix+"attn_qkv.weight", 8, 16),
			tensorInfo(prefix+"attn_output.weight", 8, 8),
			tensorInfo(prefix+"post_attention_norm.weight", 8),
			tensorInfo(prefix+"ffn_norm.weight", 8),
			tensorInfo(prefix+"ffn_up.weight", 8, 24),
			tensorInfo(prefix+"ffn_down.weight", 12, 8),
			tensorInfo(prefix+"post_ffw_norm.weight", 8),
		)
	}
	tensors = append(tensors,
		tensorInfo("blk.1.nextn.eh_proj.weight", 16, 8),
		tensorInfo("blk.1.nextn.enorm.weight", 8),
		tensorInfo("blk.1.nextn.hnorm.weight", 8),
		tensorInfo("blk.1.nextn.shared_head_norm.weight", 8),
	)
	weights, err := ReadWeights(&gguf.File{Tensors: tensors}, spec)
	if err != nil {
		t.Fatal(err)
	}
	if len(weights.Layers) != 1 || len(weights.NextNMTP) != 1 ||
		weights.NextNMTP[0].Layer.AttentionQKV == nil ||
		weights.NextNMTP[0].OutputNorm == nil {
		t.Fatalf("unexpected GLM4 NextN catalog: %+v", weights.NextNMTP)
	}
}

func TestReadWeightsGraniteDense(t *testing.T) {
	spec := Spec{CommonSpec: CommonSpec{Architecture: "granite",
		BlockCount:    1,
		ContextLength: 4,

		EmbeddingLength:   8,
		FeedForwardLength: 16,

		VocabularySize: 32}, AttentionSpec: AttentionSpec{OriginalContextLength: 2,

		HeadCount:          2,
		HeadCountKV:        1,
		KeyLength:          4,
		ValueLength:        4,
		RopeDimensionCount: 4,
		RopeScalingType:    "longrope"},
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
		tensorInfo("blk.0.rope_factors_long.weight", 2),
		tensorInfo("blk.0.rope_factors_short.weight", 2),
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
		weights.Layers[0].FeedForwardGateBias == nil ||
		weights.Layers[0].RopeFactors == nil ||
		weights.Layers[0].RopeFactors.Name != "blk.0.rope_factors_long.weight" {
		t.Fatalf("unexpected dense Granite weights: %+v", weights)
	}
}

func TestReadWeightsMaincoder(t *testing.T) {
	spec := Spec{CommonSpec: CommonSpec{Architecture: "maincoder",
		BlockCount:        1,
		EmbeddingLength:   8,
		FeedForwardLength: 16,

		VocabularySize: 32}, AttentionSpec: AttentionSpec{HeadCount: 2,
		HeadCountKV: 1,
		KeyLength:   4,
		ValueLength: 4},
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
	spec := Spec{CommonSpec: CommonSpec{Architecture: "mistral3",
		BlockCount:        1,
		EmbeddingLength:   8,
		FeedForwardLength: 16,

		VocabularySize: 32}, AttentionSpec: AttentionSpec{HeadCount: 2,
		HeadCountKV: 1,
		KeyLength:   4,
		ValueLength: 4},
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

func TestReadWeightsMistral3MoE(t *testing.T) {
	spec := Spec{CommonSpec: CommonSpec{Architecture: "mistral3", BlockCount: 1, EmbeddingLength: 8,
		FeedForwardLength: 16,
		VocabularySize:    32}, AttentionSpec: AttentionSpec{HeadCount: 2, HeadCountKV: 1,
		KeyLength: 4, ValueLength: 4}, MoESpec: MoESpec{ExpertCount: 4, ExpertUsedCount: 2, ExpertFeedForward: 16,
		ExpertWeightsScale: 1, ExpertWeightsNorm: true},
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
		tensorInfo("blk.0.ffn_gate_inp.weight", 8, 4),
		tensorInfo("blk.0.ffn_gate_exps.weight", 8, 16, 4),
		tensorInfo("blk.0.ffn_up_exps.weight", 8, 16, 4),
		tensorInfo("blk.0.ffn_down_exps.weight", 16, 8, 4),
	}}
	weights, err := ReadWeights(file, spec)
	if err != nil {
		t.Fatal(err)
	}
	layer := weights.Layers[0]
	if layer.FeedForwardRouter == nil || layer.FeedForwardGateExperts == nil ||
		layer.FeedForwardUpExperts == nil || layer.FeedForwardDownExperts == nil ||
		layer.FeedForwardGate.Name != "" || layer.FeedForwardUp.Name != "" ||
		layer.FeedForwardDown.Name != "" {
		t.Fatalf("unexpected Mistral 3 MoE weights: %+v", layer)
	}
}

func TestReadWeightsOrion(t *testing.T) {
	spec := Spec{CommonSpec: CommonSpec{Architecture: "orion",
		BlockCount:        1,
		EmbeddingLength:   8,
		FeedForwardLength: 16,

		VocabularySize:   32,
		LayerNormEpsilon: 1e-5}, AttentionSpec: AttentionSpec{HeadCount: 2,
		HeadCountKV: 1,
		KeyLength:   4,
		ValueLength: 4},
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
	spec := Spec{CommonSpec: CommonSpec{Architecture: "starcoder2",
		BlockCount:        1,
		EmbeddingLength:   8,
		FeedForwardLength: 16,

		VocabularySize:   32,
		LayerNormEpsilon: 1e-5}, AttentionSpec: AttentionSpec{HeadCount: 2,
		HeadCountKV: 1,
		KeyLength:   4,
		ValueLength: 4},
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
	spec := Spec{CommonSpec: CommonSpec{Architecture: "codeshell",
		BlockCount:        1,
		EmbeddingLength:   8,
		FeedForwardLength: 16,

		VocabularySize:   32,
		LayerNormEpsilon: 1e-5}, AttentionSpec: AttentionSpec{HeadCount: 2,
		HeadCountKV: 1,
		KeyLength:   4,
		ValueLength: 4},
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
	spec := Spec{CommonSpec: CommonSpec{Architecture: "baichuan",
		BlockCount:        1,
		EmbeddingLength:   8,
		FeedForwardLength: 16,

		VocabularySize: 32,
		RMSNormEpsilon: 1e-6}, AttentionSpec: AttentionSpec{HeadCount: 2,
		HeadCountKV: 2,
		KeyLength:   4,
		ValueLength: 4},
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
	spec := Spec{CommonSpec: CommonSpec{Architecture: "arcee",
		BlockCount:        1,
		EmbeddingLength:   8,
		FeedForwardLength: 16,

		VocabularySize: 32,
		RMSNormEpsilon: 1e-5}, AttentionSpec: AttentionSpec{HeadCount: 2,
		HeadCountKV: 1,
		KeyLength:   4,
		ValueLength: 4},
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
	spec := Spec{CommonSpec: CommonSpec{Architecture: "nemotron",
		BlockCount:        1,
		EmbeddingLength:   8,
		FeedForwardLength: 16,

		VocabularySize:   32,
		LayerNormEpsilon: 1e-5}, AttentionSpec: AttentionSpec{HeadCount: 2,
		HeadCountKV: 1,
		KeyLength:   4,
		ValueLength: 4},
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
	spec := Spec{CommonSpec: CommonSpec{Architecture: "jais2", BlockCount: 1, EmbeddingLength: 8,
		FeedForwardLength: 16,
		VocabularySize:    32, LayerNormEpsilon: 1e-5}, AttentionSpec: AttentionSpec{HeadCount: 2, HeadCountKV: 2,
		KeyLength: 4, ValueLength: 4},
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

func TestReadWeightsJais(t *testing.T) {
	spec := Spec{CommonSpec: CommonSpec{Architecture: "jais", BlockCount: 1, EmbeddingLength: 8,
		FeedForwardLength: 12,
		VocabularySize:    32, LayerNormEpsilon: 1e-5}, AttentionSpec: AttentionSpec{HeadCount: 2, HeadCountKV: 2,
		KeyLength: 4, ValueLength: 4},
	}
	file := &gguf.File{Tensors: []gguf.TensorInfo{
		tensorInfo("token_embd.weight", 8, 32),
		tensorInfo("output_norm.weight", 8), tensorInfo("output_norm.bias", 8),
		tensorInfo("output.weight", 8, 32),
		tensorInfo("blk.0.attn_norm.weight", 8), tensorInfo("blk.0.attn_norm.bias", 8),
		tensorInfo("blk.0.attn_qkv.weight", 8, 24), tensorInfo("blk.0.attn_qkv.bias", 24),
		tensorInfo("blk.0.attn_output.weight", 8, 8), tensorInfo("blk.0.attn_output.bias", 8),
		tensorInfo("blk.0.ffn_norm.weight", 8), tensorInfo("blk.0.ffn_norm.bias", 8),
		tensorInfo("blk.0.ffn_gate.weight", 8, 12), tensorInfo("blk.0.ffn_gate.bias", 12),
		tensorInfo("blk.0.ffn_up.weight", 8, 12), tensorInfo("blk.0.ffn_up.bias", 12),
		tensorInfo("blk.0.ffn_down.weight", 12, 8), tensorInfo("blk.0.ffn_down.bias", 8),
	}}
	weights, err := ReadWeights(file, spec)
	if err != nil {
		t.Fatal(err)
	}
	layer := weights.Layers[0]
	if weights.Output == nil || weights.OutputNormBias == nil || layer.AttentionQKV == nil ||
		layer.AttentionQKVBias == nil || layer.AttentionQ.Name != "" ||
		layer.AttentionNormBias == nil || layer.FeedForwardNormBias == nil ||
		layer.AttentionOutputBias == nil || layer.FeedForwardGateBias == nil ||
		layer.FeedForwardUpBias == nil || layer.FeedForwardDownBias == nil {
		t.Fatalf("unexpected Jais weights: %+v", weights)
	}
}

func TestReadWeightsOLMoWithoutNormTensors(t *testing.T) {
	spec := Spec{CommonSpec: CommonSpec{Architecture: "olmo", BlockCount: 1, EmbeddingLength: 8,
		FeedForwardLength: 16,
		VocabularySize:    32, LayerNormEpsilon: 1e-5}, AttentionSpec: AttentionSpec{HeadCount: 2, HeadCountKV: 1,
		KeyLength: 4, ValueLength: 4},
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
	spec := Spec{CommonSpec: CommonSpec{Architecture: "seed_oss", BlockCount: 1, EmbeddingLength: 8,
		FeedForwardLength: 16,
		VocabularySize:    32, RMSNormEpsilon: 1e-5}, AttentionSpec: AttentionSpec{HeadCount: 2, HeadCountKV: 1,
		KeyLength: 4, ValueLength: 4},
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
	spec := Spec{CommonSpec: CommonSpec{Architecture: "cohere2", BlockCount: 1, EmbeddingLength: 8,
		FeedForwardLength: 16,

		VocabularySize: 32, LayerNormEpsilon: 1e-5}, AttentionSpec: AttentionSpec{HeadCount: 2, HeadCountKV: 1,
		KeyLength: 4, ValueLength: 4, RopeDimensionCount: 2},
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

func TestReadWeightsCohere2MoEDensePrefixAndFusedExperts(t *testing.T) {
	spec := Spec{CommonSpec: CommonSpec{Architecture: "cohere2moe", BlockCount: 2, EmbeddingLength: 8,
		FeedForwardLength: 12,
		VocabularySize:    32,
		RMSNormEpsilon:    1e-5}, AttentionSpec: AttentionSpec{HeadCount: 2, HeadCountKV: 1,
		KeyLength: 4, ValueLength: 4}, MoESpec: MoESpec{LeadingDenseBlocks: 1,
		ExpertCount: 4, ExpertUsedCount: 2, ExpertFeedForward: 6,
		SharedExpertCount: 1, SharedExpertFF: 6},
	}
	file := &gguf.File{Tensors: []gguf.TensorInfo{
		tensorInfo("token_embd.weight", 8, 32), tensorInfo("output_norm.weight", 8),
		tensorInfo("output.weight", 8, 32),
		tensorInfo("blk.0.attn_norm.weight", 8),
		tensorInfo("blk.0.attn_qkv.weight", 8, 16),
		tensorInfo("blk.0.attn_output.weight", 8, 8),
		tensorInfo("blk.0.ffn_gate.weight", 8, 12),
		tensorInfo("blk.0.ffn_up.weight", 8, 12),
		tensorInfo("blk.0.ffn_down.weight", 12, 8),
		tensorInfo("blk.1.attn_norm.weight", 8),
		tensorInfo("blk.1.attn_q.weight", 8, 8),
		tensorInfo("blk.1.attn_k.weight", 8, 4),
		tensorInfo("blk.1.attn_v.weight", 8, 4),
		tensorInfo("blk.1.attn_output.weight", 8, 8),
		tensorInfo("blk.1.ffn_gate_inp.weight", 8, 4),
		tensorInfo("blk.1.ffn_gate_up_exps.weight", 8, 12, 4),
		tensorInfo("blk.1.ffn_down_exps.weight", 6, 8, 4),
		tensorInfo("blk.1.ffn_gate_shexp.weight", 8, 6),
		tensorInfo("blk.1.ffn_up_shexp.weight", 8, 6),
		tensorInfo("blk.1.ffn_down_shexp.weight", 6, 8),
	}}
	weights, err := ReadWeights(file, spec)
	if err != nil {
		t.Fatal(err)
	}
	dense, moe := weights.Layers[0], weights.Layers[1]
	if weights.Output == nil || dense.AttentionQKV == nil || dense.FeedForwardGate.Name == "" ||
		dense.FeedForwardRouter != nil || moe.AttentionQ.Name == "" || moe.FeedForwardNorm.Name != "" ||
		moe.FeedForwardGateUpExperts == nil || moe.FeedForwardGateExperts != nil ||
		moe.FeedForwardUpExperts != nil || moe.FeedForwardDownExperts == nil ||
		moe.FeedForwardSharedGate == nil || moe.FeedForwardSharedUp == nil || moe.FeedForwardSharedDown == nil {
		t.Fatalf("unexpected Cohere2-MoE catalog: %+v", weights)
	}
}

func TestReadWeightsCohere2MoEMTP(t *testing.T) {
	spec := Spec{CommonSpec: CommonSpec{Architecture: "cohere2moe", BlockCount: 1, NextNPredictLayers: 1,
		EmbeddingLength: 8, FeedForwardLength: 12, VocabularySize: 32,

		RMSNormEpsilon: 1e-5}, AttentionSpec: AttentionSpec{HeadCount: 2, HeadCountKV: 1, KeyLength: 4, ValueLength: 4}, MoESpec: MoESpec{LeadingDenseBlocks: 1,
		ExpertCount: 4, ExpertUsedCount: 2, ExpertFeedForward: 6,
		SharedExpertCount: 1, SharedExpertFF: 6},
	}
	common := []gguf.TensorInfo{
		tensorInfo("token_embd.weight", 8, 32), tensorInfo("output_norm.weight", 8),
		tensorInfo("output.weight", 8, 32),
	}
	trunk := []gguf.TensorInfo{
		tensorInfo("blk.0.attn_norm.weight", 8), tensorInfo("blk.0.attn_q.weight", 8, 8),
		tensorInfo("blk.0.attn_k.weight", 8, 4), tensorInfo("blk.0.attn_v.weight", 8, 4),
		tensorInfo("blk.0.attn_output.weight", 8, 8), tensorInfo("blk.0.ffn_gate.weight", 8, 12),
		tensorInfo("blk.0.ffn_up.weight", 8, 12), tensorInfo("blk.0.ffn_down.weight", 12, 8),
	}
	mtp := []gguf.TensorInfo{
		tensorInfo("blk.1.attn_norm.weight", 8), tensorInfo("blk.1.attn_q.weight", 8, 8),
		tensorInfo("blk.1.attn_k.weight", 8, 4), tensorInfo("blk.1.attn_v.weight", 8, 4),
		tensorInfo("blk.1.attn_output.weight", 8, 8), tensorInfo("blk.1.ffn_gate_inp.weight", 8, 4),
		tensorInfo("blk.1.ffn_gate_up_exps.weight", 8, 12, 4), tensorInfo("blk.1.ffn_down_exps.weight", 6, 8, 4),
		tensorInfo("blk.1.ffn_gate_shexp.weight", 8, 6), tensorInfo("blk.1.ffn_up_shexp.weight", 8, 6),
		tensorInfo("blk.1.ffn_down_shexp.weight", 6, 8), tensorInfo("blk.1.nextn.eh_proj.weight", 16, 8),
		tensorInfo("blk.1.nextn.enorm.weight", 8), tensorInfo("blk.1.nextn.hnorm.weight", 8),
		tensorInfo("blk.1.nextn.shared_head_norm.weight", 8),
		tensorInfo("blk.1.nextn.shared_head_head.weight", 8, 32),
	}
	combined, err := ReadWeights(&gguf.File{Tensors: append(append(common, trunk...), mtp...)}, spec)
	if err != nil {
		t.Fatal(err)
	}
	if len(combined.Layers) != 1 || combined.Cohere2MTP == nil || combined.Cohere2MTP.MTPOnly ||
		combined.Cohere2MTP.Layer.FeedForwardGateUpExperts == nil ||
		combined.Cohere2MTP.OutputNorm == nil || combined.Cohere2MTP.Output == nil {
		t.Fatalf("unexpected combined Cohere2-MoE MTP catalog: %+v", combined.Cohere2MTP)
	}
	sidecar, err := ReadWeights(&gguf.File{Tensors: append(common, mtp...)}, spec)
	if err != nil {
		t.Fatal(err)
	}
	if len(sidecar.Layers) != 0 || sidecar.Cohere2MTP == nil || !sidecar.Cohere2MTP.MTPOnly {
		t.Fatalf("unexpected Cohere2-MoE MTP sidecar catalog: %+v", sidecar)
	}
}

func TestReadWeightsHYV3DetectsDenseAndFusedMoELayers(t *testing.T) {
	spec := Spec{CommonSpec: CommonSpec{Architecture: "hy_v3", BlockCount: 2, EmbeddingLength: 8,
		FeedForwardLength: 12,

		VocabularySize: 32, RMSNormEpsilon: 1e-5}, AttentionSpec: AttentionSpec{HeadCount: 2, HeadCountKV: 1,
		KeyLength: 4, ValueLength: 4, RopeDimensionCount: 4}, MoESpec: MoESpec{ExpertCount: 4, ExpertUsedCount: 2, ExpertFeedForward: 6,
		SharedExpertFF: 6, ExpertGatingFunc: 2},
	}
	file := &gguf.File{Tensors: []gguf.TensorInfo{
		tensorInfo("token_embd.weight", 8, 32), tensorInfo("output_norm.weight", 8),
		tensorInfo("output.weight", 8, 32),
		tensorInfo("blk.0.attn_norm.weight", 8),
		tensorInfo("blk.0.attn_qkv.weight", 8, 16),
		tensorInfo("blk.0.attn_output.weight", 8, 8),
		tensorInfo("blk.0.attn_q_norm.weight", 4), tensorInfo("blk.0.attn_k_norm.weight", 4),
		tensorInfo("blk.0.ffn_norm.weight", 8),
		tensorInfo("blk.0.ffn_gate.weight", 8, 12),
		tensorInfo("blk.0.ffn_up.weight", 8, 12),
		tensorInfo("blk.0.ffn_down.weight", 12, 8),
		tensorInfo("blk.1.attn_norm.weight", 8),
		tensorInfo("blk.1.attn_q.weight", 8, 8),
		tensorInfo("blk.1.attn_k.weight", 8, 4),
		tensorInfo("blk.1.attn_v.weight", 8, 4),
		tensorInfo("blk.1.attn_output.weight", 8, 8),
		tensorInfo("blk.1.attn_q_norm.weight", 4), tensorInfo("blk.1.attn_k_norm.weight", 4),
		tensorInfo("blk.1.ffn_norm.weight", 8),
		tensorInfo("blk.1.ffn_gate_inp.weight", 8, 4),
		tensorInfo("blk.1.exp_probs_b", 4),
		tensorInfo("blk.1.ffn_gate_up_exps.weight", 8, 12, 4),
		tensorInfo("blk.1.ffn_down_exps.weight", 6, 8, 4),
		tensorInfo("blk.1.ffn_gate_shexp.weight", 8, 6),
		tensorInfo("blk.1.ffn_up_shexp.weight", 8, 6),
		tensorInfo("blk.1.ffn_down_shexp.weight", 6, 8),
	}}
	weights, err := ReadWeights(file, spec)
	if err != nil {
		t.Fatal(err)
	}
	dense, moe := weights.Layers[0], weights.Layers[1]
	if dense.AttentionQKV == nil || dense.AttentionQNorm == nil || dense.AttentionKNorm == nil ||
		dense.FeedForwardGate.Name == "" || dense.FeedForwardRouter != nil ||
		moe.AttentionQ.Name == "" || moe.AttentionQNorm == nil || moe.AttentionKNorm == nil ||
		moe.FeedForwardGateUpExperts == nil || moe.FeedForwardGateExperts != nil ||
		moe.FeedForwardUpExperts != nil || moe.FeedForwardDownExperts == nil ||
		moe.FeedForwardExpertBias == nil || moe.FeedForwardSharedGate == nil ||
		moe.FeedForwardSharedUp == nil || moe.FeedForwardSharedDown == nil {
		t.Fatalf("unexpected HY-V3 catalog: %+v", weights)
	}
}

func TestReadWeightsHYV3MTPHeads(t *testing.T) {
	spec := Spec{CommonSpec: CommonSpec{Architecture: "hy_v3", BlockCount: 1, NextNPredictLayers: 1,
		EmbeddingLength: 8, FeedForwardLength: 12, VocabularySize: 32}, AttentionSpec: AttentionSpec{HeadCount: 2, HeadCountKV: 1, KeyLength: 4, ValueLength: 4,
		RopeDimensionCount: 4},
	}
	tensors := []gguf.TensorInfo{
		tensorInfo("token_embd.weight", 8, 32), tensorInfo("output_norm.weight", 8),
		tensorInfo("output.weight", 8, 32),
	}
	for block := 0; block < 2; block++ {
		prefix := fmt.Sprintf("blk.%d.", block)
		tensors = append(tensors,
			tensorInfo(prefix+"attn_norm.weight", 8),
			tensorInfo(prefix+"attn_q.weight", 8, 8),
			tensorInfo(prefix+"attn_k.weight", 8, 4),
			tensorInfo(prefix+"attn_v.weight", 8, 4),
			tensorInfo(prefix+"attn_output.weight", 8, 8),
			tensorInfo(prefix+"attn_q_norm.weight", 4),
			tensorInfo(prefix+"attn_k_norm.weight", 4),
			tensorInfo(prefix+"ffn_norm.weight", 8),
			tensorInfo(prefix+"ffn_gate.weight", 8, 12),
			tensorInfo(prefix+"ffn_up.weight", 8, 12),
			tensorInfo(prefix+"ffn_down.weight", 12, 8),
		)
	}
	tensors = append(tensors,
		tensorInfo("blk.1.nextn.eh_proj.weight", 16, 8),
		tensorInfo("blk.1.nextn.enorm.weight", 8),
		tensorInfo("blk.1.nextn.hnorm.weight", 8),
		tensorInfo("blk.1.nextn.shared_head_norm.weight", 8),
	)
	weights, err := ReadWeights(&gguf.File{Tensors: tensors}, spec)
	if err != nil {
		t.Fatal(err)
	}
	if len(weights.Layers) != 1 || len(weights.HYV3MTP) != 1 ||
		weights.HYV3MTP[0].Layer.AttentionQNorm == nil ||
		weights.HYV3MTP[0].OutputNorm == nil || weights.HYV3MTP[0].Output != nil {
		t.Fatalf("unexpected HY-V3 MTP catalog: %+v", weights.HYV3MTP)
	}
}

func TestReadWeightsDeepSeek2OCRDensePrefixAndFusedExperts(t *testing.T) {
	spec := Spec{CommonSpec: CommonSpec{Architecture: "deepseek2-ocr", BlockCount: 2, EmbeddingLength: 8,
		FeedForwardLength: 12,

		VocabularySize: 32, RMSNormEpsilon: 1e-6}, AttentionSpec: AttentionSpec{HeadCount: 2, HeadCountKV: 2,
		KeyLength: 4, ValueLength: 4, RopeDimensionCount: 4}, MoESpec: MoESpec{LeadingDenseBlocks: 1,
		ExpertCount: 4, ExpertUsedCount: 2, ExpertFeedForward: 6,
		SharedExpertCount: 2, SharedExpertFF: 12, ExpertGatingFunc: 1},
	}
	file := &gguf.File{Tensors: []gguf.TensorInfo{
		tensorInfo("token_embd.weight", 8, 32), tensorInfo("output_norm.weight", 8),
		tensorInfo("output.weight", 8, 32),
		tensorInfo("blk.0.attn_norm.weight", 8),
		tensorInfo("blk.0.attn_q.weight", 8, 8),
		tensorInfo("blk.0.attn_k.weight", 8, 8),
		tensorInfo("blk.0.attn_v.weight", 8, 8),
		tensorInfo("blk.0.attn_output.weight", 8, 8),
		tensorInfo("blk.0.ffn_norm.weight", 8),
		tensorInfo("blk.0.ffn_gate.weight", 8, 12),
		tensorInfo("blk.0.ffn_up.weight", 8, 12),
		tensorInfo("blk.0.ffn_down.weight", 12, 8),
		tensorInfo("blk.1.attn_norm.weight", 8),
		tensorInfo("blk.1.attn_q.weight", 8, 8),
		tensorInfo("blk.1.attn_k.weight", 8, 8),
		tensorInfo("blk.1.attn_v.weight", 8, 8),
		tensorInfo("blk.1.attn_output.weight", 8, 8),
		tensorInfo("blk.1.ffn_norm.weight", 8),
		tensorInfo("blk.1.ffn_gate_inp.weight", 8, 4),
		tensorInfo("blk.1.exp_probs_b.bias", 4),
		tensorInfo("blk.1.ffn_gate_up_exps.weight", 8, 12, 4),
		tensorInfo("blk.1.ffn_down_exps.weight", 6, 8, 4),
		tensorInfo("blk.1.ffn_gate_shexp.weight", 8, 12),
		tensorInfo("blk.1.ffn_up_shexp.weight", 8, 12),
		tensorInfo("blk.1.ffn_down_shexp.weight", 12, 8),
	}}
	weights, err := ReadWeights(file, spec)
	if err != nil {
		t.Fatal(err)
	}
	dense, moe := weights.Layers[0], weights.Layers[1]
	if dense.AttentionQ.Name == "" || dense.AttentionK.Name == "" || dense.AttentionV.Name == "" ||
		dense.FeedForwardGate.Name == "" || dense.FeedForwardRouter != nil ||
		moe.AttentionQ.Name == "" || moe.AttentionQNorm != nil || moe.AttentionKNorm != nil ||
		moe.FeedForwardGateUpExperts == nil || moe.FeedForwardGateExperts != nil ||
		moe.FeedForwardUpExperts != nil || moe.FeedForwardDownExperts == nil ||
		moe.FeedForwardExpertBias == nil || moe.FeedForwardSharedGate == nil ||
		moe.FeedForwardSharedUp == nil || moe.FeedForwardSharedDown == nil {
		t.Fatalf("unexpected DeepSeek2-OCR catalog: %+v", weights)
	}
}

func TestReadWeightsCommandR64UsesPerHeadQKNorms(t *testing.T) {
	spec := Spec{CommonSpec: CommonSpec{Architecture: "command-r", BlockCount: 64, EmbeddingLength: 8,
		FeedForwardLength: 16,
		VocabularySize:    32, LayerNormEpsilon: 1e-5}, AttentionSpec: AttentionSpec{HeadCount: 2, HeadCountKV: 1,
		KeyLength: 4, ValueLength: 4},
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
	spec := Spec{CommonSpec: CommonSpec{Architecture: "plamo", BlockCount: 1, EmbeddingLength: 8,
		FeedForwardLength: 16,
		VocabularySize:    32, RMSNormEpsilon: 1e-5}, AttentionSpec: AttentionSpec{HeadCount: 2, HeadCountKV: 1,
		KeyLength: 4, ValueLength: 4},
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

func TestReadWeightsPLaMo3PerLayerWidths(t *testing.T) {
	spec := Spec{CommonSpec: CommonSpec{Architecture: "plamo3", BlockCount: 2, EmbeddingLength: 8,
		FeedForwardLength: 12,
		VocabularySize:    32, RMSNormEpsilon: 1e-6}, AttentionSpec: AttentionSpec{HeadCount: 2, HeadCountKV: 1,
		KeyLength: 2, ValueLength: 2,
		LayerHeadCounts: []uint32{2, 4}, LayerKVHeadCounts: []uint32{1, 2}}, MoESpec: MoESpec{LayerFeedForward: []uint32{12, 16}},
	}
	tensors := []gguf.TensorInfo{
		tensorInfo("token_embd.weight", 8, 32), tensorInfo("output_norm.weight", 8),
	}
	for block := range uint32(2) {
		prefix := fmt.Sprintf("blk.%d.", block)
		heads := uint64(spec.LayerHeadCount(block))
		kvHeads := uint64(spec.LayerKVHeadCount(block))
		ff := uint64(spec.LayerFeedForwardLength(block))
		tensors = append(tensors,
			tensorInfo(prefix+"attn_norm.weight", 8),
			tensorInfo(prefix+"attn_qkv.weight", 8, 2*heads+4*kvHeads),
			tensorInfo(prefix+"attn_q_norm.weight", 2),
			tensorInfo(prefix+"attn_k_norm.weight", 2),
			tensorInfo(prefix+"attn_output.weight", 2*heads, 8),
			tensorInfo(prefix+"post_attention_norm.weight", 8),
			tensorInfo(prefix+"ffn_norm.weight", 8),
			tensorInfo(prefix+"ffn_up.weight", 8, 2*ff),
			tensorInfo(prefix+"ffn_down.weight", ff, 8),
			tensorInfo(prefix+"post_ffw_norm.weight", 8),
		)
	}
	weights, err := ReadWeights(&gguf.File{Tensors: tensors}, spec)
	if err != nil {
		t.Fatal(err)
	}
	for block, layer := range weights.Layers {
		ff := uint64(spec.LayerFeedForwardLength(uint32(block)))
		if layer.AttentionQKV == nil || layer.AttentionQNorm == nil || layer.AttentionKNorm == nil ||
			layer.AttentionPostNorm == nil || layer.FeedForwardNorm.Name == "" ||
			layer.FeedForwardGate.Name != "" || layer.FeedForwardUp.Shape[1] != 2*ff ||
			layer.FeedForwardPostNorm == nil {
			t.Fatalf("unexpected PLaMo 3 layer %d catalog: %+v", block, layer)
		}
	}
}

func TestReadWeightsStableLMOptionalNormLayouts(t *testing.T) {
	for _, sequential := range []bool{false, true} {
		spec := Spec{CommonSpec: CommonSpec{Architecture: "stablelm", BlockCount: 1, EmbeddingLength: 8,
			FeedForwardLength: 16,

			VocabularySize: 32, LayerNormEpsilon: 1e-5}, AttentionSpec: AttentionSpec{HeadCount: 2, HeadCountKV: 1,
			KeyLength: 4, ValueLength: 4, RopeDimensionCount: 2},
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
	spec := Spec{CommonSpec: CommonSpec{Architecture: "phi2", BlockCount: 1, EmbeddingLength: 8,
		FeedForwardLength: 16,

		VocabularySize: 32, LayerNormEpsilon: 1e-5}, AttentionSpec: AttentionSpec{HeadCount: 2, HeadCountKV: 2,
		KeyLength: 4, ValueLength: 4, RopeDimensionCount: 2},
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
	spec := Spec{CommonSpec: CommonSpec{Architecture: "phi3", BlockCount: 1, ContextLength: 128,
		EmbeddingLength: 8, FeedForwardLength: 16,

		VocabularySize: 32, RMSNormEpsilon: 1e-5}, AttentionSpec: AttentionSpec{OriginalContextLength: 32,
		HeadCount: 2, HeadCountKV: 2, KeyLength: 4, ValueLength: 4,
		RopeDimensionCount: 4, RopeScalingType: "longrope",
		RopeAttentionFactor: 1.1},
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

func TestReadWeightsApertus(t *testing.T) {
	spec := Spec{CommonSpec: CommonSpec{Architecture: "apertus", BlockCount: 1, ContextLength: 128,
		EmbeddingLength: 8, FeedForwardLength: 16,

		VocabularySize: 32, RMSNormEpsilon: 1e-5}, AttentionSpec: AttentionSpec{OriginalContextLength: 32,
		HeadCount: 2, HeadCountKV: 1, KeyLength: 4, ValueLength: 4,
		RopeDimensionCount: 4, RopeScalingType: "longrope",
		RopeAttentionFactor: 1.1},
	}
	file := &gguf.File{Tensors: []gguf.TensorInfo{
		tensorInfo("token_embd.weight", 8, 32), tensorInfo("output_norm.weight", 8),
		tensorInfo("output.weight", 8, 32), tensorInfo("blk.0.attn_norm.weight", 8),
		tensorInfo("blk.0.attn_qkv.weight", 8, 16), tensorInfo("blk.0.attn_qkv.bias", 16),
		tensorInfo("blk.0.attn_q_norm.weight", 4), tensorInfo("blk.0.attn_k_norm.weight", 4),
		tensorInfo("blk.0.attn_output.weight", 8, 8), tensorInfo("blk.0.attn_output.bias", 8),
		tensorInfo("blk.0.ffn_norm.weight", 8), tensorInfo("blk.0.ffn_up.weight", 8, 16),
		tensorInfo("blk.0.ffn_down.weight", 16, 8),
		tensorInfo("blk.0.rope_factors_long.weight", 2),
		tensorInfo("blk.0.rope_factors_short.weight", 2),
	}}
	weights, err := ReadWeights(file, spec)
	if err != nil {
		t.Fatal(err)
	}
	layer := weights.Layers[0]
	if weights.Output == nil || layer.AttentionQKV == nil || layer.AttentionQKVBias == nil ||
		layer.AttentionQNorm == nil || layer.AttentionKNorm == nil ||
		layer.AttentionOutputBias == nil || layer.FeedForwardGate.Name != "" ||
		layer.RopeFactors == nil || layer.RopeFactors.Name != "blk.0.rope_factors_long.weight" {
		t.Fatalf("unexpected Apertus weights: %+v", weights)
	}
}

func TestReadWeightsGPTNeoX(t *testing.T) {
	spec := Spec{CommonSpec: CommonSpec{Architecture: "gptneox", BlockCount: 1, EmbeddingLength: 8,
		FeedForwardLength: 16,

		VocabularySize: 32, LayerNormEpsilon: 1e-5}, AttentionSpec: AttentionSpec{HeadCount: 2, HeadCountKV: 2,
		KeyLength: 4, ValueLength: 4, RopeDimensionCount: 2},
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

func TestReadWeightsGLM4(t *testing.T) {
	spec := Spec{CommonSpec: CommonSpec{Architecture: "glm4", BlockCount: 1, EmbeddingLength: 8,
		FeedForwardLength: 12,

		VocabularySize: 32, RMSNormEpsilon: 1e-5}, AttentionSpec: AttentionSpec{HeadCount: 2, HeadCountKV: 1,
		KeyLength: 4, ValueLength: 4, RopeDimensionCount: 4},
	}
	file := &gguf.File{Tensors: []gguf.TensorInfo{
		tensorInfo("token_embd.weight", 8, 32), tensorInfo("output_norm.weight", 8),
		tensorInfo("blk.0.attn_norm.weight", 8),
		tensorInfo("blk.0.attn_qkv.weight", 8, 16), tensorInfo("blk.0.attn_qkv.bias", 16),
		tensorInfo("blk.0.attn_output.weight", 8, 8),
		tensorInfo("blk.0.post_attention_norm.weight", 8),
		tensorInfo("blk.0.ffn_norm.weight", 8), tensorInfo("blk.0.ffn_up.weight", 8, 24),
		tensorInfo("blk.0.ffn_down.weight", 12, 8), tensorInfo("blk.0.post_ffw_norm.weight", 8),
		tensorInfo("blk.0.rope_freqs.weight", 2),
	}}
	weights, err := ReadWeights(file, spec)
	if err != nil {
		t.Fatal(err)
	}
	layer := weights.Layers[0]
	if weights.Output != nil || layer.AttentionQKV == nil || layer.AttentionQKVBias == nil ||
		layer.AttentionQ.Name != "" || layer.AttentionPostNorm == nil ||
		layer.FeedForwardPostNorm == nil || layer.FeedForwardGate.Name != "" ||
		layer.FeedForwardUp.Shape[1] != 24 || layer.RopeFactors == nil {
		t.Fatalf("unexpected GLM4 weights: %+v", weights)
	}
}

func TestReadWeightsEXAONE4(t *testing.T) {
	spec := Spec{CommonSpec: CommonSpec{Architecture: "exaone4", BlockCount: 1, EmbeddingLength: 8,
		FeedForwardLength: 12,

		VocabularySize: 32, RMSNormEpsilon: 1e-5}, AttentionSpec: AttentionSpec{HeadCount: 2, HeadCountKV: 1,
		KeyLength: 4, ValueLength: 4, RopeDimensionCount: 4},
	}
	file := &gguf.File{Tensors: []gguf.TensorInfo{
		tensorInfo("token_embd.weight", 8, 32), tensorInfo("output_norm.weight", 8),
		tensorInfo("blk.0.attn_qkv.weight", 8, 16), tensorInfo("blk.0.attn_qkv.bias", 16),
		tensorInfo("blk.0.attn_q_norm.weight", 4), tensorInfo("blk.0.attn_k_norm.weight", 4),
		tensorInfo("blk.0.attn_output.weight", 8, 8),
		tensorInfo("blk.0.post_attention_norm.weight", 8),
		tensorInfo("blk.0.ffn_gate.weight", 8, 12), tensorInfo("blk.0.ffn_up.weight", 8, 12),
		tensorInfo("blk.0.ffn_down.weight", 12, 8), tensorInfo("blk.0.post_ffw_norm.weight", 8),
		tensorInfo("blk.0.rope_freqs.weight", 2),
	}}
	weights, err := ReadWeights(file, spec)
	if err != nil {
		t.Fatal(err)
	}
	layer := weights.Layers[0]
	if weights.Output != nil || layer.AttentionNorm.Name != "" ||
		layer.FeedForwardNorm.Name != "" || layer.AttentionQKV == nil ||
		layer.AttentionQNorm == nil || layer.AttentionKNorm == nil ||
		layer.AttentionPostNorm == nil || layer.FeedForwardPostNorm == nil ||
		layer.FeedForwardGate.Name == "" || layer.RopeFactors == nil {
		t.Fatalf("unexpected EXAONE 4 weights: %+v", weights)
	}
}

func TestReadWeightsFalcon40B(t *testing.T) {
	spec := Spec{CommonSpec: CommonSpec{Architecture: "falcon", BlockCount: 1, EmbeddingLength: 8,
		FeedForwardLength: 16,

		VocabularySize: 32, LayerNormEpsilon: 1e-5}, AttentionSpec: AttentionSpec{HeadCount: 2, HeadCountKV: 1,
		KeyLength: 4, ValueLength: 4, RopeDimensionCount: 4},
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
	spec := Spec{CommonSpec: CommonSpec{Architecture: "bitnet", BlockCount: 1, EmbeddingLength: 8,
		FeedForwardLength: 16,
		VocabularySize:    32, RMSNormEpsilon: 1e-5}, AttentionSpec: AttentionSpec{HeadCount: 2, HeadCountKV: 1,
		KeyLength: 4, ValueLength: 4},
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
	spec := Spec{CommonSpec: CommonSpec{Architecture: "gemma2",
		BlockCount:        1,
		EmbeddingLength:   8,
		FeedForwardLength: 16,

		VocabularySize: 32}, AttentionSpec: AttentionSpec{HeadCount: 2,
		HeadCountKV: 1,
		KeyLength:   4,
		ValueLength: 4},
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
	spec := Spec{CommonSpec: CommonSpec{Architecture: "gemma",
		BlockCount:        1,
		EmbeddingLength:   8,
		FeedForwardLength: 16,

		VocabularySize: 32}, AttentionSpec: AttentionSpec{HeadCount: 2,
		HeadCountKV: 1,
		KeyLength:   4,
		ValueLength: 4},
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
	spec := Spec{CommonSpec: CommonSpec{Architecture: "llama",
		BlockCount:        1,
		EmbeddingLength:   8,
		FeedForwardLength: 16,

		VocabularySize: 32}, AttentionSpec: AttentionSpec{HeadCount: 2,
		HeadCountKV: 1,
		KeyLength:   4,
		ValueLength: 4},
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
	spec := Spec{CommonSpec: CommonSpec{EmbeddingLength: 8, VocabularySize: 32}}
	file := &gguf.File{Tensors: []gguf.TensorInfo{
		tensorInfo("token_embd.weight", 7, 32),
	}}
	_, err := ReadWeights(file, spec)
	if err == nil || !strings.Contains(err.Error(), "dimension 0") {
		t.Fatalf("error = %v", err)
	}
}

func TestReadWeightsQwen35Hybrid(t *testing.T) {
	spec := Spec{CommonSpec: CommonSpec{Architecture: "qwen35",
		BlockCount:        4,
		EmbeddingLength:   8,
		FeedForwardLength: 16,

		VocabularySize: 32}, AttentionSpec: AttentionSpec{HeadCount: 2,
		HeadCountKV: 1,
		KeyLength:   4,
		ValueLength: 4}, RecurrentSpec: RecurrentSpec{SSMConvKernel: 3,
		SSMInnerSize:          8,
		SSMStateSize:          2,
		SSMTimeStepRank:       4,
		SSMGroupCount:         2,
		FullAttentionInterval: 4},
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

func TestReadWeightsQwen35MTP(t *testing.T) {
	spec := Spec{CommonSpec: CommonSpec{Architecture: "qwen35", BlockCount: 1, NextNPredictLayers: 1,
		EmbeddingLength: 8, FeedForwardLength: 16,
		VocabularySize: 32}, AttentionSpec: AttentionSpec{HeadCount: 2, HeadCountKV: 1,
		KeyLength: 4, ValueLength: 4}, RecurrentSpec: RecurrentSpec{SSMConvKernel: 3, SSMInnerSize: 8, SSMStateSize: 2,
		SSMTimeStepRank: 4, SSMGroupCount: 2, FullAttentionInterval: 1},
	}
	tensors := []gguf.TensorInfo{
		tensorInfo("token_embd.weight", 8, 32), tensorInfo("output_norm.weight", 8),
	}
	for block := range uint32(2) {
		prefix := fmt.Sprintf("blk.%d.", block)
		tensors = append(tensors,
			tensorInfo(prefix+"attn_norm.weight", 8),
			tensorInfo(prefix+"post_attention_norm.weight", 8),
			tensorInfo(prefix+"attn_q.weight", 8, 16),
			tensorInfo(prefix+"attn_k.weight", 8, 4),
			tensorInfo(prefix+"attn_v.weight", 8, 4),
			tensorInfo(prefix+"attn_output.weight", 8, 8),
			tensorInfo(prefix+"attn_q_norm.weight", 4),
			tensorInfo(prefix+"attn_k_norm.weight", 4),
			tensorInfo(prefix+"ffn_gate.weight", 8, 16),
			tensorInfo(prefix+"ffn_up.weight", 8, 16),
			tensorInfo(prefix+"ffn_down.weight", 16, 8),
		)
	}
	tensors = append(tensors,
		tensorInfo("blk.1.nextn.eh_proj.weight", 16, 8),
		tensorInfo("blk.1.nextn.enorm.weight", 8),
		tensorInfo("blk.1.nextn.hnorm.weight", 8),
		tensorInfo("blk.1.nextn.embed_tokens.weight", 8, 32),
		tensorInfo("blk.1.nextn.shared_head_norm.weight", 8),
		tensorInfo("blk.1.nextn.shared_head_head.weight", 8, 32),
	)
	weights, err := ReadWeights(&gguf.File{Tensors: tensors}, spec)
	if err != nil {
		t.Fatal(err)
	}
	mtp := weights.Qwen35MTP
	if len(weights.Layers) != 1 || mtp == nil || mtp.Layer.Recurrent ||
		mtp.Layer.AttentionQNorm == nil || mtp.EHProjection.Name == "" ||
		mtp.TokenEmbedding == nil || mtp.OutputNorm == nil || mtp.Output == nil {
		t.Fatalf("unexpected Qwen3.5 MTP catalog: %+v", mtp)
	}
	sidecarTensors := make([]gguf.TensorInfo, 0, len(tensors))
	for _, item := range tensors {
		if !strings.HasPrefix(item.Name, "blk.0.") {
			sidecarTensors = append(sidecarTensors, item)
		}
	}
	sidecar, err := ReadWeights(&gguf.File{Tensors: sidecarTensors}, spec)
	if err != nil {
		t.Fatal(err)
	}
	if len(sidecar.Layers) != 0 || sidecar.Qwen35MTP == nil || !sidecar.Qwen35MTP.MTPOnly {
		t.Fatalf("unexpected Qwen3.5 MTP-only catalog: %+v", sidecar)
	}
}

func TestReadWeightsKimiLinearHybrid(t *testing.T) {
	spec := Spec{CommonSpec: CommonSpec{Architecture: "kimi-linear", BlockCount: 2, EmbeddingLength: 8,
		FeedForwardLength: 12,
		VocabularySize:    32}, AttentionSpec: AttentionSpec{HeadCount: 2, HeadCountKV: 1, KeyLength: 4,
		ValueLength: 2, KVLoRARank: 3, RopeDimensionCount: 2,

		LayerKVHeadCounts: []uint32{0, 1}}, MoESpec: MoESpec{ExpertCount: 4, ExpertUsedCount: 2, ExpertFeedForward: 6,
		SharedExpertFF: 6, LeadingDenseBlocks: 1}, RecurrentSpec: RecurrentSpec{KDAHeadDim: 2, SSMConvKernel: 3, SSMInnerSize: 4,

		RecurrentLayers: []bool{true, false}},
	}
	tensors := []gguf.TensorInfo{
		tensorInfo("token_embd.weight", 8, 32), tensorInfo("output_norm.weight", 8),
		tensorInfo("output.weight", 8, 32),
		tensorInfo("blk.0.attn_norm.weight", 8), tensorInfo("blk.0.attn_q.weight", 8, 4),
		tensorInfo("blk.0.attn_k.weight", 8, 4), tensorInfo("blk.0.attn_v.weight", 8, 4),
		tensorInfo("blk.0.attn_output.weight", 4, 8),
		tensorInfo("blk.0.ssm_conv1d_q.weight", 3, 1, 4, 1),
		tensorInfo("blk.0.ssm_conv1d_k.weight", 3, 1, 4),
		tensorInfo("blk.0.ssm_conv1d_v.weight", 3, 1, 4, 1),
		tensorInfo("blk.0.ssm_f_a.weight", 8, 2), tensorInfo("blk.0.ssm_f_b.weight", 2, 4),
		tensorInfo("blk.0.ssm_beta.weight", 8, 2), tensorInfo("blk.0.ssm_a", 1, 2, 1, 1),
		tensorInfo("blk.0.ssm_dt.bias", 4), tensorInfo("blk.0.ssm_g_a.weight", 8, 2),
		tensorInfo("blk.0.ssm_g_b.weight", 2, 4), tensorInfo("blk.0.ssm_norm.weight", 2),
		tensorInfo("blk.0.ffn_norm.weight", 8), tensorInfo("blk.0.ffn_gate.weight", 8, 12),
		tensorInfo("blk.0.ffn_up.weight", 8, 12), tensorInfo("blk.0.ffn_down.weight", 12, 8),
		tensorInfo("blk.1.attn_norm.weight", 8), tensorInfo("blk.1.attn_q.weight", 8, 8),
		tensorInfo("blk.1.attn_kv_a_mqa.weight", 8, 5), tensorInfo("blk.1.attn_kv_a_norm.weight", 3),
		tensorInfo("blk.1.attn_k_b.weight", 2, 3, 2), tensorInfo("blk.1.attn_v_b.weight", 3, 2, 2),
		tensorInfo("blk.1.attn_output.weight", 4, 8), tensorInfo("blk.1.ffn_norm.weight", 8),
		tensorInfo("blk.1.ffn_gate_inp.weight", 8, 4), tensorInfo("blk.1.ffn_gate_exps.weight", 8, 6, 4),
		tensorInfo("blk.1.ffn_up_exps.weight", 8, 6, 4), tensorInfo("blk.1.ffn_down_exps.weight", 6, 8, 4),
		tensorInfo("blk.1.exp_probs_b.bias", 4), tensorInfo("blk.1.ffn_gate_shexp.weight", 8, 6),
		tensorInfo("blk.1.ffn_up_shexp.weight", 8, 6), tensorInfo("blk.1.ffn_down_shexp.weight", 6, 8),
	}
	weights, err := ReadWeights(&gguf.File{Tensors: tensors}, spec)
	if err != nil {
		t.Fatal(err)
	}
	if !weights.Layers[0].Recurrent || weights.Layers[1].Recurrent ||
		weights.Layers[0].SSMQueryConv == nil || weights.Layers[0].SSMOutputGateB == nil ||
		weights.Layers[1].AttentionKB == nil || weights.Layers[1].FeedForwardExpertBias == nil {
		t.Fatalf("unexpected Kimi Linear catalog: %+v", weights.Layers)
	}
}

func TestReadWeightsRWKV6Qwen2(t *testing.T) {
	spec := Spec{CommonSpec: CommonSpec{Architecture: "rwkv6qwen2", BlockCount: 1, EmbeddingLength: 8,
		FeedForwardLength: 12,

		VocabularySize: 32}, AttentionSpec: AttentionSpec{HeadCount: 2, HeadCountKV: 1}, RecurrentSpec: RecurrentSpec{WKVHeadSize: 4, TimeMixExtraDim: 3, TimeDecayExtraDim: 2},
	}
	tensors := []gguf.TensorInfo{
		tensorInfo("token_embd.weight", 8, 32), tensorInfo("output_norm.weight", 8),
		tensorInfo("output_norm.bias", 8), tensorInfo("output.weight", 8, 32),
		tensorInfo("blk.0.attn_norm.weight", 8),
		tensorInfo("blk.0.time_mix_w1.weight", 8, 15),
		tensorInfo("blk.0.time_mix_w2.weight", 3, 8, 5),
		tensorInfo("blk.0.time_mix_lerp_x.weight", 8, 1, 1),
		tensorInfo("blk.0.time_mix_lerp_fused.weight", 8, 1, 1, 5),
		tensorInfo("blk.0.time_mix_decay.weight", 8),
		tensorInfo("blk.0.time_mix_decay_w1.weight", 8, 2),
		tensorInfo("blk.0.time_mix_decay_w2.weight", 2, 8),
		tensorInfo("blk.0.time_mix_key.weight", 8, 4),
		tensorInfo("blk.0.time_mix_value.weight", 8, 4),
		tensorInfo("blk.0.time_mix_receptance.weight", 8, 8),
		tensorInfo("blk.0.time_mix_gate.weight", 8, 8),
		tensorInfo("blk.0.time_mix_output.weight", 8, 8),
		tensorInfo("blk.0.time_mix_key.bias", 4),
		tensorInfo("blk.0.time_mix_value.bias", 4),
		tensorInfo("blk.0.time_mix_receptance.bias", 8),
		tensorInfo("blk.0.ffn_norm.weight", 8), tensorInfo("blk.0.ffn_gate.weight", 8, 12),
		tensorInfo("blk.0.ffn_up.weight", 8, 12), tensorInfo("blk.0.ffn_down.weight", 12, 8),
	}
	weights, err := ReadWeights(&gguf.File{Tensors: tensors}, spec)
	if err != nil {
		t.Fatal(err)
	}
	layer := weights.Layers[0]
	if !layer.Recurrent || layer.TimeMixW1 == nil || layer.TimeMixOutput == nil ||
		layer.AttentionQBias == nil || layer.AttentionKBias == nil || layer.AttentionVBias == nil ||
		weights.OutputNormBias == nil || weights.Output == nil {
		t.Fatalf("unexpected RWKV6-Qwen2 catalog: %+v", layer)
	}
}

func TestReadWeightsRWKV6(t *testing.T) {
	spec := Spec{CommonSpec: CommonSpec{Architecture: "rwkv6", BlockCount: 1, EmbeddingLength: 8,
		FeedForwardLength: 12,

		VocabularySize: 32}, AttentionSpec: AttentionSpec{HeadCount: 2, HeadCountKV: 2}, RecurrentSpec: RecurrentSpec{WKVHeadSize: 4, TimeMixExtraDim: 3, TimeDecayExtraDim: 2},
	}
	tensors := []gguf.TensorInfo{
		tensorInfo("token_embd.weight", 8, 32), tensorInfo("token_embd_norm.weight", 8),
		tensorInfo("token_embd_norm.bias", 8), tensorInfo("output_norm.weight", 8),
		tensorInfo("output_norm.bias", 8), tensorInfo("output.weight", 8, 32),
		tensorInfo("blk.0.attn_norm.weight", 8), tensorInfo("blk.0.attn_norm.bias", 8),
		tensorInfo("blk.0.attn_norm_2.weight", 8), tensorInfo("blk.0.attn_norm_2.bias", 8),
		tensorInfo("blk.0.time_mix_w1.weight", 8, 15), tensorInfo("blk.0.time_mix_w2.weight", 3, 8, 5),
		tensorInfo("blk.0.time_mix_lerp_x.weight", 8, 1, 1), tensorInfo("blk.0.time_mix_lerp_fused.weight", 8, 1, 1, 5),
		tensorInfo("blk.0.time_mix_first.weight", 4, 2), tensorInfo("blk.0.time_mix_decay.weight", 8),
		tensorInfo("blk.0.time_mix_decay_w1.weight", 8, 2), tensorInfo("blk.0.time_mix_decay_w2.weight", 2, 8),
		tensorInfo("blk.0.time_mix_key.weight", 8, 8), tensorInfo("blk.0.time_mix_value.weight", 8, 8),
		tensorInfo("blk.0.time_mix_receptance.weight", 8, 8), tensorInfo("blk.0.time_mix_gate.weight", 8, 8),
		tensorInfo("blk.0.time_mix_ln.weight", 8), tensorInfo("blk.0.time_mix_ln.bias", 8),
		tensorInfo("blk.0.time_mix_output.weight", 8, 8), tensorInfo("blk.0.channel_mix_lerp_k.weight", 8, 1, 1),
		tensorInfo("blk.0.channel_mix_lerp_r.weight", 8, 1, 1), tensorInfo("blk.0.channel_mix_key.weight", 8, 12),
		tensorInfo("blk.0.channel_mix_value.weight", 12, 8), tensorInfo("blk.0.channel_mix_receptance.weight", 8, 8),
	}
	weights, err := ReadWeights(&gguf.File{Tensors: tensors}, spec)
	if err != nil {
		t.Fatal(err)
	}
	layer := weights.Layers[0]
	if !layer.Recurrent || layer.TimeMixFirst == nil || layer.TimeMixLN == nil ||
		layer.ChannelMixKey == nil || layer.AttentionNorm2Bias == nil ||
		weights.TokenEmbeddingNorm == nil || weights.TokenEmbeddingNormBias == nil ||
		weights.OutputNormBias == nil || weights.Output == nil {
		t.Fatalf("unexpected RWKV6 catalog: %+v", layer)
	}
	legacy := make([]gguf.TensorInfo, 0, len(tensors)+4)
	for _, item := range tensors {
		if item.Name != "blk.0.time_mix_lerp_fused.weight" {
			legacy = append(legacy, item)
		}
	}
	for _, suffix := range []string{"w", "k", "v", "r", "g"} {
		legacy = append(legacy, tensorInfo("blk.0.time_mix_lerp_"+suffix+".weight", 8, 1, 1))
	}
	legacyWeights, err := ReadWeights(&gguf.File{Tensors: legacy}, spec)
	if err != nil {
		t.Fatal(err)
	}
	if legacyWeights.Layers[0].TimeMixLerpFused != nil || legacyWeights.Layers[0].TimeMixLerpW == nil {
		t.Fatalf("unexpected legacy RWKV6 lerp catalog: %+v", legacyWeights.Layers[0])
	}
}
