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

func TestReadWeightsQwen3MoE(t *testing.T) {
	spec := Spec{
		Architecture:       "qwen3moe",
		BlockCount:         1,
		EmbeddingLength:    8,
		FeedForwardLength:  24,
		ExpertCount:        4,
		ExpertUsedCount:    2,
		ExpertFeedForward:  12,
		ExpertWeightsScale: 1,
		HeadCount:          2,
		HeadCountKV:        1,
		KeyLength:          4,
		ValueLength:        4,
		VocabularySize:     32,
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
		tensorInfo("blk.0.ffn_gate_inp.weight", 8, 4),
		tensorInfo("blk.0.ffn_gate_exps.weight", 8, 12, 4),
		tensorInfo("blk.0.ffn_up_exps.weight", 8, 12, 4),
		tensorInfo("blk.0.ffn_down_exps.weight", 12, 8, 4),
	}}
	weights, err := ReadWeights(file, spec)
	if err != nil {
		t.Fatal(err)
	}
	layer := weights.Layers[0]
	if layer.FeedForwardRouter == nil || layer.FeedForwardGateExperts == nil ||
		layer.FeedForwardUpExperts == nil || layer.FeedForwardDownExperts == nil ||
		layer.FeedForwardUp.Name != "" || layer.FeedForwardDown.Name != "" {
		t.Fatalf("unexpected Qwen3-MoE layer catalog: %+v", layer)
	}
}

func TestReadWeightsLlama4InterleavedMoE(t *testing.T) {
	spec := Spec{
		Architecture: "llama4", BlockCount: 2, EmbeddingLength: 8, FeedForwardLength: 12,
		HeadCount: 2, HeadCountKV: 1, KeyLength: 4, ValueLength: 4, VocabularySize: 32,
		ExpertCount: 4, ExpertUsedCount: 2, ExpertFeedForward: 6, SharedExpertFF: 6,
		ExpertWeightsScale: 1, MoELayerStep: 2,
	}
	tensors := []gguf.TensorInfo{tensorInfo("token_embd.weight", 8, 32), tensorInfo("output_norm.weight", 8)}
	for block := 0; block < 2; block++ {
		prefix := fmt.Sprintf("blk.%d.", block)
		tensors = append(tensors,
			tensorInfo(prefix+"attn_norm.weight", 8),
			tensorInfo(prefix+"attn_q.weight", 8, 8),
			tensorInfo(prefix+"attn_k.weight", 8, 4),
			tensorInfo(prefix+"attn_v.weight", 8, 4),
			tensorInfo(prefix+"attn_output.weight", 8, 8),
			tensorInfo(prefix+"ffn_norm.weight", 8),
		)
		if block == 0 {
			tensors = append(tensors,
				tensorInfo(prefix+"ffn_gate.weight", 8, 12),
				tensorInfo(prefix+"ffn_up.weight", 8, 12),
				tensorInfo(prefix+"ffn_down.weight", 12, 8),
			)
		} else {
			tensors = append(tensors,
				tensorInfo(prefix+"ffn_gate_inp.weight", 8, 4),
				tensorInfo(prefix+"ffn_gate_exps.weight", 8, 6, 4),
				tensorInfo(prefix+"ffn_up_exps.weight", 8, 6, 4),
				tensorInfo(prefix+"ffn_down_exps.weight", 6, 8, 4),
				tensorInfo(prefix+"ffn_gate_shexp.weight", 8, 6),
				tensorInfo(prefix+"ffn_up_shexp.weight", 8, 6),
				tensorInfo(prefix+"ffn_down_shexp.weight", 6, 8),
			)
		}
	}
	weights, err := ReadWeights(&gguf.File{Tensors: tensors}, spec)
	if err != nil {
		t.Fatal(err)
	}
	if weights.Layers[0].FeedForwardGate.Name == "" || weights.Layers[0].FeedForwardRouter != nil ||
		weights.Layers[1].FeedForwardRouter == nil || weights.Layers[1].FeedForwardSharedGate == nil ||
		weights.Layers[1].FeedForwardSharedUp == nil || weights.Layers[1].FeedForwardSharedDown == nil {
		t.Fatalf("unexpected Llama 4 catalog: %+v", weights.Layers)
	}
}

func TestReadWeightsGPTOSSBiasedExperts(t *testing.T) {
	spec := Spec{
		Architecture: "gpt-oss", BlockCount: 1, EmbeddingLength: 8, FeedForwardLength: 8,
		HeadCount: 2, HeadCountKV: 1, KeyLength: 4, ValueLength: 4, VocabularySize: 32,
		ExpertCount: 4, ExpertUsedCount: 2, ExpertFeedForward: 6, ExpertWeightsScale: 1,
	}
	prefix := "blk.0."
	file := &gguf.File{Tensors: []gguf.TensorInfo{
		tensorInfo("token_embd.weight", 8, 32), tensorInfo("output_norm.weight", 8), tensorInfo("output.weight", 8, 32),
		tensorInfo(prefix+"attn_norm.weight", 8), tensorInfo(prefix+"post_attention_norm.weight", 8),
		tensorInfo(prefix+"attn_q.weight", 8, 8), tensorInfo(prefix+"attn_k.weight", 8, 4),
		tensorInfo(prefix+"attn_v.weight", 8, 4), tensorInfo(prefix+"attn_output.weight", 8, 8),
		tensorInfo(prefix+"attn_output.bias", 8), tensorInfo(prefix+"attn_sinks.weight", 2),
		tensorInfo(prefix+"ffn_gate_inp.weight", 8, 4), tensorInfo(prefix+"ffn_gate_inp.bias", 4),
		tensorInfo(prefix+"ffn_gate_exps.weight", 8, 6, 4), tensorInfo(prefix+"ffn_gate_exps.bias", 6, 4),
		tensorInfo(prefix+"ffn_up_exps.weight", 8, 6, 4), tensorInfo(prefix+"ffn_up_exps.bias", 6, 4),
		tensorInfo(prefix+"ffn_down_exps.weight", 6, 8, 4), tensorInfo(prefix+"ffn_down_exps.bias", 8, 4),
	}}
	weights, err := ReadWeights(file, spec)
	if err != nil {
		t.Fatal(err)
	}
	layer := weights.Layers[0]
	if weights.Output == nil || layer.AttentionPostNorm == nil || layer.AttentionSinks == nil ||
		layer.AttentionOutputBias == nil || layer.FeedForwardRouterBias == nil ||
		layer.FeedForwardGateBias == nil || layer.FeedForwardUpBias == nil || layer.FeedForwardDownBias == nil {
		t.Fatalf("unexpected GPT-OSS catalog: %+v", layer)
	}
}

func TestReadWeightsGroveMoE(t *testing.T) {
	spec := Spec{
		Architecture: "grovemoe", BlockCount: 1, EmbeddingLength: 8,
		FeedForwardLength: 24, ExpertCount: 4, ExpertUsedCount: 2,
		ExpertFeedForward: 6, ExpertChunkFeedForward: 3, ExpertsPerGroup: 2,
		ExpertWeightsScale: 1, HeadCount: 2, HeadCountKV: 1,
		KeyLength: 4, ValueLength: 4, VocabularySize: 32,
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
		tensorInfo("blk.0.ffn_gate_inp.weight", 8, 4),
		tensorInfo("blk.0.ffn_gate_exps.weight", 8, 6, 4),
		tensorInfo("blk.0.ffn_up_exps.weight", 8, 6, 4),
		tensorInfo("blk.0.ffn_down_exps.weight", 6, 8, 4),
		tensorInfo("blk.0.ffn_gate_chexps.weight", 8, 3, 2),
		tensorInfo("blk.0.ffn_up_chexps.weight", 8, 3, 2),
		tensorInfo("blk.0.ffn_down_chexps.weight", 3, 8, 2),
	}}
	weights, err := ReadWeights(file, spec)
	if err != nil {
		t.Fatal(err)
	}
	layer := weights.Layers[0]
	if layer.FeedForwardGateChunkExperts == nil || layer.FeedForwardUpChunkExperts == nil ||
		layer.FeedForwardDownChunkExperts == nil || layer.AttentionQNorm == nil ||
		layer.AttentionKNorm == nil {
		t.Fatalf("unexpected GroveMoE catalog: %+v", layer)
	}
}

func TestReadWeightsGLM4MoE(t *testing.T) {
	spec := Spec{
		Architecture: "glm4moe", BlockCount: 2, LeadingDenseBlocks: 1,
		EmbeddingLength: 8, FeedForwardLength: 12, ExpertCount: 4,
		ExpertUsedCount: 2, ExpertFeedForward: 6, SharedExpertCount: 2,
		SharedExpertFF: 12, ExpertWeightsScale: 1, HeadCount: 2,
		HeadCountKV: 1, KeyLength: 4, ValueLength: 4, VocabularySize: 32,
	}
	tensors := []gguf.TensorInfo{
		tensorInfo("token_embd.weight", 8, 32), tensorInfo("output_norm.weight", 8),
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
			tensorInfo(prefix+"attn_post_norm.weight", 8),
		)
	}
	tensors = append(tensors,
		tensorInfo("blk.0.ffn_gate.weight", 8, 12),
		tensorInfo("blk.0.ffn_up.weight", 8, 12),
		tensorInfo("blk.0.ffn_down.weight", 12, 8),
		tensorInfo("blk.1.ffn_gate_inp.weight", 8, 4),
		tensorInfo("blk.1.exp_probs_b.bias", 4),
		tensorInfo("blk.1.ffn_gate_exps.weight", 8, 6, 4),
		tensorInfo("blk.1.ffn_up_exps.weight", 8, 6, 4),
		tensorInfo("blk.1.ffn_down_exps.weight", 6, 8, 4),
		tensorInfo("blk.1.ffn_gate_shexp.weight", 8, 12),
		tensorInfo("blk.1.ffn_up_shexp.weight", 8, 12),
		tensorInfo("blk.1.ffn_down_shexp.weight", 12, 8),
	)
	weights, err := ReadWeights(&gguf.File{Tensors: tensors}, spec)
	if err != nil {
		t.Fatal(err)
	}
	dense, moe := weights.Layers[0], weights.Layers[1]
	if dense.FeedForwardGate.Name == "" || dense.FeedForwardRouter != nil ||
		moe.FeedForwardRouter == nil || moe.FeedForwardExpertBias == nil ||
		moe.FeedForwardSharedGate == nil || moe.FeedForwardNorm.Name != "blk.1.attn_post_norm.weight" ||
		moe.AttentionQNorm == nil || moe.AttentionKNorm == nil {
		t.Fatalf("unexpected GLM4-MoE catalog: dense=%+v moe=%+v", dense, moe)
	}
}

func TestReadWeightsMiMo2MixedDenseAndMoE(t *testing.T) {
	spec := Spec{
		Architecture: "mimo2", BlockCount: 2, EmbeddingLength: 8,
		FeedForwardLength: 12, ExpertCount: 4, ExpertUsedCount: 2,
		ExpertFeedForward: 6, ExpertWeightsScale: 1, HeadCount: 2,
		HeadCountKV: 1, LayerKVHeadCounts: []uint32{1, 1},
		KeyLength: 4, ValueLength: 3, VocabularySize: 32,
	}
	tensors := []gguf.TensorInfo{
		tensorInfo("token_embd.weight", 8, 32),
		tensorInfo("output_norm.weight", 8),
		tensorInfo("output.weight", 8, 32),
	}
	for block := 0; block < 2; block++ {
		prefix := fmt.Sprintf("blk.%d.", block)
		tensors = append(tensors,
			tensorInfo(prefix+"attn_norm.weight", 8),
			tensorInfo(prefix+"attn_qkv.weight", 8, 15),
			tensorInfo(prefix+"attn_output.weight", 6, 8),
			tensorInfo(prefix+"ffn_norm.weight", 8),
		)
	}
	tensors = append(tensors,
		tensorInfo("blk.0.attn_sinks.weight", 2),
		tensorInfo("blk.0.ffn_gate.weight", 8, 12),
		tensorInfo("blk.0.ffn_up.weight", 8, 12),
		tensorInfo("blk.0.ffn_down.weight", 12, 8),
		tensorInfo("blk.1.ffn_gate_inp.weight", 8, 4),
		tensorInfo("blk.1.ffn_gate_exps.weight", 8, 6, 4),
		tensorInfo("blk.1.ffn_up_exps.weight", 8, 6, 4),
		tensorInfo("blk.1.ffn_down_exps.weight", 6, 8, 4),
		tensorInfo("blk.1.exp_probs_b.bias", 4),
	)
	weights, err := ReadWeights(&gguf.File{Tensors: tensors}, spec)
	if err != nil {
		t.Fatal(err)
	}
	dense, moe := weights.Layers[0], weights.Layers[1]
	if weights.Output == nil || dense.AttentionQKV == nil || dense.AttentionSinks == nil ||
		dense.FeedForwardGate.Name == "" || dense.FeedForwardRouter != nil ||
		moe.AttentionQKV == nil || moe.AttentionSinks != nil || moe.FeedForwardRouter == nil ||
		moe.FeedForwardExpertBias == nil || moe.FeedForwardGateExperts == nil ||
		moe.FeedForwardUp.Name != "" {
		t.Fatalf("unexpected MiMo2 catalog: dense=%+v moe=%+v", dense, moe)
	}
	withoutBias := make([]gguf.TensorInfo, 0, len(tensors)-1)
	for _, item := range tensors {
		if item.Name != "blk.1.exp_probs_b.bias" {
			withoutBias = append(withoutBias, item)
		}
	}
	weights, err = ReadWeights(&gguf.File{Tensors: withoutBias}, spec)
	if err != nil {
		t.Fatal(err)
	}
	if weights.Layers[1].FeedForwardExpertBias != nil {
		t.Fatal("MiMo2 optional expert correction bias remained present")
	}
}

func TestReadWeightsStep35MixedDenseAndMoE(t *testing.T) {
	spec := Spec{
		Architecture: "step35", BlockCount: 2, EmbeddingLength: 8,
		FeedForwardLength: 12, ExpertCount: 4, ExpertUsedCount: 2,
		ExpertFeedForward: 6, SharedExpertFF: 8, ExpertWeightsScale: 1.25,
		HeadCount: 2, HeadCountKV: 1, LayerHeadCounts: []uint32{2, 4},
		LayerKVHeadCounts: []uint32{1, 2}, KeyLength: 4, ValueLength: 4,
		RopeDimensionCount: 4, VocabularySize: 32,
	}
	tensors := []gguf.TensorInfo{
		tensorInfo("token_embd.weight", 8, 32), tensorInfo("output_norm.weight", 8),
		tensorInfo("output.weight", 8, 32), tensorInfo("rope_freqs.weight", 2),
	}
	for block := 0; block < 2; block++ {
		prefix := fmt.Sprintf("blk.%d.", block)
		heads := spec.LayerHeadCount(uint32(block))
		kvHeads := spec.LayerKVHeadCount(uint32(block))
		tensors = append(tensors,
			tensorInfo(prefix+"attn_norm.weight", 8),
			tensorInfo(prefix+"attn_q.weight", 8, uint64(heads)*4),
			tensorInfo(prefix+"attn_k.weight", 8, uint64(kvHeads)*4),
			tensorInfo(prefix+"attn_v.weight", 8, uint64(kvHeads)*4),
			tensorInfo(prefix+"attn_output.weight", uint64(heads)*4, 8),
			tensorInfo(prefix+"ffn_norm.weight", 8),
		)
	}
	tensors = append(tensors,
		tensorInfo("blk.0.ffn_gate.weight", 8, 12),
		tensorInfo("blk.0.ffn_up.weight", 8, 12),
		tensorInfo("blk.0.ffn_down.weight", 12, 8),
		tensorInfo("blk.1.attn_q_norm.weight", 4),
		tensorInfo("blk.1.attn_k_norm.weight", 4),
		tensorInfo("blk.1.attn_gate.weight", 8, 4),
		tensorInfo("blk.1.ffn_gate_inp.weight", 8, 4),
		tensorInfo("blk.1.ffn_gate_exps.weight", 8, 6, 4),
		tensorInfo("blk.1.ffn_up_exps.weight", 8, 6, 4),
		tensorInfo("blk.1.ffn_down_exps.weight", 6, 8, 4),
		tensorInfo("blk.1.exp_probs_b.bias", 4),
		tensorInfo("blk.1.ffn_gate_shexp.weight", 8, 8),
		tensorInfo("blk.1.ffn_up_shexp.weight", 8, 8),
		tensorInfo("blk.1.ffn_down_shexp.weight", 8, 8),
	)
	weights, err := ReadWeights(&gguf.File{Tensors: tensors}, spec)
	if err != nil {
		t.Fatal(err)
	}
	dense, moe := weights.Layers[0], weights.Layers[1]
	if weights.Output == nil || dense.FeedForwardGate.Name == "" || dense.FeedForwardRouter != nil ||
		dense.RopeFactors == nil || moe.RopeFactors == nil ||
		moe.AttentionQNorm == nil || moe.AttentionKNorm == nil || moe.AttentionOutputGate == nil ||
		moe.FeedForwardRouter == nil || moe.FeedForwardExpertBias == nil ||
		moe.FeedForwardSharedGate == nil || moe.FeedForwardSharedUp == nil || moe.FeedForwardSharedDown == nil {
		t.Fatalf("unexpected Step3.5 catalog: dense=%+v moe=%+v", dense, moe)
	}
}

func TestReadWeightsDBRX(t *testing.T) {
	spec := Spec{
		Architecture: "dbrx", BlockCount: 1, EmbeddingLength: 8, FeedForwardLength: 6,
		ExpertCount: 4, ExpertUsedCount: 2, ExpertFeedForward: 6, ExpertWeightsScale: 1,
		HeadCount: 2, HeadCountKV: 1, KeyLength: 4, ValueLength: 4, VocabularySize: 32,
	}
	file := &gguf.File{Tensors: []gguf.TensorInfo{
		tensorInfo("token_embd.weight", 8, 32),
		tensorInfo("output_norm.weight", 8),
		tensorInfo("output.weight", 8, 32),
		tensorInfo("blk.0.attn_norm.weight", 8),
		tensorInfo("blk.0.attn_qkv.weight", 8, 16),
		tensorInfo("blk.0.attn_output.weight", 8, 8),
		tensorInfo("blk.0.attn_output_norm.weight", 8),
		tensorInfo("blk.0.ffn_gate_inp.weight", 8, 4),
		tensorInfo("blk.0.ffn_gate_exps.weight", 8, 6, 4),
		tensorInfo("blk.0.ffn_up_exps.weight", 8, 6, 4),
		tensorInfo("blk.0.ffn_down_exps.weight", 6, 8, 4),
	}}
	weights, err := ReadWeights(file, spec)
	if err != nil {
		t.Fatal(err)
	}
	layer := weights.Layers[0]
	if weights.Output == nil || weights.OutputNormBias != nil || layer.AttentionQKV == nil ||
		layer.AttentionNormBias != nil || layer.FeedForwardNorm.Name != "blk.0.attn_output_norm.weight" ||
		layer.FeedForwardNormBias != nil || layer.FeedForwardRouter == nil ||
		layer.FeedForwardGateExperts == nil || layer.FeedForwardUpExperts == nil ||
		layer.FeedForwardDownExperts == nil || layer.FeedForwardUp.Name != "" {
		t.Fatalf("unexpected DBRX catalog: %+v", weights)
	}
}

func TestReadWeightsArcticParallelDenseAndMoE(t *testing.T) {
	spec := Spec{
		Architecture: "arctic", BlockCount: 1, EmbeddingLength: 8,
		FeedForwardLength: 12, ExpertCount: 4, ExpertUsedCount: 2,
		ExpertFeedForward: 12, ExpertWeightsScale: 1,
		HeadCount: 2, HeadCountKV: 1, KeyLength: 4, ValueLength: 4, VocabularySize: 32,
	}
	file := &gguf.File{Tensors: []gguf.TensorInfo{
		tensorInfo("token_embd.weight", 8, 32), tensorInfo("output_norm.weight", 8),
		tensorInfo("blk.0.attn_norm.weight", 8),
		tensorInfo("blk.0.attn_q.weight", 8, 8), tensorInfo("blk.0.attn_k.weight", 8, 4),
		tensorInfo("blk.0.attn_v.weight", 8, 4), tensorInfo("blk.0.attn_output.weight", 8, 8),
		tensorInfo("blk.0.ffn_norm.weight", 8), tensorInfo("blk.0.ffn_norm_exps.weight", 8),
		tensorInfo("blk.0.ffn_gate.weight", 8, 8), tensorInfo("blk.0.ffn_up.weight", 8, 8),
		tensorInfo("blk.0.ffn_down.weight", 8, 8),
		tensorInfo("blk.0.ffn_gate_inp.weight", 8, 4),
		tensorInfo("blk.0.ffn_gate_exps.weight", 8, 12, 4),
		tensorInfo("blk.0.ffn_up_exps.weight", 8, 12, 4),
		tensorInfo("blk.0.ffn_down_exps.weight", 12, 8, 4),
	}}
	weights, err := ReadWeights(file, spec)
	if err != nil {
		t.Fatal(err)
	}
	layer := weights.Layers[0]
	if weights.Output != nil || layer.FeedForwardExpertNorm == nil ||
		layer.FeedForwardGate.Name == "" || layer.FeedForwardUp.Name == "" ||
		layer.FeedForwardDown.Name == "" || layer.FeedForwardRouter == nil ||
		layer.FeedForwardGateExperts == nil || layer.FeedForwardUpExperts == nil ||
		layer.FeedForwardDownExperts == nil {
		t.Fatalf("unexpected Arctic catalog: %+v", weights)
	}
}

func TestReadWeightsOpenELMPerLayerWidths(t *testing.T) {
	spec := Spec{
		Architecture: "openelm", BlockCount: 2, EmbeddingLength: 8,
		FeedForwardLength: 12, LayerFeedForward: []uint32{12, 16},
		HeadCount: 2, HeadCountKV: 1, LayerHeadCounts: []uint32{2, 4},
		LayerKVHeadCounts: []uint32{1, 2}, KeyLength: 4, ValueLength: 4,
		VocabularySize: 32,
	}
	tensors := []gguf.TensorInfo{
		tensorInfo("token_embd.weight", 8, 32), tensorInfo("output_norm.weight", 8),
	}
	for block, widths := range []struct{ qkv, output, ffn uint64 }{{16, 8, 12}, {32, 16, 16}} {
		prefix := fmt.Sprintf("blk.%d.", block)
		tensors = append(tensors,
			tensorInfo(prefix+"attn_norm.weight", 8),
			tensorInfo(prefix+"attn_qkv.weight", 8, widths.qkv),
			tensorInfo(prefix+"attn_q_norm.weight", 4),
			tensorInfo(prefix+"attn_k_norm.weight", 4),
			tensorInfo(prefix+"attn_output.weight", widths.output, 8),
			tensorInfo(prefix+"ffn_norm.weight", 8),
			tensorInfo(prefix+"ffn_gate.weight", 8, widths.ffn),
			tensorInfo(prefix+"ffn_up.weight", 8, widths.ffn),
			tensorInfo(prefix+"ffn_down.weight", widths.ffn, 8),
		)
	}
	weights, err := ReadWeights(&gguf.File{Tensors: tensors}, spec)
	if err != nil {
		t.Fatal(err)
	}
	if weights.Output != nil || weights.Layers[0].AttentionQKV == nil ||
		weights.Layers[1].AttentionQKV == nil || weights.Layers[1].AttentionQNorm == nil ||
		weights.Layers[1].AttentionKNorm == nil || weights.Layers[1].FeedForwardUp.Shape[1] != 16 ||
		weights.Layers[1].AttentionOutput.Shape[0] != 16 {
		t.Fatalf("unexpected OpenELM catalog: %+v", weights)
	}
}

func TestReadWeightsDeciSparseLayers(t *testing.T) {
	spec := Spec{
		Architecture: "deci", BlockCount: 4, EmbeddingLength: 8,
		FeedForwardLength: 12, LayerFeedForward: []uint32{12, 12, 12, 0},
		HeadCount: 2, HeadCountKV: 1, LayerHeadCounts: []uint32{2, 2, 0, 0},
		LayerKVHeadCounts: []uint32{1, 0, 0, 0}, KeyLength: 4, ValueLength: 4,
		VocabularySize: 32,
	}
	tensors := []gguf.TensorInfo{
		tensorInfo("token_embd.weight", 8, 32), tensorInfo("output_norm.weight", 8),
		tensorInfo("blk.0.attn_norm.weight", 8), tensorInfo("blk.0.attn_qkv.weight", 8, 16),
		tensorInfo("blk.0.attn_output.weight", 8, 8),
		tensorInfo("blk.1.attn_norm.weight", 8), tensorInfo("blk.1.attn_output.weight", 8, 8),
	}
	for block := 0; block < 3; block++ {
		prefix := fmt.Sprintf("blk.%d.", block)
		tensors = append(tensors,
			tensorInfo(prefix+"ffn_norm.weight", 8),
			tensorInfo(prefix+"ffn_gate.weight", 8, 12),
			tensorInfo(prefix+"ffn_up.weight", 8, 12),
			tensorInfo(prefix+"ffn_down.weight", 12, 8),
		)
	}
	weights, err := ReadWeights(&gguf.File{Tensors: tensors}, spec)
	if err != nil {
		t.Fatal(err)
	}
	if weights.Layers[0].AttentionQKV == nil || weights.Layers[0].AttentionOutput.Name == "" ||
		weights.Layers[1].AttentionQ.Name != "" || weights.Layers[1].AttentionOutput.Name == "" ||
		weights.Layers[2].AttentionNorm.Name != "" || weights.Layers[2].FeedForwardUp.Name == "" ||
		weights.Layers[3].AttentionNorm.Name != "" || weights.Layers[3].FeedForwardNorm.Name != "" ||
		weights.Layers[3].FeedForwardUp.Name != "" {
		t.Fatalf("unexpected Deci catalog: %+v", weights)
	}
}

func TestReadWeightsGrokOptionalDenseBranch(t *testing.T) {
	for _, gated := range []bool{false, true} {
		for _, dense := range []bool{false, true} {
			t.Run(fmt.Sprintf("gated=%t/dense=%t", gated, dense), func(t *testing.T) {
				spec := Spec{
					Architecture: "grok", BlockCount: 1, EmbeddingLength: 8,
					FeedForwardLength: 12, ExpertCount: 4, ExpertUsedCount: 2,
					ExpertFeedForward: 6, ExpertWeightsScale: 1, HeadCount: 2,
					HeadCountKV: 1, KeyLength: 4, ValueLength: 4, VocabularySize: 32,
				}
				tensors := []gguf.TensorInfo{
					tensorInfo("token_embd.weight", 8, 32), tensorInfo("output_norm.weight", 8),
					tensorInfo("blk.0.attn_norm.weight", 8), tensorInfo("blk.0.attn_qkv.weight", 8, 16),
					tensorInfo("blk.0.attn_output.weight", 8, 8), tensorInfo("blk.0.attn_output_norm.weight", 8),
					tensorInfo("blk.0.ffn_norm.weight", 8), tensorInfo("blk.0.ffn_gate_inp.weight", 8, 4),
					tensorInfo("blk.0.ffn_up_exps.weight", 8, 6, 4),
					tensorInfo("blk.0.ffn_down_exps.weight", 6, 8, 4),
					tensorInfo("blk.0.layer_output_norm.weight", 8),
				}
				if gated {
					tensors = append(tensors, tensorInfo("blk.0.ffn_gate_exps.weight", 8, 6, 4))
				}
				if dense {
					tensors = append(tensors,
						tensorInfo("blk.0.ffn_gate.weight", 8, 12),
						tensorInfo("blk.0.ffn_up.weight", 8, 12),
						tensorInfo("blk.0.ffn_down.weight", 12, 8),
					)
				}
				weights, err := ReadWeights(&gguf.File{Tensors: tensors}, spec)
				if err != nil {
					t.Fatal(err)
				}
				layer := weights.Layers[0]
				if layer.AttentionQKV == nil || layer.AttentionPostNorm == nil ||
					layer.FeedForwardPostNorm == nil || layer.FeedForwardRouter == nil ||
					(layer.FeedForwardGateExperts != nil) != gated || layer.FeedForwardUpExperts == nil ||
					layer.FeedForwardDownExperts == nil || (layer.FeedForwardUp.Name != "") != dense {
					t.Fatalf("unexpected Grok catalog: %+v", weights)
				}
			})
		}
	}
}

func TestReadWeightsGrokRejectsPartialDenseBranch(t *testing.T) {
	spec := Spec{
		Architecture: "grok", BlockCount: 1, EmbeddingLength: 8,
		FeedForwardLength: 12, ExpertCount: 4, ExpertUsedCount: 2,
		ExpertFeedForward: 6, HeadCount: 2, HeadCountKV: 1,
		KeyLength: 4, ValueLength: 4, VocabularySize: 32,
	}
	tensors := []gguf.TensorInfo{
		tensorInfo("token_embd.weight", 8, 32), tensorInfo("output_norm.weight", 8),
		tensorInfo("blk.0.attn_norm.weight", 8), tensorInfo("blk.0.attn_qkv.weight", 8, 16),
		tensorInfo("blk.0.attn_output.weight", 8, 8), tensorInfo("blk.0.attn_output_norm.weight", 8),
		tensorInfo("blk.0.ffn_norm.weight", 8), tensorInfo("blk.0.ffn_gate_inp.weight", 8, 4),
		tensorInfo("blk.0.ffn_up_exps.weight", 8, 6, 4),
		tensorInfo("blk.0.ffn_down_exps.weight", 6, 8, 4),
		tensorInfo("blk.0.ffn_post_norm.weight", 8),
		tensorInfo("blk.0.ffn_up.weight", 8, 12),
	}
	if _, err := ReadWeights(&gguf.File{Tensors: tensors}, spec); err == nil {
		t.Fatal("partial Grok dense branch accepted")
	}
}

func TestReadWeightsMellum(t *testing.T) {
	spec := Spec{
		Architecture: "mellum", BlockCount: 1, EmbeddingLength: 8,
		FeedForwardLength: 12, ExpertCount: 4, ExpertUsedCount: 2,
		ExpertFeedForward: 6, HeadCount: 2, HeadCountKV: 1,
		KeyLength: 4, ValueLength: 4, VocabularySize: 32,
	}
	tensors := []gguf.TensorInfo{
		tensorInfo("token_embd.weight", 8, 32), tensorInfo("output_norm.weight", 8),
		tensorInfo("output.weight", 8, 32), tensorInfo("blk.0.attn_norm.weight", 8),
		tensorInfo("blk.0.attn_q.weight", 8, 8), tensorInfo("blk.0.attn_k.weight", 8, 4),
		tensorInfo("blk.0.attn_v.weight", 8, 4), tensorInfo("blk.0.attn_output.weight", 8, 8),
		tensorInfo("blk.0.attn_q_norm.weight", 4), tensorInfo("blk.0.attn_k_norm.weight", 4),
		tensorInfo("blk.0.ffn_norm.weight", 8), tensorInfo("blk.0.ffn_gate_inp.weight", 8, 4),
		tensorInfo("blk.0.ffn_gate_exps.weight", 8, 6, 4),
		tensorInfo("blk.0.ffn_up_exps.weight", 8, 6, 4),
		tensorInfo("blk.0.ffn_down_exps.weight", 6, 8, 4),
	}
	weights, err := ReadWeights(&gguf.File{Tensors: tensors}, spec)
	if err != nil {
		t.Fatal(err)
	}
	layer := weights.Layers[0]
	if weights.Output == nil || layer.AttentionQNorm == nil || layer.AttentionKNorm == nil ||
		layer.FeedForwardRouter == nil || layer.FeedForwardGateExperts == nil ||
		layer.FeedForwardUpExperts == nil || layer.FeedForwardDownExperts == nil {
		t.Fatalf("unexpected Mellum catalog: %+v", weights)
	}
}

func TestReadWeightsQwen(t *testing.T) {
	spec := Spec{Architecture: "qwen", BlockCount: 1, EmbeddingLength: 8,
		FeedForwardLength: 12, HeadCount: 2, HeadCountKV: 2, KeyLength: 4,
		ValueLength: 4, VocabularySize: 32}
	tensors := []gguf.TensorInfo{
		tensorInfo("token_embd.weight", 8, 32), tensorInfo("output_norm.weight", 8),
		tensorInfo("output.weight", 8, 32), tensorInfo("blk.0.attn_norm.weight", 8),
		tensorInfo("blk.0.attn_qkv.weight", 8, 24), tensorInfo("blk.0.attn_qkv.bias", 24),
		tensorInfo("blk.0.attn_output.weight", 8, 8), tensorInfo("blk.0.ffn_norm.weight", 8),
		tensorInfo("blk.0.ffn_gate.weight", 8, 12), tensorInfo("blk.0.ffn_up.weight", 8, 12),
		tensorInfo("blk.0.ffn_down.weight", 12, 8),
	}
	weights, err := ReadWeights(&gguf.File{Tensors: tensors}, spec)
	if err != nil {
		t.Fatal(err)
	}
	if weights.Output == nil || weights.Layers[0].AttentionQKV == nil ||
		weights.Layers[0].AttentionQKVBias == nil {
		t.Fatalf("unexpected Qwen catalog: %+v", weights)
	}
}

func TestReadWeightsChatGLM(t *testing.T) {
	s := Spec{Architecture: "chatglm", BlockCount: 1, EmbeddingLength: 8, FeedForwardLength: 12, HeadCount: 2, HeadCountKV: 1, KeyLength: 4, ValueLength: 4, VocabularySize: 32}
	tensors := []gguf.TensorInfo{tensorInfo("token_embd.weight", 8, 32), tensorInfo("output_norm.weight", 8), tensorInfo("blk.0.attn_norm.weight", 8), tensorInfo("blk.0.attn_qkv.weight", 8, 16), tensorInfo("blk.0.attn_output.weight", 8, 8), tensorInfo("blk.0.ffn_norm.weight", 8), tensorInfo("blk.0.ffn_up.weight", 8, 24), tensorInfo("blk.0.ffn_down.weight", 12, 8)}
	w, err := ReadWeights(&gguf.File{Tensors: tensors}, s)
	if err != nil {
		t.Fatal(err)
	}
	if w.Layers[0].AttentionQKV == nil || w.Layers[0].FeedForwardUp.Name == "" || w.Layers[0].FeedForwardGate.Name != "" {
		t.Fatalf("unexpected ChatGLM catalog: %+v", w)
	}
}

func TestReadWeightsHunyuanDense(t *testing.T) {
	s := Spec{Architecture: "hunyuan-dense", BlockCount: 1, EmbeddingLength: 8, FeedForwardLength: 12, HeadCount: 2, HeadCountKV: 1, KeyLength: 4, ValueLength: 4, VocabularySize: 32}
	tensors := []gguf.TensorInfo{tensorInfo("token_embd.weight", 8, 32), tensorInfo("output_norm.weight", 8), tensorInfo("blk.0.attn_norm.weight", 8), tensorInfo("blk.0.attn_qkv.weight", 8, 16), tensorInfo("blk.0.attn_output.weight", 8, 8), tensorInfo("blk.0.attn_q_norm.weight", 4), tensorInfo("blk.0.attn_k_norm.weight", 4), tensorInfo("blk.0.ffn_norm.weight", 8), tensorInfo("blk.0.ffn_gate.weight", 8, 12), tensorInfo("blk.0.ffn_up.weight", 8, 12), tensorInfo("blk.0.ffn_down.weight", 12, 8)}
	w, err := ReadWeights(&gguf.File{Tensors: tensors}, s)
	if err != nil {
		t.Fatal(err)
	}
	if w.Layers[0].AttentionQKV == nil || w.Layers[0].AttentionQNorm == nil || w.Layers[0].AttentionKNorm == nil {
		t.Fatalf("unexpected Hunyuan-Dense catalog: %+v", w)
	}
}

func TestReadWeightsHunyuanVL(t *testing.T) {
	s := Spec{Architecture: "hunyuan_vl", BlockCount: 1, EmbeddingLength: 8, FeedForwardLength: 12, HeadCount: 2, HeadCountKV: 1, KeyLength: 4, ValueLength: 4, VocabularySize: 32}
	tensors := []gguf.TensorInfo{tensorInfo("token_embd.weight", 8, 32), tensorInfo("output_norm.weight", 8), tensorInfo("blk.0.attn_norm.weight", 8), tensorInfo("blk.0.attn_qkv.weight", 8, 16), tensorInfo("blk.0.attn_output.weight", 8, 8), tensorInfo("blk.0.attn_q_norm.weight", 4), tensorInfo("blk.0.attn_k_norm.weight", 4), tensorInfo("blk.0.ffn_norm.weight", 8), tensorInfo("blk.0.ffn_gate.weight", 8, 12), tensorInfo("blk.0.ffn_up.weight", 8, 12), tensorInfo("blk.0.ffn_down.weight", 12, 8)}
	w, err := ReadWeights(&gguf.File{Tensors: tensors}, s)
	if err != nil {
		t.Fatal(err)
	}
	if w.Layers[0].AttentionQKV == nil || w.Layers[0].AttentionQNorm == nil || w.Layers[0].AttentionKNorm == nil {
		t.Fatalf("unexpected Hunyuan-VL catalog: %+v", w)
	}
}

func TestReadWeightsCogVLMRequiresTextAndVisualExperts(t *testing.T) {
	s := Spec{Architecture: "cogvlm", BlockCount: 1, EmbeddingLength: 8, FeedForwardLength: 12, HeadCount: 2, HeadCountKV: 2, KeyLength: 4, ValueLength: 4, VocabularySize: 32}
	tensors := []gguf.TensorInfo{
		tensorInfo("token_embd.weight", 8, 32), tensorInfo("output_norm.weight", 8),
		tensorInfo("blk.0.attn_norm.weight", 8), tensorInfo("blk.0.attn_qkv.weight", 8, 24),
		tensorInfo("blk.0.attn_output.weight", 8, 8), tensorInfo("blk.0.ffn_norm.weight", 8),
		tensorInfo("blk.0.ffn_gate.weight", 8, 12), tensorInfo("blk.0.ffn_up.weight", 8, 12),
		tensorInfo("blk.0.ffn_down.weight", 12, 8), tensorInfo("blk.0.vis_attn_qkv.weight", 8, 24),
		tensorInfo("blk.0.vis_attn_output.weight", 8, 8), tensorInfo("blk.0.vis_gate.weight", 8, 12),
		tensorInfo("blk.0.vis_up.weight", 8, 12), tensorInfo("blk.0.vis_down.weight", 12, 8),
	}
	w, err := ReadWeights(&gguf.File{Tensors: tensors}, s)
	if err != nil {
		t.Fatal(err)
	}
	layer := w.Layers[0]
	if layer.AttentionQKV == nil || layer.VisualAttentionQKV == nil ||
		layer.VisualAttentionOutput == nil || layer.VisualFeedForwardGate == nil ||
		layer.VisualFeedForwardUp == nil || layer.VisualFeedForwardDown == nil {
		t.Fatalf("unexpected CogVLM catalog: %+v", layer)
	}
}

func TestReadWeightsBailingMoE(t *testing.T) {
	spec := Spec{
		Architecture: "bailingmoe", BlockCount: 1, EmbeddingLength: 8,
		FeedForwardLength: 16, ExpertCount: 4, ExpertUsedCount: 2,
		ExpertFeedForward: 6, SharedExpertCount: 2, SharedExpertFF: 12,
		ExpertWeightsScale: 1.25, HeadCount: 2, HeadCountKV: 1,
		KeyLength: 4, ValueLength: 4, VocabularySize: 32,
	}
	file := &gguf.File{Tensors: []gguf.TensorInfo{
		tensorInfo("token_embd.weight", 8, 32), tensorInfo("output_norm.weight", 8),
		tensorInfo("output.weight", 8, 32), tensorInfo("blk.0.attn_norm.weight", 8),
		tensorInfo("blk.0.attn_q.weight", 8, 8), tensorInfo("blk.0.attn_k.weight", 8, 4),
		tensorInfo("blk.0.attn_v.weight", 8, 4), tensorInfo("blk.0.attn_output.weight", 8, 8),
		tensorInfo("blk.0.ffn_norm.weight", 8), tensorInfo("blk.0.ffn_gate_inp.weight", 8, 4),
		tensorInfo("blk.0.ffn_gate_exps.weight", 8, 6, 4),
		tensorInfo("blk.0.ffn_up_exps.weight", 8, 6, 4),
		tensorInfo("blk.0.ffn_down_exps.weight", 6, 8, 4),
		tensorInfo("blk.0.ffn_gate_shexp.weight", 8, 12),
		tensorInfo("blk.0.ffn_up_shexp.weight", 8, 12),
		tensorInfo("blk.0.ffn_down_shexp.weight", 12, 8),
	}}
	weights, err := ReadWeights(file, spec)
	if err != nil {
		t.Fatal(err)
	}
	layer := weights.Layers[0]
	if weights.Output == nil || layer.FeedForwardRouter == nil ||
		layer.FeedForwardGateExperts == nil || layer.FeedForwardUpExperts == nil ||
		layer.FeedForwardDownExperts == nil || layer.FeedForwardSharedGate == nil ||
		layer.FeedForwardSharedUp == nil || layer.FeedForwardSharedDown == nil ||
		layer.FeedForwardExpertBias != nil {
		t.Fatalf("unexpected BailingMoE catalog: %+v", weights)
	}
}

func TestReadWeightsDeepSeekDenseThenMoEWithTiedOutput(t *testing.T) {
	spec := Spec{
		Architecture: "deepseek", BlockCount: 2, LeadingDenseBlocks: 1,
		EmbeddingLength: 8, FeedForwardLength: 16, ExpertCount: 4, ExpertUsedCount: 2,
		ExpertFeedForward: 6, SharedExpertCount: 2, SharedExpertFF: 12,
		ExpertWeightsScale: 1.3, HeadCount: 2, HeadCountKV: 1,
		KeyLength: 4, ValueLength: 4, VocabularySize: 32,
	}
	tensors := []gguf.TensorInfo{
		tensorInfo("token_embd.weight", 8, 32), tensorInfo("output_norm.weight", 8),
	}
	for block := 0; block < 2; block++ {
		prefix := fmt.Sprintf("blk.%d.", block)
		tensors = append(tensors,
			tensorInfo(prefix+"attn_norm.weight", 8),
			tensorInfo(prefix+"attn_q.weight", 8, 8),
			tensorInfo(prefix+"attn_k.weight", 8, 4),
			tensorInfo(prefix+"attn_v.weight", 8, 4),
			tensorInfo(prefix+"attn_output.weight", 8, 8),
			tensorInfo(prefix+"ffn_norm.weight", 8),
		)
	}
	tensors = append(tensors,
		tensorInfo("blk.0.ffn_gate.weight", 8, 16),
		tensorInfo("blk.0.ffn_up.weight", 8, 16),
		tensorInfo("blk.0.ffn_down.weight", 16, 8),
		tensorInfo("blk.1.ffn_gate_inp.weight", 8, 4),
		tensorInfo("blk.1.ffn_gate_exps.weight", 8, 6, 4),
		tensorInfo("blk.1.ffn_up_exps.weight", 8, 6, 4),
		tensorInfo("blk.1.ffn_down_exps.weight", 6, 8, 4),
		tensorInfo("blk.1.ffn_gate_shexp.weight", 8, 12),
		tensorInfo("blk.1.ffn_up_shexp.weight", 8, 12),
		tensorInfo("blk.1.ffn_down_shexp.weight", 12, 8),
	)
	weights, err := ReadWeights(&gguf.File{Tensors: tensors}, spec)
	if err != nil {
		t.Fatal(err)
	}
	dense, moe := weights.Layers[0], weights.Layers[1]
	if weights.Output != nil || dense.FeedForwardGate.Name == "" || dense.FeedForwardRouter != nil ||
		moe.FeedForwardRouter == nil || moe.FeedForwardGateExperts == nil ||
		moe.FeedForwardUpExperts == nil || moe.FeedForwardDownExperts == nil ||
		moe.FeedForwardSharedGate == nil || moe.FeedForwardSharedUp == nil ||
		moe.FeedForwardSharedDown == nil || moe.FeedForwardExpertBias != nil {
		t.Fatalf("unexpected DeepSeek catalog: %+v", weights)
	}
}

func TestReadWeightsGraniteMoEUngatedWithSharedExpert(t *testing.T) {
	spec := Spec{
		Architecture: "granitemoe", BlockCount: 1, EmbeddingLength: 8,
		FeedForwardLength: 6, ExpertCount: 4, ExpertUsedCount: 2,
		ExpertFeedForward: 6, SharedExpertFF: 5, ExpertWeightsScale: 1,
		HeadCount: 2, HeadCountKV: 1, KeyLength: 4, ValueLength: 4, VocabularySize: 32,
	}
	file := &gguf.File{Tensors: []gguf.TensorInfo{
		tensorInfo("token_embd.weight", 8, 32), tensorInfo("output_norm.weight", 8),
		tensorInfo("blk.0.attn_norm.weight", 8),
		tensorInfo("blk.0.attn_q.weight", 8, 8), tensorInfo("blk.0.attn_k.weight", 8, 4),
		tensorInfo("blk.0.attn_v.weight", 8, 4), tensorInfo("blk.0.attn_output.weight", 8, 8),
		tensorInfo("blk.0.attn_output.bias", 8), tensorInfo("blk.0.ffn_norm.weight", 8),
		tensorInfo("blk.0.ffn_gate_inp.weight", 8, 4),
		tensorInfo("blk.0.ffn_up_exps.weight", 8, 6, 4),
		tensorInfo("blk.0.ffn_down_exps.weight", 6, 8, 4),
		tensorInfo("blk.0.ffn_gate_shexp.weight", 8, 5),
		tensorInfo("blk.0.ffn_up_shexp.weight", 8, 5),
		tensorInfo("blk.0.ffn_down_shexp.weight", 5, 8),
	}}
	weights, err := ReadWeights(file, spec)
	if err != nil {
		t.Fatal(err)
	}
	layer := weights.Layers[0]
	if weights.Output != nil || layer.AttentionOutputBias == nil ||
		layer.FeedForwardRouter == nil || layer.FeedForwardGateExperts != nil ||
		layer.FeedForwardUpExperts == nil || layer.FeedForwardDownExperts == nil ||
		layer.FeedForwardSharedGate == nil || layer.FeedForwardSharedUp == nil ||
		layer.FeedForwardSharedDown == nil {
		t.Fatalf("unexpected GraniteMoE catalog: %+v", weights)
	}
}

func TestReadWeightsSmallThinkerFusedQKV(t *testing.T) {
	spec := Spec{
		Architecture: "smallthinker", BlockCount: 1, EmbeddingLength: 8,
		FeedForwardLength: 6, ExpertCount: 4, ExpertUsedCount: 2,
		ExpertFeedForward: 6, ExpertWeightsScale: 1, ExpertGatingFunc: 1,
		HeadCount: 2, HeadCountKV: 1, KeyLength: 4, ValueLength: 4, VocabularySize: 32,
	}
	file := &gguf.File{Tensors: []gguf.TensorInfo{
		tensorInfo("token_embd.weight", 8, 32), tensorInfo("output_norm.weight", 8),
		tensorInfo("blk.0.attn_norm.weight", 8), tensorInfo("blk.0.attn_qkv.weight", 8, 16),
		tensorInfo("blk.0.attn_qkv.bias", 16), tensorInfo("blk.0.attn_output.weight", 8, 8),
		tensorInfo("blk.0.ffn_norm.weight", 8), tensorInfo("blk.0.ffn_gate_inp.weight", 8, 4),
		tensorInfo("blk.0.ffn_gate_exps.weight", 8, 6, 4),
		tensorInfo("blk.0.ffn_up_exps.weight", 8, 6, 4),
		tensorInfo("blk.0.ffn_down_exps.weight", 6, 8, 4),
	}}
	weights, err := ReadWeights(file, spec)
	if err != nil {
		t.Fatal(err)
	}
	layer := weights.Layers[0]
	if weights.Output != nil || layer.AttentionQKV == nil || layer.AttentionQKVBias == nil ||
		layer.FeedForwardRouter == nil || layer.FeedForwardGateExperts == nil ||
		layer.FeedForwardUpExperts == nil || layer.FeedForwardDownExperts == nil {
		t.Fatalf("unexpected SmallThinker catalog: %+v", weights)
	}
}

func TestReadWeightsDOTS1DenseThenMoE(t *testing.T) {
	spec := Spec{
		Architecture: "dots1", BlockCount: 2, LeadingDenseBlocks: 1,
		EmbeddingLength: 8, FeedForwardLength: 12, ExpertCount: 4, ExpertUsedCount: 2,
		ExpertFeedForward: 6, SharedExpertCount: 2, SharedExpertFF: 12,
		ExpertWeightsScale: 1.25, ExpertWeightsNorm: true, ExpertGatingFunc: 2,
		HeadCount: 2, HeadCountKV: 2, KeyLength: 4, ValueLength: 4, VocabularySize: 32,
	}
	tensors := []gguf.TensorInfo{
		tensorInfo("token_embd.weight", 8, 32), tensorInfo("output_norm.weight", 8),
		tensorInfo("output.weight", 8, 32),
	}
	for block := 0; block < 2; block++ {
		prefix := fmt.Sprintf("blk.%d.", block)
		tensors = append(tensors,
			tensorInfo(prefix+"attn_norm.weight", 8), tensorInfo(prefix+"attn_qkv.weight", 8, 24),
			tensorInfo(prefix+"attn_output.weight", 8, 8), tensorInfo(prefix+"attn_q_norm.weight", 4),
			tensorInfo(prefix+"attn_k_norm.weight", 4), tensorInfo(prefix+"ffn_norm.weight", 8),
		)
	}
	tensors = append(tensors,
		tensorInfo("blk.0.ffn_gate.weight", 8, 12), tensorInfo("blk.0.ffn_up.weight", 8, 12),
		tensorInfo("blk.0.ffn_down.weight", 12, 8), tensorInfo("blk.1.ffn_gate_inp.weight", 8, 4),
		tensorInfo("blk.1.exp_probs_b.bias", 4), tensorInfo("blk.1.ffn_gate_exps.weight", 8, 6, 4),
		tensorInfo("blk.1.ffn_up_exps.weight", 8, 6, 4), tensorInfo("blk.1.ffn_down_exps.weight", 6, 8, 4),
		tensorInfo("blk.1.ffn_gate_shexp.weight", 8, 12), tensorInfo("blk.1.ffn_up_shexp.weight", 8, 12),
		tensorInfo("blk.1.ffn_down_shexp.weight", 12, 8),
	)
	weights, err := ReadWeights(&gguf.File{Tensors: tensors}, spec)
	if err != nil {
		t.Fatal(err)
	}
	dense, moe := weights.Layers[0], weights.Layers[1]
	if weights.Output == nil || dense.AttentionQKV == nil || dense.AttentionQNorm == nil ||
		dense.AttentionKNorm == nil || dense.FeedForwardGate.Name == "" || dense.FeedForwardRouter != nil ||
		moe.FeedForwardRouter == nil || moe.FeedForwardExpertBias == nil ||
		moe.FeedForwardSharedGate == nil || moe.FeedForwardSharedUp == nil ||
		moe.FeedForwardSharedDown == nil {
		t.Fatalf("unexpected DOTS1 catalog: %+v", weights)
	}
}

func TestReadWeightsMiniMaxM2(t *testing.T) {
	spec := Spec{
		Architecture: "minimax-m2", BlockCount: 1, EmbeddingLength: 8,
		FeedForwardLength: 6, ExpertCount: 4, ExpertUsedCount: 2,
		ExpertFeedForward: 6, ExpertWeightsScale: 1, ExpertGatingFunc: 2,
		HeadCount: 2, HeadCountKV: 1, KeyLength: 4, ValueLength: 4, VocabularySize: 32,
	}
	file := &gguf.File{Tensors: []gguf.TensorInfo{
		tensorInfo("token_embd.weight", 8, 32), tensorInfo("output_norm.weight", 8),
		tensorInfo("output.weight", 8, 32), tensorInfo("blk.0.attn_norm.weight", 8),
		tensorInfo("blk.0.attn_qkv.weight", 8, 16), tensorInfo("blk.0.attn_output.weight", 8, 8),
		tensorInfo("blk.0.attn_q_norm.weight", 8), tensorInfo("blk.0.attn_k_norm.weight", 4),
		tensorInfo("blk.0.ffn_norm.weight", 8), tensorInfo("blk.0.ffn_gate_inp.weight", 8, 4),
		tensorInfo("blk.0.exp_probs_b.bias", 4), tensorInfo("blk.0.ffn_gate_exps.weight", 8, 6, 4),
		tensorInfo("blk.0.ffn_up_exps.weight", 8, 6, 4),
		tensorInfo("blk.0.ffn_down_exps.weight", 6, 8, 4),
	}}
	weights, err := ReadWeights(file, spec)
	if err != nil {
		t.Fatal(err)
	}
	layer := weights.Layers[0]
	if weights.Output == nil || layer.AttentionQKV == nil || layer.AttentionQNorm == nil ||
		layer.AttentionKNorm == nil || layer.FeedForwardRouter == nil ||
		layer.FeedForwardExpertBias == nil || layer.FeedForwardGateExperts == nil {
		t.Fatalf("unexpected MiniMax-M2 catalog: %+v", weights)
	}
}

func TestReadWeightsBailingMoE2DenseThenMoE(t *testing.T) {
	spec := Spec{
		Architecture: "bailingmoe2", BlockCount: 2, LeadingDenseBlocks: 1,
		EmbeddingLength: 8, FeedForwardLength: 16, ExpertCount: 4, ExpertUsedCount: 2,
		ExpertFeedForward: 6, SharedExpertCount: 2, SharedExpertFF: 10,
		ExpertWeightsScale: 1.25, ExpertGatingFunc: 2,
		HeadCount: 2, HeadCountKV: 1, KeyLength: 4, ValueLength: 4, VocabularySize: 32,
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
			tensorInfo(prefix+"attn_q_norm.weight", 4),
			tensorInfo(prefix+"attn_k_norm.weight", 4),
			tensorInfo(prefix+"ffn_norm.weight", 8),
		)
	}
	tensors = append(tensors,
		tensorInfo("blk.0.ffn_gate.weight", 8, 16),
		tensorInfo("blk.0.ffn_up.weight", 8, 16),
		tensorInfo("blk.0.ffn_down.weight", 16, 8),
		tensorInfo("blk.1.ffn_gate_inp.weight", 8, 4),
		tensorInfo("blk.1.exp_probs_b.bias", 4),
		tensorInfo("blk.1.ffn_gate_exps.weight", 8, 6, 4),
		tensorInfo("blk.1.ffn_up_exps.weight", 8, 6, 4),
		tensorInfo("blk.1.ffn_down_exps.weight", 6, 8, 4),
		tensorInfo("blk.1.ffn_gate_shexp.weight", 8, 10),
		tensorInfo("blk.1.ffn_up_shexp.weight", 8, 10),
		tensorInfo("blk.1.ffn_down_shexp.weight", 10, 8),
	)
	weights, err := ReadWeights(&gguf.File{Tensors: tensors}, spec)
	if err != nil {
		t.Fatal(err)
	}
	dense, moe := weights.Layers[0], weights.Layers[1]
	if weights.Output == nil || dense.AttentionQKV == nil || dense.AttentionQ.Name != "" ||
		dense.FeedForwardGate.Name == "" || dense.FeedForwardRouter != nil ||
		moe.AttentionQKV == nil || moe.AttentionQNorm == nil || moe.AttentionKNorm == nil ||
		moe.FeedForwardRouter == nil || moe.FeedForwardExpertBias == nil ||
		moe.FeedForwardSharedGate == nil || moe.FeedForwardSharedUp == nil ||
		moe.FeedForwardSharedDown == nil {
		t.Fatalf("unexpected BailingMoE2 catalog: %+v", weights)
	}
}

func TestReadWeightsDream(t *testing.T) {
	spec := Spec{
		Architecture:      "dream",
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
		tensorInfo("output.bias", 32),
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
	if len(weights.Layers) != 1 || weights.OutputBias == nil ||
		weights.Layers[0].AttentionQBias != nil {
		t.Fatalf("unexpected Dream weights: %+v", weights)
	}
}

func TestReadWeightsLlamaEmbedDenseAndMoE(t *testing.T) {
	base := Spec{
		Architecture: "llama-embed", BlockCount: 1, EmbeddingLength: 8,
		FeedForwardLength: 16, HeadCount: 2, HeadCountKV: 1,
		KeyLength: 4, ValueLength: 4, VocabularySize: 32,
		RMSNormEpsilon: 1e-6, NonCausalAttention: true,
	}
	common := []gguf.TensorInfo{
		tensorInfo("token_embd.weight", 8, 32), tensorInfo("output_norm.weight", 8),
		tensorInfo("blk.0.attn_norm.weight", 8), tensorInfo("blk.0.attn_q.weight", 8, 8),
		tensorInfo("blk.0.attn_k.weight", 8, 4), tensorInfo("blk.0.attn_v.weight", 8, 4),
		tensorInfo("blk.0.attn_output.weight", 8, 8), tensorInfo("blk.0.ffn_norm.weight", 8),
	}

	denseFile := &gguf.File{Tensors: append(append([]gguf.TensorInfo{}, common...),
		tensorInfo("blk.0.ffn_gate.weight", 8, 16),
		tensorInfo("blk.0.ffn_up.weight", 8, 16),
		tensorInfo("blk.0.ffn_down.weight", 16, 8),
	)}
	dense, err := ReadWeights(denseFile, base)
	if err != nil {
		t.Fatal(err)
	}
	if dense.OutputNorm.Name != "output_norm.weight" || dense.Output != nil ||
		dense.Layers[0].FeedForwardGate.Name == "" || dense.Layers[0].FeedForwardRouter != nil {
		t.Fatalf("unexpected dense Llama Embed catalog: %+v", dense)
	}

	moeSpec := base
	moeSpec.ExpertCount, moeSpec.ExpertUsedCount, moeSpec.ExpertFeedForward = 4, 2, 16
	moeSpec.ExpertWeightsScale = 1
	moeFile := &gguf.File{Tensors: append(append([]gguf.TensorInfo{}, common...),
		tensorInfo("blk.0.ffn_gate_inp.weight", 8, 4),
		tensorInfo("blk.0.ffn_gate_exps.weight", 8, 16, 4),
		tensorInfo("blk.0.ffn_up_exps.weight", 8, 16, 4),
		tensorInfo("blk.0.ffn_down_exps.weight", 16, 8, 4),
	)}
	moe, err := ReadWeights(moeFile, moeSpec)
	if err != nil {
		t.Fatal(err)
	}
	if moe.Layers[0].FeedForwardRouter == nil || moe.Layers[0].FeedForwardGateExperts == nil ||
		moe.Layers[0].FeedForwardUpExperts == nil || moe.Layers[0].FeedForwardDownExperts == nil ||
		moe.Layers[0].FeedForwardGate.Name != "" {
		t.Fatalf("unexpected MoE Llama Embed catalog: %+v", moe.Layers[0])
	}
}

func TestReadWeightsPanguEmbeddedRequiresOutputBiasAndSelectsLongRoPE(t *testing.T) {
	spec := Spec{
		Architecture: "pangu-embedded", BlockCount: 1, ContextLength: 4096,
		OriginalContextLength: 2048, EmbeddingLength: 8, FeedForwardLength: 16,
		HeadCount: 2, HeadCountKV: 1, KeyLength: 4, ValueLength: 4,
		RopeDimensionCount: 4, RopeScalingType: "longrope", VocabularySize: 32,
		RMSNormEpsilon: 1e-6,
	}
	file := &gguf.File{Tensors: []gguf.TensorInfo{
		tensorInfo("token_embd.weight", 8, 32), tensorInfo("output_norm.weight", 8),
		tensorInfo("blk.0.attn_norm.weight", 8), tensorInfo("blk.0.attn_q.weight", 8, 8),
		tensorInfo("blk.0.attn_k.weight", 8, 4), tensorInfo("blk.0.attn_v.weight", 8, 4),
		tensorInfo("blk.0.attn_output.weight", 8, 8), tensorInfo("blk.0.attn_output.bias", 8),
		tensorInfo("blk.0.ffn_norm.weight", 8), tensorInfo("blk.0.ffn_gate.weight", 8, 16),
		tensorInfo("blk.0.ffn_up.weight", 8, 16), tensorInfo("blk.0.ffn_down.weight", 16, 8),
		tensorInfo("blk.0.rope_factors_long.weight", 2), tensorInfo("blk.0.rope_factors_short.weight", 2),
	}}
	weights, err := ReadWeights(file, spec)
	if err != nil {
		t.Fatal(err)
	}
	layer := weights.Layers[0]
	if weights.Output != nil || weights.OutputNorm.Name != "output_norm.weight" ||
		layer.AttentionOutputBias == nil || layer.RopeFactors == nil ||
		layer.RopeFactors.Name != "blk.0.rope_factors_long.weight" {
		t.Fatalf("unexpected Pangu Embedded catalog: %+v", weights)
	}

	withoutBias := make([]gguf.TensorInfo, 0, len(file.Tensors)-1)
	for _, item := range file.Tensors {
		if item.Name != "blk.0.attn_output.bias" {
			withoutBias = append(withoutBias, item)
		}
	}
	if _, err := ReadWeights(&gguf.File{Tensors: withoutBias}, spec); err == nil ||
		!strings.Contains(err.Error(), "attn_output.bias") {
		t.Fatalf("missing Pangu Embedded output bias error = %v", err)
	}

	fusedTensors := make([]gguf.TensorInfo, 0, len(file.Tensors))
	for _, item := range file.Tensors {
		if item.Name != "blk.0.attn_q.weight" && item.Name != "blk.0.attn_k.weight" && item.Name != "blk.0.attn_v.weight" {
			fusedTensors = append(fusedTensors, item)
		}
	}
	fusedTensors = append(fusedTensors,
		tensorInfo("blk.0.attn_qkv.weight", 8, 16),
		tensorInfo("blk.0.attn_qkv.bias", 16),
	)
	fused, err := ReadWeights(&gguf.File{Tensors: fusedTensors}, spec)
	if err != nil {
		t.Fatal(err)
	}
	if fused.Layers[0].AttentionQKV == nil || fused.Layers[0].AttentionQKVBias == nil ||
		fused.Layers[0].AttentionQ.Name != "" {
		t.Fatalf("unexpected fused Pangu Embedded catalog: %+v", fused.Layers[0])
	}
}

func TestReadWeightsModernBERTUsesOptionalFirstNormAndFusedGEGLU(t *testing.T) {
	spec := Spec{
		Architecture: "modern-bert", BlockCount: 2, ContextLength: 8192,
		EmbeddingLength: 8, FeedForwardLength: 16, HeadCount: 2, HeadCountKV: 2,
		KeyLength: 4, ValueLength: 4, RopeDimensionCount: 4, VocabularySize: 32,
		LayerNormEpsilon: 1e-5, NonCausalAttention: true, HiddenActivation: "gelu",
	}
	tensors := []gguf.TensorInfo{
		tensorInfo("token_embd.weight", 8, 32), tensorInfo("token_embd_norm.weight", 8),
		tensorInfo("output_norm.weight", 8),
	}
	for block := 0; block < 2; block++ {
		prefix := fmt.Sprintf("blk.%d.", block)
		if block > 0 {
			tensors = append(tensors, tensorInfo(prefix+"attn_norm.weight", 8))
		}
		tensors = append(tensors,
			tensorInfo(prefix+"attn_qkv.weight", 8, 24),
			tensorInfo(prefix+"attn_output.weight", 8, 8),
			tensorInfo(prefix+"ffn_norm.weight", 8),
			tensorInfo(prefix+"ffn_up.weight", 8, 32),
			tensorInfo(prefix+"ffn_down.weight", 16, 8),
		)
	}
	weights, err := ReadWeights(&gguf.File{Tensors: tensors}, spec)
	if err != nil {
		t.Fatal(err)
	}
	if weights.TokenEmbeddingNorm == nil || weights.TokenEmbeddingNormBias != nil ||
		weights.OutputNorm.Name != "output_norm.weight" || weights.Output != nil ||
		weights.Layers[0].AttentionNorm.Name != "" || weights.Layers[1].AttentionNorm.Name == "" ||
		weights.Layers[0].AttentionQKV == nil || weights.Layers[0].FeedForwardUp.Shape[1] != 32 ||
		weights.Layers[0].FeedForwardGate.Name != "" {
		t.Fatalf("unexpected ModernBERT catalog: %+v", weights)
	}
}

func TestReadWeightsGemmaEmbeddingLoadsProjectionAndPostNormCatalog(t *testing.T) {
	spec := Spec{
		Architecture: "gemma-embedding", BlockCount: 1, ContextLength: 2048,
		EmbeddingLength: 8, FeedForwardLength: 16, HeadCount: 2, HeadCountKV: 1,
		KeyLength: 4, ValueLength: 4, RopeDimensionCount: 4, VocabularySize: 32,
		RMSNormEpsilon: 1e-6, NonCausalAttention: true,
		Dense2FeatureIn: 8, Dense2FeatureOut: 6, Dense3FeatureIn: 6, Dense3FeatureOut: 8,
	}
	file := &gguf.File{Tensors: []gguf.TensorInfo{
		tensorInfo("token_embd.weight", 8, 32), tensorInfo("output_norm.weight", 8),
		tensorInfo("dense_2.weight", 8, 6), tensorInfo("dense_3.weight", 6, 8),
		tensorInfo("blk.0.attn_norm.weight", 8), tensorInfo("blk.0.attn_qkv.weight", 8, 16),
		tensorInfo("blk.0.attn_qkv.bias", 16), tensorInfo("blk.0.attn_output.weight", 8, 8),
		tensorInfo("blk.0.attn_q_norm.weight", 4), tensorInfo("blk.0.attn_k_norm.weight", 4),
		tensorInfo("blk.0.post_attention_norm.weight", 8), tensorInfo("blk.0.ffn_norm.weight", 8),
		tensorInfo("blk.0.ffn_gate.weight", 8, 16), tensorInfo("blk.0.ffn_up.weight", 8, 16),
		tensorInfo("blk.0.ffn_down.weight", 16, 8), tensorInfo("blk.0.post_ffw_norm.weight", 8),
	}}
	weights, err := ReadWeights(file, spec)
	if err != nil {
		t.Fatal(err)
	}
	layer := weights.Layers[0]
	if weights.Dense2Output == nil || weights.Dense3Output == nil || weights.Output != nil ||
		layer.AttentionQKV == nil || layer.AttentionQKVBias == nil ||
		layer.AttentionQNorm == nil || layer.AttentionKNorm == nil ||
		layer.AttentionPostNorm == nil || layer.FeedForwardPostNorm == nil {
		t.Fatalf("unexpected Gemma embedding catalog: %+v", weights)
	}
}

func TestReadWeightsEuroBERTFusedQKVWithoutOutput(t *testing.T) {
	spec := Spec{
		Architecture: "eurobert", BlockCount: 1, EmbeddingLength: 8,
		FeedForwardLength: 16, HeadCount: 2, HeadCountKV: 1,
		KeyLength: 4, ValueLength: 4, VocabularySize: 32,
		RMSNormEpsilon: 1e-6, NonCausalAttention: true,
	}
	file := &gguf.File{Tensors: []gguf.TensorInfo{
		tensorInfo("token_embd.weight", 8, 32), tensorInfo("output_norm.weight", 8),
		tensorInfo("blk.0.attn_norm.weight", 8),
		tensorInfo("blk.0.attn_qkv.weight", 8, 16),
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
	if weights.Output != nil || layer.AttentionQKV == nil || layer.AttentionQ.Name != "" ||
		layer.FeedForwardGate.Name == "" || layer.FeedForwardUp.Name == "" ||
		layer.FeedForwardDown.Name == "" {
		t.Fatalf("unexpected EuroBERT catalog: %+v", weights)
	}
}

func TestReadWeightsBERTPostNormEncoder(t *testing.T) {
	spec := Spec{
		Architecture: "bert", BlockCount: 1, ContextLength: 512,
		EmbeddingLength: 8, FeedForwardLength: 16, HeadCount: 2, HeadCountKV: 2,
		KeyLength: 4, ValueLength: 4, VocabularySize: 32, TokenTypeCount: 2,
		LayerNormEpsilon: 1e-5, NonCausalAttention: true, RopeDisabled: true,
	}
	file := &gguf.File{Tensors: []gguf.TensorInfo{
		tensorInfo("token_embd.weight", 8, 32),
		tensorInfo("token_types.weight", 8, 2),
		tensorInfo("position_embd.weight", 8, 512),
		tensorInfo("token_embd_norm.weight", 8), tensorInfo("token_embd_norm.bias", 8),
		tensorInfo("blk.0.attn_qkv.weight", 8, 24), tensorInfo("blk.0.attn_qkv.bias", 24),
		tensorInfo("blk.0.attn_output.weight", 8, 8), tensorInfo("blk.0.attn_output.bias", 8),
		tensorInfo("blk.0.attn_output_norm.weight", 8), tensorInfo("blk.0.attn_output_norm.bias", 8),
		tensorInfo("blk.0.ffn_up.weight", 8, 16), tensorInfo("blk.0.ffn_up.bias", 16),
		tensorInfo("blk.0.ffn_down.weight", 16, 8), tensorInfo("blk.0.ffn_down.bias", 8),
		tensorInfo("blk.0.layer_output_norm.weight", 8), tensorInfo("blk.0.layer_output_norm.bias", 8),
	}}
	weights, err := ReadWeights(file, spec)
	if err != nil {
		t.Fatal(err)
	}
	layer := weights.Layers[0]
	if weights.OutputNorm.Name != "" || weights.Output != nil || weights.PositionEmbedding == nil ||
		weights.TokenTypeEmbedding == nil || weights.TokenEmbeddingNorm == nil ||
		weights.TokenEmbeddingNormBias == nil || layer.AttentionQKV == nil ||
		layer.AttentionQKVBias == nil || layer.AttentionNorm.Name != "" ||
		layer.AttentionPostNorm == nil || layer.AttentionPostNormBias == nil ||
		layer.FeedForwardNorm.Name != "" || layer.FeedForwardGate.Name != "" ||
		layer.FeedForwardUpBias == nil || layer.FeedForwardDownBias == nil ||
		layer.FeedForwardPostNorm == nil || layer.FeedForwardPostNormBias == nil {
		t.Fatalf("unexpected BERT catalog: %+v", weights)
	}
}

func TestReadWeightsNeoBERTFusedQKVAndSwiGLU(t *testing.T) {
	spec := Spec{
		Architecture: "neo-bert", BlockCount: 1, EmbeddingLength: 8,
		FeedForwardLength: 16, HeadCount: 2, HeadCountKV: 2,
		KeyLength: 4, ValueLength: 4, VocabularySize: 32, RMSNormEpsilon: 1e-6,
	}
	file := &gguf.File{Tensors: []gguf.TensorInfo{
		tensorInfo("token_embd.weight", 8, 32), tensorInfo("enc.output_norm.weight", 8),
		tensorInfo("blk.0.attn_norm.weight", 8), tensorInfo("blk.0.attn_qkv.weight", 8, 24),
		tensorInfo("blk.0.attn_output.weight", 8, 8), tensorInfo("blk.0.ffn_norm.weight", 8),
		tensorInfo("blk.0.ffn_up.weight", 8, 32), tensorInfo("blk.0.ffn_down.weight", 16, 8),
	}}
	weights, err := ReadWeights(file, spec)
	if err != nil {
		t.Fatal(err)
	}
	layer := weights.Layers[0]
	if weights.OutputNorm.Name != "enc.output_norm.weight" || weights.Output != nil ||
		layer.AttentionQKV == nil || layer.AttentionQ.Name != "" ||
		layer.FeedForwardGate.Name != "" || layer.FeedForwardUp.Shape[1] != 32 {
		t.Fatalf("unexpected NeoBERT catalog: %+v", weights)
	}
}

func TestReadWeightsNomicBERTPostNormSwiGLU(t *testing.T) {
	spec := Spec{
		Architecture: "nomic-bert", BlockCount: 1, ContextLength: 8192,
		EmbeddingLength: 8, FeedForwardLength: 16, HeadCount: 2, HeadCountKV: 2,
		KeyLength: 4, ValueLength: 4, VocabularySize: 32, TokenTypeCount: 2,
		LayerNormEpsilon: 1e-5, NonCausalAttention: true, RopeDimensionCount: 4,
	}
	file := &gguf.File{Tensors: []gguf.TensorInfo{
		tensorInfo("token_embd.weight", 8, 32), tensorInfo("token_types.weight", 8, 2),
		tensorInfo("token_embd_norm.weight", 8), tensorInfo("token_embd_norm.bias", 8),
		tensorInfo("blk.0.attn_qkv.weight", 8, 24), tensorInfo("blk.0.attn_qkv.bias", 24),
		tensorInfo("blk.0.attn_output.weight", 8, 8), tensorInfo("blk.0.attn_output.bias", 8),
		tensorInfo("blk.0.attn_output_norm.weight", 8), tensorInfo("blk.0.attn_output_norm.bias", 8),
		tensorInfo("blk.0.ffn_gate.weight", 8, 16), tensorInfo("blk.0.ffn_up.weight", 8, 16),
		tensorInfo("blk.0.ffn_up.bias", 16), tensorInfo("blk.0.ffn_down.weight", 16, 8),
		tensorInfo("blk.0.ffn_down.bias", 8), tensorInfo("blk.0.layer_output_norm.weight", 8),
		tensorInfo("blk.0.layer_output_norm.bias", 8),
	}}
	weights, err := ReadWeights(file, spec)
	if err != nil {
		t.Fatal(err)
	}
	layer := weights.Layers[0]
	if weights.OutputNorm.Name != "" || weights.Output != nil || weights.PositionEmbedding != nil ||
		weights.TokenTypeEmbedding == nil || weights.TokenEmbeddingNorm == nil ||
		weights.TokenEmbeddingNormBias == nil || layer.AttentionQKV == nil ||
		layer.AttentionQKVBias == nil || layer.AttentionNorm.Name != "" ||
		layer.AttentionPostNorm == nil || layer.AttentionPostNormBias == nil ||
		layer.FeedForwardNorm.Name != "" || layer.FeedForwardGate.Name == "" ||
		layer.FeedForwardUpBias == nil || layer.FeedForwardDownBias == nil ||
		layer.FeedForwardPostNorm == nil || layer.FeedForwardPostNormBias == nil {
		t.Fatalf("unexpected NomicBERT catalog: %+v", weights)
	}
}

func TestReadWeightsJinaBERTV2OptionalNormsAndFusedGEGLU(t *testing.T) {
	spec := Spec{
		Architecture: "jina-bert-v2", BlockCount: 1, ContextLength: 8192,
		EmbeddingLength: 8, FeedForwardLength: 16, HeadCount: 2, HeadCountKV: 2,
		KeyLength: 4, ValueLength: 4, VocabularySize: 32, TokenTypeCount: 2,
		LayerNormEpsilon: 1e-5, NonCausalAttention: true, RopeDisabled: true, MaxALiBiBias: 8,
	}
	file := &gguf.File{Tensors: []gguf.TensorInfo{
		tensorInfo("token_embd.weight", 8, 32), tensorInfo("token_types.weight", 8, 2),
		tensorInfo("token_embd_norm.weight", 8), tensorInfo("token_embd_norm.bias", 8),
		tensorInfo("blk.0.attn_qkv.weight", 8, 24), tensorInfo("blk.0.attn_qkv.bias", 24),
		tensorInfo("blk.0.attn_q_norm.weight", 8), tensorInfo("blk.0.attn_q_norm.bias", 8),
		tensorInfo("blk.0.attn_k_norm.weight", 8), tensorInfo("blk.0.attn_k_norm.bias", 8),
		tensorInfo("blk.0.attn_output.weight", 8, 8), tensorInfo("blk.0.attn_output.bias", 8),
		tensorInfo("blk.0.attn_output_norm.weight", 8), tensorInfo("blk.0.attn_output_norm.bias", 8),
		tensorInfo("blk.0.attn_norm_2.weight", 8), tensorInfo("blk.0.attn_norm_2.bias", 8),
		tensorInfo("blk.0.ffn_up.weight", 8, 32), tensorInfo("blk.0.ffn_up.bias", 32),
		tensorInfo("blk.0.ffn_down.weight", 16, 8), tensorInfo("blk.0.ffn_down.bias", 8),
		tensorInfo("blk.0.layer_output_norm.weight", 8), tensorInfo("blk.0.layer_output_norm.bias", 8),
	}}
	weights, err := ReadWeights(file, spec)
	if err != nil {
		t.Fatal(err)
	}
	layer := weights.Layers[0]
	if weights.OutputNorm.Name != "" || weights.Output != nil || weights.PositionEmbedding != nil ||
		weights.TokenTypeEmbedding == nil || weights.TokenEmbeddingNorm == nil ||
		weights.TokenEmbeddingNormBias == nil || layer.AttentionQKV == nil ||
		layer.AttentionQKVBias == nil || layer.AttentionQNorm == nil ||
		layer.AttentionKNorm == nil || layer.AttentionQNormBias == nil ||
		layer.AttentionKNormBias == nil || layer.AttentionNorm2 == nil ||
		layer.AttentionNorm2Bias == nil || layer.AttentionOutputBias == nil ||
		layer.AttentionPostNorm == nil || layer.AttentionPostNormBias == nil ||
		layer.FeedForwardGate.Name != "" || layer.FeedForwardUp.Shape[1] != 32 ||
		layer.FeedForwardUpBias == nil || layer.FeedForwardDownBias == nil ||
		layer.FeedForwardPostNorm == nil || layer.FeedForwardPostNormBias == nil {
		t.Fatalf("unexpected JinaBERT v2 catalog: %+v", weights)
	}
}

func TestReadWeightsJinaBERTV2SeparateGate(t *testing.T) {
	spec := Spec{
		Architecture: "jina-bert-v2", BlockCount: 1, ContextLength: 8192,
		EmbeddingLength: 8, FeedForwardLength: 16, HeadCount: 2, HeadCountKV: 2,
		KeyLength: 4, ValueLength: 4, VocabularySize: 32, TokenTypeCount: 2,
		LayerNormEpsilon: 1e-5, NonCausalAttention: true, RopeDisabled: true, MaxALiBiBias: 8,
	}
	file := &gguf.File{Tensors: []gguf.TensorInfo{
		tensorInfo("token_embd.weight", 8, 32), tensorInfo("token_types.weight", 8, 2),
		tensorInfo("token_embd_norm.weight", 8), tensorInfo("token_embd_norm.bias", 8),
		tensorInfo("blk.0.attn_q.weight", 8, 8), tensorInfo("blk.0.attn_k.weight", 8, 8),
		tensorInfo("blk.0.attn_v.weight", 8, 8), tensorInfo("blk.0.attn_output.weight", 8, 8),
		tensorInfo("blk.0.attn_output.bias", 8), tensorInfo("blk.0.attn_output_norm.weight", 8),
		tensorInfo("blk.0.attn_output_norm.bias", 8), tensorInfo("blk.0.ffn_gate.weight", 8, 16),
		tensorInfo("blk.0.ffn_up.weight", 8, 16), tensorInfo("blk.0.ffn_down.weight", 16, 8),
		tensorInfo("blk.0.ffn_down.bias", 8), tensorInfo("blk.0.layer_output_norm.weight", 8),
		tensorInfo("blk.0.layer_output_norm.bias", 8),
	}}
	weights, err := ReadWeights(file, spec)
	if err != nil {
		t.Fatal(err)
	}
	if weights.Layers[0].FeedForwardGate.Name == "" || weights.Layers[0].FeedForwardUp.Shape[1] != 16 {
		t.Fatalf("unexpected JinaBERT v2 separate gate catalog: %+v", weights.Layers[0])
	}
}

func TestReadWeightsJinaBERTV3PostNormGELU(t *testing.T) {
	spec := Spec{
		Architecture: "jina-bert-v3", BlockCount: 1, ContextLength: 8192,
		EmbeddingLength: 8, FeedForwardLength: 16, HeadCount: 2, HeadCountKV: 2,
		KeyLength: 4, ValueLength: 4, VocabularySize: 32, TokenTypeCount: 2,
		LayerNormEpsilon: 1e-5, NonCausalAttention: true, RopeDimensionCount: 4,
	}
	file := &gguf.File{Tensors: []gguf.TensorInfo{
		tensorInfo("token_embd.weight", 8, 32), tensorInfo("token_types.weight", 8, 2),
		tensorInfo("token_embd_norm.weight", 8), tensorInfo("token_embd_norm.bias", 8),
		tensorInfo("blk.0.attn_qkv.weight", 8, 24), tensorInfo("blk.0.attn_qkv.bias", 24),
		tensorInfo("blk.0.attn_output.weight", 8, 8), tensorInfo("blk.0.attn_output.bias", 8),
		tensorInfo("blk.0.attn_output_norm.weight", 8), tensorInfo("blk.0.attn_output_norm.bias", 8),
		tensorInfo("blk.0.ffn_up.weight", 8, 16), tensorInfo("blk.0.ffn_up.bias", 16),
		tensorInfo("blk.0.ffn_down.weight", 16, 8), tensorInfo("blk.0.ffn_down.bias", 8),
		tensorInfo("blk.0.layer_output_norm.weight", 8), tensorInfo("blk.0.layer_output_norm.bias", 8),
	}}
	weights, err := ReadWeights(file, spec)
	if err != nil {
		t.Fatal(err)
	}
	layer := weights.Layers[0]
	if weights.OutputNorm.Name != "" || weights.Output != nil || weights.PositionEmbedding != nil ||
		weights.TokenTypeEmbedding == nil || weights.TokenEmbeddingNorm == nil ||
		weights.TokenEmbeddingNormBias == nil || layer.AttentionQKV == nil ||
		layer.AttentionQKVBias == nil || layer.AttentionPostNorm == nil ||
		layer.AttentionPostNormBias == nil || layer.FeedForwardGate.Name != "" ||
		layer.FeedForwardUpBias == nil || layer.FeedForwardDownBias == nil ||
		layer.FeedForwardPostNorm == nil || layer.FeedForwardPostNormBias == nil {
		t.Fatalf("unexpected JinaBERT v3 catalog: %+v", weights)
	}
}

func TestReadWeightsNomicBERTMoEAlternatesDenseAndExperts(t *testing.T) {
	spec := Spec{
		Architecture: "nomic-bert-moe", BlockCount: 2, ContextLength: 8192,
		EmbeddingLength: 8, FeedForwardLength: 16, HeadCount: 2, HeadCountKV: 2,
		KeyLength: 4, ValueLength: 4, VocabularySize: 32, TokenTypeCount: 2,
		ExpertCount: 4, ExpertUsedCount: 2, ExpertFeedForward: 16, ExpertWeightsScale: 1,
		MoELayerStep: 2, LayerNormEpsilon: 1e-5, NonCausalAttention: true, RopeDimensionCount: 4,
	}
	tensors := []gguf.TensorInfo{
		tensorInfo("token_embd.weight", 8, 32), tensorInfo("token_types.weight", 8, 2),
		tensorInfo("token_embd_norm.weight", 8), tensorInfo("token_embd_norm.bias", 8),
	}
	for block := 0; block < 2; block++ {
		prefix := fmt.Sprintf("blk.%d.", block)
		tensors = append(tensors,
			tensorInfo(prefix+"attn_qkv.weight", 8, 24), tensorInfo(prefix+"attn_output.weight", 8, 8),
			tensorInfo(prefix+"attn_output_norm.weight", 8), tensorInfo(prefix+"attn_output_norm.bias", 8),
			tensorInfo(prefix+"layer_output_norm.weight", 8), tensorInfo(prefix+"layer_output_norm.bias", 8),
		)
	}
	tensors = append(tensors,
		tensorInfo("blk.0.ffn_up.weight", 8, 16), tensorInfo("blk.0.ffn_down.weight", 16, 8),
		tensorInfo("blk.1.ffn_gate_inp.weight", 8, 4),
		tensorInfo("blk.1.ffn_up_exps.weight", 8, 16, 4),
		tensorInfo("blk.1.ffn_down_exps.weight", 16, 8, 4),
	)
	weights, err := ReadWeights(&gguf.File{Tensors: tensors}, spec)
	if err != nil {
		t.Fatal(err)
	}
	dense, moe := weights.Layers[0], weights.Layers[1]
	if dense.FeedForwardUp.Name == "" || dense.FeedForwardDown.Name == "" || dense.FeedForwardRouter != nil ||
		moe.FeedForwardRouter == nil || moe.FeedForwardUpExperts == nil || moe.FeedForwardDownExperts == nil ||
		moe.FeedForwardGateExperts != nil || moe.FeedForwardUp.Name != "" {
		t.Fatalf("unexpected NomicBERT-MoE catalog: %+v / %+v", dense, moe)
	}
}

func TestReadWeightsRND1(t *testing.T) {
	spec := Spec{
		Architecture: "rnd1", BlockCount: 1, EmbeddingLength: 8,
		FeedForwardLength: 24, ExpertCount: 4, ExpertUsedCount: 2,
		ExpertFeedForward: 12, ExpertWeightsScale: 1,
		HeadCount: 2, HeadCountKV: 1, KeyLength: 4, ValueLength: 4,
		VocabularySize: 32, NonCausalAttention: true,
	}
	file := &gguf.File{Tensors: []gguf.TensorInfo{
		tensorInfo("token_embd.weight", 8, 32), tensorInfo("output_norm.weight", 8),
		tensorInfo("blk.0.attn_norm.weight", 8), tensorInfo("blk.0.attn_q.weight", 8, 8),
		tensorInfo("blk.0.attn_k.weight", 8, 4), tensorInfo("blk.0.attn_v.weight", 8, 4),
		tensorInfo("blk.0.attn_output.weight", 8, 8),
		tensorInfo("blk.0.attn_q_norm.weight", 4), tensorInfo("blk.0.attn_k_norm.weight", 4),
		tensorInfo("blk.0.ffn_norm.weight", 8), tensorInfo("blk.0.ffn_gate_inp.weight", 8, 4),
		tensorInfo("blk.0.ffn_gate_exps.weight", 8, 12, 4),
		tensorInfo("blk.0.ffn_up_exps.weight", 8, 12, 4),
		tensorInfo("blk.0.ffn_down_exps.weight", 12, 8, 4),
	}}
	weights, err := ReadWeights(file, spec)
	if err != nil {
		t.Fatal(err)
	}
	layer := weights.Layers[0]
	if layer.AttentionQNorm == nil || layer.AttentionKNorm == nil ||
		layer.FeedForwardRouter == nil || layer.FeedForwardGateExperts == nil ||
		layer.FeedForwardUpExperts == nil || layer.FeedForwardDownExperts == nil {
		t.Fatalf("unexpected RND1 weights: %+v", layer)
	}
}

func TestReadWeightsLLaDAFamilies(t *testing.T) {
	denseSpec := Spec{
		Architecture: "llada", BlockCount: 1, EmbeddingLength: 8, FeedForwardLength: 12,
		HeadCount: 2, HeadCountKV: 1, KeyLength: 4, ValueLength: 4, VocabularySize: 32,
	}
	common := []gguf.TensorInfo{
		tensorInfo("token_embd.weight", 8, 32), tensorInfo("output_norm.weight", 8),
		tensorInfo("blk.0.attn_norm.weight", 8), tensorInfo("blk.0.attn_q.weight", 8, 8),
		tensorInfo("blk.0.attn_k.weight", 8, 4), tensorInfo("blk.0.attn_v.weight", 8, 4),
		tensorInfo("blk.0.attn_output.weight", 8, 8), tensorInfo("blk.0.ffn_norm.weight", 8),
	}
	denseTensors := append(append([]gguf.TensorInfo{}, common...),
		tensorInfo("blk.0.ffn_gate.weight", 8, 12), tensorInfo("blk.0.ffn_up.weight", 8, 12),
		tensorInfo("blk.0.ffn_down.weight", 12, 8),
	)
	dense, err := ReadWeights(&gguf.File{Tensors: denseTensors}, denseSpec)
	if err != nil {
		t.Fatal(err)
	}
	if dense.Output != nil || dense.Layers[0].FeedForwardGate.Name == "" {
		t.Fatalf("unexpected LLaDA catalog: %+v", dense)
	}
	moeSpec := denseSpec
	moeSpec.Architecture = "llada-moe"
	moeSpec.ExpertCount, moeSpec.ExpertUsedCount, moeSpec.ExpertFeedForward = 4, 2, 6
	moeTensors := append(append([]gguf.TensorInfo{}, common...),
		tensorInfo("output.weight", 8, 32),
		tensorInfo("blk.0.attn_q_norm.weight", 4), tensorInfo("blk.0.attn_k_norm.weight", 4),
		tensorInfo("blk.0.ffn_gate_inp.weight", 8, 4),
		tensorInfo("blk.0.ffn_gate_exps.weight", 8, 6, 4),
		tensorInfo("blk.0.ffn_up_exps.weight", 8, 6, 4),
		tensorInfo("blk.0.ffn_down_exps.weight", 6, 8, 4),
	)
	moe, err := ReadWeights(&gguf.File{Tensors: moeTensors}, moeSpec)
	if err != nil {
		t.Fatal(err)
	}
	if moe.Output == nil || moe.Layers[0].AttentionQNorm == nil ||
		moe.Layers[0].AttentionKNorm == nil || moe.Layers[0].FeedForwardRouter == nil {
		t.Fatalf("unexpected LLaDA-MoE catalog: %+v", moe)
	}
}

func TestReadWeightsLagunaDenseThenMoE(t *testing.T) {
	spec := Spec{
		Architecture: "laguna", BlockCount: 2, EmbeddingLength: 8,
		FeedForwardLength: 16, LeadingDenseBlocks: 1,
		ExpertCount: 4, ExpertUsedCount: 2, ExpertFeedForward: 12,
		SharedExpertFF: 10, ExpertWeightsScale: 1,
		HeadCount: 2, HeadCountKV: 1, LayerHeadCounts: []uint32{2, 4},
		LayerKVHeadCounts: []uint32{1, 1}, KeyLength: 4, ValueLength: 4,
		VocabularySize: 32,
	}
	tensors := []gguf.TensorInfo{
		tensorInfo("token_embd.weight", 8, 32), tensorInfo("output_norm.weight", 8),
	}
	for block, heads := range []uint64{2, 4} {
		prefix := fmt.Sprintf("blk.%d.", block)
		tensors = append(tensors,
			tensorInfo(prefix+"attn_norm.weight", 8),
			tensorInfo(prefix+"attn_q.weight", 8, heads*4),
			tensorInfo(prefix+"attn_k.weight", 8, 4),
			tensorInfo(prefix+"attn_v.weight", 8, 4),
			tensorInfo(prefix+"attn_output.weight", heads*4, 8),
			tensorInfo(prefix+"attn_q_norm.weight", 4),
			tensorInfo(prefix+"attn_k_norm.weight", 4),
			tensorInfo(prefix+"attn_gate.weight", 8, heads),
			tensorInfo(prefix+"ffn_norm.weight", 8),
		)
	}
	tensors = append(tensors,
		tensorInfo("blk.0.ffn_gate.weight", 8, 16),
		tensorInfo("blk.0.ffn_up.weight", 8, 16),
		tensorInfo("blk.0.ffn_down.weight", 16, 8),
		tensorInfo("blk.1.ffn_gate_inp.weight", 8, 4),
		tensorInfo("blk.1.ffn_gate_exps.weight", 8, 12, 4),
		tensorInfo("blk.1.ffn_up_exps.weight", 8, 12, 4),
		tensorInfo("blk.1.ffn_down_exps.weight", 12, 8, 4),
		tensorInfo("blk.1.exp_probs_b.bias", 4),
		tensorInfo("blk.1.ffn_gate_shexp.weight", 8, 10),
		tensorInfo("blk.1.ffn_up_shexp.weight", 8, 10),
		tensorInfo("blk.1.ffn_down_shexp.weight", 10, 8),
	)
	weights, err := ReadWeights(&gguf.File{Tensors: tensors}, spec)
	if err != nil {
		t.Fatal(err)
	}
	if weights.Layers[0].FeedForwardRouter != nil || weights.Layers[0].FeedForwardGate.Name == "" ||
		weights.Layers[1].FeedForwardRouter == nil || weights.Layers[1].FeedForwardExpertBias == nil ||
		weights.Layers[1].FeedForwardSharedDown == nil ||
		weights.Layers[1].AttentionQ.Shape[1] != 16 {
		t.Fatalf("unexpected Laguna catalog: %+v", weights.Layers)
	}
}

func TestReadWeightsAFMoE(t *testing.T) {
	spec := Spec{
		Architecture: "afmoe", BlockCount: 1, EmbeddingLength: 8,
		FeedForwardLength: 16, ExpertCount: 4, ExpertUsedCount: 2,
		ExpertFeedForward: 6, SharedExpertCount: 2, SharedExpertFF: 12,
		ExpertWeightsScale: 2.826, HeadCount: 2, HeadCountKV: 1,
		KeyLength: 4, ValueLength: 4, VocabularySize: 32,
	}
	file := &gguf.File{Tensors: []gguf.TensorInfo{
		tensorInfo("token_embd.weight", 8, 32), tensorInfo("output_norm.weight", 8),
		tensorInfo("blk.0.attn_norm.weight", 8), tensorInfo("blk.0.post_attention_norm.weight", 8),
		tensorInfo("blk.0.attn_q.weight", 8, 8), tensorInfo("blk.0.attn_k.weight", 8, 4),
		tensorInfo("blk.0.attn_v.weight", 8, 4), tensorInfo("blk.0.attn_output.weight", 8, 8),
		tensorInfo("blk.0.attn_q_norm.weight", 4), tensorInfo("blk.0.attn_k_norm.weight", 4),
		tensorInfo("blk.0.attn_gate.weight", 8, 8), tensorInfo("blk.0.ffn_norm.weight", 8),
		tensorInfo("blk.0.post_ffw_norm.weight", 8), tensorInfo("blk.0.ffn_gate_inp.weight", 8, 4),
		tensorInfo("blk.0.ffn_gate_exps.weight", 8, 6, 4),
		tensorInfo("blk.0.ffn_up_exps.weight", 8, 6, 4),
		tensorInfo("blk.0.ffn_down_exps.weight", 6, 8, 4),
		tensorInfo("blk.0.exp_probs_b.bias", 4),
		tensorInfo("blk.0.ffn_gate_shexp.weight", 8, 12),
		tensorInfo("blk.0.ffn_up_shexp.weight", 8, 12),
		tensorInfo("blk.0.ffn_down_shexp.weight", 12, 8),
	}}
	weights, err := ReadWeights(file, spec)
	if err != nil {
		t.Fatal(err)
	}
	layer := weights.Layers[0]
	if layer.AttentionPostNorm == nil || layer.FeedForwardPostNorm == nil ||
		layer.AttentionOutputGate == nil || layer.FeedForwardExpertBias == nil ||
		layer.FeedForwardSharedDown == nil {
		t.Fatalf("unexpected AFMoE catalog: %+v", layer)
	}
}

func TestReadWeightsOLMoE(t *testing.T) {
	spec := Spec{
		Architecture: "olmoe", BlockCount: 1, EmbeddingLength: 8,
		FeedForwardLength: 12, ExpertCount: 4, ExpertUsedCount: 2,
		ExpertFeedForward: 12, ExpertWeightsScale: 1,
		HeadCount: 2, HeadCountKV: 2, KeyLength: 4, ValueLength: 4,
		VocabularySize: 32,
	}
	file := &gguf.File{Tensors: []gguf.TensorInfo{
		tensorInfo("token_embd.weight", 8, 32), tensorInfo("output_norm.weight", 8),
		tensorInfo("output.weight", 8, 32), tensorInfo("blk.0.attn_norm.weight", 8),
		tensorInfo("blk.0.attn_q.weight", 8, 8), tensorInfo("blk.0.attn_k.weight", 8, 8),
		tensorInfo("blk.0.attn_v.weight", 8, 8), tensorInfo("blk.0.attn_output.weight", 8, 8),
		tensorInfo("blk.0.attn_q_norm.weight", 8), tensorInfo("blk.0.attn_k_norm.weight", 8),
		tensorInfo("blk.0.ffn_norm.weight", 8), tensorInfo("blk.0.ffn_gate_inp.weight", 8, 4),
		tensorInfo("blk.0.ffn_gate_exps.weight", 8, 12, 4),
		tensorInfo("blk.0.ffn_up_exps.weight", 8, 12, 4),
		tensorInfo("blk.0.ffn_down_exps.weight", 12, 8, 4),
	}}
	weights, err := ReadWeights(file, spec)
	if err != nil {
		t.Fatal(err)
	}
	layer := weights.Layers[0]
	if layer.AttentionQNorm == nil || layer.AttentionQNorm.Shape[0] != 8 ||
		layer.AttentionKNorm == nil || layer.FeedForwardRouter == nil {
		t.Fatalf("unexpected OLMoE catalog: %+v", layer)
	}
}

func TestReadWeightsPhiMoE(t *testing.T) {
	spec := Spec{
		Architecture: "phimoe", BlockCount: 1, ContextLength: 128,
		OriginalContextLength: 32, EmbeddingLength: 8, FeedForwardLength: 12,
		ExpertCount: 4, ExpertUsedCount: 2, ExpertFeedForward: 12, ExpertWeightsScale: 1,
		HeadCount: 2, HeadCountKV: 2, KeyLength: 4, ValueLength: 4,
		RopeDimensionCount: 4, RopeScalingType: "longrope",
		RopeAttentionFactor: 1.1, VocabularySize: 32, RMSNormEpsilon: 1e-5,
	}
	file := &gguf.File{Tensors: []gguf.TensorInfo{
		tensorInfo("token_embd.weight", 8, 32),
		tensorInfo("output_norm.weight", 8), tensorInfo("output_norm.bias", 8),
		tensorInfo("output.weight", 8, 32), tensorInfo("output.bias", 32),
		tensorInfo("blk.0.attn_norm.weight", 8), tensorInfo("blk.0.attn_norm.bias", 8),
		tensorInfo("blk.0.attn_q.weight", 8, 8), tensorInfo("blk.0.attn_k.weight", 8, 8),
		tensorInfo("blk.0.attn_v.weight", 8, 8), tensorInfo("blk.0.attn_output.weight", 8, 8),
		tensorInfo("blk.0.attn_output.bias", 8),
		tensorInfo("blk.0.ffn_norm.weight", 8), tensorInfo("blk.0.ffn_norm.bias", 8),
		tensorInfo("blk.0.ffn_gate_inp.weight", 8, 4),
		tensorInfo("blk.0.ffn_gate_exps.weight", 8, 12, 4),
		tensorInfo("blk.0.ffn_up_exps.weight", 8, 12, 4),
		tensorInfo("blk.0.ffn_down_exps.weight", 12, 8, 4),
		tensorInfo("blk.0.rope_factors_long.weight", 2),
		tensorInfo("blk.0.rope_factors_short.weight", 2),
	}}
	weights, err := ReadWeights(file, spec)
	if err != nil {
		t.Fatal(err)
	}
	layer := weights.Layers[0]
	if weights.Output == nil || weights.OutputNormBias == nil || weights.OutputBias == nil ||
		layer.AttentionNormBias == nil || layer.FeedForwardNormBias == nil ||
		layer.AttentionOutputBias == nil || layer.FeedForwardRouter == nil ||
		layer.RopeFactors == nil || layer.RopeFactors.Name != "blk.0.rope_factors_long.weight" {
		t.Fatalf("unexpected PhiMoE catalog: %+v", weights)
	}
}

func TestReadWeightsEXAOneMoE(t *testing.T) {
	spec := Spec{
		Architecture: "exaone-moe", BlockCount: 1, EmbeddingLength: 8, FeedForwardLength: 16,
		ExpertCount: 4, ExpertUsedCount: 2, ExpertFeedForward: 6, ExpertWeightsScale: 1.5,
		SharedExpertFF: 12, ExpertGatingFunc: 2,
		HeadCount: 2, HeadCountKV: 1, KeyLength: 4, ValueLength: 4, VocabularySize: 32,
	}
	file := &gguf.File{Tensors: []gguf.TensorInfo{
		tensorInfo("token_embd.weight", 8, 32), tensorInfo("output_norm.weight", 8),
		tensorInfo("output.weight", 8, 32), tensorInfo("blk.0.attn_norm.weight", 8),
		tensorInfo("blk.0.attn_q.weight", 8, 8), tensorInfo("blk.0.attn_k.weight", 8, 4),
		tensorInfo("blk.0.attn_v.weight", 8, 4), tensorInfo("blk.0.attn_output.weight", 8, 8),
		tensorInfo("blk.0.attn_q_norm.weight", 4), tensorInfo("blk.0.attn_k_norm.weight", 4),
		tensorInfo("blk.0.ffn_norm.weight", 8), tensorInfo("blk.0.ffn_gate_inp.weight", 8, 4),
		tensorInfo("blk.0.ffn_gate_exps.weight", 8, 6, 4),
		tensorInfo("blk.0.ffn_up_exps.weight", 8, 6, 4),
		tensorInfo("blk.0.ffn_down_exps.weight", 6, 8, 4),
		tensorInfo("blk.0.exp_probs_b.bias", 4),
		tensorInfo("blk.0.ffn_gate_shexp.weight", 8, 12),
		tensorInfo("blk.0.ffn_up_shexp.weight", 8, 12),
		tensorInfo("blk.0.ffn_down_shexp.weight", 12, 8),
	}}
	weights, err := ReadWeights(file, spec)
	if err != nil {
		t.Fatal(err)
	}
	layer := weights.Layers[0]
	if layer.AttentionQNorm == nil || layer.FeedForwardRouter == nil ||
		layer.FeedForwardExpertBias == nil || layer.FeedForwardSharedDown == nil {
		t.Fatalf("unexpected EXAONE-MoE catalog: %+v", layer)
	}
}

func TestReadWeightsChameleon(t *testing.T) {
	spec := Spec{
		Architecture: "chameleon", BlockCount: 1, EmbeddingLength: 8,
		FeedForwardLength: 16, HeadCount: 2, HeadCountKV: 1,
		KeyLength: 4, ValueLength: 4, VocabularySize: 8200,
	}
	file := &gguf.File{Tensors: []gguf.TensorInfo{
		tensorInfo("token_embd.weight", 8, 8200), tensorInfo("output_norm.weight", 8),
		tensorInfo("blk.0.attn_norm.weight", 8), tensorInfo("blk.0.attn_q.weight", 8, 8),
		tensorInfo("blk.0.attn_k.weight", 8, 4), tensorInfo("blk.0.attn_v.weight", 8, 4),
		tensorInfo("blk.0.attn_output.weight", 8, 8),
		tensorInfo("blk.0.attn_q_norm.weight", 4, 2), tensorInfo("blk.0.attn_q_norm.bias", 4, 2),
		tensorInfo("blk.0.attn_k_norm.weight", 4, 1), tensorInfo("blk.0.attn_k_norm.bias", 4, 1),
		tensorInfo("blk.0.ffn_norm.weight", 8), tensorInfo("blk.0.ffn_gate.weight", 8, 16),
		tensorInfo("blk.0.ffn_up.weight", 8, 16), tensorInfo("blk.0.ffn_down.weight", 16, 8),
	}}
	weights, err := ReadWeights(file, spec)
	if err != nil {
		t.Fatal(err)
	}
	layer := weights.Layers[0]
	if layer.AttentionQNorm == nil || layer.AttentionKNorm == nil ||
		layer.AttentionQNormBias == nil || layer.AttentionKNormBias == nil {
		t.Fatalf("unexpected Chameleon weights: %+v", layer)
	}
}

func TestReadWeightsLFM2Hybrid(t *testing.T) {
	spec := Spec{
		Architecture: "lfm2", BlockCount: 2, EmbeddingLength: 8,
		FeedForwardLength: 16, HeadCount: 2, HeadCountKV: 1,
		KeyLength: 4, ValueLength: 4, VocabularySize: 32,
		ShortConvCacheLength: 4, RecurrentLayers: []bool{true, false},
	}
	tensors := []gguf.TensorInfo{
		tensorInfo("token_embd.weight", 8, 32),
		tensorInfo("token_embd_norm.weight", 8),
	}
	for block := 0; block < 2; block++ {
		prefix := fmt.Sprintf("blk.%d.", block)
		tensors = append(tensors,
			tensorInfo(prefix+"attn_norm.weight", 8),
			tensorInfo(prefix+"ffn_norm.weight", 8),
			tensorInfo(prefix+"ffn_gate.weight", 8, 16),
			tensorInfo(prefix+"ffn_up.weight", 8, 16),
			tensorInfo(prefix+"ffn_down.weight", 16, 8),
		)
	}
	tensors = append(tensors,
		tensorInfo("blk.0.shortconv.conv.weight", 4, 8),
		tensorInfo("blk.0.shortconv.in_proj.weight", 8, 24),
		tensorInfo("blk.0.shortconv.out_proj.weight", 8, 8),
		tensorInfo("blk.1.attn_q.weight", 8, 8),
		tensorInfo("blk.1.attn_k.weight", 8, 4),
		tensorInfo("blk.1.attn_v.weight", 8, 4),
		tensorInfo("blk.1.attn_output.weight", 8, 8),
		tensorInfo("blk.1.attn_q_norm.weight", 4),
		tensorInfo("blk.1.attn_k_norm.weight", 4),
	)
	weights, err := ReadWeights(&gguf.File{Tensors: tensors}, spec)
	if err != nil {
		t.Fatal(err)
	}
	if !weights.Layers[0].Recurrent || weights.Layers[0].ShortConvKernel == nil ||
		weights.Layers[1].Recurrent || weights.Layers[1].AttentionQNorm == nil {
		t.Fatalf("unexpected LFM2 weights: %+v", weights.Layers)
	}
}

func TestReadWeightsLFM2MoEDenseThenHybridMoE(t *testing.T) {
	spec := Spec{
		Architecture: "lfm2moe", BlockCount: 3, LeadingDenseBlocks: 1,
		EmbeddingLength: 8, FeedForwardLength: 16,
		ExpertCount: 4, ExpertUsedCount: 2, ExpertFeedForward: 6,
		ExpertWeightsScale: 1.25, ExpertGatingFunc: 2,
		HeadCount: 2, HeadCountKV: 1, KeyLength: 4, ValueLength: 4, VocabularySize: 32,
		ShortConvCacheLength: 4, RecurrentLayers: []bool{true, false, true},
	}
	tensors := []gguf.TensorInfo{
		tensorInfo("token_embd.weight", 8, 32), tensorInfo("token_embd_norm.weight", 8),
	}
	for block := 0; block < 3; block++ {
		prefix := fmt.Sprintf("blk.%d.", block)
		tensors = append(tensors,
			tensorInfo(prefix+"attn_norm.weight", 8),
			tensorInfo(prefix+"ffn_norm.weight", 8),
		)
		if block == 0 {
			tensors = append(tensors,
				tensorInfo(prefix+"ffn_gate.weight", 8, 16),
				tensorInfo(prefix+"ffn_up.weight", 8, 16),
				tensorInfo(prefix+"ffn_down.weight", 16, 8),
			)
		} else {
			tensors = append(tensors,
				tensorInfo(prefix+"ffn_gate_inp.weight", 8, 4),
				tensorInfo(prefix+"ffn_gate_exps.weight", 8, 6, 4),
				tensorInfo(prefix+"ffn_up_exps.weight", 8, 6, 4),
				tensorInfo(prefix+"ffn_down_exps.weight", 6, 8, 4),
				tensorInfo(prefix+"exp_probs_b.bias", 4),
			)
		}
		if spec.RecurrentLayers[block] {
			tensors = append(tensors,
				tensorInfo(prefix+"shortconv.conv.weight", 4, 8),
				tensorInfo(prefix+"shortconv.in_proj.weight", 8, 24),
				tensorInfo(prefix+"shortconv.out_proj.weight", 8, 8),
			)
		} else {
			tensors = append(tensors,
				tensorInfo(prefix+"attn_q.weight", 8, 8),
				tensorInfo(prefix+"attn_k.weight", 8, 4),
				tensorInfo(prefix+"attn_v.weight", 8, 4),
				tensorInfo(prefix+"attn_output.weight", 8, 8),
				tensorInfo(prefix+"attn_q_norm.weight", 4),
				tensorInfo(prefix+"attn_k_norm.weight", 4),
			)
		}
	}
	weights, err := ReadWeights(&gguf.File{Tensors: tensors}, spec)
	if err != nil {
		t.Fatal(err)
	}
	if !weights.Layers[0].Recurrent || weights.Layers[0].FeedForwardRouter != nil ||
		weights.Layers[1].Recurrent || weights.Layers[1].AttentionQNorm == nil ||
		weights.Layers[1].FeedForwardRouter == nil || weights.Layers[1].FeedForwardExpertBias == nil ||
		!weights.Layers[2].Recurrent || weights.Layers[2].ShortConvKernel == nil ||
		weights.Layers[2].FeedForwardRouter == nil || weights.Layers[2].FeedForwardExpertBias == nil {
		t.Fatalf("unexpected LFM2-MoE weights: %+v", weights.Layers)
	}
}

func TestReadWeightsPLMMLA(t *testing.T) {
	spec := Spec{
		Architecture: "plm", BlockCount: 1, EmbeddingLength: 8,
		FeedForwardLength: 16, HeadCount: 2, HeadCountKV: 2,
		KeyLength: 6, ValueLength: 4, VocabularySize: 32,
		KVLoRARank: 3, RopeDimensionCount: 2,
	}
	file := &gguf.File{Tensors: []gguf.TensorInfo{
		tensorInfo("token_embd.weight", 8, 32), tensorInfo("output_norm.weight", 8),
		tensorInfo("blk.0.attn_norm.weight", 8), tensorInfo("blk.0.attn_q.weight", 8, 12),
		tensorInfo("blk.0.attn_kv_a_mqa.weight", 8, 5),
		tensorInfo("blk.0.attn_kv_a_norm.weight", 3),
		tensorInfo("blk.0.attn_kv_b.weight", 3, 16),
		tensorInfo("blk.0.attn_output.weight", 8, 8),
		tensorInfo("blk.0.ffn_norm.weight", 8), tensorInfo("blk.0.ffn_up.weight", 8, 16),
		tensorInfo("blk.0.ffn_down.weight", 16, 8),
	}}
	weights, err := ReadWeights(file, spec)
	if err != nil {
		t.Fatal(err)
	}
	if weights.Layers[0].AttentionKVAMQA == nil || weights.Layers[0].AttentionKVANorm == nil ||
		weights.Layers[0].AttentionKVB == nil || weights.Layers[0].FeedForwardGate.Name != "" {
		t.Fatalf("unexpected PLM weights: %+v", weights.Layers[0])
	}
}

func TestReadWeightsMiniCPM3(t *testing.T) {
	spec := Spec{Architecture: "minicpm3", BlockCount: 1, ContextLength: 4096, OriginalContextLength: 4096,
		EmbeddingLength: 8, FeedForwardLength: 12, HeadCount: 2, HeadCountKV: 2,
		KeyLength: 6, ValueLength: 4, QLoRARank: 3, KVLoRARank: 3,
		RopeDimensionCount: 2, VocabularySize: 32}
	tensors := []gguf.TensorInfo{
		tensorInfo("token_embd.weight", 8, 32), tensorInfo("output_norm.weight", 8),
		tensorInfo("blk.0.attn_norm.weight", 8), tensorInfo("blk.0.attn_q_a.weight", 8, 3),
		tensorInfo("blk.0.attn_q_a_norm.weight", 3), tensorInfo("blk.0.attn_q_b.weight", 3, 12),
		tensorInfo("blk.0.attn_kv_a_mqa.weight", 8, 5), tensorInfo("blk.0.attn_kv_a_norm.weight", 3),
		tensorInfo("blk.0.attn_kv_b.weight", 3, 16), tensorInfo("blk.0.attn_output.weight", 8, 8),
		tensorInfo("blk.0.ffn_norm.weight", 8), tensorInfo("blk.0.ffn_gate.weight", 8, 12),
		tensorInfo("blk.0.ffn_up.weight", 8, 12), tensorInfo("blk.0.ffn_down.weight", 12, 8),
		tensorInfo("blk.0.rope_factors_long.weight", 1), tensorInfo("blk.0.rope_factors_short.weight", 1),
	}
	weights, err := ReadWeights(&gguf.File{Tensors: tensors}, spec)
	if err != nil {
		t.Fatal(err)
	}
	layer := weights.Layers[0]
	if weights.Output != nil || layer.AttentionQB == nil || layer.AttentionQNorm == nil ||
		layer.AttentionKVAMQA == nil || layer.RopeFactors == nil ||
		layer.RopeFactors.Name != "blk.0.rope_factors_short.weight" {
		t.Fatalf("unexpected MiniCPM3 catalog: %+v", weights)
	}
}

func TestReadWeightsDeepSeek2AbsorbedMLA(t *testing.T) {
	testReadWeightsDeepSeek2FamilyAbsorbedMLA(t, "deepseek2")
}

func TestReadWeightsMistral4AbsorbedMLA(t *testing.T) {
	testReadWeightsDeepSeek2FamilyAbsorbedMLA(t, "mistral4")
}

func TestReadWeightsMamba(t *testing.T) {
	spec := Spec{Architecture: "mamba", BlockCount: 1, EmbeddingLength: 4,
		SSMConvKernel: 3, SSMInnerSize: 8, SSMStateSize: 2, SSMTimeStepRank: 2,
		VocabularySize: 32}
	file := &gguf.File{Tensors: []gguf.TensorInfo{
		tensorInfo("token_embd.weight", 4, 32), tensorInfo("output_norm.weight", 4),
		tensorInfo("blk.0.attn_norm.weight", 4), tensorInfo("blk.0.ssm_in.weight", 4, 16),
		tensorInfo("blk.0.ssm_conv1d.weight", 3, 8), tensorInfo("blk.0.ssm_conv1d.bias", 8),
		tensorInfo("blk.0.ssm_x.weight", 8, 6), tensorInfo("blk.0.ssm_dt.weight", 2, 8),
		tensorInfo("blk.0.ssm_dt.bias", 8), tensorInfo("blk.0.ssm_a", 2, 8),
		tensorInfo("blk.0.ssm_d", 8), tensorInfo("blk.0.ssm_out.weight", 8, 4),
	}}
	weights, err := ReadWeights(file, spec)
	if err != nil {
		t.Fatal(err)
	}
	layer := weights.Layers[0]
	if !layer.Recurrent || layer.SSMInput == nil || layer.SSMConv1D == nil ||
		layer.SSMConv1DBias == nil || layer.SSMX == nil || layer.SSMTimeStepWeight == nil ||
		layer.SSMTimeStep == nil || layer.SSMA == nil || layer.SSMD == nil || layer.SSMOutput == nil ||
		layer.AttentionQ.Name != "" || layer.FeedForwardNorm.Name != "" {
		t.Fatalf("unexpected Mamba catalog: %+v", layer)
	}
}

func TestReadWeightsMamba2(t *testing.T) {
	spec := Spec{Architecture: "mamba2", BlockCount: 1, EmbeddingLength: 4,
		SSMConvKernel: 3, SSMInnerSize: 8, SSMStateSize: 2, SSMTimeStepRank: 4,
		SSMGroupCount: 2, VocabularySize: 32}
	file := &gguf.File{Tensors: []gguf.TensorInfo{
		tensorInfo("token_embd.weight", 4, 32), tensorInfo("output_norm.weight", 4),
		tensorInfo("blk.0.attn_norm.weight", 4), tensorInfo("blk.0.ssm_in.weight", 4, 28),
		tensorInfo("blk.0.ssm_conv1d.weight", 3, 16), tensorInfo("blk.0.ssm_conv1d.bias", 16),
		tensorInfo("blk.0.ssm_dt.bias", 4), tensorInfo("blk.0.ssm_a", 1, 4),
		tensorInfo("blk.0.ssm_d", 1, 4), tensorInfo("blk.0.ssm_norm.weight", 4, 2),
		tensorInfo("blk.0.ssm_out.weight", 8, 4),
	}}
	weights, err := ReadWeights(file, spec)
	if err != nil {
		t.Fatal(err)
	}
	layer := weights.Layers[0]
	if !layer.Recurrent || layer.SSMInput == nil || layer.SSMConv1D == nil ||
		layer.SSMConv1DBias == nil || layer.SSMTimeStep == nil || layer.SSMA == nil ||
		layer.SSMD == nil || layer.SSMNorm == nil || layer.SSMOutput == nil ||
		layer.SSMX != nil || layer.SSMTimeStepWeight != nil || layer.AttentionQ.Name != "" ||
		layer.FeedForwardNorm.Name != "" {
		t.Fatalf("unexpected Mamba2 catalog: %+v", layer)
	}
}

func TestReadWeightsJamba(t *testing.T) {
	spec := Spec{Architecture: "jamba", BlockCount: 2, EmbeddingLength: 4,
		FeedForwardLength: 6, HeadCount: 2, HeadCountKV: 1, KeyLength: 2, ValueLength: 2,
		LayerKVHeadCounts: []uint32{0, 1}, RecurrentLayers: []bool{true, false},
		SSMConvKernel: 3, SSMInnerSize: 8, SSMStateSize: 2, SSMTimeStepRank: 2,
		SSMGroupCount: 1, ExpertCount: 4, ExpertUsedCount: 2, ExpertFeedForward: 6,
		VocabularySize: 32}
	file := &gguf.File{Tensors: []gguf.TensorInfo{
		tensorInfo("token_embd.weight", 4, 32), tensorInfo("output_norm.weight", 4),
		tensorInfo("blk.0.attn_norm.weight", 4), tensorInfo("blk.0.ssm_in.weight", 4, 16),
		tensorInfo("blk.0.ssm_conv1d.weight", 3, 8), tensorInfo("blk.0.ssm_conv1d.bias", 8),
		tensorInfo("blk.0.ssm_x.weight", 8, 6), tensorInfo("blk.0.ssm_dt_norm.weight", 2),
		tensorInfo("blk.0.ssm_dt.weight", 2, 8), tensorInfo("blk.0.ssm_dt.bias", 8),
		tensorInfo("blk.0.ssm_b_norm.weight", 2), tensorInfo("blk.0.ssm_c_norm.weight", 2),
		tensorInfo("blk.0.ssm_a", 2, 8), tensorInfo("blk.0.ssm_d", 8),
		tensorInfo("blk.0.ssm_out.weight", 8, 4), tensorInfo("blk.0.ffn_norm.weight", 4),
		tensorInfo("blk.0.ffn_gate.weight", 4, 6), tensorInfo("blk.0.ffn_up.weight", 4, 6),
		tensorInfo("blk.0.ffn_down.weight", 6, 4), tensorInfo("blk.1.attn_norm.weight", 4),
		tensorInfo("blk.1.attn_q.weight", 4, 4), tensorInfo("blk.1.attn_k.weight", 4, 2),
		tensorInfo("blk.1.attn_v.weight", 4, 2), tensorInfo("blk.1.attn_output.weight", 4, 4),
		tensorInfo("blk.1.ffn_norm.weight", 4), tensorInfo("blk.1.ffn_gate_inp.weight", 4, 4),
		tensorInfo("blk.1.ffn_gate_exps.weight", 4, 6, 4), tensorInfo("blk.1.ffn_up_exps.weight", 4, 6, 4),
		tensorInfo("blk.1.ffn_down_exps.weight", 6, 4, 4),
	}}
	weights, err := ReadWeights(file, spec)
	if err != nil {
		t.Fatal(err)
	}
	if !weights.Layers[0].Recurrent || weights.Layers[0].SSMTimeStepNorm == nil ||
		weights.Layers[0].SSMBNorm == nil || weights.Layers[0].SSMCNorm == nil ||
		weights.Layers[0].FeedForwardGate.Name == "" || weights.Layers[1].Recurrent ||
		weights.Layers[1].AttentionQ.Name == "" || weights.Layers[1].FeedForwardRouter == nil {
		t.Fatalf("unexpected Jamba catalog: %+v", weights.Layers)
	}
}

func TestReadWeightsGraniteHybrid(t *testing.T) {
	spec := Spec{Architecture: "granitehybrid", BlockCount: 2, EmbeddingLength: 4,
		FeedForwardLength: 6, HeadCount: 2, HeadCountKV: 1, KeyLength: 2, ValueLength: 2,
		LayerKVHeadCounts: []uint32{0, 1}, RecurrentLayers: []bool{true, false},
		SSMConvKernel: 3, SSMInnerSize: 8, SSMStateSize: 2, SSMTimeStepRank: 4,
		SSMGroupCount: 2, ExpertCount: 4, ExpertUsedCount: 2, ExpertFeedForward: 6,
		SharedExpertFF: 5, VocabularySize: 32}
	file := &gguf.File{Tensors: []gguf.TensorInfo{
		tensorInfo("token_embd.weight", 4, 32), tensorInfo("output_norm.weight", 4),
		tensorInfo("blk.0.attn_norm.weight", 4), tensorInfo("blk.0.ssm_in.weight", 4, 28),
		tensorInfo("blk.0.ssm_conv1d.weight", 3, 16), tensorInfo("blk.0.ssm_dt.bias", 4),
		tensorInfo("blk.0.ssm_a", 1, 4), tensorInfo("blk.0.ssm_d", 1, 4),
		tensorInfo("blk.0.ssm_norm.weight", 4, 2), tensorInfo("blk.0.ssm_out.weight", 8, 4),
		tensorInfo("blk.0.ffn_norm.weight", 4), tensorInfo("blk.0.ffn_gate_inp.weight", 4, 4),
		tensorInfo("blk.0.ffn_up_exps.weight", 4, 6, 4), tensorInfo("blk.0.ffn_down_exps.weight", 6, 4, 4),
		tensorInfo("blk.0.ffn_gate_shexp.weight", 4, 5), tensorInfo("blk.0.ffn_up_shexp.weight", 4, 5),
		tensorInfo("blk.0.ffn_down_shexp.weight", 5, 4),
		tensorInfo("blk.1.attn_norm.weight", 4), tensorInfo("blk.1.attn_q.weight", 4, 4),
		tensorInfo("blk.1.attn_k.weight", 4, 2), tensorInfo("blk.1.attn_v.weight", 4, 2),
		tensorInfo("blk.1.attn_output.weight", 4, 4), tensorInfo("blk.1.attn_output.bias", 4),
		tensorInfo("blk.1.ffn_norm.weight", 4), tensorInfo("blk.1.ffn_gate_inp.weight", 4, 4),
		tensorInfo("blk.1.ffn_gate_exps.weight", 4, 6, 4), tensorInfo("blk.1.ffn_up_exps.weight", 4, 6, 4),
		tensorInfo("blk.1.ffn_down_exps.weight", 6, 4, 4), tensorInfo("blk.1.ffn_gate_shexp.weight", 4, 5),
		tensorInfo("blk.1.ffn_up_shexp.weight", 4, 5), tensorInfo("blk.1.ffn_down_shexp.weight", 5, 4),
	}}
	weights, err := ReadWeights(file, spec)
	if err != nil {
		t.Fatal(err)
	}
	recurrent, attention := weights.Layers[0], weights.Layers[1]
	if !recurrent.Recurrent || recurrent.SSMInput == nil || recurrent.SSMConv1DBias != nil ||
		recurrent.SSMNorm == nil || recurrent.FeedForwardRouter == nil ||
		recurrent.FeedForwardGateExperts != nil || recurrent.FeedForwardSharedDown == nil ||
		attention.Recurrent || attention.AttentionQ.Name == "" || attention.AttentionOutputBias == nil ||
		attention.FeedForwardGateExperts == nil || attention.FeedForwardSharedDown == nil {
		t.Fatalf("unexpected Granite Hybrid catalog: %+v", weights.Layers)
	}
}

func TestReadWeightsPLaMo2(t *testing.T) {
	spec := Spec{Architecture: "plamo2", BlockCount: 2, EmbeddingLength: 4,
		FeedForwardLength: 6, HeadCount: 2, HeadCountKV: 1, KeyLength: 2, ValueLength: 2,
		LayerKVHeadCounts: []uint32{0, 1}, RecurrentLayers: []bool{true, false},
		SSMConvKernel: 3, SSMInnerSize: 8, SSMStateSize: 2, SSMTimeStepRank: 4,
		SSMGroupCount: 0, VocabularySize: 32}
	file := &gguf.File{Tensors: []gguf.TensorInfo{
		tensorInfo("token_embd.weight", 4, 32), tensorInfo("output_norm.weight", 4),
		tensorInfo("blk.0.attn_norm.weight", 4), tensorInfo("blk.0.ssm_in.weight", 4, 16),
		tensorInfo("blk.0.ssm_conv1d.weight", 3, 8), tensorInfo("blk.0.ssm_x.weight", 8, 68),
		tensorInfo("blk.0.ssm_dt.weight", 64, 4), tensorInfo("blk.0.ssm_dt.bias", 4),
		tensorInfo("blk.0.ssm_a", 4), tensorInfo("blk.0.ssm_d", 4),
		tensorInfo("blk.0.ssm_out.weight", 8, 4), tensorInfo("blk.0.ssm_dt_norm.weight", 64),
		tensorInfo("blk.0.ssm_b_norm.weight", 2), tensorInfo("blk.0.ssm_c_norm.weight", 2),
		tensorInfo("blk.0.post_attention_norm.weight", 4), tensorInfo("blk.0.ffn_norm.weight", 4),
		tensorInfo("blk.0.ffn_up.weight", 4, 12), tensorInfo("blk.0.ffn_down.weight", 6, 4),
		tensorInfo("blk.0.post_ffw_norm.weight", 4),
		tensorInfo("blk.1.attn_norm.weight", 4), tensorInfo("blk.1.attn_qkv.weight", 4, 8),
		tensorInfo("blk.1.attn_q_norm.weight", 2, 2), tensorInfo("blk.1.attn_k_norm.weight", 2, 1),
		tensorInfo("blk.1.attn_output.weight", 4, 4), tensorInfo("blk.1.post_attention_norm.weight", 4),
		tensorInfo("blk.1.ffn_norm.weight", 4), tensorInfo("blk.1.ffn_up.weight", 4, 12),
		tensorInfo("blk.1.ffn_down.weight", 6, 4), tensorInfo("blk.1.post_ffw_norm.weight", 4),
	}}
	weights, err := ReadWeights(file, spec)
	if err != nil {
		t.Fatal(err)
	}
	recurrent, attention := weights.Layers[0], weights.Layers[1]
	if !recurrent.Recurrent || recurrent.SSMInput == nil || recurrent.SSMConv1DBias != nil ||
		recurrent.SSMX == nil || recurrent.SSMTimeStepNorm == nil || recurrent.SSMBNorm == nil ||
		recurrent.SSMCNorm == nil || recurrent.AttentionPostNorm == nil || recurrent.FeedForwardPostNorm == nil ||
		recurrent.FeedForwardGate.Name != "" || recurrent.FeedForwardUp.Shape[1] != 12 ||
		attention.Recurrent || attention.AttentionQKV == nil || attention.AttentionQNorm == nil ||
		attention.AttentionKNorm == nil || attention.FeedForwardUp.Shape[1] != 12 {
		t.Fatalf("unexpected PLaMo2 catalog: %+v", weights.Layers)
	}
}

func testReadWeightsDeepSeek2FamilyAbsorbedMLA(t *testing.T, architecture string) {
	spec := Spec{Architecture: architecture, BlockCount: 2, EmbeddingLength: 8,
		FeedForwardLength: 12, HeadCount: 2, HeadCountKV: 2, KeyLength: 6, ValueLength: 4,
		QLoRARank: 3, KVLoRARank: 3, RopeDimensionCount: 2, VocabularySize: 32,
		LeadingDenseBlocks: 1, ExpertCount: 4, ExpertUsedCount: 2, ExpertFeedForward: 6,
		SharedExpertCount: 1, SharedExpertFF: 6}
	tensors := []gguf.TensorInfo{
		tensorInfo("token_embd.weight", 8, 32), tensorInfo("output_norm.weight", 8),
	}
	for block := range uint32(2) {
		prefix := fmt.Sprintf("blk.%d.", block)
		tensors = append(tensors,
			tensorInfo(prefix+"attn_norm.weight", 8), tensorInfo(prefix+"attn_q_a.weight", 8, 3),
			tensorInfo(prefix+"attn_q_a_norm.weight", 3), tensorInfo(prefix+"attn_q_b.weight", 3, 12),
			tensorInfo(prefix+"attn_kv_a_mqa.weight", 8, 5), tensorInfo(prefix+"attn_kv_a_norm.weight", 3),
			tensorInfo(prefix+"attn_k_b.weight", 4, 3, 2), tensorInfo(prefix+"attn_v_b.weight", 3, 4, 2),
			tensorInfo(prefix+"attn_output.weight", 8, 8), tensorInfo(prefix+"ffn_norm.weight", 8),
		)
	}
	tensors = append(tensors,
		tensorInfo("blk.0.ffn_gate.weight", 8, 12), tensorInfo("blk.0.ffn_up.weight", 8, 12),
		tensorInfo("blk.0.ffn_down.weight", 12, 8), tensorInfo("blk.1.ffn_gate_inp.weight", 8, 4),
		tensorInfo("blk.1.ffn_gate_exps.weight", 8, 6, 4), tensorInfo("blk.1.ffn_up_exps.weight", 8, 6, 4),
		tensorInfo("blk.1.ffn_down_exps.weight", 6, 8, 4), tensorInfo("blk.1.exp_probs_b.bias", 4),
		tensorInfo("blk.1.ffn_gate_shexp.weight", 8, 6), tensorInfo("blk.1.ffn_up_shexp.weight", 8, 6),
		tensorInfo("blk.1.ffn_down_shexp.weight", 6, 8),
	)
	weights, err := ReadWeights(&gguf.File{Tensors: tensors}, spec)
	if err != nil {
		t.Fatal(err)
	}
	if weights.Layers[0].AttentionKB == nil || weights.Layers[0].AttentionVB == nil ||
		weights.Layers[0].AttentionKVB != nil || weights.Layers[0].FeedForwardRouter != nil ||
		weights.Layers[1].FeedForwardRouter == nil || weights.Layers[1].FeedForwardExpertBias == nil ||
		weights.Layers[1].FeedForwardSharedDown == nil {
		t.Fatalf("unexpected %s catalog: %+v", architecture, weights.Layers)
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

func TestReadWeightsGPT2AndStarCoder(t *testing.T) {
	for _, architecture := range []string{"gpt2", "starcoder"} {
		t.Run(architecture, func(t *testing.T) {
			spec := Spec{
				Architecture: architecture, BlockCount: 1, ContextLength: 16,
				EmbeddingLength: 8, FeedForwardLength: 16, HeadCount: 2,
				HeadCountKV: 2, KeyLength: 4, ValueLength: 4, VocabularySize: 32,
				LayerNormEpsilon: 1e-5, RopeDisabled: true,
			}
			file := &gguf.File{Tensors: []gguf.TensorInfo{
				tensorInfo("token_embd.weight", 8, 32),
				tensorInfo("position_embd.weight", 8, 16),
				tensorInfo("output_norm.weight", 8), tensorInfo("output_norm.bias", 8),
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
			if weights.PositionEmbedding == nil || layer.AttentionQKV == nil ||
				layer.AttentionQKVBias == nil || layer.FeedForwardGate.Name != "" {
				t.Fatalf("unexpected %s weights: %+v", architecture, weights)
			}
		})
	}
}

func TestReadWeightsBloom(t *testing.T) {
	spec := Spec{
		Architecture: "bloom", BlockCount: 1, ContextLength: 16,
		EmbeddingLength: 8, FeedForwardLength: 16, HeadCount: 2,
		HeadCountKV: 2, KeyLength: 4, ValueLength: 4, VocabularySize: 32,
		LayerNormEpsilon: 1e-5, RopeDisabled: true, MaxALiBiBias: 8,
	}
	file := &gguf.File{Tensors: []gguf.TensorInfo{
		tensorInfo("token_embd.weight", 8, 32),
		tensorInfo("token_embd_norm.weight", 8), tensorInfo("token_embd_norm.bias", 8),
		tensorInfo("output_norm.weight", 8), tensorInfo("output_norm.bias", 8),
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
	if weights.TokenEmbeddingNorm == nil || weights.TokenEmbeddingNormBias == nil ||
		weights.Layers[0].AttentionQKV == nil || weights.Layers[0].AttentionQKVBias == nil {
		t.Fatalf("unexpected Bloom weights: %+v", weights)
	}
}

func TestReadWeightsMPTBiasFreeVariant(t *testing.T) {
	spec := Spec{
		Architecture: "mpt", BlockCount: 1, ContextLength: 16,
		EmbeddingLength: 8, FeedForwardLength: 16, HeadCount: 2,
		HeadCountKV: 2, KeyLength: 4, ValueLength: 4, VocabularySize: 32,
		LayerNormEpsilon: 1e-5, RopeDisabled: true, MaxALiBiBias: 8,
	}
	file := &gguf.File{Tensors: []gguf.TensorInfo{
		tensorInfo("token_embd.weight", 8, 32), tensorInfo("position_embd.weight", 8, 16),
		tensorInfo("output_norm.weight", 8),
		tensorInfo("blk.0.attn_norm.weight", 8),
		tensorInfo("blk.0.attn_qkv.weight", 8, 24),
		tensorInfo("blk.0.attn_output.weight", 8, 8),
		tensorInfo("blk.0.ffn_norm.weight", 8),
		tensorInfo("blk.0.ffn_up.weight", 8, 16),
		tensorInfo("blk.0.ffn_down.weight", 16, 8),
	}}
	weights, err := ReadWeights(file, spec)
	if err != nil {
		t.Fatal(err)
	}
	if weights.PositionEmbedding == nil || weights.OutputNormBias != nil ||
		weights.Layers[0].AttentionQKV == nil || weights.Layers[0].AttentionNormBias != nil {
		t.Fatalf("unexpected MPT weights: %+v", weights)
	}
}

func TestReadWeightsDenseRefact(t *testing.T) {
	spec := Spec{
		Architecture: "refact", BlockCount: 1, ContextLength: 16,
		EmbeddingLength: 8, FeedForwardLength: 16, HeadCount: 2,
		HeadCountKV: 1, KeyLength: 4, ValueLength: 4, VocabularySize: 32,
		RMSNormEpsilon: 1e-5, RopeDisabled: true, MaxALiBiBias: 8,
	}
	file := &gguf.File{Tensors: []gguf.TensorInfo{
		tensorInfo("token_embd.weight", 8, 32), tensorInfo("output_norm.weight", 8),
		tensorInfo("blk.0.attn_norm.weight", 8),
		tensorInfo("blk.0.attn_q.weight", 8, 8), tensorInfo("blk.0.attn_k.weight", 8, 4),
		tensorInfo("blk.0.attn_v.weight", 8, 4), tensorInfo("blk.0.attn_output.weight", 8, 8),
		tensorInfo("blk.0.ffn_norm.weight", 8), tensorInfo("blk.0.ffn_gate.weight", 8, 16),
		tensorInfo("blk.0.ffn_up.weight", 8, 16), tensorInfo("blk.0.ffn_down.weight", 16, 8),
	}}
	weights, err := ReadWeights(file, spec)
	if err != nil {
		t.Fatal(err)
	}
	if len(weights.Layers) != 1 || weights.Layers[0].FeedForwardGate.Name == "" {
		t.Fatalf("unexpected Refact weights: %+v", weights)
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
		Architecture:          "granite",
		BlockCount:            1,
		ContextLength:         4,
		OriginalContextLength: 2,
		EmbeddingLength:       8,
		FeedForwardLength:     16,
		HeadCount:             2,
		HeadCountKV:           1,
		KeyLength:             4,
		ValueLength:           4,
		RopeDimensionCount:    4,
		RopeScalingType:       "longrope",
		VocabularySize:        32,
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

func TestReadWeightsJais(t *testing.T) {
	spec := Spec{
		Architecture: "jais", BlockCount: 1, EmbeddingLength: 8,
		FeedForwardLength: 12, HeadCount: 2, HeadCountKV: 2,
		KeyLength: 4, ValueLength: 4, VocabularySize: 32, LayerNormEpsilon: 1e-5,
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

func TestReadWeightsCohere2MoEDensePrefixAndFusedExperts(t *testing.T) {
	spec := Spec{
		Architecture: "cohere2moe", BlockCount: 2, EmbeddingLength: 8,
		FeedForwardLength: 12, HeadCount: 2, HeadCountKV: 1,
		KeyLength: 4, ValueLength: 4, VocabularySize: 32,
		RMSNormEpsilon: 1e-5, LeadingDenseBlocks: 1,
		ExpertCount: 4, ExpertUsedCount: 2, ExpertFeedForward: 6,
		SharedExpertCount: 1, SharedExpertFF: 6,
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

func TestReadWeightsHYV3DetectsDenseAndFusedMoELayers(t *testing.T) {
	spec := Spec{
		Architecture: "hy_v3", BlockCount: 2, EmbeddingLength: 8,
		FeedForwardLength: 12, HeadCount: 2, HeadCountKV: 1,
		KeyLength: 4, ValueLength: 4, RopeDimensionCount: 4,
		VocabularySize: 32, RMSNormEpsilon: 1e-5,
		ExpertCount: 4, ExpertUsedCount: 2, ExpertFeedForward: 6,
		SharedExpertFF: 6, ExpertGatingFunc: 2,
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

func TestReadWeightsDeepSeek2OCRDensePrefixAndFusedExperts(t *testing.T) {
	spec := Spec{
		Architecture: "deepseek2-ocr", BlockCount: 2, EmbeddingLength: 8,
		FeedForwardLength: 12, HeadCount: 2, HeadCountKV: 2,
		KeyLength: 4, ValueLength: 4, RopeDimensionCount: 4,
		VocabularySize: 32, RMSNormEpsilon: 1e-6, LeadingDenseBlocks: 1,
		ExpertCount: 4, ExpertUsedCount: 2, ExpertFeedForward: 6,
		SharedExpertCount: 2, SharedExpertFF: 12, ExpertGatingFunc: 1,
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

func TestReadWeightsPLaMo3PerLayerWidths(t *testing.T) {
	spec := Spec{
		Architecture: "plamo3", BlockCount: 2, EmbeddingLength: 8,
		FeedForwardLength: 12, HeadCount: 2, HeadCountKV: 1,
		KeyLength: 2, ValueLength: 2, VocabularySize: 32, RMSNormEpsilon: 1e-6,
		LayerHeadCounts: []uint32{2, 4}, LayerKVHeadCounts: []uint32{1, 2},
		LayerFeedForward: []uint32{12, 16},
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

func TestReadWeightsApertus(t *testing.T) {
	spec := Spec{
		Architecture: "apertus", BlockCount: 1, ContextLength: 128,
		OriginalContextLength: 32, EmbeddingLength: 8, FeedForwardLength: 16,
		HeadCount: 2, HeadCountKV: 1, KeyLength: 4, ValueLength: 4,
		RopeDimensionCount: 4, RopeScalingType: "longrope",
		RopeAttentionFactor: 1.1, VocabularySize: 32, RMSNormEpsilon: 1e-5,
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

func TestReadWeightsGLM4(t *testing.T) {
	spec := Spec{
		Architecture: "glm4", BlockCount: 1, EmbeddingLength: 8,
		FeedForwardLength: 12, HeadCount: 2, HeadCountKV: 1,
		KeyLength: 4, ValueLength: 4, RopeDimensionCount: 4,
		VocabularySize: 32, RMSNormEpsilon: 1e-5,
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
	spec := Spec{
		Architecture: "exaone4", BlockCount: 1, EmbeddingLength: 8,
		FeedForwardLength: 12, HeadCount: 2, HeadCountKV: 1,
		KeyLength: 4, ValueLength: 4, RopeDimensionCount: 4,
		VocabularySize: 32, RMSNormEpsilon: 1e-5,
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

func TestReadWeightsQwen35MoEAttention(t *testing.T) {
	spec := Spec{
		Architecture: "qwen35moe", BlockCount: 1, EmbeddingLength: 8,
		FeedForwardLength: 12, HeadCount: 2, HeadCountKV: 1, KeyLength: 4,
		ValueLength: 4, VocabularySize: 32, ExpertCount: 4, ExpertUsedCount: 2,
		ExpertFeedForward: 6, SharedExpertFF: 10, ExpertWeightsScale: 1.25,
		SSMConvKernel: 3, SSMInnerSize: 4, SSMStateSize: 2, SSMTimeStepRank: 2,
		SSMGroupCount: 1, RecurrentLayers: []bool{false},
	}
	file := &gguf.File{Tensors: []gguf.TensorInfo{
		tensorInfo("token_embd.weight", 8, 32),
		tensorInfo("output_norm.weight", 8),
		tensorInfo("blk.0.attn_norm.weight", 8),
		tensorInfo("blk.0.post_attention_norm.weight", 8),
		tensorInfo("blk.0.attn_q.weight", 8, 16),
		tensorInfo("blk.0.attn_k.weight", 8, 4),
		tensorInfo("blk.0.attn_v.weight", 8, 4),
		tensorInfo("blk.0.attn_output.weight", 8, 8),
		tensorInfo("blk.0.attn_q_norm.weight", 4),
		tensorInfo("blk.0.attn_k_norm.weight", 4),
		tensorInfo("blk.0.ffn_gate_inp.weight", 8, 4),
		tensorInfo("blk.0.ffn_gate_up_exps.weight", 8, 12, 4),
		tensorInfo("blk.0.ffn_down_exps.weight", 6, 8, 4),
		tensorInfo("blk.0.ffn_gate_inp_shexp.weight", 8),
		tensorInfo("blk.0.ffn_gate_shexp.weight", 8, 10),
		tensorInfo("blk.0.ffn_up_shexp.weight", 8, 10),
		tensorInfo("blk.0.ffn_down_shexp.weight", 10, 8),
	}}
	weights, err := ReadWeights(file, spec)
	if err != nil {
		t.Fatal(err)
	}
	layer := weights.Layers[0]
	if layer.Recurrent || layer.FeedForwardRouter == nil ||
		layer.FeedForwardGateUpExperts == nil || layer.FeedForwardDownExperts == nil ||
		layer.FeedForwardSharedRouter == nil || layer.FeedForwardSharedGate == nil ||
		layer.FeedForwardSharedUp == nil || layer.FeedForwardSharedDown == nil {
		t.Fatalf("unexpected Qwen3.5-MoE layer catalog: %+v", layer)
	}
}

func TestReadWeightsQwen3NextRecurrentLayouts(t *testing.T) {
	for _, legacy := range []bool{false, true} {
		name := "optimized"
		if legacy {
			name = "legacy_qkvz"
		}
		t.Run(name, func(t *testing.T) {
			spec := Spec{
				Architecture: "qwen3next", BlockCount: 1, EmbeddingLength: 8,
				FeedForwardLength: 12, HeadCount: 2, HeadCountKV: 1, KeyLength: 4,
				ValueLength: 4, VocabularySize: 32, ExpertCount: 4, ExpertUsedCount: 2,
				ExpertFeedForward: 6, SharedExpertFF: 10, ExpertWeightsScale: 1.25,
				SSMConvKernel: 3, SSMInnerSize: 4, SSMStateSize: 2, SSMTimeStepRank: 2,
				SSMGroupCount: 1, RecurrentLayers: []bool{true},
			}
			tensors := []gguf.TensorInfo{
				tensorInfo("token_embd.weight", 8, 32),
				tensorInfo("output_norm.weight", 8),
				tensorInfo("blk.0.attn_norm.weight", 8),
				tensorInfo("blk.0.post_attention_norm.weight", 8),
				tensorInfo("blk.0.ssm_conv1d.weight", 3, 8),
				tensorInfo("blk.0.ssm_dt.bias", 2),
				tensorInfo("blk.0.ssm_a", 2),
				tensorInfo("blk.0.ssm_ba.weight", 8, 4),
				tensorInfo("blk.0.ssm_norm.weight", 2),
				tensorInfo("blk.0.ssm_out.weight", 4, 8),
				tensorInfo("blk.0.ffn_gate_inp.weight", 8, 4),
				tensorInfo("blk.0.ffn_gate_up_exps.weight", 8, 12, 4),
				tensorInfo("blk.0.ffn_down_exps.weight", 6, 8, 4),
				tensorInfo("blk.0.ffn_gate_inp_shexp.weight", 8),
				tensorInfo("blk.0.ffn_gate_shexp.weight", 8, 10),
				tensorInfo("blk.0.ffn_up_shexp.weight", 8, 10),
				tensorInfo("blk.0.ffn_down_shexp.weight", 10, 8),
			}
			if legacy {
				tensors = append(tensors, tensorInfo("blk.0.ssm_in.weight", 8, 12))
			} else {
				tensors = append(tensors,
					tensorInfo("blk.0.attn_qkv.weight", 8, 8),
					tensorInfo("blk.0.attn_gate.weight", 8, 4),
				)
			}
			weights, err := ReadWeights(&gguf.File{Tensors: tensors}, spec)
			if err != nil {
				t.Fatal(err)
			}
			layer := weights.Layers[0]
			if !layer.Recurrent || layer.AttentionQKV == nil || layer.SSMBetaAlpha == nil ||
				(layer.AttentionGate == nil) != legacy || layer.FeedForwardGateUpExperts == nil ||
				layer.FeedForwardSharedRouter == nil {
				t.Fatalf("unexpected Qwen3-Next catalog: %+v", layer)
			}
		})
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

func TestReadWeightsErnie45MoEInterleavesDenseAndExpertLayers(t *testing.T) {
	spec := Spec{
		Architecture: "ernie4_5-moe", BlockCount: 4, EmbeddingLength: 8,
		FeedForwardLength: 12, HeadCount: 2, HeadCountKV: 1,
		KeyLength: 4, ValueLength: 4, VocabularySize: 32,
		RMSNormEpsilon: 1e-6, ExpertCount: 4, ExpertUsedCount: 2,
		ExpertFeedForward: 6, ExpertWeightsScale: 1.25,
		LeadingDenseBlocks: 1, MoELayerStep: 2, SharedExpertFF: 5,
	}
	tensors := []gguf.TensorInfo{
		tensorInfo("token_embd.weight", 8, 32), tensorInfo("output_norm.weight", 8),
	}
	for block := range uint32(4) {
		prefix := fmt.Sprintf("blk.%d.", block)
		tensors = append(tensors,
			tensorInfo(prefix+"attn_norm.weight", 8),
			tensorInfo(prefix+"attn_qkv.weight", 8, 16),
			tensorInfo(prefix+"attn_output.weight", 8, 8),
			tensorInfo(prefix+"attn_output.bias", 8),
			tensorInfo(prefix+"ffn_norm.weight", 8),
		)
		if spec.IsInterleavedMoELayer(block) {
			tensors = append(tensors,
				tensorInfo(prefix+"ffn_gate_inp.weight", 8, 4),
				tensorInfo(prefix+"ffn_up_exps.weight", 8, 6, 4),
				tensorInfo(prefix+"ffn_down_exps.weight", 6, 8, 4),
				tensorInfo(prefix+"exp_probs_b.bias", 4),
				tensorInfo(prefix+"ffn_gate_shexp.weight", 8, 5),
				tensorInfo(prefix+"ffn_up_shexp.weight", 8, 5),
				tensorInfo(prefix+"ffn_down_shexp.weight", 5, 8),
			)
			if block == 3 {
				tensors = append(tensors, tensorInfo(prefix+"ffn_gate_exps.weight", 8, 6, 4))
			}
		} else {
			tensors = append(tensors,
				tensorInfo(prefix+"ffn_gate.weight", 8, 12),
				tensorInfo(prefix+"ffn_up.weight", 8, 12),
				tensorInfo(prefix+"ffn_down.weight", 12, 8),
			)
		}
	}
	weights, err := ReadWeights(&gguf.File{Tensors: tensors}, spec)
	if err != nil {
		t.Fatal(err)
	}
	for block, layer := range weights.Layers {
		if layer.AttentionOutputBias != nil {
			t.Fatalf("ERNIE layer %d retained ignored attention output bias", block)
		}
		if spec.IsInterleavedMoELayer(uint32(block)) {
			if layer.FeedForwardRouter == nil || layer.FeedForwardUpExperts == nil ||
				layer.FeedForwardDownExperts == nil || layer.FeedForwardExpertBias == nil ||
				layer.FeedForwardSharedGate == nil || layer.FeedForwardGate.Name != "" {
				t.Fatalf("ERNIE MoE layer %d catalog: %+v", block, layer)
			}
			if (block == 1) != (layer.FeedForwardGateExperts == nil) {
				t.Fatalf("ERNIE optional expert gate mismatch at layer %d", block)
			}
		} else if layer.FeedForwardGate.Name == "" || layer.FeedForwardRouter != nil {
			t.Fatalf("ERNIE dense layer %d catalog: %+v", block, layer)
		}
	}
}

func TestReadWeightsPaddleOCR(t *testing.T) {
	testReadWeightsMRoPETextDecoder(t, "paddleocr")
}

func TestReadWeightsGemma4SharedKVMoEAndPerLayerInputs(t *testing.T) {
	spec := Spec{
		Architecture: "gemma4", BlockCount: 4, EmbeddingLength: 8,
		FeedForwardLength: 10, LayerFeedForward: []uint32{10, 11, 12, 13},
		HeadCount: 2, HeadCountKV: 1, LayerKVHeadCounts: []uint32{1, 1, 1, 1},
		KeyLength: 4, ValueLength: 4, KeyLengthSWA: 2, ValueLengthSWA: 2,
		RopeDimensionCount: 4, RopeDimensionSWA: 2, VocabularySize: 32,
		RMSNormEpsilon: 1e-6, SlidingLayers: []bool{true, false, true, false},
		SharedKVLayers: 2, EmbeddingPerLayer: 3,
		ExpertCount: 4, ExpertUsedCount: 2, ExpertFeedForward: 3, ExpertWeightsScale: 1,
	}
	tensors := []gguf.TensorInfo{
		tensorInfo("token_embd.weight", 8, 32),
		tensorInfo("output_norm.weight", 8),
		tensorInfo("rope_freqs.weight", 2),
		tensorInfo("per_layer_token_embd.weight", 12, 32),
		tensorInfo("per_layer_model_proj.weight", 8, 12),
		tensorInfo("per_layer_proj_norm.weight", 3),
	}
	for block := range uint32(4) {
		prefix := fmt.Sprintf("blk.%d.", block)
		keyWidth := uint64(spec.LayerKeyLength(block))
		valueWidth := uint64(spec.LayerValueLength(block))
		ffWidth := uint64(spec.LayerFeedForwardLength(block))
		tensors = append(tensors,
			tensorInfo(prefix+"attn_norm.weight", 8),
			tensorInfo(prefix+"attn_q.weight", 8, 2*keyWidth),
			tensorInfo(prefix+"attn_q_norm.weight", keyWidth),
			tensorInfo(prefix+"attn_output.weight", 2*valueWidth, 8),
			tensorInfo(prefix+"post_attention_norm.weight", 8),
			tensorInfo(prefix+"ffn_norm.weight", 8),
			tensorInfo(prefix+"ffn_gate.weight", 8, ffWidth),
			tensorInfo(prefix+"ffn_up.weight", 8, ffWidth),
			tensorInfo(prefix+"ffn_down.weight", ffWidth, 8),
			tensorInfo(prefix+"post_ffw_norm.weight", 8),
			tensorInfo(prefix+"per_layer_inp_gate.weight", 8, 3),
			tensorInfo(prefix+"per_layer_proj.weight", 3, 8),
			tensorInfo(prefix+"per_layer_post_norm.weight", 8),
		)
		if spec.LayerHasKV(block) {
			tensors = append(tensors,
				tensorInfo(prefix+"attn_k.weight", 8, keyWidth),
				tensorInfo(prefix+"attn_k_norm.weight", keyWidth),
			)
		}
		if block == 2 {
			tensors = append(tensors,
				tensorInfo(prefix+"ffn_gate_inp.weight", 8, 4),
				tensorInfo(prefix+"ffn_gate_inp.scale", 8),
				tensorInfo(prefix+"ffn_gate_up_exps.weight", 8, 6, 4),
				tensorInfo(prefix+"ffn_down_exps.weight", 3, 8, 4),
				tensorInfo(prefix+"ffn_down_exps.scale", 4),
				tensorInfo(prefix+"pre_ffw_norm_2.weight", 8),
				tensorInfo(prefix+"post_ffw_norm_1.weight", 8),
				tensorInfo(prefix+"post_ffw_norm_2.weight", 8),
			)
		}
	}
	weights, err := ReadWeights(&gguf.File{Tensors: tensors}, spec)
	if err != nil {
		t.Fatal(err)
	}
	if weights.PerLayerTokenEmbedding == nil || weights.Layers[0].AttentionV.Name != "" ||
		weights.Layers[2].AttentionK.Name != "" || weights.Layers[2].FeedForwardRouter == nil ||
		weights.Layers[2].FeedForwardDownExpertsScale == nil ||
		weights.Layers[3].PerLayerProjection == nil {
		t.Fatalf("unexpected Gemma 4 catalog: %+v", weights)
	}
}

func TestReadWeightsQwen2VL(t *testing.T) {
	testReadWeightsMRoPETextDecoder(t, "qwen2vl")
}

func TestReadWeightsQwen3VL(t *testing.T) {
	testReadWeightsMRoPETextDecoder(t, "qwen3vl")
}

func TestReadWeightsQwen3VLMoE(t *testing.T) {
	spec := Spec{
		Architecture: "qwen3vlmoe", BlockCount: 1, EmbeddingLength: 8,
		FeedForwardLength: 24, ExpertCount: 4, ExpertUsedCount: 2,
		ExpertFeedForward: 12, ExpertWeightsScale: 1.25,
		HeadCount: 2, HeadCountKV: 1, KeyLength: 4, ValueLength: 4,
		VocabularySize: 32, RMSNormEpsilon: 1e-6,
	}
	file := &gguf.File{Tensors: []gguf.TensorInfo{
		tensorInfo("token_embd.weight", 8, 32), tensorInfo("output_norm.weight", 8),
		tensorInfo("blk.0.attn_norm.weight", 8), tensorInfo("blk.0.attn_qkv.weight", 8, 16),
		tensorInfo("blk.0.attn_output.weight", 8, 8),
		tensorInfo("blk.0.attn_q_norm.weight", 4), tensorInfo("blk.0.attn_k_norm.weight", 4),
		tensorInfo("blk.0.ffn_norm.weight", 8), tensorInfo("blk.0.ffn_gate_inp.weight", 8, 4),
		tensorInfo("blk.0.ffn_gate_exps.weight", 8, 12, 4),
		tensorInfo("blk.0.ffn_up_exps.weight", 8, 12, 4),
		tensorInfo("blk.0.ffn_down_exps.weight", 12, 8, 4),
	}}
	weights, err := ReadWeights(file, spec)
	if err != nil {
		t.Fatal(err)
	}
	layer := weights.Layers[0]
	if layer.AttentionQKV == nil || layer.AttentionQNorm == nil || layer.AttentionKNorm == nil ||
		layer.FeedForwardRouter == nil || layer.FeedForwardGateExperts == nil ||
		layer.FeedForwardUpExperts == nil || layer.FeedForwardDownExperts == nil ||
		layer.FeedForwardGate.Name != "" || layer.FeedForwardUp.Name != "" || layer.FeedForwardDown.Name != "" {
		t.Fatalf("unexpected Qwen3-VL-MoE catalog: %+v", weights)
	}
}

func testReadWeightsMRoPETextDecoder(t *testing.T, architecture string) {
	spec := Spec{
		Architecture: architecture, BlockCount: 1, EmbeddingLength: 8,
		FeedForwardLength: 12, HeadCount: 2, HeadCountKV: 1,
		KeyLength: 4, ValueLength: 4, VocabularySize: 32, RMSNormEpsilon: 1e-6,
	}
	file := &gguf.File{Tensors: []gguf.TensorInfo{
		tensorInfo("token_embd.weight", 8, 32), tensorInfo("output_norm.weight", 8),
		tensorInfo("blk.0.attn_norm.weight", 8), tensorInfo("blk.0.attn_qkv.weight", 8, 16),
		tensorInfo("blk.0.attn_output.weight", 8, 8), tensorInfo("blk.0.attn_output.bias", 8),
		tensorInfo("blk.0.ffn_norm.weight", 8), tensorInfo("blk.0.ffn_gate.weight", 8, 12),
		tensorInfo("blk.0.ffn_up.weight", 8, 12), tensorInfo("blk.0.ffn_down.weight", 12, 8),
	}}
	if architecture == "qwen3vl" {
		file.Tensors = append(file.Tensors,
			tensorInfo("blk.0.attn_q_norm.weight", 4),
			tensorInfo("blk.0.attn_k_norm.weight", 4),
		)
	}
	weights, err := ReadWeights(file, spec)
	if err != nil {
		t.Fatal(err)
	}
	layer := weights.Layers[0]
	if weights.Output != nil || layer.AttentionQKV == nil || layer.AttentionOutputBias == nil ||
		layer.FeedForwardNorm.Name == "" || layer.FeedForwardGate.Name == "" ||
		(architecture == "qwen3vl" && (layer.AttentionQNorm == nil || layer.AttentionKNorm == nil)) {
		t.Fatalf("unexpected %s catalog: %+v", architecture, weights)
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

func TestReadWeightsTalkieUsesEmbeddingSkipCatalog(t *testing.T) {
	spec := Spec{
		Architecture: "talkie", BlockCount: 1, EmbeddingLength: 8,
		FeedForwardLength: 12, HeadCount: 2, HeadCountKV: 1,
		KeyLength: 4, ValueLength: 4, VocabularySize: 32, RMSNormEpsilon: 1e-6,
	}
	file := &gguf.File{Tensors: []gguf.TensorInfo{
		tensorInfo("token_embd.weight", 8, 32), tensorInfo("output.weight", 8, 32),
		tensorInfo("blk.0.attn_qkv.weight", 8, 16), tensorInfo("blk.0.attn_qkv.bias", 16),
		tensorInfo("blk.0.attn_output.weight", 8, 8), tensorInfo("blk.0.attn_q_norm.weight", 1, 2),
		tensorInfo("blk.0.ffn_gate.weight", 8, 12), tensorInfo("blk.0.ffn_up.weight", 8, 12),
		tensorInfo("blk.0.ffn_down.weight", 12, 8), tensorInfo("blk.0.layer_out_scale.weight", 1),
	}}
	weights, err := ReadWeights(file, spec)
	if err != nil {
		t.Fatal(err)
	}
	layer := weights.Layers[0]
	if weights.Output == nil || weights.OutputNorm.Name != "" || layer.AttentionNorm.Name != "" ||
		layer.FeedForwardNorm.Name != "" || layer.AttentionQKV == nil ||
		layer.AttentionQNorm == nil || layer.AttentionKNorm != nil || layer.LayerOutputScale == nil {
		t.Fatalf("unexpected Talkie catalog: %+v", weights)
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
