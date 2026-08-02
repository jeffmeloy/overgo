package model

import (
	"fmt"

	"llamacpp2go/internal/gguf"

	"strings"

	"testing"
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

func TestReadWeightsNemotronHMoEThreeWayLayers(t *testing.T) {
	spec := Spec{
		Architecture: "nemotron_h_moe", BlockCount: 3, EmbeddingLength: 8,
		FeedForwardLength: 6, LayerFeedForward: []uint32{0, 0, 6},
		HeadCount: 2, HeadCountKV: 1, LayerHeadCounts: []uint32{2, 0, 2},
		LayerKVHeadCounts: []uint32{1, 0, 1}, RecurrentLayers: []bool{false, true, false},
		KeyLength: 4, ValueLength: 4, VocabularySize: 32, RMSNormEpsilon: 1e-5,
		SSMConvKernel: 3, SSMInnerSize: 16, SSMStateSize: 2, SSMTimeStepRank: 4, SSMGroupCount: 2,
		ExpertCount: 4, ExpertUsedCount: 2, ExpertFeedForward: 6, SharedExpertFF: 5,
		ExpertWeightsScale: 1.25, ExpertWeightsNorm: true, MoELatentSize: 4,
	}
	tensors := []gguf.TensorInfo{
		tensorInfo("token_embd.weight", 8, 32), tensorInfo("output_norm.weight", 8),
		tensorInfo("blk.0.attn_norm.weight", 8),
		tensorInfo("blk.0.attn_q.weight", 8, 8), tensorInfo("blk.0.attn_k.weight", 8, 4),
		tensorInfo("blk.0.attn_v.weight", 8, 4), tensorInfo("blk.0.attn_output.weight", 8, 8),
		tensorInfo("blk.0.attn_output.bias", 8),
		tensorInfo("blk.1.attn_norm.weight", 8), tensorInfo("blk.1.ssm_in.weight", 8, 44),
		tensorInfo("blk.1.ssm_conv1d.weight", 3, 24), tensorInfo("blk.1.ssm_conv1d.bias", 24),
		tensorInfo("blk.1.ssm_dt.bias", 4), tensorInfo("blk.1.ssm_a", 1, 4),
		tensorInfo("blk.1.ssm_d", 1, 4), tensorInfo("blk.1.ssm_norm.weight", 8, 2),
		tensorInfo("blk.1.ssm_out.weight", 16, 8),
		tensorInfo("blk.2.attn_norm.weight", 8), tensorInfo("blk.2.ffn_gate_inp.weight", 8, 4),
		tensorInfo("blk.2.exp_probs_b.bias", 4), tensorInfo("blk.2.ffn_latent_down.weight", 8, 4),
		tensorInfo("blk.2.ffn_latent_up.weight", 4, 8), tensorInfo("blk.2.ffn_up_exps.weight", 4, 6, 4),
		tensorInfo("blk.2.ffn_down_exps.weight", 6, 4, 4), tensorInfo("blk.2.ffn_up_shexp.weight", 8, 5),
		tensorInfo("blk.2.ffn_down_shexp.weight", 5, 8),
	}
	weights, err := ReadWeights(&gguf.File{Tensors: tensors}, spec)
	if err != nil {
		t.Fatal(err)
	}
	attention, recurrent, moe := weights.Layers[0], weights.Layers[1], weights.Layers[2]
	if attention.AttentionQ.Name == "" || attention.AttentionOutputBias == nil || attention.FeedForwardUp.Name != "" ||
		!recurrent.Recurrent || recurrent.SSMInput == nil || recurrent.SSMConv1DBias == nil ||
		moe.Recurrent || moe.FeedForwardRouter == nil || moe.FeedForwardExpertBias == nil ||
		moe.FeedForwardLatentDown == nil || moe.FeedForwardLatentUp == nil ||
		moe.FeedForwardUpExperts == nil || moe.FeedForwardDownExperts == nil ||
		moe.FeedForwardSharedUp == nil || moe.FeedForwardSharedDown == nil || moe.AttentionQ.Name != "" {
		t.Fatalf("unexpected Nemotron-H MoE catalog: %+v", weights.Layers)
	}
}

func TestReadWeightsNemotronHDenseFFNLayer(t *testing.T) {
	spec := Spec{
		Architecture: "nemotron_h", BlockCount: 1, EmbeddingLength: 8,
		FeedForwardLength: 6, LayerFeedForward: []uint32{6},
		HeadCount: 2, HeadCountKV: 1, LayerHeadCounts: []uint32{2}, LayerKVHeadCounts: []uint32{1},
		KeyLength: 4, ValueLength: 4, VocabularySize: 32, RMSNormEpsilon: 1e-5,
	}
	weights, err := ReadWeights(&gguf.File{Tensors: []gguf.TensorInfo{
		tensorInfo("token_embd.weight", 8, 32), tensorInfo("output_norm.weight", 8),
		tensorInfo("blk.0.attn_norm.weight", 8), tensorInfo("blk.0.ffn_up.weight", 8, 6),
		tensorInfo("blk.0.ffn_up.bias", 6), tensorInfo("blk.0.ffn_down.weight", 6, 8),
		tensorInfo("blk.0.ffn_down.bias", 8),
	}}, spec)
	if err != nil {
		t.Fatal(err)
	}
	layer := weights.Layers[0]
	if layer.AttentionNorm.Name == "" || layer.FeedForwardUp.Name == "" || layer.FeedForwardDown.Name == "" ||
		layer.FeedForwardUpBias == nil || layer.FeedForwardDownBias == nil || layer.AttentionQ.Name != "" {
		t.Fatalf("unexpected dense Nemotron-H catalog: %+v", layer)
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

func TestReadWeightsStep35MTPHeads(t *testing.T) {
	spec := Spec{
		Architecture: "step35", BlockCount: 1, NextNPredictLayers: 2,
		EmbeddingLength: 8, FeedForwardLength: 12, VocabularySize: 32,
		HeadCount: 2, HeadCountKV: 1,
		LayerHeadCounts: []uint32{2, 4, 2}, LayerKVHeadCounts: []uint32{1, 2, 1},
		KeyLength: 4, ValueLength: 4, RopeDimensionCount: 4,
	}
	tensors := []gguf.TensorInfo{
		tensorInfo("token_embd.weight", 8, 32), tensorInfo("output_norm.weight", 8),
		tensorInfo("output.weight", 8, 32), tensorInfo("rope_freqs.weight", 2),
	}
	for block := uint32(0); block < 3; block++ {
		prefix := fmt.Sprintf("blk.%d.", block)
		heads, kvHeads := spec.LayerHeadCount(block), spec.LayerKVHeadCount(block)
		tensors = append(tensors,
			tensorInfo(prefix+"attn_norm.weight", 8),
			tensorInfo(prefix+"attn_q.weight", 8, uint64(heads)*4),
			tensorInfo(prefix+"attn_k.weight", 8, uint64(kvHeads)*4),
			tensorInfo(prefix+"attn_v.weight", 8, uint64(kvHeads)*4),
			tensorInfo(prefix+"attn_output.weight", uint64(heads)*4, 8),
			tensorInfo(prefix+"ffn_norm.weight", 8),
			tensorInfo(prefix+"ffn_gate.weight", 8, 12),
			tensorInfo(prefix+"ffn_up.weight", 8, 12),
			tensorInfo(prefix+"ffn_down.weight", 12, 8),
		)
		if block > 0 {
			tensors = append(tensors,
				tensorInfo(prefix+"nextn.eh_proj.weight", 16, 8),
				tensorInfo(prefix+"nextn.enorm.weight", 8),
				tensorInfo(prefix+"nextn.hnorm.weight", 8),
			)
		}
	}
	tensors = append(tensors,
		tensorInfo("blk.1.nextn.embed_tokens.weight", 8, 32),
		tensorInfo("blk.2.nextn.shared_head_norm.weight", 8),
		tensorInfo("blk.2.nextn.shared_head_head.weight", 8, 32),
	)
	weights, err := ReadWeights(&gguf.File{Tensors: tensors}, spec)
	if err != nil {
		t.Fatal(err)
	}
	if len(weights.Layers) != 1 || len(weights.Step35MTP) != 2 ||
		weights.Step35MTP[0].Layer.AttentionQ.Shape[1] != 16 ||
		weights.Step35MTP[0].TokenEmbedding == nil || weights.Step35MTP[0].OutputNorm != nil ||
		weights.Step35MTP[1].Layer.AttentionQ.Shape[1] != 8 ||
		weights.Step35MTP[1].OutputNorm == nil || weights.Step35MTP[1].Output == nil {
		t.Fatalf("unexpected Step3.5 MTP catalog: %+v", weights.Step35MTP)
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
	for _, architecture := range []string{"granitemoe", "granite"} {
		t.Run(architecture, func(t *testing.T) {
			spec := Spec{
				Architecture: architecture, BlockCount: 1, EmbeddingLength: 8,
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
		})
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
