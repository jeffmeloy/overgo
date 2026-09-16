package model

import (
	"fmt"
	"slices"

	"overgo/internal/gguf"

	"overgo/internal/tensor/dtype"

	"strings"

	"testing"
)

func TestReadWeightsJinaBERTV2OptionalNormsAndFusedGEGLU(t *testing.T) {
	spec := Spec{CommonSpec: CommonSpec{Architecture: "jina-bert-v2", BlockCount: 1, ContextLength: 8192,
		EmbeddingLength: 8, FeedForwardLength: 16,
		VocabularySize:   32,
		LayerNormEpsilon: 1e-5}, AttentionSpec: AttentionSpec{HeadCount: 2, HeadCountKV: 2,
		KeyLength: 4, ValueLength: 4,
		NonCausalAttention: true, RopeDisabled: true, MaxALiBiBias: 8}, EncoderSpec: EncoderSpec{TokenTypeCount: 2},
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
	weights, err := readFixtureWeights(file, spec)
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
		layer.FeedForwardGate != nil || layer.FeedForwardUp.Shape[1] != 32 ||
		layer.FeedForwardUpBias == nil || layer.FeedForwardDownBias == nil ||
		layer.FeedForwardPostNorm == nil || layer.FeedForwardPostNormBias == nil {
		t.Fatalf("unexpected JinaBERT v2 catalog: %+v", weights)
	}
}

func TestReadWeightsJinaBERTV2SeparateGate(t *testing.T) {
	spec := Spec{CommonSpec: CommonSpec{Architecture: "jina-bert-v2", BlockCount: 1, ContextLength: 8192,
		EmbeddingLength: 8, FeedForwardLength: 16,
		VocabularySize:   32,
		LayerNormEpsilon: 1e-5}, AttentionSpec: AttentionSpec{HeadCount: 2, HeadCountKV: 2,
		KeyLength: 4, ValueLength: 4,
		NonCausalAttention: true, RopeDisabled: true, MaxALiBiBias: 8}, EncoderSpec: EncoderSpec{TokenTypeCount: 2},
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
	weights, err := readFixtureWeights(file, spec)
	if err != nil {
		t.Fatal(err)
	}
	if weights.Layers[0].FeedForwardGate == nil || weights.Layers[0].FeedForwardUp.Shape[1] != 16 {
		t.Fatalf("unexpected JinaBERT v2 separate gate catalog: %+v", weights.Layers[0])
	}
}

func TestReadWeightsJinaBERTV3PostNormGELU(t *testing.T) {
	spec := Spec{CommonSpec: CommonSpec{Architecture: "jina-bert-v3", BlockCount: 1, ContextLength: 8192,
		EmbeddingLength: 8, FeedForwardLength: 16,
		VocabularySize:   32,
		LayerNormEpsilon: 1e-5}, AttentionSpec: AttentionSpec{HeadCount: 2, HeadCountKV: 2,
		KeyLength: 4, ValueLength: 4,
		NonCausalAttention: true, RopeDimensionCount: 4}, EncoderSpec: EncoderSpec{TokenTypeCount: 2},
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
	weights, err := readFixtureWeights(file, spec)
	if err != nil {
		t.Fatal(err)
	}
	layer := weights.Layers[0]
	if weights.OutputNorm.Name != "" || weights.Output != nil || weights.PositionEmbedding != nil ||
		weights.TokenTypeEmbedding == nil || weights.TokenEmbeddingNorm == nil ||
		weights.TokenEmbeddingNormBias == nil || layer.AttentionQKV == nil ||
		layer.AttentionQKVBias == nil || layer.AttentionPostNorm == nil ||
		layer.AttentionPostNormBias == nil || layer.FeedForwardGate != nil ||
		layer.FeedForwardUpBias == nil || layer.FeedForwardDownBias == nil ||
		layer.FeedForwardPostNorm == nil || layer.FeedForwardPostNormBias == nil {
		t.Fatalf("unexpected JinaBERT v3 catalog: %+v", weights)
	}
}

func TestReadWeightsBERTMoEAlternatesDenseAndExperts(t *testing.T) {
	for _, architecture := range []string{"jina-bert-v3", "nomic-bert-moe"} {
		t.Run(architecture, func(t *testing.T) {
			spec := Spec{CommonSpec: CommonSpec{Architecture: architecture, BlockCount: 2, ContextLength: 8192,
				EmbeddingLength: 8, FeedForwardLength: 16,
				VocabularySize: 32,

				LayerNormEpsilon: 1e-5}, AttentionSpec: AttentionSpec{HeadCount: 2, HeadCountKV: 2,
				KeyLength: 4, ValueLength: 4,

				NonCausalAttention: true, RopeDimensionCount: 4}, MoESpec: MoESpec{ExpertCount: 4, ExpertUsedCount: 2, ExpertFeedForward: 16, ExpertWeightsScale: 1,
				MoELayerStep: 2}, EncoderSpec: EncoderSpec{TokenTypeCount: 2},
			}
			tensors := []gguf.TensorInfo{
				tensorInfo("token_embd.weight", 8, 32), tensorInfo("token_types.weight", 8, 2),
				tensorInfo("token_embd_norm.weight", 8), tensorInfo("token_embd_norm.bias", 8),
			}
			for block := range 2 {
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
			weights, err := readFixtureWeights(&gguf.File{Tensors: tensors}, spec)
			if err != nil {
				t.Fatal(err)
			}
			dense, moe := weights.Layers[0], weights.Layers[1]
			if dense.FeedForwardUp == nil || dense.FeedForwardDown == nil || dense.FeedForwardRouter != nil ||
				moe.FeedForwardRouter == nil || moe.FeedForwardUpExperts == nil || moe.FeedForwardDownExperts == nil ||
				moe.FeedForwardGateExperts != nil || moe.FeedForwardUp != nil {
				t.Fatalf("unexpected %s MoE catalog: %+v / %+v", architecture, dense, moe)
			}
		})
	}
}

func TestReadWeightsRND1(t *testing.T) {
	spec := Spec{CommonSpec: CommonSpec{Architecture: "rnd1", BlockCount: 1, EmbeddingLength: 8,
		FeedForwardLength: 24,

		VocabularySize: 32}, AttentionSpec: AttentionSpec{HeadCount: 2, HeadCountKV: 1, KeyLength: 4, ValueLength: 4,
		NonCausalAttention: true}, MoESpec: MoESpec{ExpertCount: 4, ExpertUsedCount: 2,
		ExpertFeedForward: 12, ExpertWeightsScale: 1},
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
	weights, err := readFixtureWeights(file, spec)
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
	denseSpec := Spec{CommonSpec: CommonSpec{Architecture: "llada", BlockCount: 1, EmbeddingLength: 8, FeedForwardLength: 12,
		VocabularySize: 32}, AttentionSpec: AttentionSpec{HeadCount: 2, HeadCountKV: 1, KeyLength: 4, ValueLength: 4},
	}
	common := []gguf.TensorInfo{
		tensorInfo("token_embd.weight", 8, 32), tensorInfo("output_norm.weight", 8),
		tensorInfo("blk.0.attn_norm.weight", 8), tensorInfo("blk.0.attn_q.weight", 8, 8),
		tensorInfo("blk.0.attn_k.weight", 8, 4), tensorInfo("blk.0.attn_v.weight", 8, 4),
		tensorInfo("blk.0.attn_output.weight", 8, 8), tensorInfo("blk.0.ffn_norm.weight", 8),
	}
	denseTensors := append(slices.Clone(common),
		tensorInfo("blk.0.ffn_gate.weight", 8, 12), tensorInfo("blk.0.ffn_up.weight", 8, 12),
		tensorInfo("blk.0.ffn_down.weight", 12, 8),
	)
	dense, err := readFixtureWeights(&gguf.File{Tensors: denseTensors}, denseSpec)
	if err != nil {
		t.Fatal(err)
	}
	if dense.Output != nil || dense.Layers[0].FeedForwardGate == nil {
		t.Fatalf("unexpected LLaDA catalog: %+v", dense)
	}
	moeSpec := denseSpec
	moeSpec.Architecture = "llada-moe"
	moeSpec.ExpertCount, moeSpec.ExpertUsedCount, moeSpec.ExpertFeedForward = 4, 2, 6
	moeTensors := append(slices.Clone(common),
		tensorInfo("output.weight", 8, 32),
		tensorInfo("blk.0.attn_q_norm.weight", 4), tensorInfo("blk.0.attn_k_norm.weight", 4),
		tensorInfo("blk.0.ffn_gate_inp.weight", 8, 4),
		tensorInfo("blk.0.ffn_gate_exps.weight", 8, 6, 4),
		tensorInfo("blk.0.ffn_up_exps.weight", 8, 6, 4),
		tensorInfo("blk.0.ffn_down_exps.weight", 6, 8, 4),
	)
	moe, err := readFixtureWeights(&gguf.File{Tensors: moeTensors}, moeSpec)
	if err != nil {
		t.Fatal(err)
	}
	if moe.Output == nil || moe.Layers[0].AttentionQNorm == nil ||
		moe.Layers[0].AttentionKNorm == nil || moe.Layers[0].FeedForwardRouter == nil {
		t.Fatalf("unexpected LLaDA-MoE catalog: %+v", moe)
	}
}

func TestReadWeightsLagunaDenseThenMoE(t *testing.T) {
	spec := Spec{CommonSpec: CommonSpec{Architecture: "laguna", BlockCount: 2, EmbeddingLength: 8,
		FeedForwardLength: 16,

		VocabularySize: 32}, AttentionSpec: AttentionSpec{HeadCount: 2, HeadCountKV: 1, LayerHeadCounts: []uint32{2, 4},
		LayerKVHeadCounts: []uint32{1, 1}, KeyLength: 4, ValueLength: 4}, MoESpec: MoESpec{LeadingDenseBlocks: 1,
		ExpertCount: 4, ExpertUsedCount: 2, ExpertFeedForward: 12,
		SharedExpertFF: 10, ExpertWeightsScale: 1},
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
	weights, err := readFixtureWeights(&gguf.File{Tensors: tensors}, spec)
	if err != nil {
		t.Fatal(err)
	}
	if weights.Layers[0].FeedForwardRouter != nil || weights.Layers[0].FeedForwardGate == nil ||
		weights.Layers[1].FeedForwardRouter == nil || weights.Layers[1].FeedForwardExpertBias == nil ||
		weights.Layers[1].FeedForwardSharedDown == nil ||
		weights.Layers[1].AttentionQ.Shape[1] != 16 {
		t.Fatalf("unexpected Laguna catalog: %+v", weights.Layers)
	}
}

func TestReadWeightsAFMoE(t *testing.T) {
	spec := Spec{CommonSpec: CommonSpec{Architecture: "afmoe", BlockCount: 1, EmbeddingLength: 8,
		FeedForwardLength: 16,

		VocabularySize: 32}, AttentionSpec: AttentionSpec{HeadCount: 2, HeadCountKV: 1,
		KeyLength: 4, ValueLength: 4}, MoESpec: MoESpec{ExpertCount: 4, ExpertUsedCount: 2,
		ExpertFeedForward: 6, SharedExpertCount: 2, SharedExpertFF: 12,
		ExpertWeightsScale: 2.826},
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
	weights, err := readFixtureWeights(file, spec)
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
	spec := Spec{CommonSpec: CommonSpec{Architecture: "olmoe", BlockCount: 1, EmbeddingLength: 8,
		FeedForwardLength: 12,

		VocabularySize: 32}, AttentionSpec: AttentionSpec{HeadCount: 2, HeadCountKV: 2, KeyLength: 4, ValueLength: 4}, MoESpec: MoESpec{ExpertCount: 4, ExpertUsedCount: 2,
		ExpertFeedForward: 12, ExpertWeightsScale: 1},
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
	weights, err := readFixtureWeights(file, spec)
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
	spec := Spec{CommonSpec: CommonSpec{Architecture: "phimoe", BlockCount: 1, ContextLength: 128,
		EmbeddingLength: 8, FeedForwardLength: 12,

		VocabularySize: 32, RMSNormEpsilon: 1e-5}, AttentionSpec: AttentionSpec{OriginalContextLength: 32,

		HeadCount: 2, HeadCountKV: 2, KeyLength: 4, ValueLength: 4,
		RopeDimensionCount: 4, RopeScalingType: "longrope",
		RopeAttentionFactor: 1.1}, MoESpec: MoESpec{ExpertCount: 4, ExpertUsedCount: 2, ExpertFeedForward: 12, ExpertWeightsScale: 1},
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
	weights, err := readFixtureWeights(file, spec)
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
	spec := Spec{CommonSpec: CommonSpec{Architecture: "exaone-moe", BlockCount: 1, NextNPredictLayers: 1, EmbeddingLength: 8, FeedForwardLength: 16,

		VocabularySize: 32}, AttentionSpec: AttentionSpec{HeadCount: 2, HeadCountKV: 1, KeyLength: 4, ValueLength: 4}, MoESpec: MoESpec{ExpertCount: 4, ExpertUsedCount: 2, ExpertFeedForward: 6, ExpertWeightsScale: 1.5,
		SharedExpertFF: 12, ExpertGatingFunc: expertGatingSigmoid},
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
		tensorInfo("blk.1.attn_norm.weight", 8),
		tensorInfo("blk.1.attn_q.weight", 8, 8), tensorInfo("blk.1.attn_k.weight", 8, 4),
		tensorInfo("blk.1.attn_v.weight", 8, 4), tensorInfo("blk.1.attn_output.weight", 8, 8),
		tensorInfo("blk.1.attn_q_norm.weight", 4), tensorInfo("blk.1.attn_k_norm.weight", 4),
		tensorInfo("blk.1.ffn_norm.weight", 8), tensorInfo("blk.1.ffn_gate.weight", 8, 16),
		tensorInfo("blk.1.ffn_up.weight", 8, 16), tensorInfo("blk.1.ffn_down.weight", 16, 8),
		tensorInfo("blk.1.nextn.eh_proj.weight", 16, 8),
		tensorInfo("blk.1.nextn.enorm.weight", 8), tensorInfo("blk.1.nextn.hnorm.weight", 8),
	}}
	weights, err := readFixtureWeights(file, spec)
	if err != nil {
		t.Fatal(err)
	}
	layer := weights.Layers[0]
	if layer.AttentionQNorm == nil || layer.FeedForwardRouter == nil ||
		layer.FeedForwardExpertBias == nil || layer.FeedForwardSharedDown == nil ||
		len(weights.AppendedSingleDraft) != 1 || weights.AppendedSingleDraft[0].Layer.FeedForwardRouter != nil ||
		weights.AppendedSingleDraft[0].Layer.FeedForwardGate == nil {
		t.Fatalf("unexpected EXAONE-MoE catalog: %+v", layer)
	}
}

func TestReadWeightsChameleon(t *testing.T) {
	spec := Spec{CommonSpec: CommonSpec{Architecture: "chameleon", BlockCount: 1, EmbeddingLength: 8,
		FeedForwardLength: 16,
		VocabularySize:    8200}, AttentionSpec: AttentionSpec{HeadCount: 2, HeadCountKV: 1,
		KeyLength: 4, ValueLength: 4},
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
	weights, err := readFixtureWeights(file, spec)
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
	spec := Spec{CommonSpec: CommonSpec{Architecture: "lfm2", BlockCount: 2, EmbeddingLength: 8,
		FeedForwardLength: 16,
		VocabularySize:    32}, AttentionSpec: AttentionSpec{HeadCount: 2, HeadCountKV: 1,
		KeyLength: 4, ValueLength: 4}, RecurrentSpec: RecurrentSpec{ShortConvCacheLength: 4, RecurrentLayers: []bool{true, false}},
	}
	tensors := []gguf.TensorInfo{
		tensorInfo("token_embd.weight", 8, 32),
		tensorInfo("token_embd_norm.weight", 8),
	}
	for block := range 2 {
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
	weights, err := readFixtureWeights(&gguf.File{Tensors: tensors}, spec)
	if err != nil {
		t.Fatal(err)
	}
	if !weights.Layers[0].Recurrent || weights.Layers[0].ShortConvKernel == nil ||
		weights.Layers[1].Recurrent || weights.Layers[1].AttentionQNorm == nil {
		t.Fatalf("unexpected LFM2 weights: %+v", weights.Layers)
	}
}

func TestReadWeightsLFM2MoEDenseThenHybridMoE(t *testing.T) {
	spec := Spec{CommonSpec: CommonSpec{Architecture: "lfm2moe", BlockCount: 3,
		EmbeddingLength: 8, FeedForwardLength: 16,

		VocabularySize: 32}, AttentionSpec: AttentionSpec{HeadCount: 2, HeadCountKV: 1, KeyLength: 4, ValueLength: 4}, MoESpec: MoESpec{LeadingDenseBlocks: 1,

		ExpertCount: 4, ExpertUsedCount: 2, ExpertFeedForward: 6,
		ExpertWeightsScale: 1.25, ExpertGatingFunc: expertGatingSigmoid}, RecurrentSpec: RecurrentSpec{ShortConvCacheLength: 4, RecurrentLayers: []bool{true, false, true}},
	}
	tensors := []gguf.TensorInfo{
		tensorInfo("token_embd.weight", 8, 32), tensorInfo("token_embd_norm.weight", 8),
	}
	for block := range 3 {
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
	weights, err := readFixtureWeights(&gguf.File{Tensors: tensors}, spec)
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
	spec := Spec{CommonSpec: CommonSpec{Architecture: "plm", BlockCount: 1, EmbeddingLength: 8,
		FeedForwardLength: 16,
		VocabularySize:    32}, AttentionSpec: AttentionSpec{HeadCount: 2, HeadCountKV: 2,
		KeyLength: 6, ValueLength: 4,
		KVLoRARank: 3, RopeDimensionCount: 2},
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
	weights, err := readFixtureWeights(file, spec)
	if err != nil {
		t.Fatal(err)
	}
	if weights.Layers[0].AttentionKVAMQA == nil || weights.Layers[0].AttentionKVANorm == nil ||
		weights.Layers[0].AttentionKVB == nil || weights.Layers[0].FeedForwardGate != nil {
		t.Fatalf("unexpected PLM weights: %+v", weights.Layers[0])
	}
}

func TestReadWeightsMiniCPM3(t *testing.T) {
	spec := Spec{CommonSpec: CommonSpec{Architecture: "minicpm3", BlockCount: 1, ContextLength: 4096,
		EmbeddingLength: 8, FeedForwardLength: 12,

		VocabularySize: 32}, AttentionSpec: AttentionSpec{OriginalContextLength: 4096,
		HeadCount: 2, HeadCountKV: 2,
		KeyLength: 6, ValueLength: 4, QLoRARank: 3, KVLoRARank: 3,
		RopeDimensionCount: 2}}
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
	weights, err := readFixtureWeights(&gguf.File{Tensors: tensors}, spec)
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

func TestReadWeightsGLMDSAIndexer(t *testing.T) {
	spec := Spec{CommonSpec: CommonSpec{Architecture: "glm-dsa", BlockCount: 2, EmbeddingLength: 8,
		FeedForwardLength: 12, NextNPredictLayers: 1,
		VocabularySize: 32}, AttentionSpec: AttentionSpec{HeadCount: 2, HeadCountKV: 1, KeyLength: 6, ValueLength: 4,
		QLoRARank: 3, KVLoRARank: 3, RopeDimensionCount: 2,

		IndexerHeadCount: 2, IndexerKeyLength: 8,
		IndexerTopK: 4, IndexerFullLayers: []bool{true, false}}, MoESpec: MoESpec{LeadingDenseBlocks: 1, ExpertCount: 4, ExpertUsedCount: 2, ExpertFeedForward: 6,
		SharedExpertCount: 1, SharedExpertFF: 6},
	}
	tensors := []gguf.TensorInfo{tensorInfo("token_embd.weight", 8, 32), tensorInfo("output_norm.weight", 8)}
	for block := range uint32(3) {
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
		tensorInfo("blk.0.indexer.k_norm.weight", 8), tensorInfo("blk.0.indexer.k_norm.bias", 8),
		tensorInfo("blk.0.indexer.proj.weight", 8, 2), tensorInfo("blk.0.indexer.attn_k.weight", 8, 8),
		tensorInfo("blk.0.indexer.attn_q_b.weight", 3, 16),
		tensorInfo("blk.0.ffn_gate.weight", 8, 12), tensorInfo("blk.0.ffn_up.weight", 8, 12),
		tensorInfo("blk.0.ffn_down.weight", 12, 8), tensorInfo("blk.1.ffn_gate_inp.weight", 8, 4),
		tensorInfo("blk.1.ffn_gate_exps.weight", 8, 6, 4), tensorInfo("blk.1.ffn_up_exps.weight", 8, 6, 4),
		tensorInfo("blk.1.ffn_down_exps.weight", 6, 8, 4), tensorInfo("blk.1.ffn_gate_shexp.weight", 8, 6),
		tensorInfo("blk.1.ffn_up_shexp.weight", 8, 6), tensorInfo("blk.1.ffn_down_shexp.weight", 6, 8),
		tensorInfo("blk.2.ffn_gate_inp.weight", 8, 4),
		tensorInfo("blk.2.ffn_gate_exps.weight", 8, 6, 4), tensorInfo("blk.2.ffn_up_exps.weight", 8, 6, 4),
		tensorInfo("blk.2.ffn_down_exps.weight", 6, 8, 4), tensorInfo("blk.2.ffn_gate_shexp.weight", 8, 6),
		tensorInfo("blk.2.ffn_up_shexp.weight", 8, 6), tensorInfo("blk.2.ffn_down_shexp.weight", 6, 8),
		tensorInfo("blk.2.nextn.eh_proj.weight", 16, 8),
		tensorInfo("blk.2.nextn.enorm.weight", 8), tensorInfo("blk.2.nextn.hnorm.weight", 8),
	)
	weights, err := readFixtureWeights(&gguf.File{Tensors: tensors}, spec)
	if err != nil {
		t.Fatal(err)
	}
	if weights.Layers[0].IndexerAttentionQB == nil || weights.Layers[0].IndexerKNormBias == nil ||
		weights.Layers[1].IndexerAttentionQB != nil || weights.Layers[1].FeedForwardRouter == nil ||
		len(weights.AppendedSingleDraft) != 1 || weights.AppendedSingleDraft[0].Layer.IndexerAttentionQB != nil ||
		weights.AppendedSingleDraft[0].Layer.FeedForwardRouter == nil {
		t.Fatalf("unexpected GLM-DSA catalog: %+v", weights.Layers)
	}
}

func TestReadWeightsDeepSeek32IndexerEveryLayer(t *testing.T) {
	spec := Spec{CommonSpec: CommonSpec{Architecture: "deepseek32", BlockCount: 2, EmbeddingLength: 8,
		FeedForwardLength: 12, NextNPredictLayers: 1,
		VocabularySize: 32}, AttentionSpec: AttentionSpec{HeadCount: 2, HeadCountKV: 1, KeyLength: 6, ValueLength: 4,
		QLoRARank: 3, KVLoRARank: 3, RopeDimensionCount: 2,

		IndexerHeadCount: 2, IndexerKeyLength: 8, IndexerTopK: 4,
		IndexerFullLayers: []bool{true, true}}, MoESpec: MoESpec{LeadingDenseBlocks: 2,
		ExpertCount: 4, ExpertUsedCount: 2, ExpertFeedForward: 6, SharedExpertCount: 1, SharedExpertFF: 6},
	}
	tensors := []gguf.TensorInfo{tensorInfo("token_embd.weight", 8, 32), tensorInfo("output_norm.weight", 8)}
	for block := range uint32(2) {
		prefix := fmt.Sprintf("blk.%d.", block)
		tensors = append(tensors,
			tensorInfo(prefix+"attn_norm.weight", 8), tensorInfo(prefix+"attn_q_a.weight", 8, 3),
			tensorInfo(prefix+"attn_q_a_norm.weight", 3), tensorInfo(prefix+"attn_q_b.weight", 3, 12),
			tensorInfo(prefix+"attn_kv_a_mqa.weight", 8, 5), tensorInfo(prefix+"attn_kv_a_norm.weight", 3),
			tensorInfo(prefix+"attn_k_b.weight", 4, 3, 2), tensorInfo(prefix+"attn_v_b.weight", 3, 4, 2),
			tensorInfo(prefix+"attn_output.weight", 8, 8), tensorInfo(prefix+"ffn_norm.weight", 8),
			tensorInfo(prefix+"ffn_gate.weight", 8, 12), tensorInfo(prefix+"ffn_up.weight", 8, 12),
			tensorInfo(prefix+"ffn_down.weight", 12, 8),
			tensorInfo(prefix+"indexer.k_norm.weight", 8), tensorInfo(prefix+"indexer.k_norm.bias", 8),
			tensorInfo(prefix+"indexer.proj.weight", 8, 2), tensorInfo(prefix+"indexer.attn_k.weight", 8, 8),
			tensorInfo(prefix+"indexer.attn_q_b.weight", 3, 16),
		)
	}
	tensors = append(tensors,
		tensorInfo("blk.2.attn_norm.weight", 8), tensorInfo("blk.2.attn_q_a.weight", 8, 3),
		tensorInfo("blk.2.attn_q_a_norm.weight", 3), tensorInfo("blk.2.attn_q_b.weight", 3, 12),
		tensorInfo("blk.2.attn_kv_a_mqa.weight", 8, 5), tensorInfo("blk.2.attn_kv_a_norm.weight", 3),
		tensorInfo("blk.2.attn_k_b.weight", 4, 3, 2), tensorInfo("blk.2.attn_v_b.weight", 3, 4, 2),
		tensorInfo("blk.2.attn_output.weight", 8, 8), tensorInfo("blk.2.ffn_norm.weight", 8),
		tensorInfo("blk.2.ffn_gate_inp.weight", 8, 4),
		tensorInfo("blk.2.ffn_gate_exps.weight", 8, 6, 4),
		tensorInfo("blk.2.ffn_up_exps.weight", 8, 6, 4),
		tensorInfo("blk.2.ffn_down_exps.weight", 6, 8, 4),
		tensorInfo("blk.2.ffn_gate_shexp.weight", 8, 6),
		tensorInfo("blk.2.ffn_up_shexp.weight", 8, 6),
		tensorInfo("blk.2.ffn_down_shexp.weight", 6, 8),
		tensorInfo("blk.2.indexer.k_norm.weight", 8), tensorInfo("blk.2.indexer.k_norm.bias", 8),
		tensorInfo("blk.2.indexer.proj.weight", 8, 2), tensorInfo("blk.2.indexer.attn_k.weight", 8, 8),
		tensorInfo("blk.2.indexer.attn_q_b.weight", 3, 16),
		tensorInfo("blk.2.nextn.eh_proj.weight", 16, 8),
		tensorInfo("blk.2.nextn.enorm.weight", 8), tensorInfo("blk.2.nextn.hnorm.weight", 8),
	)
	weights, err := readFixtureWeights(&gguf.File{Tensors: tensors}, spec)
	if err != nil {
		t.Fatal(err)
	}
	for block, layer := range weights.Layers {
		if layer.IndexerAttentionQB == nil || layer.IndexerKNormBias == nil {
			t.Fatalf("DeepSeek 3.2 layer %d lacks indexer tensors", block)
		}
	}
	if len(weights.AppendedSingleDraft) != 1 || weights.AppendedSingleDraft[0].Layer.IndexerAttentionQB == nil ||
		weights.AppendedSingleDraft[0].Layer.FeedForwardRouter == nil {
		t.Fatalf("DeepSeek 3.2 NextN tail is incomplete: %+v", weights.AppendedSingleDraft)
	}
}

func TestReadWeightsDeepSeek4(t *testing.T) {
	spec := Spec{CommonSpec: CommonSpec{Architecture: "deepseek4", BlockCount: 3, EmbeddingLength: 8, VocabularySize: 32,

		HyperConnectionCount: 4,

		HashLayerCount: 1}, AttentionSpec: AttentionSpec{HeadCount: 2, HeadCountKV: 1, KeyLength: 4, ValueLength: 4, QLoRARank: 3,
		AttentionOutputGroups: 1, AttentionOutputRank: 3,

		IndexerHeadCount: 2, IndexerKeyLength: 8, CompressRatios: []uint32{0, 4, 128}}, MoESpec: MoESpec{ExpertCount: 4, ExpertUsedCount: 2, ExpertFeedForward: 8, SharedExpertFF: 8},
	}
	tensors := []gguf.TensorInfo{
		tensorInfo("token_embd.weight", 8, 32), tensorInfo("output_norm.weight", 8),
		tensorInfo("output.weight", 8, 32), tensorInfo("output_hc_fn.weight", 32, 4),
		tensorInfo("output_hc_base.weight", 4), tensorInfo("output_hc_scale.weight", 1),
	}
	for block, ratio := range spec.CompressRatios {
		prefix := fmt.Sprintf("blk.%d.", block)
		tensors = append(tensors,
			tensorInfo(prefix+"attn_norm.weight", 8), tensorInfo(prefix+"attn_sinks.weight", 2),
			tensorInfo(prefix+"attn_q_a.weight", 8, 3), tensorInfo(prefix+"attn_q_a_norm.weight", 3),
			tensorInfo(prefix+"attn_q_b.weight", 3, 8), tensorInfo(prefix+"attn_kv.weight", 8, 4),
			tensorInfo(prefix+"attn_kv_a_norm.weight", 4), tensorInfo(prefix+"attn_output_a.weight", 8, 3),
			tensorInfo(prefix+"attn_output.weight", 3, 8),
			tensorInfo(prefix+"hc_attn_fn.weight", 32, 24), tensorInfo(prefix+"hc_attn_base.weight", 24),
			tensorInfo(prefix+"hc_attn_scale.weight", 3), tensorInfo(prefix+"hc_ffn_fn.weight", 32, 24),
			tensorInfo(prefix+"hc_ffn_base.weight", 24), tensorInfo(prefix+"hc_ffn_scale.weight", 3),
			tensorInfo(prefix+"ffn_norm.weight", 8), tensorInfo(prefix+"ffn_gate_inp.weight", 8, 4),
			tensorInfo(prefix+"ffn_gate_exps.weight", 8, 8, 4), tensorInfo(prefix+"ffn_up_exps.weight", 8, 8, 4),
			tensorInfo(prefix+"ffn_down_exps.weight", 8, 8, 4), tensorInfo(prefix+"ffn_gate_shexp.weight", 8, 8),
			tensorInfo(prefix+"ffn_up_shexp.weight", 8, 8), tensorInfo(prefix+"ffn_down_shexp.weight", 8, 8),
		)
		if block == 0 {
			hash := tensorInfo(prefix+"ffn_gate_tid2eid.weight", 2, 32)
			hash.Type = dtype.I32
			tensors = append(tensors, hash)
		} else {
			tensors = append(tensors, tensorInfo(prefix+"exp_probs_b.bias", 4))
		}
		if ratio != 0 {
			coefficient := uint64(1)
			if ratio == 4 {
				coefficient = 2
			}
			tensors = append(tensors,
				tensorInfo(prefix+"attn_compressor_kv.weight", 8, coefficient*4),
				tensorInfo(prefix+"attn_compressor_gate.weight", 8, coefficient*4),
				tensorInfo(prefix+"attn_compressor_ape.weight", coefficient*4, uint64(ratio)),
				tensorInfo(prefix+"attn_compressor_norm.weight", 4),
			)
		}
		if ratio == 4 {
			tensors = append(tensors,
				tensorInfo(prefix+"indexer.proj.weight", 8, 2), tensorInfo(prefix+"indexer.attn_q_b.weight", 3, 16),
				tensorInfo(prefix+"indexer_compressor_kv.weight", 8, 16), tensorInfo(prefix+"indexer_compressor_gate.weight", 8, 16),
				tensorInfo(prefix+"indexer_compressor_ape.weight", 16, 4), tensorInfo(prefix+"indexer_compressor_norm.weight", 8),
			)
		}
	}
	weights, err := readFixtureWeights(&gguf.File{Tensors: tensors}, spec)
	if err != nil {
		t.Fatal(err)
	}
	if weights.Output == nil || weights.Layers[0].FeedForwardHashExperts == nil ||
		weights.Layers[0].FeedForwardRouterBias != nil || weights.Layers[1].IndexerCompressorNorm == nil ||
		weights.Layers[2].AttentionCompressorNorm == nil || weights.Layers[2].IndexerCompressorNorm != nil ||
		weights.Layers[2].HyperHeadFN == nil {
		t.Fatalf("unexpected DeepSeek 4 catalog: %+v", weights)
	}
}

func TestReadWeightsMamba(t *testing.T) {
	spec := Spec{CommonSpec: CommonSpec{Architecture: "mamba", BlockCount: 1, EmbeddingLength: 4,

		VocabularySize: 32}, RecurrentSpec: RecurrentSpec{SSMConvKernel: 3, SSMInnerSize: 8, SSMStateSize: 2, SSMTimeStepRank: 2},
	}
	fixtures := []gguf.TensorInfo{
		tensorInfo("token_embd.weight", 4, 32), tensorInfo("output_norm.weight", 4),
		tensorInfo("blk.0.attn_norm.weight", 4),
	}
	fixtures = append(fixtures, tensorBindingFixtures(
		"blk.0.", selectiveScanTensorRequirements(spec, &LayerWeights{}), nil,
	)...)
	file := &gguf.File{Tensors: fixtures}
	weights, err := readFixtureWeights(file, spec)
	if err != nil {
		t.Fatal(err)
	}
	layer := weights.Layers[0]
	if !layer.Recurrent || layer.SSMInput == nil || layer.SSMConv1D == nil ||
		layer.SSMConv1DBias == nil || layer.SSMX == nil || layer.SSMTimeStepWeight == nil ||
		layer.SSMTimeStep == nil || layer.SSMA == nil || layer.SSMD == nil || layer.SSMOutput == nil ||
		layer.AttentionQ != nil || layer.FeedForwardNorm != nil {
		t.Fatalf("unexpected Mamba catalog: %+v", layer)
	}
}

func TestReadWeightsMamba2(t *testing.T) {
	spec := Spec{CommonSpec: CommonSpec{Architecture: "mamba2", BlockCount: 1, EmbeddingLength: 4,

		VocabularySize: 32}, RecurrentSpec: RecurrentSpec{SSMConvKernel: 3, SSMInnerSize: 8, SSMStateSize: 2, SSMTimeStepRank: 4,
		SSMGroupCount: 2}}
	file := &gguf.File{Tensors: []gguf.TensorInfo{
		tensorInfo("token_embd.weight", 4, 32), tensorInfo("output_norm.weight", 4),
		tensorInfo("blk.0.attn_norm.weight", 4), tensorInfo("blk.0.ssm_in.weight", 4, 28),
		tensorInfo("blk.0.ssm_conv1d.weight", 3, 16), tensorInfo("blk.0.ssm_conv1d.bias", 16),
		tensorInfo("blk.0.ssm_dt.bias", 4), tensorInfo("blk.0.ssm_a", 1, 4),
		tensorInfo("blk.0.ssm_d", 1, 4), tensorInfo("blk.0.ssm_norm.weight", 4, 2),
		tensorInfo("blk.0.ssm_out.weight", 8, 4),
	}}
	weights, err := readFixtureWeights(file, spec)
	if err != nil {
		t.Fatal(err)
	}
	layer := weights.Layers[0]
	if !layer.Recurrent || layer.SSMInput == nil || layer.SSMConv1D == nil ||
		layer.SSMConv1DBias == nil || layer.SSMTimeStep == nil || layer.SSMA == nil ||
		layer.SSMD == nil || layer.SSMNorm == nil || layer.SSMOutput == nil ||
		layer.SSMX != nil || layer.SSMTimeStepWeight != nil || layer.AttentionQ != nil ||
		layer.FeedForwardNorm != nil {
		t.Fatalf("unexpected Mamba2 catalog: %+v", layer)
	}
}

func TestReadWeightsFalconH1(t *testing.T) {
	spec := Spec{CommonSpec: CommonSpec{Architecture: "falcon-h1", BlockCount: 1, EmbeddingLength: 4,
		FeedForwardLength: 6,

		VocabularySize: 32}, AttentionSpec: AttentionSpec{HeadCount: 2, HeadCountKV: 1, KeyLength: 2, ValueLength: 2,
		RopeDimensionCount: 2}, RecurrentSpec: RecurrentSpec{SSMConvKernel: 3, SSMInnerSize: 8, SSMStateSize: 2,
		SSMTimeStepRank: 4, SSMGroupCount: 2}}
	file := &gguf.File{Tensors: []gguf.TensorInfo{
		tensorInfo("token_embd.weight", 4, 32), tensorInfo("output_norm.weight", 4),
		tensorInfo("blk.0.attn_norm.weight", 4), tensorInfo("blk.0.attn_q.weight", 4, 4),
		tensorInfo("blk.0.attn_k.weight", 4, 2), tensorInfo("blk.0.attn_v.weight", 4, 2),
		tensorInfo("blk.0.attn_output.weight", 4, 4), tensorInfo("blk.0.ssm_in.weight", 4, 28),
		tensorInfo("blk.0.ssm_conv1d.weight", 3, 16), tensorInfo("blk.0.ssm_dt.bias", 4),
		tensorInfo("blk.0.ssm_a", 1, 4), tensorInfo("blk.0.ssm_d", 1, 4),
		tensorInfo("blk.0.ssm_out.weight", 8, 4), tensorInfo("blk.0.ffn_norm", 4),
		tensorInfo("blk.0.ffn_gate.weight", 4, 6), tensorInfo("blk.0.ffn_up.weight", 4, 6),
		tensorInfo("blk.0.ffn_down.weight", 6, 4),
	}}
	weights, err := readFixtureWeights(file, spec)
	if err != nil {
		t.Fatal(err)
	}
	layer := weights.Layers[0]
	if layer.Recurrent || layer.SSMInput == nil || layer.SSMConv1D == nil || layer.SSMTimeStep == nil ||
		layer.SSMA == nil || layer.SSMD == nil || layer.SSMOutput == nil || layer.AttentionQ == nil ||
		layer.AttentionK == nil || layer.AttentionV == nil || layer.AttentionOutput == nil ||
		layer.FeedForwardNorm.Name != "blk.0.ffn_norm" {
		t.Fatalf("unexpected Falcon-H1 catalog: %+v", layer)
	}
}

func TestReadWeightsJamba(t *testing.T) {
	spec := Spec{CommonSpec: CommonSpec{Architecture: "jamba", BlockCount: 2, EmbeddingLength: 4,
		FeedForwardLength: 6,

		VocabularySize: 32}, AttentionSpec: AttentionSpec{HeadCount: 2, HeadCountKV: 1, KeyLength: 2, ValueLength: 2,
		LayerKVHeadCounts: []uint32{0, 1}}, MoESpec: MoESpec{ExpertCount: 4, ExpertUsedCount: 2, ExpertFeedForward: 6}, RecurrentSpec: RecurrentSpec{RecurrentLayers: []bool{true, false},
		SSMConvKernel: 3, SSMInnerSize: 8, SSMStateSize: 2, SSMTimeStepRank: 2,
		SSMGroupCount: 1},
	}
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
	weights, err := readFixtureWeights(file, spec)
	if err != nil {
		t.Fatal(err)
	}
	if !weights.Layers[0].Recurrent || weights.Layers[0].SSMTimeStepNorm == nil ||
		weights.Layers[0].SSMBNorm == nil || weights.Layers[0].SSMCNorm == nil ||
		weights.Layers[0].FeedForwardGate == nil || weights.Layers[1].Recurrent ||
		weights.Layers[1].AttentionQ == nil || weights.Layers[1].FeedForwardRouter == nil {
		t.Fatalf("unexpected Jamba catalog: %+v", weights.Layers)
	}
}

func TestReadWeightsGraniteHybrid(t *testing.T) {
	spec := Spec{CommonSpec: CommonSpec{Architecture: "granitehybrid", BlockCount: 2, EmbeddingLength: 4,
		FeedForwardLength: 6,

		VocabularySize: 32}, AttentionSpec: AttentionSpec{HeadCount: 2, HeadCountKV: 1, KeyLength: 2, ValueLength: 2,
		LayerKVHeadCounts: []uint32{0, 1}}, MoESpec: MoESpec{ExpertCount: 4, ExpertUsedCount: 2, ExpertFeedForward: 6,
		SharedExpertFF: 5}, RecurrentSpec: RecurrentSpec{RecurrentLayers: []bool{true, false},
		SSMConvKernel: 3, SSMInnerSize: 8, SSMStateSize: 2, SSMTimeStepRank: 4,
		SSMGroupCount: 2},
	}
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
	weights, err := readFixtureWeights(file, spec)
	if err != nil {
		t.Fatal(err)
	}
	recurrent, attention := weights.Layers[0], weights.Layers[1]
	if !recurrent.Recurrent || recurrent.SSMInput == nil || recurrent.SSMConv1DBias != nil ||
		recurrent.SSMNorm == nil || recurrent.FeedForwardRouter == nil ||
		recurrent.FeedForwardGateExperts != nil || recurrent.FeedForwardSharedDown == nil ||
		attention.Recurrent || attention.AttentionQ == nil || attention.AttentionOutputBias == nil ||
		attention.FeedForwardGateExperts == nil || attention.FeedForwardSharedDown == nil {
		t.Fatalf("unexpected Granite Hybrid catalog: %+v", weights.Layers)
	}
}

func TestReadWeightsPLaMo2(t *testing.T) {
	spec := Spec{CommonSpec: CommonSpec{Architecture: "plamo2", BlockCount: 2, EmbeddingLength: 4,
		FeedForwardLength: 6,

		VocabularySize: 32}, AttentionSpec: AttentionSpec{HeadCount: 2, HeadCountKV: 1, KeyLength: 2, ValueLength: 2,
		LayerKVHeadCounts: []uint32{0, 1}}, RecurrentSpec: RecurrentSpec{RecurrentLayers: []bool{true, false},
		SSMConvKernel: 3, SSMInnerSize: 8, SSMStateSize: 2, SSMTimeStepRank: 4,
		SSMGroupCount: 0}}
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
	weights, err := readFixtureWeights(file, spec)
	if err != nil {
		t.Fatal(err)
	}
	recurrent, attention := weights.Layers[0], weights.Layers[1]
	if !recurrent.Recurrent || recurrent.SSMInput == nil || recurrent.SSMConv1DBias != nil ||
		recurrent.SSMX == nil || recurrent.SSMTimeStepNorm == nil || recurrent.SSMBNorm == nil ||
		recurrent.SSMCNorm == nil || recurrent.AttentionPostNorm == nil || recurrent.FeedForwardPostNorm == nil ||
		recurrent.FeedForwardGate != nil || recurrent.FeedForwardUp.Shape[1] != 12 ||
		attention.Recurrent || attention.AttentionQKV == nil || attention.AttentionQNorm == nil ||
		attention.AttentionKNorm == nil || attention.FeedForwardUp.Shape[1] != 12 {
		t.Fatalf("unexpected PLaMo2 catalog: %+v", weights.Layers)
	}
}

func testReadWeightsDeepSeek2FamilyAbsorbedMLA(t *testing.T, architecture string) {
	spec := Spec{CommonSpec: CommonSpec{Architecture: architecture, BlockCount: 2, EmbeddingLength: 8,
		FeedForwardLength: 12,
		VocabularySize:    32}, AttentionSpec: AttentionSpec{HeadCount: 2, HeadCountKV: 2, KeyLength: 6, ValueLength: 4,
		QLoRARank: 3, KVLoRARank: 3, RopeDimensionCount: 2}, MoESpec: MoESpec{LeadingDenseBlocks: 1, ExpertCount: 4, ExpertUsedCount: 2, ExpertFeedForward: 6,
		SharedExpertCount: 1, SharedExpertFF: 6}}
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
	weights, err := readFixtureWeights(&gguf.File{Tensors: tensors}, spec)
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
	spec := Spec{CommonSpec: CommonSpec{Architecture: "qwen2",
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
	weights, err := readFixtureWeights(file, spec)
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
			spec := Spec{CommonSpec: CommonSpec{Architecture: architecture, BlockCount: 1, ContextLength: 16,
				EmbeddingLength: 8, FeedForwardLength: 16,
				VocabularySize:   32,
				LayerNormEpsilon: 1e-5}, AttentionSpec: AttentionSpec{HeadCount: 2,
				HeadCountKV: 2, KeyLength: 4, ValueLength: 4,
				RopeDisabled: true},
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
			weights, err := readFixtureWeights(file, spec)
			if err != nil {
				t.Fatal(err)
			}
			layer := weights.Layers[0]
			if weights.PositionEmbedding == nil || layer.AttentionQKV == nil ||
				layer.AttentionQKVBias == nil || layer.FeedForwardGate != nil {
				t.Fatalf("unexpected %s weights: %+v", architecture, weights)
			}
		})
	}
}

func TestReadWeightsBloom(t *testing.T) {
	spec := Spec{CommonSpec: CommonSpec{Architecture: "bloom", BlockCount: 1, ContextLength: 16,
		EmbeddingLength: 8, FeedForwardLength: 16,
		VocabularySize:   32,
		LayerNormEpsilon: 1e-5}, AttentionSpec: AttentionSpec{HeadCount: 2,
		HeadCountKV: 2, KeyLength: 4, ValueLength: 4,
		RopeDisabled: true, MaxALiBiBias: 8},
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
	weights, err := readFixtureWeights(file, spec)
	if err != nil {
		t.Fatal(err)
	}
	if weights.TokenEmbeddingNorm == nil || weights.TokenEmbeddingNormBias == nil ||
		weights.Layers[0].AttentionQKV == nil || weights.Layers[0].AttentionQKVBias == nil {
		t.Fatalf("unexpected Bloom weights: %+v", weights)
	}
}

func TestReadWeightsMPTBiasFreeVariant(t *testing.T) {
	spec := Spec{CommonSpec: CommonSpec{Architecture: "mpt", BlockCount: 1, ContextLength: 16,
		EmbeddingLength: 8, FeedForwardLength: 16,
		VocabularySize:   32,
		LayerNormEpsilon: 1e-5}, AttentionSpec: AttentionSpec{HeadCount: 2,
		HeadCountKV: 2, KeyLength: 4, ValueLength: 4,
		RopeDisabled: true, MaxALiBiBias: 8},
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
	weights, err := readFixtureWeights(file, spec)
	if err != nil {
		t.Fatal(err)
	}
	if weights.PositionEmbedding == nil || weights.OutputNormBias != nil ||
		weights.Layers[0].AttentionQKV == nil || weights.Layers[0].AttentionNormBias != nil {
		t.Fatalf("unexpected MPT weights: %+v", weights)
	}
}

func TestReadWeightsMPTQKLayerNormVariant(t *testing.T) {
	spec := Spec{CommonSpec: CommonSpec{Architecture: "mpt", BlockCount: 1, ContextLength: 16,
		EmbeddingLength: 8, FeedForwardLength: 16,
		VocabularySize:   32,
		LayerNormEpsilon: 1e-5}, AttentionSpec: AttentionSpec{HeadCount: 2,
		HeadCountKV: 2, KeyLength: 4, ValueLength: 4,
		RopeDisabled: true, MaxALiBiBias: 8},
	}
	file := &gguf.File{Tensors: []gguf.TensorInfo{
		tensorInfo("token_embd.weight", 8, 32), tensorInfo("output_norm.weight", 8),
		tensorInfo("blk.0.attn_norm.weight", 8),
		tensorInfo("blk.0.attn_qkv.weight", 8, 24),
		tensorInfo("blk.0.attn_q_norm.weight", 8), tensorInfo("blk.0.attn_q_norm.bias", 8),
		tensorInfo("blk.0.attn_k_norm.weight", 8), tensorInfo("blk.0.attn_k_norm.bias", 8),
		tensorInfo("blk.0.attn_output.weight", 8, 8),
		tensorInfo("blk.0.ffn_norm.weight", 8), tensorInfo("blk.0.ffn_up.weight", 8, 16),
		tensorInfo("blk.0.ffn_act.scales", 16),
		tensorInfo("blk.0.ffn_down.weight", 16, 8),
	}}
	weights, err := readFixtureWeights(file, spec)
	if err != nil {
		t.Fatal(err)
	}
	layer := weights.Layers[0]
	if layer.AttentionQNorm == nil || layer.AttentionQNormBias == nil ||
		layer.AttentionKNorm == nil || layer.AttentionKNormBias == nil ||
		layer.FeedForwardActivationScale == nil {
		t.Fatalf("MPT variant tensors were not admitted: %+v", layer)
	}
}

func TestReadWeightsRejectsIncompleteMPTVariants(t *testing.T) {
	spec := Spec{CommonSpec: CommonSpec{Architecture: "mpt", BlockCount: 1, ContextLength: 16,
		EmbeddingLength: 8, FeedForwardLength: 16,
		VocabularySize:   32,
		LayerNormEpsilon: 1e-5}, AttentionSpec: AttentionSpec{HeadCount: 2,
		HeadCountKV: 2, KeyLength: 4, ValueLength: 4,
		RopeDisabled: true},
	}
	base := []gguf.TensorInfo{
		tensorInfo("token_embd.weight", 8, 32), tensorInfo("output_norm.weight", 8),
		tensorInfo("blk.0.attn_norm.weight", 8), tensorInfo("blk.0.attn_qkv.weight", 8, 24),
		tensorInfo("blk.0.attn_output.weight", 8, 8), tensorInfo("blk.0.ffn_norm.weight", 8),
		tensorInfo("blk.0.ffn_up.weight", 8, 16), tensorInfo("blk.0.ffn_down.weight", 16, 8),
	}
	for _, test := range []struct {
		name  string
		extra gguf.TensorInfo
	}{
		{name: "Q norm without K norm", extra: tensorInfo("blk.0.attn_q_norm.weight", 8)},
		{name: "activation scale width", extra: tensorInfo("blk.0.ffn_act.scales", 8)},
	} {
		t.Run(test.name, func(t *testing.T) {
			items := append(append([]gguf.TensorInfo(nil), base...), test.extra)
			if _, err := readFixtureWeights(&gguf.File{Tensors: items}, spec); err == nil {
				t.Fatal("invalid MPT variant was accepted")
			}
		})
	}
}

func TestReadWeightsDenseRefact(t *testing.T) {
	spec := Spec{CommonSpec: CommonSpec{Architecture: "refact", BlockCount: 1, ContextLength: 16,
		EmbeddingLength: 8, FeedForwardLength: 16,
		VocabularySize: 32,
		RMSNormEpsilon: 1e-5}, AttentionSpec: AttentionSpec{HeadCount: 2,
		HeadCountKV: 1, KeyLength: 4, ValueLength: 4,
		RopeDisabled: true, MaxALiBiBias: 8},
	}
	file := &gguf.File{Tensors: []gguf.TensorInfo{
		tensorInfo("token_embd.weight", 8, 32), tensorInfo("output_norm.weight", 8),
		tensorInfo("blk.0.attn_norm.weight", 8),
		tensorInfo("blk.0.attn_q.weight", 8, 8), tensorInfo("blk.0.attn_k.weight", 8, 4),
		tensorInfo("blk.0.attn_v.weight", 8, 4), tensorInfo("blk.0.attn_output.weight", 8, 8),
		tensorInfo("blk.0.ffn_norm.weight", 8), tensorInfo("blk.0.ffn_gate.weight", 8, 16),
		tensorInfo("blk.0.ffn_up.weight", 8, 16), tensorInfo("blk.0.ffn_down.weight", 16, 8),
	}}
	weights, err := readFixtureWeights(file, spec)
	if err != nil {
		t.Fatal(err)
	}
	if len(weights.Layers) != 1 || weights.Layers[0].FeedForwardGate == nil {
		t.Fatalf("unexpected Refact weights: %+v", weights)
	}
}

func TestReadWeightsRefactExperts(t *testing.T) {
	spec := Spec{CommonSpec: CommonSpec{Architecture: "refact", BlockCount: 1, ContextLength: 16,
		EmbeddingLength: 8, FeedForwardLength: 16, VocabularySize: 32,
		RMSNormEpsilon: 1e-5}, AttentionSpec: AttentionSpec{HeadCount: 2,
		HeadCountKV: 1, KeyLength: 4, ValueLength: 4, RopeDisabled: true, MaxALiBiBias: 8},
		MoESpec: MoESpec{ExpertCount: 4, ExpertUsedCount: 2, ExpertFeedForward: 16,
			ExpertWeightsNorm: true, ExpertWeightsScale: 1},
	}
	tensors := []gguf.TensorInfo{
		tensorInfo("token_embd.weight", 8, 32), tensorInfo("output_norm.weight", 8),
		tensorInfo("blk.0.attn_norm.weight", 8),
		tensorInfo("blk.0.attn_q.weight", 8, 8), tensorInfo("blk.0.attn_k.weight", 8, 4),
		tensorInfo("blk.0.attn_v.weight", 8, 4), tensorInfo("blk.0.attn_output.weight", 8, 8),
		tensorInfo("blk.0.ffn_norm.weight", 8), tensorInfo("blk.0.ffn_gate_inp.weight", 8, 4),
		tensorInfo("blk.0.ffn_up_exps.weight", 8, 16, 4),
		tensorInfo("blk.0.ffn_down_exps.weight", 16, 8, 4),
	}
	weights, err := readFixtureWeights(&gguf.File{Tensors: tensors}, spec)
	if err != nil {
		t.Fatal(err)
	}
	layer := weights.Layers[0]
	if layer.FeedForwardRouter == nil || layer.FeedForwardUpExperts == nil ||
		layer.FeedForwardDownExperts == nil || layer.FeedForwardGateExperts != nil {
		t.Fatalf("unexpected ungated Refact experts: %+v", layer)
	}
	tensors = append(tensors, tensorInfo("blk.0.ffn_gate_exps.weight", 8, 16, 4))
	weights, err = readFixtureWeights(&gguf.File{Tensors: tensors}, spec)
	if err != nil {
		t.Fatal(err)
	}
	if weights.Layers[0].FeedForwardGateExperts == nil {
		t.Fatal("gated Refact experts were not loaded")
	}
}

func TestReadWeightsInternLM2EXAONEAndXVERSE(t *testing.T) {
	for _, architecture := range []string{"internlm2", "exaone", "xverse"} {
		t.Run(architecture, func(t *testing.T) {
			spec := Spec{CommonSpec: CommonSpec{Architecture: architecture,
				BlockCount:        1,
				EmbeddingLength:   8,
				FeedForwardLength: 16,

				VocabularySize: 32}, AttentionSpec: AttentionSpec{HeadCount: 2,
				HeadCountKV: 1,
				KeyLength:   4,
				ValueLength: 4},
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
			weights, err := readFixtureWeights(&gguf.File{Tensors: tensors}, spec)
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
		spec := Spec{CommonSpec: CommonSpec{Architecture: architecture, EmbeddingLength: 8, VocabularySize: 32}}
		_, err := readFixtureWeights(&gguf.File{Tensors: []gguf.TensorInfo{
			tensorInfo("token_embd.weight", 8, 32),
			tensorInfo("output_norm.weight", 8),
		}}, spec)
		if err == nil || !strings.Contains(err.Error(), "output.weight") {
			t.Fatalf("%s error = %v, want required output weight", architecture, err)
		}
	}
}

func TestReadWeightsOLMo2PostNormalizedBlock(t *testing.T) {
	spec := Spec{CommonSpec: CommonSpec{Architecture: "olmo2",
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
	weights, err := readFixtureWeights(file, spec)
	if err != nil {
		t.Fatal(err)
	}
	layer := weights.Layers[0]
	if layer.AttentionNorm != nil ||
		layer.FeedForwardNorm != nil ||
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
	spec := Spec{CommonSpec: CommonSpec{Architecture: "smollm3",
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
	weights, err := readFixtureWeights(file, spec)
	if err != nil {
		t.Fatal(err)
	}
	if weights.Output != nil || len(weights.Layers) != 1 {
		t.Fatalf("unexpected tied-output SmolLM3 weights: %+v", weights)
	}
}

func TestReadWeightsMiniCPM(t *testing.T) {
	spec, file := denseBiasedWeightFixture("minicpm")
	weights, err := readFixtureWeights(file, spec)
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

func TestReadWeightsNormalRoPELongRoPESelectsContextFactors(t *testing.T) {
	for _, architecture := range []string{"llama", "llama-embed", "minicpm", "mistral3"} {
		t.Run(architecture, func(t *testing.T) {
			tensors := []gguf.TensorInfo{
				tensorInfo("token_embd.weight", 8, 32), tensorInfo("output_norm.weight", 8),
				tensorInfo("blk.0.attn_norm.weight", 8), tensorInfo("blk.0.attn_q.weight", 8, 8),
				tensorInfo("blk.0.attn_k.weight", 8, 4), tensorInfo("blk.0.attn_v.weight", 8, 4),
				tensorInfo("blk.0.attn_output.weight", 8, 8), tensorInfo("blk.0.ffn_norm.weight", 8),
				tensorInfo("blk.0.ffn_gate.weight", 8, 16), tensorInfo("blk.0.ffn_up.weight", 8, 16),
				tensorInfo("blk.0.ffn_down.weight", 16, 8),
				tensorInfo("blk.0.rope_factors_long.weight", 2),
				tensorInfo("blk.0.rope_factors_short.weight", 2),
			}
			for _, context := range []uint32{2048, 8192} {
				spec := Spec{CommonSpec: CommonSpec{Architecture: architecture, BlockCount: 1, ContextLength: context,
					EmbeddingLength: 8, FeedForwardLength: 16,

					VocabularySize: 32}, AttentionSpec: AttentionSpec{OriginalContextLength: 4096,
					HeadCount: 2, HeadCountKV: 1, KeyLength: 4, ValueLength: 4,
					RopeDimensionCount: 4, RopeScalingType: "longrope"},
				}
				weights, err := readFixtureWeights(&gguf.File{Tensors: tensors}, spec)
				if err != nil {
					t.Fatal(err)
				}
				want := "blk.0.rope_factors_short.weight"
				if context > spec.OriginalContextLength {
					want = "blk.0.rope_factors_long.weight"
				}
				if weights.Layers[0].RopeFactors == nil || weights.Layers[0].RopeFactors.Name != want {
					t.Fatalf("context %d factors = %v, want %q", context, weights.Layers[0].RopeFactors, want)
				}
			}
		})
	}
}
