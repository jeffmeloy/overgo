package model

import (
	"fmt"

	"overgo/internal/gguf"

	"overgo/internal/tensor/dtype"

	"os"

	"testing"
)

func TestReadWeightsRWKV7Family(t *testing.T) {
	for _, architecture := range []string{"rwkv7", "arwkv7"} {
		t.Run(architecture, func(t *testing.T) {
			classic := architecture == "rwkv7"
			spec := Spec{CommonSpec: CommonSpec{Architecture: architecture, BlockCount: 1, EmbeddingLength: 8,
				FeedForwardLength: 12,

				VocabularySize: 32}, AttentionSpec: AttentionSpec{HeadCount: 2, HeadCountKV: 2}, RecurrentSpec: RecurrentSpec{WKVHeadSize: 4, DecayLoRARank: 3, ICLRLoRARank: 2,
				ValueMixLoRARank: 3},
			}
			if classic {
				spec.GateLoRARank = 2
			}
			tensors := []gguf.TensorInfo{
				tensorInfo("token_embd.weight", 8, 32), tensorInfo("output_norm.weight", 8),
				tensorInfo("output.weight", 8, 32), tensorInfo("blk.0.attn_norm.weight", 8),
				tensorInfo("blk.0.time_mix_w0.weight", 8), tensorInfo("blk.0.time_mix_w1.weight", 8, 3),
				tensorInfo("blk.0.time_mix_w2.weight", 3, 8), tensorInfo("blk.0.time_mix_a0.weight", 8),
				tensorInfo("blk.0.time_mix_a1.weight", 8, 2), tensorInfo("blk.0.time_mix_a2.weight", 2, 8),
				tensorInfo("blk.0.time_mix_v0.weight", 8), tensorInfo("blk.0.time_mix_v1.weight", 8, 2),
				tensorInfo("blk.0.time_mix_v2.weight", 2, 8),
				tensorInfo("blk.0.time_mix_k_k.weight", 8), tensorInfo("blk.0.time_mix_k_a.weight", 8),
				tensorInfo("blk.0.time_mix_r_k.weight", 8), tensorInfo("blk.0.time_mix_key.weight", 8, 8),
				tensorInfo("blk.0.time_mix_value.weight", 8, 8), tensorInfo("blk.0.time_mix_receptance.weight", 8, 8),
				tensorInfo("blk.0.time_mix_output.weight", 8, 8),
			}
			if classic {
				tensors = append(tensors,
					tensorInfo("token_embd_norm.weight", 8), tensorInfo("token_embd_norm.bias", 8),
					tensorInfo("output_norm.bias", 8), tensorInfo("blk.0.attn_norm.bias", 8),
					tensorInfo("blk.0.time_mix_lerp_fused.weight", 8, 1, 1, 6),
					tensorInfo("blk.0.time_mix_g1.weight", 8, 2), tensorInfo("blk.0.time_mix_g2.weight", 2, 8),
					tensorInfo("blk.0.attn_norm_2.weight", 8), tensorInfo("blk.0.attn_norm_2.bias", 8),
					tensorInfo("blk.0.time_mix_ln.weight", 8), tensorInfo("blk.0.time_mix_ln.bias", 8),
					tensorInfo("blk.0.channel_mix_lerp_k.weight", 8, 1, 1),
					tensorInfo("blk.0.channel_mix_key.weight", 8, 12), tensorInfo("blk.0.channel_mix_value.weight", 12, 8),
				)
			} else {
				tensors = append(tensors,
					tensorInfo("blk.0.time_mix_lerp_fused.weight", 8, 1, 1, 5),
					tensorInfo("blk.0.ffn_norm.weight", 8), tensorInfo("blk.0.ffn_gate.weight", 8, 12),
					tensorInfo("blk.0.ffn_up.weight", 8, 12), tensorInfo("blk.0.ffn_down.weight", 12, 8),
				)
			}
			weights, err := readFixtureWeights(&gguf.File{Tensors: tensors}, spec)
			if err != nil {
				t.Fatal(err)
			}
			layer := weights.Layers[0]
			if !layer.Recurrent || layer.TimeMixW0 == nil || layer.TimeMixA2 == nil ||
				layer.TimeMixV2 == nil || layer.TimeMixKK == nil || layer.TimeMixRK == nil ||
				layer.TimeMixLerpFused == nil || weights.Output == nil {
				t.Fatalf("unexpected %s catalog: %+v", architecture, layer)
			}
			if classic && (layer.TimeMixG2 == nil || layer.TimeMixLN == nil || layer.ChannelMixKey == nil ||
				weights.TokenEmbeddingNorm == nil || weights.OutputNormBias == nil) {
				t.Fatalf("incomplete WKV7 catalog: %+v", layer)
			}
			if !classic && (layer.TimeMixG1 != nil || layer.FeedForwardGate == nil || weights.TokenEmbeddingNorm != nil) {
				t.Fatalf("incomplete ARWKV7 catalog: %+v", layer)
			}
		})
	}
}

func TestReadWeightsQwen35MoEAttention(t *testing.T) {
	spec := Spec{CommonSpec: CommonSpec{Architecture: "qwen35moe", BlockCount: 1, EmbeddingLength: 8,
		FeedForwardLength: 12,
		VocabularySize:    32}, AttentionSpec: AttentionSpec{HeadCount: 2, HeadCountKV: 1, KeyLength: 4,
		ValueLength: 4}, MoESpec: MoESpec{ExpertCount: 4, ExpertUsedCount: 2,
		ExpertFeedForward: 6, SharedExpertFF: 10, ExpertWeightsScale: 1.25}, RecurrentSpec: RecurrentSpec{SSMConvKernel: 3, SSMInnerSize: 4, SSMStateSize: 2, SSMTimeStepRank: 2,
		SSMGroupCount: 1, RecurrentLayers: []bool{false}},
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
	weights, err := readFixtureWeights(file, spec)
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
			spec := Spec{CommonSpec: CommonSpec{Architecture: "qwen3next", BlockCount: 1, EmbeddingLength: 8,
				FeedForwardLength: 12,
				VocabularySize:    32}, AttentionSpec: AttentionSpec{HeadCount: 2, HeadCountKV: 1, KeyLength: 4,
				ValueLength: 4}, MoESpec: MoESpec{ExpertCount: 4, ExpertUsedCount: 2,
				ExpertFeedForward: 6, SharedExpertFF: 10, ExpertWeightsScale: 1.25}, RecurrentSpec: RecurrentSpec{SSMConvKernel: 3, SSMInnerSize: 4, SSMStateSize: 2, SSMTimeStepRank: 2,
				SSMGroupCount: 1, RecurrentLayers: []bool{true}},
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
			weights, err := readFixtureWeights(&gguf.File{Tensors: tensors}, spec)
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
	requireIntegration(t)
	path := os.Getenv("OVERGO_QWEN35_MODEL")
	if path == "" {
		t.Skip("set OVERGO_QWEN35_MODEL to run real Qwen3.5 catalog validation")
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
	weights, err := readFixtureWeights(file, spec)
	if err != nil {
		t.Fatal(err)
	}
	if spec.Architecture != "qwen35" || len(weights.Layers) != 32 ||
		!weights.Layers[0].Recurrent || weights.Layers[3].Recurrent {
		t.Fatalf("unexpected real Qwen3.5 catalog: spec=%+v layers=%d", spec, len(weights.Layers))
	}
}

func TestReadRealKimiLinearCatalog(t *testing.T) {
	requireIntegration(t)
	path := os.Getenv("OVERGO_KIMI_LINEAR_MODEL")
	if path == "" {
		t.Skip("set OVERGO_KIMI_LINEAR_MODEL to run real Kimi Linear catalog validation")
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
	weights, err := readFixtureWeights(file, spec)
	if err != nil {
		t.Fatal(err)
	}
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

func TestReadRealGemma3Catalog(t *testing.T) {
	requireIntegration(t)
	path := os.Getenv("OVERGO_GEMMA3_MODEL")
	if path == "" {
		t.Skip("set OVERGO_GEMMA3_MODEL to run real Gemma 3 catalog validation")
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
	weights, err := readFixtureWeights(file, spec)
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
	spec := Spec{CommonSpec: CommonSpec{Architecture: "t5encoder",
		BlockCount:        1,
		EmbeddingLength:   8,
		FeedForwardLength: 16,

		VocabularySize: 32}, AttentionSpec: AttentionSpec{HeadCount: 2,
		HeadCountKV: 2,
		KeyLength:   4,
		ValueLength: 4}, EncoderSpec: EncoderSpec{RelativeBuckets: 4},
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
	weights, err := readFixtureWeights(file, spec)
	if err != nil {
		t.Fatal(err)
	}
	if weights.OutputNorm.Name != "enc.output_norm.weight" ||
		len(weights.Layers) != 1 ||
		weights.Layers[0].AttentionRelativeBias == nil {
		t.Fatalf("unexpected T5 encoder weights: %+v", weights)
	}
}

func TestReadWeightsT5EncoderDecoder(t *testing.T) {
	spec := Spec{CommonSpec: CommonSpec{Architecture: "t5", BlockCount: 2,
		EmbeddingLength: 8, FeedForwardLength: 16,

		VocabularySize: 32}, AttentionSpec: AttentionSpec{HeadCount: 2, HeadCountKV: 2, KeyLength: 4, ValueLength: 4}, EncoderSpec: EncoderSpec{DecoderBlockCount: 2,

		RelativeBuckets: 4},
	}
	tensors := []gguf.TensorInfo{
		tensorInfo("token_embd.weight", 8, 32),
		tensorInfo("enc.output_norm.weight", 8),
		tensorInfo("dec.output_norm.weight", 8),
	}
	for block := 0; block < 2; block++ {
		for _, prefix := range []string{fmt.Sprintf("enc.blk.%d.", block), fmt.Sprintf("dec.blk.%d.", block)} {
			tensors = append(tensors,
				tensorInfo(prefix+"attn_norm.weight", 8),
				tensorInfo(prefix+"attn_q.weight", 8, 8),
				tensorInfo(prefix+"attn_k.weight", 8, 8),
				tensorInfo(prefix+"attn_v.weight", 8, 8),
				tensorInfo(prefix+"attn_o.weight", 8, 8),
				tensorInfo(prefix+"ffn_norm.weight", 8),
				tensorInfo(prefix+"ffn_up.weight", 8, 16),
				tensorInfo(prefix+"ffn_down.weight", 16, 8),
			)
			if block == 0 {
				tensors = append(tensors, tensorInfo(prefix+"attn_rel_b.weight", 2, 4))
			}
		}
		prefix := fmt.Sprintf("dec.blk.%d.", block)
		tensors = append(tensors,
			tensorInfo(prefix+"cross_attn_norm.weight", 8),
			tensorInfo(prefix+"cross_attn_q.weight", 8, 8),
			tensorInfo(prefix+"cross_attn_k.weight", 8, 8),
			tensorInfo(prefix+"cross_attn_v.weight", 8, 8),
			tensorInfo(prefix+"cross_attn_o.weight", 8, 8),
		)
	}
	weights, err := readFixtureWeights(&gguf.File{Tensors: tensors}, spec)
	if err != nil {
		t.Fatal(err)
	}
	if len(weights.EncoderLayers) != 2 || len(weights.Layers) != 2 ||
		weights.EncoderOutputNorm == nil || weights.Layers[0].CrossAttentionQ == nil ||
		weights.EncoderLayers[1].AttentionRelativeBias == nil ||
		weights.Layers[1].AttentionRelativeBias == nil ||
		weights.Layers[0].FeedForwardGate != nil {
		t.Fatalf("unexpected T5 weights: %+v", weights)
	}
}

func TestReadWeightsWavTokenizerDecoder(t *testing.T) {
	spec := Spec{CommonSpec: CommonSpec{Architecture: "wavtokenizer-dec", EmbeddingLength: 4, VocabularySize: 8,
		OutputEmbeddingLength: 3,
		FeedForwardLength:     10}, MultimodalSpec: MultimodalSpec{PosNetEmbeddingLength: 6, PosNetBlockCount: 6,
		ConvNextEmbeddingLength: 6, ConvNextBlockCount: 2},
	}
	tensors := []gguf.TensorInfo{
		tensorInfo("token_embd.weight", 4, 8),
		tensorInfo("conv1d.weight", 7, 4, 6), tensorInfo("conv1d.bias", 1, 6),
		tensorInfo("token_embd_norm.weight", 6), tensorInfo("token_embd_norm.bias", 6),
		tensorInfo("output_norm.weight", 6), tensorInfo("output_norm.bias", 6),
		tensorInfo("output.weight", 6, 3), tensorInfo("output.bias", 3),
	}
	for _, block := range []int{0, 1, 3, 4} {
		prefix := fmt.Sprintf("posnet.%d.", block)
		tensors = append(tensors,
			tensorInfo(prefix+"norm1.weight", 1, 6), tensorInfo(prefix+"norm1.bias", 1, 6),
			tensorInfo(prefix+"conv1.weight", 3, 6, 6), tensorInfo(prefix+"conv1.bias", 1, 6),
			tensorInfo(prefix+"norm2.weight", 1, 6), tensorInfo(prefix+"norm2.bias", 1, 6),
			tensorInfo(prefix+"conv2.weight", 3, 6, 6), tensorInfo(prefix+"conv2.bias", 1, 6),
		)
	}
	tensors = append(tensors,
		tensorInfo("posnet.2.attn_norm.weight", 1, 6), tensorInfo("posnet.2.attn_norm.bias", 1, 6),
		tensorInfo("posnet.2.attn_q.weight", 1, 6, 6), tensorInfo("posnet.2.attn_q.bias", 1, 6),
		tensorInfo("posnet.2.attn_k.weight", 1, 6, 6), tensorInfo("posnet.2.attn_k.bias", 1, 6),
		tensorInfo("posnet.2.attn_v.weight", 1, 6, 6), tensorInfo("posnet.2.attn_v.bias", 1, 6),
		tensorInfo("posnet.2.attn_output.weight", 1, 6, 6), tensorInfo("posnet.2.attn_output.bias", 1, 6),
		tensorInfo("posnet.5.attn_norm.weight", 1, 6), tensorInfo("posnet.5.attn_norm.bias", 1, 6),
	)
	for block := 0; block < 2; block++ {
		prefix := fmt.Sprintf("convnext.%d.", block)
		tensors = append(tensors,
			tensorInfo(prefix+"dw.weight", 7, 1, 6), tensorInfo(prefix+"dw.bias", 1, 6),
			tensorInfo(prefix+"norm.weight", 6), tensorInfo(prefix+"norm.bias", 6),
			tensorInfo(prefix+"pw1.weight", 6, 10), tensorInfo(prefix+"pw1.bias", 10),
			tensorInfo(prefix+"pw2.weight", 10, 6), tensorInfo(prefix+"pw2.bias", 6),
			tensorInfo(prefix+"gamma.weight", 6),
		)
	}
	weights, err := readFixtureWeights(&gguf.File{Tensors: tensors}, spec)
	if err != nil {
		t.Fatal(err)
	}
	if weights.AudioDecoder == nil || len(weights.AudioDecoder.PosNet) != 6 ||
		len(weights.AudioDecoder.ConvNext) != 2 || weights.AudioDecoder.PosNet[2].AttentionQ.Name == "" ||
		weights.AudioDecoder.PosNet[5].AttentionNorm.Name == "" || weights.Output == nil ||
		weights.Output.Shape[1] != 3 {
		t.Fatalf("unexpected AudioDecoder decoder weights: %+v", weights.AudioDecoder)
	}
}

func TestReadWeightsDFlash(t *testing.T) {
	spec := Spec{CommonSpec: CommonSpec{Architecture: "dflash", BlockCount: 2, EmbeddingLength: 8,
		FeedForwardLength: 16,
		TargetLayers:      []int32{2, 7}}, AttentionSpec: AttentionSpec{HeadCount: 2, HeadCountKV: 1,
		KeyLength: 4, ValueLength: 4},
	}
	tensors := []gguf.TensorInfo{
		tensorInfo("fc.weight", 16, 8), tensorInfo("enc.output_norm.weight", 8),
		tensorInfo("output_norm.weight", 8),
	}
	for block := 0; block < 2; block++ {
		prefix := fmt.Sprintf("blk.%d.", block)
		tensors = append(tensors,
			tensorInfo(prefix+"attn_norm.weight", 8),
			tensorInfo(prefix+"attn_q.weight", 8, 8), tensorInfo(prefix+"attn_k.weight", 8, 4),
			tensorInfo(prefix+"attn_v.weight", 8, 4), tensorInfo(prefix+"attn_output.weight", 8, 8),
			tensorInfo(prefix+"attn_q_norm.weight", 4), tensorInfo(prefix+"attn_k_norm.weight", 4),
			tensorInfo(prefix+"ffn_norm.weight", 8), tensorInfo(prefix+"ffn_gate.weight", 8, 16),
			tensorInfo(prefix+"ffn_up.weight", 8, 16), tensorInfo(prefix+"ffn_down.weight", 16, 8),
		)
	}
	weights, err := readFixtureWeights(&gguf.File{Tensors: tensors}, spec)
	if err != nil {
		t.Fatal(err)
	}
	if weights.FeatureProjection == nil || weights.EncoderOutputNorm == nil ||
		weights.TokenEmbedding.Name != "" || len(weights.Layers) != 2 ||
		weights.Layers[1].AttentionQNorm == nil || weights.Layers[1].AttentionKNorm == nil {
		t.Fatalf("unexpected DFlash weights: %+v", weights)
	}
}

func TestReadWeightsEagle3(t *testing.T) {
	spec := Spec{CommonSpec: CommonSpec{Architecture: "eagle3", BlockCount: 1, EmbeddingLength: 8,
		FeedForwardLength: 16,

		VocabularySize: 32, TargetHiddenSize: 12, TargetLayers: []int32{2, 7, 11}}, AttentionSpec: AttentionSpec{HeadCount: 2, HeadCountKV: 1,
		KeyLength: 4, ValueLength: 4, RopeDimensionCount: 4},
	}
	tensors := []gguf.TensorInfo{
		tensorInfo("d2t", 24), tensorInfo("fc.weight", 36, 8),
		tensorInfo("token_embd.weight", 8, 32), tensorInfo("output_norm.weight", 8),
		tensorInfo("output.weight", 8, 24),
		tensorInfo("blk.0.attn_norm.weight", 8), tensorInfo("blk.0.attn_norm_2.weight", 8),
		tensorInfo("blk.0.attn_q.weight", 16, 8), tensorInfo("blk.0.attn_k.weight", 16, 4),
		tensorInfo("blk.0.attn_v.weight", 16, 4), tensorInfo("blk.0.attn_output.weight", 8, 8),
		tensorInfo("blk.0.ffn_norm.weight", 8), tensorInfo("blk.0.ffn_gate.weight", 8, 16),
		tensorInfo("blk.0.ffn_up.weight", 8, 16), tensorInfo("blk.0.ffn_down.weight", 16, 8),
		tensorInfo("blk.0.rope_freqs.weight", 2),
	}
	tensors[0].Type = dtype.I64
	weights, err := readFixtureWeights(&gguf.File{Tensors: tensors}, spec)
	if err != nil {
		t.Fatal(err)
	}
	if weights.DraftToTarget == nil || weights.FeatureProjection == nil ||
		weights.TokenEmbedding.Name == "" || weights.Output == nil || len(weights.Layers) != 1 ||
		weights.Layers[0].AttentionNorm2 == nil || weights.Layers[0].RopeFactors == nil {
		t.Fatalf("unexpected Eagle3 weights: %+v", weights)
	}
}

func TestReadWeightsErnie45MoEInterleavesDenseAndExpertLayers(t *testing.T) {
	spec := bindFixtureSpec(Spec{CommonSpec: CommonSpec{Architecture: "ernie4_5-moe", BlockCount: 4, EmbeddingLength: 8,
		FeedForwardLength: 12,
		VocabularySize:    32,
		RMSNormEpsilon:    1e-6}, AttentionSpec: AttentionSpec{HeadCount: 2, HeadCountKV: 1,
		KeyLength: 4, ValueLength: 4}, MoESpec: MoESpec{ExpertCount: 4, ExpertUsedCount: 2,
		ExpertFeedForward: 6, ExpertWeightsScale: 1.25,
		LeadingDenseBlocks: 1, MoELayerStep: 2, SharedExpertFF: 5},
	})
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
	weights, err := readFixtureWeights(&gguf.File{Tensors: tensors}, spec)
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
				layer.FeedForwardSharedGate == nil || layer.FeedForwardGate != nil {
				t.Fatalf("ERNIE MoE layer %d catalog: %+v", block, layer)
			}
			if (block == 1) != (layer.FeedForwardGateExperts == nil) {
				t.Fatalf("ERNIE optional expert gate mismatch at layer %d", block)
			}
		} else if layer.FeedForwardGate == nil || layer.FeedForwardRouter != nil {
			t.Fatalf("ERNIE dense layer %d catalog: %+v", block, layer)
		}
	}
}

func TestReadWeightsPaddleOCR(t *testing.T) {
	testReadWeightsMRoPETextDecoder(t, "paddleocr")
}

func TestReadWeightsGemma4SharedKVMoEAndPerLayerInputs(t *testing.T) {
	spec := Spec{CommonSpec: CommonSpec{Architecture: "gemma4", BlockCount: 4, EmbeddingLength: 8,
		FeedForwardLength: 10,

		VocabularySize: 32,
		RMSNormEpsilon: 1e-6}, AttentionSpec: AttentionSpec{HeadCount: 2, HeadCountKV: 1, LayerKVHeadCounts: []uint32{1, 1, 1, 1},
		KeyLength: 4, ValueLength: 4, KeyLengthSWA: 2, ValueLengthSWA: 2,
		RopeDimensionCount: 4, RopeDimensionSWA: 2,
		SlidingLayers: []bool{true, false, true, false}}, MoESpec: MoESpec{LayerFeedForward: []uint32{10, 11, 12, 13},

		ExpertCount: 4, ExpertUsedCount: 2, ExpertFeedForward: 3, ExpertWeightsScale: 1}, MultimodalSpec: MultimodalSpec{SharedKVLayers: 2, EmbeddingPerLayer: 3},
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
	weights, err := readFixtureWeights(&gguf.File{Tensors: tensors}, spec)
	if err != nil {
		t.Fatal(err)
	}
	if weights.PerLayerTokenEmbedding == nil || weights.Layers[0].AttentionV != nil ||
		weights.Layers[0].FeedForwardGate == nil || weights.Layers[2].FeedForwardGate == nil ||
		weights.Layers[2].AttentionK != nil || weights.Layers[2].FeedForwardRouter == nil ||
		weights.Layers[2].FeedForwardDownExpertsScale == nil ||
		weights.Layers[3].PerLayerProjection == nil {
		t.Fatalf("unexpected Gemma 4 catalog: %+v", weights)
	}
}

func TestReadWeightsGemma4Assistant(t *testing.T) {
	spec := Spec{CommonSpec: CommonSpec{Architecture: "gemma4-assistant", BlockCount: 2, EmbeddingLength: 8,
		TargetHiddenSize: 12, FeedForwardLength: 16,

		VocabularySize: 32}, AttentionSpec: AttentionSpec{HeadCount: 2, HeadCountKV: 1,
		KeyLength: 4, ValueLength: 4, KeyLengthSWA: 2, ValueLengthSWA: 2,
		RopeDimensionCount: 4, RopeDimensionSWA: 2,
		SlidingLayers: []bool{true, false}},
	}
	tensors := []gguf.TensorInfo{
		tensorInfo("token_embd.weight", 8, 32),
		tensorInfo("output_norm.weight", 8),
		tensorInfo("blk.0.nextn.pre_projection.weight", 24, 8),
		tensorInfo("nextn.post_projection.weight", 8, 12),
		tensorInfo("rope_freqs.weight", 2),
	}
	for block := range uint32(2) {
		prefix := fmt.Sprintf("blk.%d.", block)
		key := uint64(spec.LayerKeyLength(block))
		tensors = append(tensors,
			tensorInfo(prefix+"attn_norm.weight", 8),
			tensorInfo(prefix+"attn_q.weight", 8, 2*key),
			tensorInfo(prefix+"attn_output.weight", 2*key, 8),
			tensorInfo(prefix+"ffn_norm.weight", 8),
			tensorInfo(prefix+"ffn_gate.weight", 8, 16),
			tensorInfo(prefix+"ffn_up.weight", 8, 16),
			tensorInfo(prefix+"ffn_down.weight", 16, 8),
			tensorInfo(prefix+"attn_q_norm.weight", key),
			tensorInfo(prefix+"post_attention_norm.weight", 8),
			tensorInfo(prefix+"post_ffw_norm.weight", 8),
			tensorInfo(prefix+"layer_output_scale.weight", 1),
		)
	}
	weights, err := readFixtureWeights(&gguf.File{Tensors: tensors}, spec)
	if err != nil {
		t.Fatal(err)
	}
	if weights.FeatureProjection == nil || weights.FeatureProjectionPost == nil ||
		weights.Output != nil || weights.Layers[0].AttentionK != nil ||
		weights.Layers[0].RopeFactors != nil || weights.Layers[1].RopeFactors == nil ||
		weights.Layers[1].LayerOutputScale == nil {
		t.Fatalf("unexpected Gemma 4 assistant catalog: %+v", weights)
	}
}

func TestReadWeightsGemma3nAltUpAndLaurel(t *testing.T) {
	spec := Spec{CommonSpec: CommonSpec{Architecture: "gemma3n", BlockCount: 21, EmbeddingLength: 8,
		FeedForwardLength: 12,
		VocabularySize:    32,
		RMSNormEpsilon:    1e-6}, AttentionSpec: AttentionSpec{HeadCount: 2, HeadCountKV: 1,
		KeyLength: 4, ValueLength: 4}, MultimodalSpec: MultimodalSpec{EmbeddingPerLayer: 3,
		AltUpCount: 4, AltUpActive: 0, LaurelRank: 2,
		KVFromStart: 20, SharedKVLayers: 1},
	}
	tensors := []gguf.TensorInfo{
		tensorInfo("token_embd.weight", 8, 32),
		tensorInfo("output_norm.weight", 8),
		tensorInfo("per_layer_token_embd.weight", 63, 32),
		tensorInfo("per_layer_model_proj.weight", 8, 63),
		tensorInfo("per_layer_proj_norm.weight", 3),
		tensorInfo("altup_proj.weight", 8, 8, 3),
		tensorInfo("altup_unembd_proj.weight", 8, 8, 3),
	}
	for block := range uint32(21) {
		prefix := fmt.Sprintf("blk.%d.", block)
		tensors = append(tensors,
			tensorInfo(prefix+"attn_norm.weight", 8),
			tensorInfo(prefix+"attn_q.weight", 8, 8),
			tensorInfo(prefix+"attn_output.weight", 8, 8),
			tensorInfo(prefix+"attn_q_norm.weight", 4),
			tensorInfo(prefix+"post_attention_norm.weight", 8),
			tensorInfo(prefix+"ffn_norm.weight", 8),
			tensorInfo(prefix+"ffn_gate.weight", 8, 12),
			tensorInfo(prefix+"ffn_up.weight", 8, 12),
			tensorInfo(prefix+"ffn_down.weight", 12, 8),
			tensorInfo(prefix+"post_ffw_norm.weight", 8),
			tensorInfo(prefix+"per_layer_inp_gate.weight", 8, 3),
			tensorInfo(prefix+"per_layer_proj.weight", 3, 8),
			tensorInfo(prefix+"per_layer_post_norm.weight", 8),
			tensorInfo(prefix+"altup_correct_coef.weight", 4, 4),
			tensorInfo(prefix+"altup_correct_scale.weight", 8),
			tensorInfo(prefix+"altup_predict_coef.weight", 4, 16),
			tensorInfo(prefix+"altup_router.weight", 8, 4),
			tensorInfo(prefix+"altup_router_norm.weight", 8),
			tensorInfo(prefix+"laurel_l.weight", 8, 2),
			tensorInfo(prefix+"laurel_r.weight", 2, 8),
			tensorInfo(prefix+"laurel_post_norm.weight", 8),
		)
		if spec.LayerHasKV(block) {
			tensors = append(tensors,
				tensorInfo(prefix+"attn_k.weight", 8, 4),
				tensorInfo(prefix+"attn_v.weight", 8, 4),
				tensorInfo(prefix+"attn_k_norm.weight", 4),
			)
		}
	}
	weights, err := readFixtureWeights(&gguf.File{Tensors: tensors}, spec)
	if err != nil {
		t.Fatal(err)
	}
	layer := weights.Layers[20]
	if weights.AltUpProjection == nil || weights.AltUpUnembedding == nil ||
		layer.AltUpCorrectCoefficient == nil || layer.AltUpPredictCoefficient == nil ||
		layer.AltUpRouter == nil || layer.LaurelLeft == nil || layer.LaurelRight == nil ||
		layer.PerLayerProjection == nil || layer.AttentionK != nil ||
		layer.AttentionV != nil || layer.AttentionKNorm != nil {
		t.Fatalf("unexpected Gemma 3n catalog: %+v", weights)
	}
}

func TestReadWeightsQwen2VL(t *testing.T) {
	testReadWeightsMRoPETextDecoder(t, "qwen2vl")
}

func TestReadWeightsQwen3VL(t *testing.T) {
	testReadWeightsMRoPETextDecoder(t, "qwen3vl")
}

func TestReadWeightsQwen3VLMoE(t *testing.T) {
	spec := Spec{CommonSpec: CommonSpec{Architecture: "qwen3vlmoe", BlockCount: 1, EmbeddingLength: 8,
		FeedForwardLength: 24,

		VocabularySize: 32, RMSNormEpsilon: 1e-6}, AttentionSpec: AttentionSpec{HeadCount: 2, HeadCountKV: 1, KeyLength: 4, ValueLength: 4}, MoESpec: MoESpec{ExpertCount: 4, ExpertUsedCount: 2,
		ExpertFeedForward: 12, ExpertWeightsScale: 1.25},
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
	weights, err := readFixtureWeights(file, spec)
	if err != nil {
		t.Fatal(err)
	}
	layer := weights.Layers[0]
	if layer.AttentionQKV == nil || layer.AttentionQNorm == nil || layer.AttentionKNorm == nil ||
		layer.FeedForwardRouter == nil || layer.FeedForwardGateExperts == nil ||
		layer.FeedForwardUpExperts == nil || layer.FeedForwardDownExperts == nil ||
		layer.FeedForwardGate != nil || layer.FeedForwardUp != nil || layer.FeedForwardDown != nil {
		t.Fatalf("unexpected Qwen3-VL-MoE catalog: %+v", weights)
	}
}

func testReadWeightsMRoPETextDecoder(t *testing.T, architecture string) {
	spec := Spec{CommonSpec: CommonSpec{Architecture: architecture, BlockCount: 1, EmbeddingLength: 8,
		FeedForwardLength: 12,
		VocabularySize:    32, RMSNormEpsilon: 1e-6}, AttentionSpec: AttentionSpec{HeadCount: 2, HeadCountKV: 1,
		KeyLength: 4, ValueLength: 4},
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
			tensorInfo("cls.output.weight", 8, 2),
		)
		spec.ClassifierLabels = []string{"no", "yes"}
	}
	weights, err := readFixtureWeights(file, spec)
	if err != nil {
		t.Fatal(err)
	}
	layer := weights.Layers[0]
	if weights.Output != nil || layer.AttentionQKV == nil || layer.AttentionOutputBias == nil ||
		layer.FeedForwardNorm == nil || layer.FeedForwardGate == nil ||
		(architecture == "qwen3vl" && (layer.AttentionQNorm == nil || layer.AttentionKNorm == nil ||
			weights.ClassifierOutput == nil)) {
		t.Fatalf("unexpected %s catalog: %+v", architecture, weights)
	}
}

func TestReadRealUMT5Catalog(t *testing.T) {
	requireIntegration(t)
	path := os.Getenv("OVERGO_UMT5_MODEL")
	if path == "" {
		t.Skip("set OVERGO_UMT5_MODEL to run real UMT5 catalog validation")
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
	weights, err := readFixtureWeights(file, spec)
	if err != nil {
		t.Fatal(err)
	}
	if spec.Architecture != "t5encoder" || len(weights.Layers) != 24 ||
		weights.Layers[0].AttentionRelativeBias == nil {
		t.Fatalf("unexpected real UMT5 catalog: spec=%+v layers=%d", spec, len(weights.Layers))
	}
}

func TestReadWeightsTalkieUsesEmbeddingSkipCatalog(t *testing.T) {
	spec := Spec{CommonSpec: CommonSpec{Architecture: "talkie", BlockCount: 1, EmbeddingLength: 8,
		FeedForwardLength: 12,
		VocabularySize:    32, RMSNormEpsilon: 1e-6}, AttentionSpec: AttentionSpec{HeadCount: 2, HeadCountKV: 1,
		KeyLength: 4, ValueLength: 4},
	}
	file := &gguf.File{Tensors: []gguf.TensorInfo{
		tensorInfo("token_embd.weight", 8, 32), tensorInfo("output.weight", 8, 32),
		tensorInfo("blk.0.attn_qkv.weight", 8, 16), tensorInfo("blk.0.attn_qkv.bias", 16),
		tensorInfo("blk.0.attn_output.weight", 8, 8), tensorInfo("blk.0.attn_q_norm.weight", 1, 2),
		tensorInfo("blk.0.ffn_gate.weight", 8, 12), tensorInfo("blk.0.ffn_up.weight", 8, 12),
		tensorInfo("blk.0.ffn_down.weight", 12, 8), tensorInfo("blk.0.layer_out_scale.weight", 1),
	}}
	weights, err := readFixtureWeights(file, spec)
	if err != nil {
		t.Fatal(err)
	}
	layer := weights.Layers[0]
	if weights.Output == nil || weights.OutputNorm.Name != "" || layer.AttentionNorm != nil ||
		layer.FeedForwardNorm != nil || layer.AttentionQKV == nil ||
		layer.AttentionQNorm == nil || layer.AttentionKNorm != nil || layer.LayerOutputScale == nil {
		t.Fatalf("unexpected Talkie catalog: %+v", weights)
	}
}

func TestReadWeightsAcceptsDenseProjectionBiases(t *testing.T) {
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
	weights, err := readFixtureWeights(file, spec)
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
