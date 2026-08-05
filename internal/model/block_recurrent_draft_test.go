package model

import (
	"llamacpp2go/internal/tensor"

	"llamacpp2go/internal/tensor/dtype"

	"testing"
)

func TestBuildKimiLinearKDAAndMLABlocks(t *testing.T) {
	spec := Spec{CommonSpec: CommonSpec{Architecture: "kimi-linear", EmbeddingLength: 8, FeedForwardLength: 12,

		RMSNormEpsilon: 1e-6}, AttentionSpec: AttentionSpec{HeadCount: 2, HeadCountKV: 1, KeyLength: 4, ValueLength: 2,
		RopeDisabled: true, RopeDimensionCount: 2,
		KVLoRARank: 3}, MoESpec: MoESpec{ExpertCount: 4, ExpertUsedCount: 2, ExpertFeedForward: 6,
		SharedExpertFF: 6, LeadingDenseBlocks: 1, ExpertGatingFunc: expertGatingSigmoid,
		ExpertWeightsScale: 1.25, ExpertWeightsNorm: true}, RecurrentSpec: RecurrentSpec{KDAHeadDim: 2, SSMConvKernel: 3, SSMInnerSize: 4},
	}
	t.Run("kda", func(t *testing.T) {
		builder := tensor.NewBuilder()
		input := builder.Input("input", dtype.F32, tensor.MustShape(8, 2))
		weights := kimiLinearCommonInputs(builder, spec, false)
		weights.AttentionQ = builder.Input("q", dtype.F32, tensor.MustShape(8, 4))
		weights.AttentionK = builder.Input("k", dtype.F32, tensor.MustShape(8, 4))
		weights.AttentionV = builder.Input("v", dtype.F32, tensor.MustShape(8, 4))
		weights.AttentionOutput = builder.Input("o", dtype.F32, tensor.MustShape(4, 8))
		weights.SSMQueryConv = builder.Input("cq", dtype.F32, tensor.MustShape(3, 1, 4, 1))
		weights.SSMKeyConv = builder.Input("ck", dtype.F32, tensor.MustShape(3, 1, 4, 1))
		weights.SSMValueConv = builder.Input("cv", dtype.F32, tensor.MustShape(3, 1, 4, 1))
		weights.SSMForgetA = builder.Input("fa", dtype.F32, tensor.MustShape(8, 2))
		weights.SSMForgetB = builder.Input("fb", dtype.F32, tensor.MustShape(2, 4))
		weights.SSMBeta = builder.Input("beta", dtype.F32, tensor.MustShape(8, 2))
		weights.SSMA = builder.Input("a", dtype.F32, tensor.MustShape(1, 2, 1, 1))
		weights.SSMTimeStep = builder.Input("dt", dtype.F32, tensor.MustShape(4))
		weights.SSMOutputGateA = builder.Input("ga", dtype.F32, tensor.MustShape(8, 2))
		weights.SSMOutputGateB = builder.Input("gb", dtype.F32, tensor.MustShape(2, 4))
		weights.SSMNorm = builder.Input("ssm_norm", dtype.F32, tensor.MustShape(2))
		conv := builder.Input("conv", dtype.F32, tensor.MustShape(2, 12))
		state := builder.Input("state", dtype.F32, tensor.MustShape(2, 2, 2, 1))
		result, err := BuildKimiLinearBlockCached(builder, input, spec, weights, []uint32{0, 1}, true, conv, state, 0)
		if err != nil {
			t.Fatal(err)
		}
		if !result.Output.Shape.Equal(input.Shape) || !result.Key.Shape.Equal(conv.Shape) || !result.Value.Shape.Equal(state.Shape) {
			t.Fatalf("unexpected KDA result: %+v", result)
		}
		nodes, err := tensor.Topological(result.Output, result.Key, result.Value)
		if err != nil {
			t.Fatal(err)
		}
		var convolutions, delta, rope int
		for _, node := range nodes {
			if node.Op == tensor.OpSSMConv {
				convolutions++
			}
			if node.Op == tensor.OpGatedDeltaNet {
				delta++
			}
			if node.Op == tensor.OpRoPENormal || node.Op == tensor.OpRoPENeoX {
				rope++
			}
		}
		if convolutions != 3 || delta != 1 || rope != 0 {
			t.Fatalf("KDA ops: convolution=%d delta=%d rope=%d", convolutions, delta, rope)
		}
	})
	t.Run("mla", func(t *testing.T) {
		builder := tensor.NewBuilder()
		input := builder.Input("input", dtype.F32, tensor.MustShape(8, 2))
		weights := kimiLinearCommonInputs(builder, spec, true)
		weights.AttentionQ = builder.Input("q", dtype.F32, tensor.MustShape(8, 8))
		weights.AttentionKVAMQA = builder.Input("kva", dtype.F32, tensor.MustShape(8, 5))
		weights.AttentionKVANorm = builder.Input("kva_norm", dtype.F32, tensor.MustShape(3))
		weights.AttentionKB = builder.Input("kb", dtype.F32, tensor.MustShape(2, 3, 2))
		weights.AttentionVB = builder.Input("vb", dtype.F32, tensor.MustShape(3, 2, 2))
		weights.AttentionOutput = builder.Input("o", dtype.F32, tensor.MustShape(4, 8))
		result, err := BuildKimiLinearBlockCached(builder, input, spec, weights, []uint32{0, 1}, false, nil, nil, 1)
		if err != nil {
			t.Fatal(err)
		}
		if !result.Key.Shape.Equal(tensor.MustShape(5, 1, 2)) || !result.Value.Shape.Equal(tensor.MustShape(3, 1, 2)) {
			t.Fatalf("unexpected MLA cache: %v %v", result.Key.Shape, result.Value.Shape)
		}
		nodes, err := tensor.Topological(result.Output, result.Key, result.Value)
		if err != nil {
			t.Fatal(err)
		}
		var moe, rope int
		for _, node := range nodes {
			if node.Op == tensor.OpMoE {
				moe++
			}
			if node.Op == tensor.OpRoPENormal || node.Op == tensor.OpRoPENeoX {
				rope++
			}
		}
		if moe != 1 || rope != 0 {
			t.Fatalf("MLA ops: MoE=%d RoPE=%d", moe, rope)
		}
	})
}

func TestBuildRWKV6Qwen2Block(t *testing.T) {
	builder := tensor.NewBuilder()
	spec := Spec{CommonSpec: CommonSpec{Architecture: "rwkv6qwen2", EmbeddingLength: 8, FeedForwardLength: 12,

		RMSNormEpsilon: 1e-6}, AttentionSpec: AttentionSpec{HeadCount: 2, HeadCountKV: 1}, RecurrentSpec: RecurrentSpec{WKVHeadSize: 4, TimeMixExtraDim: 3,
		TimeDecayExtraDim: 2, RescaleEvery: 1},
	}
	input := builder.Input("input", dtype.F32, tensor.MustShape(8, 2))
	weights := LayerGraphWeights{
		AttentionNorm:     builder.Input("attn_norm", dtype.F32, tensor.MustShape(8)),
		TimeMixW1:         builder.Input("mix_w1", dtype.F32, tensor.MustShape(8, 15)),
		TimeMixW2:         builder.Input("mix_w2", dtype.F32, tensor.MustShape(3, 8, 5)),
		TimeMixLerpX:      builder.Input("lerp_x", dtype.F32, tensor.MustShape(8, 1, 1)),
		TimeMixLerpFused:  builder.Input("lerp", dtype.F32, tensor.MustShape(8, 1, 1, 5)),
		TimeMixDecay:      builder.Input("decay", dtype.F32, tensor.MustShape(8)),
		TimeMixDecayW1:    builder.Input("decay_w1", dtype.F32, tensor.MustShape(8, 2)),
		TimeMixDecayW2:    builder.Input("decay_w2", dtype.F32, tensor.MustShape(2, 8)),
		TimeMixKey:        builder.Input("key", dtype.F32, tensor.MustShape(8, 4)),
		TimeMixValue:      builder.Input("value", dtype.F32, tensor.MustShape(8, 4)),
		TimeMixReceptance: builder.Input("receptance", dtype.F32, tensor.MustShape(8, 8)),
		TimeMixGate:       builder.Input("gate", dtype.F32, tensor.MustShape(8, 8)),
		TimeMixOutput:     builder.Input("output", dtype.F32, tensor.MustShape(8, 8)),
		FeedForwardNorm:   builder.Input("ffn_norm", dtype.F32, tensor.MustShape(8)),
		FeedForwardGate:   builder.Input("ffn_gate", dtype.F32, tensor.MustShape(8, 12)),
		FeedForwardUp:     builder.Input("ffn_up", dtype.F32, tensor.MustShape(8, 12)),
		FeedForwardDown:   builder.Input("ffn_down", dtype.F32, tensor.MustShape(12, 8)),
	}
	shift := builder.Input("shift", dtype.F32, tensor.MustShape(8))
	state := builder.Input("state", dtype.F32, tensor.MustShape(4, 4, 2, 1))
	result, err := BuildRWKV6Qwen2BlockCached(builder, input, spec, weights, shift, state, 0)
	if err != nil {
		t.Fatal(err)
	}
	if !result.Output.Shape.Equal(input.Shape) || !result.Key.Shape.Equal(shift.Shape) || !result.Value.Shape.Equal(state.Shape) {
		t.Fatalf("unexpected RWKV6-Qwen2 result: %+v", result)
	}
	nodes, err := tensor.Topological(result.Output, result.Key, result.Value)
	if err != nil {
		t.Fatal(err)
	}
	var gla, tanh, exponential int
	for _, node := range nodes {
		switch node.Op {
		case tensor.OpGatedLinearAttention:
			gla++
		case tensor.OpTanh:
			tanh++
		case tensor.OpExp:
			exponential++
		}
	}
	if gla != 1 || tanh != 2 || exponential != 2 {
		t.Fatalf("RWKV6-Qwen2 ops: GLA=%d tanh=%d exp=%d", gla, tanh, exponential)
	}
}

func TestBuildRWKV6Block(t *testing.T) {
	builder := tensor.NewBuilder()
	spec := Spec{CommonSpec: CommonSpec{Architecture: "rwkv6", EmbeddingLength: 8, FeedForwardLength: 12,

		LayerNormEpsilon: 1e-5}, AttentionSpec: AttentionSpec{HeadCount: 2, HeadCountKV: 2}, RecurrentSpec: RecurrentSpec{WKVHeadSize: 4, TimeMixExtraDim: 3,
		TimeDecayExtraDim: 2, RescaleEvery: 1},
	}
	input := builder.Input("input", dtype.F32, tensor.MustShape(8, 2))
	weights := LayerGraphWeights{
		AttentionNorm:        builder.Input("attn_norm", dtype.F32, tensor.MustShape(8)),
		AttentionNormBias:    builder.Input("attn_norm_bias", dtype.F32, tensor.MustShape(8)),
		AttentionNorm2:       builder.Input("attn_norm_2", dtype.F32, tensor.MustShape(8)),
		AttentionNorm2Bias:   builder.Input("attn_norm_2_bias", dtype.F32, tensor.MustShape(8)),
		TimeMixW1:            builder.Input("mix_w1", dtype.F32, tensor.MustShape(8, 15)),
		TimeMixW2:            builder.Input("mix_w2", dtype.F32, tensor.MustShape(3, 8, 5)),
		TimeMixLerpX:         builder.Input("lerp_x", dtype.F32, tensor.MustShape(8, 1, 1)),
		TimeMixLerpFused:     builder.Input("lerp", dtype.F32, tensor.MustShape(8, 1, 1, 5)),
		TimeMixFirst:         builder.Input("first", dtype.F32, tensor.MustShape(4, 2)),
		TimeMixDecay:         builder.Input("decay", dtype.F32, tensor.MustShape(8)),
		TimeMixDecayW1:       builder.Input("decay_w1", dtype.F32, tensor.MustShape(8, 2)),
		TimeMixDecayW2:       builder.Input("decay_w2", dtype.F32, tensor.MustShape(2, 8)),
		TimeMixKey:           builder.Input("key", dtype.F32, tensor.MustShape(8, 8)),
		TimeMixValue:         builder.Input("value", dtype.F32, tensor.MustShape(8, 8)),
		TimeMixReceptance:    builder.Input("receptance", dtype.F32, tensor.MustShape(8, 8)),
		TimeMixGate:          builder.Input("gate", dtype.F32, tensor.MustShape(8, 8)),
		TimeMixLN:            builder.Input("mix_ln", dtype.F32, tensor.MustShape(8)),
		TimeMixLNBias:        builder.Input("mix_ln_bias", dtype.F32, tensor.MustShape(8)),
		TimeMixOutput:        builder.Input("output", dtype.F32, tensor.MustShape(8, 8)),
		ChannelMixLerpK:      builder.Input("channel_lerp_k", dtype.F32, tensor.MustShape(8, 1, 1)),
		ChannelMixLerpR:      builder.Input("channel_lerp_r", dtype.F32, tensor.MustShape(8, 1, 1)),
		ChannelMixKey:        builder.Input("channel_key", dtype.F32, tensor.MustShape(8, 12)),
		ChannelMixValue:      builder.Input("channel_value", dtype.F32, tensor.MustShape(12, 8)),
		ChannelMixReceptance: builder.Input("channel_receptance", dtype.F32, tensor.MustShape(8, 8)),
	}
	shift := builder.Input("shift", dtype.F32, tensor.MustShape(8, 2))
	state := builder.Input("state", dtype.F32, tensor.MustShape(4, 4, 2, 1))
	result, err := BuildRWKV6BlockCached(builder, input, spec, weights, shift, state, 0)
	if err != nil {
		t.Fatal(err)
	}
	if !result.Output.Shape.Equal(input.Shape) || !result.Key.Shape.Equal(shift.Shape) || !result.Value.Shape.Equal(state.Shape) {
		t.Fatalf("unexpected RWKV6 result: %+v", result)
	}
	nodes, err := tensor.Topological(result.Output, result.Key, result.Value)
	if err != nil {
		t.Fatal(err)
	}
	var wkv, layerNorm, reluSquared int
	for _, node := range nodes {
		switch node.Op {
		case tensor.OpRWKV6:
			wkv++
		case tensor.OpLayerNorm:
			layerNorm++
		case tensor.OpReLUSquared:
			reluSquared++
		}
	}
	if wkv != 1 || layerNorm != 3 || reluSquared != 1 {
		t.Fatalf("RWKV6 ops: WKV=%d LayerNorm=%d ReLU2=%d", wkv, layerNorm, reluSquared)
	}
}

func TestBuildRWKV7BlockValueResidual(t *testing.T) {
	builder := tensor.NewBuilder()
	spec := Spec{CommonSpec: CommonSpec{Architecture: "rwkv7", EmbeddingLength: 8, FeedForwardLength: 12,

		LayerNormEpsilon: 1e-5}, AttentionSpec: AttentionSpec{HeadCount: 2, HeadCountKV: 2}, RecurrentSpec: RecurrentSpec{WKVHeadSize: 4, TokenShiftCount: 2,
		DecayLoRARank: 3, ICLRLoRARank: 2, ValueMixLoRARank: 3, GateLoRARank: 2},
	}
	input := builder.Input("input", dtype.F32, tensor.MustShape(8, 2))
	weights := rwkv7GraphWeights(builder, spec, true)
	shift := builder.Input("shift", dtype.F32, tensor.MustShape(8, 2))
	state := builder.Input("state", dtype.F32, tensor.MustShape(4, 4, 2, 1))
	first, err := BuildRWKV7BlockCached(builder, input, spec, weights, shift, state, 0)
	if err != nil {
		t.Fatal(err)
	}
	if first.Auxiliary == nil || !first.Auxiliary.Shape.Equal(input.Shape) ||
		!first.Output.Shape.Equal(input.Shape) || !first.Key.Shape.Equal(shift.Shape) || !first.Value.Shape.Equal(state.Shape) {
		t.Fatalf("unexpected RWKV7 first-layer result: %+v", first)
	}
	weights.PerLayerInput = builder.Input("first_value", dtype.F32, input.Shape)
	second, err := BuildRWKV7BlockCached(builder, input, spec, weights, shift, state, 1)
	if err != nil {
		t.Fatal(err)
	}
	if second.Auxiliary != nil {
		t.Fatalf("unexpected RWKV7 later-layer auxiliary: %+v", second.Auxiliary)
	}
	nodes, err := tensor.Topological(first.Output, first.Key, first.Value, first.Auxiliary, second.Output)
	if err != nil {
		t.Fatal(err)
	}
	var wkv, sumRows int
	for _, node := range nodes {
		switch node.Op {
		case tensor.OpRWKV7:
			wkv++
		case tensor.OpSumRows:
			sumRows++
		}
	}
	if wkv != 2 || sumRows != 2 {
		t.Fatalf("RWKV7 ops: WKV=%d SumRows=%d", wkv, sumRows)
	}
}

func TestBuildARWKV7UngatedBlock(t *testing.T) {
	builder := tensor.NewBuilder()
	spec := Spec{CommonSpec: CommonSpec{Architecture: "arwkv7", EmbeddingLength: 8, FeedForwardLength: 12,

		RMSNormEpsilon: 1e-6}, AttentionSpec: AttentionSpec{HeadCount: 2, HeadCountKV: 2}, RecurrentSpec: RecurrentSpec{WKVHeadSize: 4, TokenShiftCount: 1,
		DecayLoRARank: 3, ICLRLoRARank: 2, ValueMixLoRARank: 3},
	}
	input := builder.Input("input", dtype.F32, tensor.MustShape(8, 2))
	weights := rwkv7GraphWeights(builder, spec, false)
	shift := builder.Input("shift", dtype.F32, tensor.MustShape(8, 1))
	state := builder.Input("state", dtype.F32, tensor.MustShape(4, 4, 2, 1))
	result, err := BuildRWKV7BlockCached(builder, input, spec, weights, shift, state, 0)
	if err != nil {
		t.Fatal(err)
	}
	if result.Auxiliary == nil || !result.Output.Shape.Equal(input.Shape) || !result.Key.Shape.Equal(shift.Shape) {
		t.Fatalf("unexpected ARWKV7 result: %+v", result)
	}
	nodes, err := tensor.Topological(result.Output, result.Key, result.Value, result.Auxiliary)
	if err != nil {
		t.Fatal(err)
	}
	for _, node := range nodes {
		if node.Op == tensor.OpLayerNorm {
			t.Fatal("ungated ARWKV7 unexpectedly uses time group normalization")
		}
	}
}

func rwkv7GraphWeights(builder *tensor.Builder, spec Spec, classic bool) LayerGraphWeights {
	embedding := uint64(spec.EmbeddingLength)
	lerps := uint64(5)
	if spec.GateLoRARank > 0 {
		lerps = 6
	}
	weights := LayerGraphWeights{
		AttentionNorm:     builder.Input("attn_norm", dtype.F32, tensor.MustShape(embedding)),
		TimeMixW0:         builder.Input("mix_w0", dtype.F32, tensor.MustShape(embedding)),
		TimeMixW1:         builder.Input("mix_w1", dtype.F32, tensor.MustShape(embedding, uint64(spec.DecayLoRARank))),
		TimeMixW2:         builder.Input("mix_w2", dtype.F32, tensor.MustShape(uint64(spec.DecayLoRARank), embedding)),
		TimeMixA0:         builder.Input("mix_a0", dtype.F32, tensor.MustShape(embedding)),
		TimeMixA1:         builder.Input("mix_a1", dtype.F32, tensor.MustShape(embedding, uint64(spec.ICLRLoRARank))),
		TimeMixA2:         builder.Input("mix_a2", dtype.F32, tensor.MustShape(uint64(spec.ICLRLoRARank), embedding)),
		TimeMixV0:         builder.Input("mix_v0", dtype.F32, tensor.MustShape(embedding)),
		TimeMixV1:         builder.Input("mix_v1", dtype.F32, tensor.MustShape(embedding, uint64(spec.ICLRLoRARank))),
		TimeMixV2:         builder.Input("mix_v2", dtype.F32, tensor.MustShape(uint64(spec.ICLRLoRARank), embedding)),
		TimeMixLerpFused:  builder.Input("lerp", dtype.F32, tensor.MustShape(embedding, 1, 1, lerps)),
		TimeMixKK:         builder.Input("mix_kk", dtype.F32, tensor.MustShape(embedding)),
		TimeMixKA:         builder.Input("mix_ka", dtype.F32, tensor.MustShape(embedding)),
		TimeMixRK:         builder.Input("mix_rk", dtype.F32, tensor.MustShape(embedding)),
		TimeMixKey:        builder.Input("key", dtype.F32, tensor.MustShape(embedding, embedding)),
		TimeMixValue:      builder.Input("value", dtype.F32, tensor.MustShape(embedding, embedding)),
		TimeMixReceptance: builder.Input("receptance", dtype.F32, tensor.MustShape(embedding, embedding)),
		TimeMixOutput:     builder.Input("output", dtype.F32, tensor.MustShape(embedding, embedding)),
	}
	if spec.GateLoRARank > 0 {
		weights.TimeMixG1 = builder.Input("mix_g1", dtype.F32, tensor.MustShape(embedding, uint64(spec.GateLoRARank)))
		weights.TimeMixG2 = builder.Input("mix_g2", dtype.F32, tensor.MustShape(uint64(spec.GateLoRARank), embedding))
	}
	if classic {
		weights.AttentionNormBias = builder.Input("attn_norm_bias", dtype.F32, tensor.MustShape(embedding))
		weights.AttentionNorm2 = builder.Input("attn_norm_2", dtype.F32, tensor.MustShape(embedding))
		weights.AttentionNorm2Bias = builder.Input("attn_norm_2_bias", dtype.F32, tensor.MustShape(embedding))
		weights.TimeMixLN = builder.Input("mix_ln", dtype.F32, tensor.MustShape(embedding))
		weights.TimeMixLNBias = builder.Input("mix_ln_bias", dtype.F32, tensor.MustShape(embedding))
		weights.ChannelMixLerpK = builder.Input("channel_lerp", dtype.F32, tensor.MustShape(embedding, 1, 1))
		weights.ChannelMixKey = builder.Input("channel_key", dtype.F32, tensor.MustShape(embedding, uint64(spec.FeedForwardLength)))
		weights.ChannelMixValue = builder.Input("channel_value", dtype.F32, tensor.MustShape(uint64(spec.FeedForwardLength), embedding))
	} else {
		weights.FeedForwardNorm = builder.Input("ffn_norm", dtype.F32, tensor.MustShape(embedding))
		weights.FeedForwardGate = builder.Input("ffn_gate", dtype.F32, tensor.MustShape(embedding, uint64(spec.FeedForwardLength)))
		weights.FeedForwardUp = builder.Input("ffn_up", dtype.F32, tensor.MustShape(embedding, uint64(spec.FeedForwardLength)))
		weights.FeedForwardDown = builder.Input("ffn_down", dtype.F32, tensor.MustShape(uint64(spec.FeedForwardLength), embedding))
	}
	return weights
}

func kimiLinearCommonInputs(builder *tensor.Builder, spec Spec, moe bool) LayerGraphWeights {
	weights := LayerGraphWeights{
		AttentionNorm:   builder.Input("attn_norm", dtype.F32, tensor.MustShape(8)),
		FeedForwardNorm: builder.Input("ffn_norm", dtype.F32, tensor.MustShape(8)),
		FeedForwardGate: builder.Input("ffn_gate", dtype.F32, tensor.MustShape(8, 12)),
		FeedForwardUp:   builder.Input("ffn_up", dtype.F32, tensor.MustShape(8, 12)),
		FeedForwardDown: builder.Input("ffn_down", dtype.F32, tensor.MustShape(12, 8)),
	}
	if !moe {
		return weights
	}
	weights.FeedForwardGate = nil
	weights.FeedForwardUp = nil
	weights.FeedForwardDown = nil
	weights.FeedForwardRouter = builder.Input("router", dtype.F32, tensor.MustShape(8, 4))
	weights.FeedForwardGateExperts = builder.Input("eg", dtype.F32, tensor.MustShape(8, 6, 4))
	weights.FeedForwardUpExperts = builder.Input("eu", dtype.F32, tensor.MustShape(8, 6, 4))
	weights.FeedForwardDownExperts = builder.Input("ed", dtype.F32, tensor.MustShape(6, 8, 4))
	weights.FeedForwardExpertBias = builder.Input("eb", dtype.F32, tensor.MustShape(4))
	weights.FeedForwardSharedGate = builder.Input("sg", dtype.F32, tensor.MustShape(8, 6))
	weights.FeedForwardSharedUp = builder.Input("su", dtype.F32, tensor.MustShape(8, 6))
	weights.FeedForwardSharedDown = builder.Input("sd", dtype.F32, tensor.MustShape(6, 8))
	return weights
}

func TestBuildT5EncoderBlock(t *testing.T) {
	builder := tensor.NewBuilder()
	spec := Spec{CommonSpec: CommonSpec{Architecture: "t5encoder",
		EmbeddingLength:   8,
		FeedForwardLength: 16,

		RMSNormEpsilon: 1e-6}, AttentionSpec: AttentionSpec{HeadCount: 2,
		HeadCountKV: 2,
		KeyLength:   4,
		ValueLength: 4}, EncoderSpec: EncoderSpec{RelativeBuckets: 4},
	}
	input := builder.Input("input", dtype.F32, tensor.MustShape(8, 3))
	weights := LayerGraphWeights{
		AttentionNorm:         builder.Input("attn_norm", dtype.F32, tensor.MustShape(8)),
		AttentionQ:            builder.Input("attn_q", dtype.F32, tensor.MustShape(8, 8)),
		AttentionK:            builder.Input("attn_k", dtype.F32, tensor.MustShape(8, 8)),
		AttentionV:            builder.Input("attn_v", dtype.F32, tensor.MustShape(8, 8)),
		AttentionOutput:       builder.Input("attn_o", dtype.F32, tensor.MustShape(8, 8)),
		AttentionRelativeBias: builder.Input("attn_rel_b", dtype.F32, tensor.MustShape(2, 4)),
		FeedForwardNorm:       builder.Input("ffn_norm", dtype.F32, tensor.MustShape(8)),
		FeedForwardGate:       builder.Input("ffn_gate", dtype.F32, tensor.MustShape(8, 16)),
		FeedForwardUp:         builder.Input("ffn_up", dtype.F32, tensor.MustShape(8, 16)),
		FeedForwardDown:       builder.Input("ffn_down", dtype.F32, tensor.MustShape(16, 8)),
	}
	output, err := BuildT5EncoderBlock(builder, input, spec, weights)
	if err != nil {
		t.Fatal(err)
	}
	if !output.Shape.Equal(input.Shape) {
		t.Fatalf("T5 output shape = %v, want %v", output.Shape, input.Shape)
	}
	var foundRelativeAttention, foundGELU bool
	order, err := tensor.Topological(output)
	if err != nil {
		t.Fatal(err)
	}
	for _, node := range order {
		if node.Op == tensor.OpAttention {
			attributes := node.Attrs.(tensor.AttentionAttributes)
			foundRelativeAttention = len(node.Inputs) == 4 &&
				attributes.RelativeBuckets == 4 &&
				!attributes.Causal
		}
		foundGELU = foundGELU || node.Op == tensor.OpGELU
	}
	if !foundRelativeAttention || !foundGELU {
		t.Fatalf(
			"T5 graph relative attention/GELU = %t/%t",
			foundRelativeAttention,
			foundGELU,
		)
	}
}

func TestBuildT5DecoderBlockCached(t *testing.T) {
	builder := tensor.NewBuilder()
	spec := Spec{CommonSpec: CommonSpec{Architecture: "t5", EmbeddingLength: 8, FeedForwardLength: 16,

		RMSNormEpsilon: 1e-6}, AttentionSpec: AttentionSpec{HeadCount: 2, HeadCountKV: 2, KeyLength: 4, ValueLength: 4}, EncoderSpec: EncoderSpec{RelativeBuckets: 4},
	}
	input := builder.Input("input", dtype.F32, tensor.MustShape(8, 2))
	encoder := builder.Input("encoder", dtype.F32, tensor.MustShape(8, 3))
	weights := LayerGraphWeights{
		AttentionNorm:         builder.Input("attn_norm", dtype.F32, tensor.MustShape(8)),
		AttentionQ:            builder.Input("attn_q", dtype.F32, tensor.MustShape(8, 8)),
		AttentionK:            builder.Input("attn_k", dtype.F32, tensor.MustShape(8, 8)),
		AttentionV:            builder.Input("attn_v", dtype.F32, tensor.MustShape(8, 8)),
		AttentionOutput:       builder.Input("attn_o", dtype.F32, tensor.MustShape(8, 8)),
		AttentionRelativeBias: builder.Input("attn_rel_b", dtype.F32, tensor.MustShape(2, 4)),
		CrossAttentionNorm:    builder.Input("cross_norm", dtype.F32, tensor.MustShape(8)),
		CrossAttentionQ:       builder.Input("cross_q", dtype.F32, tensor.MustShape(8, 8)),
		CrossAttentionK:       builder.Input("cross_k", dtype.F32, tensor.MustShape(8, 8)),
		CrossAttentionV:       builder.Input("cross_v", dtype.F32, tensor.MustShape(8, 8)),
		CrossAttentionOutput:  builder.Input("cross_o", dtype.F32, tensor.MustShape(8, 8)),
		FeedForwardNorm:       builder.Input("ffn_norm", dtype.F32, tensor.MustShape(8)),
		FeedForwardUp:         builder.Input("ffn_up", dtype.F32, tensor.MustShape(8, 16)),
		FeedForwardDown:       builder.Input("ffn_down", dtype.F32, tensor.MustShape(16, 8)),
	}
	result, err := BuildT5DecoderBlockCached(builder, input, encoder, spec, weights, nil, nil, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	if !result.Output.Shape.Equal(input.Shape) ||
		!result.Key.Shape.Equal(tensor.MustShape(4, 2, 2)) ||
		!result.FixedStates["cross_key"].Shape.Equal(tensor.MustShape(4, 2, 3)) {
		t.Fatalf("unexpected T5 decoder result: %+v", result)
	}
	var causalRelative, crossAttention, relu bool
	order, err := tensor.Topological(result.Output)
	if err != nil {
		t.Fatal(err)
	}
	for _, node := range order {
		if node.Op == tensor.OpAttention {
			attributes := node.Attrs.(tensor.AttentionAttributes)
			causalRelative = causalRelative || attributes.Causal &&
				attributes.RelativeBuckets == 4 && !attributes.RelativeBidirectional
			crossAttention = crossAttention || !attributes.Causal && attributes.RelativeBuckets == 0
		}
		relu = relu || node.Op == tensor.OpReLU
	}
	if !causalRelative || !crossAttention || !relu {
		t.Fatalf("T5 decoder graph causal-relative/cross/ReLU = %t/%t/%t", causalRelative, crossAttention, relu)
	}
}

func TestBuildQwen35RecurrentBlock(t *testing.T) {
	builder := tensor.NewBuilder()
	spec := qwen35TestSpec()
	input := builder.Input("input", dtype.F32, tensor.MustShape(8, 2))
	convState := builder.Input("conv_state", dtype.F32, tensor.MustShape(2, 8))
	ssmState := builder.Input("ssm_state", dtype.F32, tensor.MustShape(2, 2, 2, 1))
	result, err := BuildQwen35BlockCached(
		builder,
		input,
		spec,
		qwen35RecurrentInputs(builder, spec),
		[]uint32{0, 1},
		true,
		nil,
		nil,
		convState,
		ssmState,
	)
	if err != nil {
		t.Fatal(err)
	}
	if !result.Output.Shape.Equal(input.Shape) ||
		!result.ConvState.Shape.Equal(convState.Shape) ||
		!result.SSMState.Shape.Equal(ssmState.Shape) ||
		!result.Recurrent {
		t.Fatalf("unexpected Qwen3.5 recurrent result: %+v", result)
	}
	nodes, err := tensor.Topological(result.Output, result.ConvState, result.SSMState)
	if err != nil {
		t.Fatal(err)
	}
	var conv, delta int
	for _, node := range nodes {
		if node.Op == tensor.OpSSMConv {
			conv++
		}
		if node.Op == tensor.OpGatedDeltaNet {
			delta++
		}
	}
	if conv != 1 || delta != 1 {
		t.Fatalf("Qwen3.5 recurrent ops: convolution=%d delta-net=%d", conv, delta)
	}
}

func TestBuildQwen35MoEBlocks(t *testing.T) {
	for _, recurrent := range []bool{false, true} {
		name := "attention"
		if recurrent {
			name = "recurrent"
		}
		t.Run(name, func(t *testing.T) {
			builder := tensor.NewBuilder()
			spec := qwen35TestSpec()
			spec.Architecture = "qwen35moe"
			spec.ExpertCount = 4
			spec.ExpertUsedCount = 2
			spec.ExpertFeedForward = 6
			spec.SharedExpertFF = 10
			spec.ExpertWeightsScale = 1.25
			input := builder.Input("input", dtype.F32, tensor.MustShape(8, 2))
			weights := qwen35AttentionInputs(builder, spec)
			var convState, ssmState *tensor.Tensor
			if recurrent {
				weights = qwen35RecurrentInputs(builder, spec)
				convState = builder.Input("conv_state", dtype.F32, tensor.MustShape(2, 8))
				ssmState = builder.Input("ssm_state", dtype.F32, tensor.MustShape(2, 2, 2, 1))
			}
			setQwen35MoEInputs(builder, spec, &weights)
			result, err := BuildQwen35BlockCached(
				builder, input, spec, weights, []uint32{0, 1}, recurrent,
				nil, nil, convState, ssmState,
			)
			if err != nil {
				t.Fatal(err)
			}
			nodes, err := tensor.Topological(result.Output)
			if err != nil {
				t.Fatal(err)
			}
			var moe tensor.MoEAttributes
			var moeCount, sigmoid int
			for _, node := range nodes {
				if node.Op == tensor.OpMoE {
					moeCount++
					moe = node.Attrs.(tensor.MoEAttributes)
				}
				if node.Op == tensor.OpSigmoid {
					sigmoid++
				}
			}
			if moeCount != 1 || moe.TopK != 2 || !moe.NormalizeTopKProb ||
				moe.Routing != tensor.MoERoutingSoftmax || moe.Scale != 1.25 ||
				moe.Activation != tensor.MoEActivationSiLU || sigmoid < 1 {
				t.Fatalf("Qwen3.5-MoE graph: MoE=%d attrs=%+v sigmoid=%d", moeCount, moe, sigmoid)
			}
		})
	}
}

func TestBuildQwen3NextBlocks(t *testing.T) {
	for _, recurrent := range []bool{false, true} {
		name := "attention"
		if recurrent {
			name = "recurrent"
		}
		t.Run(name, func(t *testing.T) {
			builder := tensor.NewBuilder()
			spec := qwen35TestSpec()
			spec.Architecture = "qwen3next"
			spec.ExpertCount = 4
			spec.ExpertUsedCount = 2
			spec.ExpertFeedForward = 6
			spec.SharedExpertFF = 10
			spec.ExpertWeightsScale = 1.25
			input := builder.Input("input", dtype.F32, tensor.MustShape(8, 2))
			weights := qwen35AttentionInputs(builder, spec)
			var convState, ssmState *tensor.Tensor
			if recurrent {
				weights = qwen35RecurrentInputs(builder, spec)
				weights.SSMBeta = nil
				weights.SSMAlpha = nil
				weights.SSMBetaAlpha = builder.Input("ssm_ba", dtype.F32, tensor.MustShape(8, 4))
				convState = builder.Input("conv_state", dtype.F32, tensor.MustShape(2, 8))
				ssmState = builder.Input("ssm_state", dtype.F32, tensor.MustShape(2, 2, 2, 1))
			}
			setQwen35MoEInputs(builder, spec, &weights)
			result, err := BuildQwen35BlockCached(
				builder, input, spec, weights, []uint32{0, 1}, recurrent,
				nil, nil, convState, ssmState,
			)
			if err != nil {
				t.Fatal(err)
			}
			nodes, err := tensor.Topological(result.Output)
			if err != nil {
				t.Fatal(err)
			}
			var moe, rope, repeatInterleave int
			for _, node := range nodes {
				if node.Op == tensor.OpMoE {
					moe++
				}
				if node.Op == tensor.OpRoPENeoX {
					rope++
				}
				if node.Op == tensor.OpGatedDeltaNet &&
					node.Attrs.(tensor.GatedDeltaNetAttributes).RepeatInterleave {
					repeatInterleave++
				}
			}
			if moe != 1 || (!recurrent && rope != 2) || (recurrent && repeatInterleave != 1) {
				t.Fatalf(
					"Qwen3-Next ops: MoE=%d NeoX=%d repeat-interleave=%d",
					moe, rope, repeatInterleave,
				)
			}
		})
	}
}

func TestBuildQwen3NextLegacyQKVZBlock(t *testing.T) {
	builder := tensor.NewBuilder()
	spec := qwen35TestSpec()
	spec.Architecture = "qwen3next"
	spec.ExpertCount = 4
	spec.ExpertUsedCount = 2
	spec.ExpertFeedForward = 6
	spec.SharedExpertFF = 10
	spec.ExpertWeightsScale = 1
	input := builder.Input("input", dtype.F32, tensor.MustShape(8, 2))
	weights := qwen35RecurrentInputs(builder, spec)
	weights.AttentionQKV = builder.Input("ssm_in", dtype.F32, tensor.MustShape(8, 12))
	weights.AttentionGate = nil
	weights.SSMBeta = nil
	weights.SSMAlpha = nil
	weights.SSMBetaAlpha = builder.Input("ssm_ba", dtype.F32, tensor.MustShape(8, 4))
	setQwen35MoEInputs(builder, spec, &weights)
	convState := builder.Input("conv_state", dtype.F32, tensor.MustShape(2, 8))
	ssmState := builder.Input("ssm_state", dtype.F32, tensor.MustShape(2, 2, 2, 1))
	result, err := BuildQwen35BlockCached(
		builder, input, spec, weights, []uint32{0, 1}, true,
		nil, nil, convState, ssmState,
	)
	if err != nil {
		t.Fatal(err)
	}
	nodes, err := tensor.Topological(result.Output)
	if err != nil {
		t.Fatal(err)
	}
	var groupSlices int
	for _, node := range nodes {
		if node.Op == tensor.OpGroupSlice {
			groupSlices++
		}
	}
	if groupSlices < 9 {
		t.Fatalf("Qwen3-Next legacy QKVZ group slices = %d, want at least 9", groupSlices)
	}
}

func qwen35TestSpec() Spec {
	return Spec{CommonSpec: CommonSpec{Architecture: "qwen35",
		EmbeddingLength:   8,
		FeedForwardLength: 12,

		RMSNormEpsilon: 1e-6}, AttentionSpec: AttentionSpec{HeadCount: 2,
		HeadCountKV:       1,
		KeyLength:         4,
		ValueLength:       4,
		RopeFrequencyBase: 10000,

		RopeDimensionCount: 4,
		RopeSections:       [4]int32{1, 1, 0, 0}}, RecurrentSpec: RecurrentSpec{SSMConvKernel: 3,
		SSMInnerSize:          4,
		SSMStateSize:          2,
		SSMTimeStepRank:       2,
		SSMGroupCount:         1,
		FullAttentionInterval: 4},
	}
}

func qwen35CommonInputs(builder *tensor.Builder, spec Spec) LayerGraphWeights {
	embedding := uint64(spec.EmbeddingLength)
	feedForward := uint64(spec.FeedForwardLength)
	return LayerGraphWeights{
		AttentionNorm:   builder.Input("attn_norm", dtype.F32, tensor.MustShape(embedding)),
		FeedForwardNorm: builder.Input("post_attention_norm", dtype.F32, tensor.MustShape(embedding)),
		FeedForwardGate: builder.Input("ffn_gate", dtype.F32, tensor.MustShape(embedding, feedForward)),
		FeedForwardUp:   builder.Input("ffn_up", dtype.F32, tensor.MustShape(embedding, feedForward)),
		FeedForwardDown: builder.Input("ffn_down", dtype.F32, tensor.MustShape(feedForward, embedding)),
	}
}

func setQwen35MoEInputs(builder *tensor.Builder, spec Spec, weights *LayerGraphWeights) {
	embedding := uint64(spec.EmbeddingLength)
	expertWidth := uint64(spec.ExpertFeedForward)
	experts := uint64(spec.ExpertCount)
	sharedWidth := uint64(spec.SharedExpertFF)
	weights.FeedForwardGate = nil
	weights.FeedForwardUp = nil
	weights.FeedForwardDown = nil
	weights.FeedForwardRouter = builder.Input("ffn_router", dtype.F32, tensor.MustShape(embedding, experts))
	weights.FeedForwardGateExperts = builder.Input(
		"ffn_gate_exps", dtype.F32, tensor.MustShape(embedding, expertWidth, experts),
	)
	weights.FeedForwardUpExperts = builder.Input(
		"ffn_up_exps", dtype.F32, tensor.MustShape(embedding, expertWidth, experts),
	)
	weights.FeedForwardDownExperts = builder.Input(
		"ffn_down_exps", dtype.F32, tensor.MustShape(expertWidth, embedding, experts),
	)
	weights.FeedForwardSharedRouter = builder.Input(
		"ffn_shared_router", dtype.F32, tensor.MustShape(embedding),
	)
	weights.FeedForwardSharedGate = builder.Input(
		"ffn_shared_gate", dtype.F32, tensor.MustShape(embedding, sharedWidth),
	)
	weights.FeedForwardSharedUp = builder.Input(
		"ffn_shared_up", dtype.F32, tensor.MustShape(embedding, sharedWidth),
	)
	weights.FeedForwardSharedDown = builder.Input(
		"ffn_shared_down", dtype.F32, tensor.MustShape(sharedWidth, embedding),
	)
}

func qwen35AttentionInputs(builder *tensor.Builder, spec Spec) LayerGraphWeights {
	result := qwen35CommonInputs(builder, spec)
	embedding := uint64(spec.EmbeddingLength)
	headWidth := uint64(spec.KeyLength)
	result.AttentionQ = builder.Input(
		"attn_q",
		dtype.F32,
		tensor.MustShape(embedding, 2*uint64(spec.HeadCount)*headWidth),
	)
	result.AttentionK = builder.Input(
		"attn_k",
		dtype.F32,
		tensor.MustShape(embedding, uint64(spec.HeadCountKV)*headWidth),
	)
	result.AttentionV = builder.Input(
		"attn_v",
		dtype.F32,
		tensor.MustShape(embedding, uint64(spec.HeadCountKV)*uint64(spec.ValueLength)),
	)
	result.AttentionOutput = builder.Input("attn_output", dtype.F32, tensor.MustShape(embedding, embedding))
	result.AttentionQNorm = builder.Input("attn_q_norm", dtype.F32, tensor.MustShape(headWidth))
	result.AttentionKNorm = builder.Input("attn_k_norm", dtype.F32, tensor.MustShape(headWidth))
	return result
}

func qwen35RecurrentInputs(builder *tensor.Builder, spec Spec) LayerGraphWeights {
	result := qwen35CommonInputs(builder, spec)
	embedding := uint64(spec.EmbeddingLength)
	state := uint64(spec.SSMStateSize)
	keyDimension := state * uint64(spec.SSMGroupCount)
	valueDimension := uint64(spec.SSMInnerSize)
	channels := 2*keyDimension + valueDimension
	valueHeads := uint64(spec.SSMTimeStepRank)
	result.AttentionQKV = builder.Input("attn_qkv", dtype.F32, tensor.MustShape(embedding, channels))
	result.AttentionGate = builder.Input("attn_gate", dtype.F32, tensor.MustShape(embedding, valueDimension))
	result.SSMConv1D = builder.Input(
		"ssm_conv1d",
		dtype.F32,
		tensor.MustShape(uint64(spec.SSMConvKernel), channels),
	)
	result.SSMTimeStep = builder.Input("ssm_dt", dtype.F32, tensor.MustShape(valueHeads))
	result.SSMA = builder.Input("ssm_a", dtype.F32, tensor.MustShape(valueHeads))
	result.SSMBeta = builder.Input("ssm_beta", dtype.F32, tensor.MustShape(embedding, valueHeads))
	result.SSMAlpha = builder.Input("ssm_alpha", dtype.F32, tensor.MustShape(embedding, valueHeads))
	result.SSMNorm = builder.Input("ssm_norm", dtype.F32, tensor.MustShape(state))
	result.SSMOutput = builder.Input("ssm_out", dtype.F32, tensor.MustShape(valueDimension, embedding))
	return result
}

func denseBlockInputs(builder *tensor.Builder, spec Spec) LayerGraphWeights {
	embedding := uint64(spec.EmbeddingLength)
	feedForward := uint64(spec.FeedForwardLength)
	query := uint64(spec.HeadCount) * uint64(spec.KeyLength)
	keyValue := uint64(spec.HeadCountKV) * uint64(spec.KeyLength)
	value := uint64(spec.HeadCountKV) * uint64(spec.ValueLength)
	attentionOutput := uint64(spec.HeadCount) * uint64(spec.ValueLength)
	return LayerGraphWeights{
		AttentionNorm:       builder.Input("attn_norm", dtype.F32, tensor.MustShape(embedding)),
		AttentionNormBias:   builder.Input("attn_norm_bias", dtype.F32, tensor.MustShape(embedding)),
		AttentionQ:          builder.Input("attn_q", dtype.F32, tensor.MustShape(embedding, query)),
		AttentionK:          builder.Input("attn_k", dtype.F32, tensor.MustShape(embedding, keyValue)),
		AttentionV:          builder.Input("attn_v", dtype.F32, tensor.MustShape(embedding, value)),
		AttentionOutput:     builder.Input("attn_output", dtype.F32, tensor.MustShape(attentionOutput, embedding)),
		AttentionQNorm:      builder.Input("attn_q_norm", dtype.F32, tensor.MustShape(uint64(spec.KeyLength))),
		AttentionKNorm:      builder.Input("attn_k_norm", dtype.F32, tensor.MustShape(uint64(spec.KeyLength))),
		FeedForwardNorm:     builder.Input("ffn_norm", dtype.F32, tensor.MustShape(embedding)),
		FeedForwardNormBias: builder.Input("ffn_norm_bias", dtype.F32, tensor.MustShape(embedding)),
		FeedForwardGate:     builder.Input("ffn_gate", dtype.F32, tensor.MustShape(embedding, feedForward)),
		FeedForwardUp:       builder.Input("ffn_up", dtype.F32, tensor.MustShape(embedding, feedForward)),
		FeedForwardDown:     builder.Input("ffn_down", dtype.F32, tensor.MustShape(feedForward, embedding)),
	}
}

func TestBuildLlama4AttentionAndMoE(t *testing.T) {
	spec := Spec{CommonSpec: CommonSpec{Architecture: "llama4", BlockCount: 4, EmbeddingLength: 8, FeedForwardLength: 12,

		RMSNormEpsilon: 1e-5}, AttentionSpec: AttentionSpec{HeadCount: 2, HeadCountKV: 1, KeyLength: 4, ValueLength: 4,
		RopeDimensionCount: 4, RopeFrequencyBase: 10000, RopeFrequencySWA: 10000,
		SlidingWindow: 4, SlidingPattern: 4, NoRopeLayerStep: 4}, MoESpec: MoESpec{ExpertCount: 4, ExpertUsedCount: 2, ExpertFeedForward: 6, SharedExpertFF: 6,
		ExpertWeightsScale: 1, ExpertGatingFunc: expertGatingSigmoid, MoELayerStep: 4},
	}
	for _, test := range []struct {
		name  string
		layer uint32
		moe   bool
	}{
		{name: "chunked dense", layer: 0},
		{name: "temperature MoE", layer: 3, moe: true},
	} {
		t.Run(test.name, func(t *testing.T) {
			builder := tensor.NewBuilder()
			input := builder.Input("input", dtype.F32, tensor.MustShape(8, 2))
			weights := denseBlockInputs(builder, spec)
			if test.layer == 3 {
				weights.AttentionTemperatureScale = builder.Input("temp", dtype.F32, tensor.MustShape(1, 1, 2))
			}
			if test.moe {
				weights.FeedForwardRouter = builder.Input("router", dtype.F32, tensor.MustShape(8, 4))
				weights.FeedForwardGateExperts = builder.Input("gate_exps", dtype.F32, tensor.MustShape(8, 6, 4))
				weights.FeedForwardUpExperts = builder.Input("up_exps", dtype.F32, tensor.MustShape(8, 6, 4))
				weights.FeedForwardDownExperts = builder.Input("down_exps", dtype.F32, tensor.MustShape(6, 8, 4))
				weights.FeedForwardSharedGate = builder.Input("shared_gate", dtype.F32, tensor.MustShape(8, 6))
				weights.FeedForwardSharedUp = builder.Input("shared_up", dtype.F32, tensor.MustShape(8, 6))
				weights.FeedForwardSharedDown = builder.Input("shared_down", dtype.F32, tensor.MustShape(6, 8))
			}
			result, err := BuildDenseBlockCachedForLayer(
				builder, input, spec, weights, []uint32{8192, 8193}, nil, nil, test.layer,
			)
			if err != nil {
				t.Fatal(err)
			}
			var rope, qkNorm int
			var attention tensor.AttentionAttributes
			var moe tensor.MoEAttributes
			nodes, graphErr := tensor.Topological(result.Output)
			if graphErr != nil {
				t.Fatal(graphErr)
			}
			for _, node := range nodes {
				switch node.Op {
				case tensor.OpRoPENormal:
					rope++
				case tensor.OpRMSNorm:
					if node.Shape.Rank == 3 {
						qkNorm++
					}
				case tensor.OpAttention:
					attention = node.Attrs.(tensor.AttentionAttributes)
				case tensor.OpMoE:
					moe = node.Attrs.(tensor.MoEAttributes)
				}
			}
			if test.layer == 0 && (rope != 2 || qkNorm < 2 || !attention.ChunkedWindow) {
				t.Fatalf("Llama 4 chunked graph: rope=%d qk_norm=%d attention=%+v", rope, qkNorm, attention)
			}
			if test.layer == 3 && (rope != 0 || moe.Routing != tensor.MoERoutingSigmoid || moe.NormalizeTopKProb) {
				t.Fatalf("Llama 4 MoE graph: rope=%d moe=%+v", rope, moe)
			}
		})
	}
}

func TestBuildMistral3TemperatureMoEBlock(t *testing.T) {
	spec := Spec{CommonSpec: CommonSpec{Architecture: "mistral3", EmbeddingLength: 8, FeedForwardLength: 6,

		RMSNormEpsilon: 1e-5}, AttentionSpec: AttentionSpec{HeadCount: 2, HeadCountKV: 1, KeyLength: 4, ValueLength: 4,
		RopeDimensionCount: 4, RopeFrequencyBase: 10000,
		AttentionTempScale: 0.1, AttentionTempFloor: 8}, MoESpec: MoESpec{ExpertCount: 4, ExpertUsedCount: 2, ExpertFeedForward: 6,
		ExpertWeightsScale: 1, ExpertWeightsNorm: true},
	}
	builder := tensor.NewBuilder()
	input := builder.Input("input", dtype.F32, tensor.MustShape(8, 2))
	weights := denseBlockInputs(builder, spec)
	weights.AttentionTemperatureScale = builder.Input("temperature", dtype.F32, tensor.MustShape(1, 1, 2))
	weights.FeedForwardRouter = builder.Input("router", dtype.F32, tensor.MustShape(8, 4))
	weights.FeedForwardGateExperts = builder.Input("gate_exps", dtype.F32, tensor.MustShape(8, 6, 4))
	weights.FeedForwardUpExperts = builder.Input("up_exps", dtype.F32, tensor.MustShape(8, 6, 4))
	weights.FeedForwardDownExperts = builder.Input("down_exps", dtype.F32, tensor.MustShape(6, 8, 4))
	result, err := BuildDenseBlockCachedForLayer(
		builder, input, spec, weights, []uint32{8, 9}, nil, nil, 0,
	)
	if err != nil {
		t.Fatal(err)
	}
	var temperature bool
	var moe tensor.MoEAttributes
	nodes, err := tensor.Topological(result.Output)
	if err != nil {
		t.Fatal(err)
	}
	for _, node := range nodes {
		if node.Op == tensor.OpMultiply && len(node.Inputs) == 2 &&
			(node.Inputs[0] == weights.AttentionTemperatureScale ||
				node.Inputs[1] == weights.AttentionTemperatureScale) {
			temperature = true
		}
		if node.Op == tensor.OpMoE {
			moe = node.Attrs.(tensor.MoEAttributes)
		}
	}
	if !temperature || moe.Experts != 4 || moe.TopK != 2 || !moe.NormalizeTopKProb ||
		moe.Routing != tensor.MoERoutingSoftmax || moe.Activation != tensor.MoEActivationSiLU {
		t.Fatalf("Mistral 3 graph temperature=%v MoE=%+v", temperature, moe)
	}
}

func TestBuildGPTOSSBiasedMoEBlock(t *testing.T) {
	builder := tensor.NewBuilder()
	spec := Spec{CommonSpec: CommonSpec{Architecture: "gpt-oss", BlockCount: 2, EmbeddingLength: 8, FeedForwardLength: 8,

		RMSNormEpsilon: 1e-5}, AttentionSpec: AttentionSpec{HeadCount: 2, HeadCountKV: 1, KeyLength: 4, ValueLength: 4,
		RopeDimensionCount: 4, RopeFrequencyBase: 10000, RopeFrequencySWA: 2000,
		SlidingWindow: 4, SlidingPattern: 2}, MoESpec: MoESpec{ExpertCount: 4, ExpertUsedCount: 2, ExpertFeedForward: 6,
		ExpertWeightsScale: 1, ExpertGatingFunc: expertGatingSelectedSoftmax},
	}
	input := builder.Input("input", dtype.F32, tensor.MustShape(8, 2))
	weights := denseBlockInputs(builder, spec)
	weights.AttentionPostNorm = builder.Input("post_attention_norm", dtype.F32, tensor.MustShape(8))
	weights.AttentionSinks = builder.Input("sinks", dtype.F32, tensor.MustShape(2))
	weights.AttentionOutputBias = builder.Input("attn_output_bias", dtype.F32, tensor.MustShape(8))
	weights.FeedForwardRouter = builder.Input("router", dtype.F32, tensor.MustShape(8, 4))
	weights.FeedForwardRouterBias = builder.Input("router_bias", dtype.F32, tensor.MustShape(4))
	weights.FeedForwardGateExperts = builder.Input("gate_exps", dtype.F32, tensor.MustShape(8, 6, 4))
	weights.FeedForwardGateBias = builder.Input("gate_bias", dtype.F32, tensor.MustShape(6, 4))
	weights.FeedForwardUpExperts = builder.Input("up_exps", dtype.F32, tensor.MustShape(8, 6, 4))
	weights.FeedForwardUpBias = builder.Input("up_bias", dtype.F32, tensor.MustShape(6, 4))
	weights.FeedForwardDownExperts = builder.Input("down_exps", dtype.F32, tensor.MustShape(6, 8, 4))
	weights.FeedForwardDownBias = builder.Input("down_bias", dtype.F32, tensor.MustShape(8, 4))
	result, err := BuildDenseBlockCachedForLayer(builder, input, spec, weights, []uint32{0, 1}, nil, nil, 0)
	if err != nil {
		t.Fatal(err)
	}
	nodes, err := tensor.Topological(result.Output)
	if err != nil {
		t.Fatal(err)
	}
	var attention tensor.AttentionAttributes
	var moe tensor.MoEAttributes
	for _, node := range nodes {
		switch node.Op {
		case tensor.OpAttention:
			attention = node.Attrs.(tensor.AttentionAttributes)
		case tensor.OpMoE:
			moe = node.Attrs.(tensor.MoEAttributes)
		}
	}
	if !attention.HasSinks || attention.Window != 4 || attention.Scale != 0.5 {
		t.Fatalf("unexpected GPT-OSS attention: %+v", attention)
	}
	if moe.Routing != tensor.MoERoutingSelectedSoftmax || moe.Activation != tensor.MoEActivationSwiGLUOAI ||
		!moe.HasRouterBias || !moe.HasExpertBiases || moe.NormalizeTopKProb {
		t.Fatalf("unexpected GPT-OSS MoE: %+v", moe)
	}
}

func TestBuildNemotronHMoEBlockUsesLatentSquaredReLUExperts(t *testing.T) {
	builder := tensor.NewBuilder()
	spec := Spec{CommonSpec: CommonSpec{Architecture: "nemotron_h_moe", BlockCount: 3, EmbeddingLength: 8,
		FeedForwardLength: 6,

		RMSNormEpsilon: 1e-5}, AttentionSpec: AttentionSpec{HeadCount: 2, HeadCountKV: 1, LayerHeadCounts: []uint32{2, 0, 2},
		LayerKVHeadCounts: []uint32{1, 0, 1},
		KeyLength:         4, ValueLength: 4}, MoESpec: MoESpec{LayerFeedForward: []uint32{0, 0, 6},

		ExpertCount: 4, ExpertUsedCount: 2, ExpertFeedForward: 6, SharedExpertFF: 5,
		ExpertWeightsNorm: true, ExpertWeightsScale: 1.25, MoELatentSize: 4}, RecurrentSpec: RecurrentSpec{RecurrentLayers: []bool{false, true, false}},
	}
	input := builder.Input("input", dtype.F32, tensor.MustShape(8, 2))
	weights := LayerGraphWeights{
		AttentionNorm:          builder.Input("norm", dtype.F32, tensor.MustShape(8)),
		FeedForwardRouter:      builder.Input("router", dtype.F32, tensor.MustShape(8, 4)),
		FeedForwardExpertBias:  builder.Input("bias", dtype.F32, tensor.MustShape(4)),
		FeedForwardLatentDown:  builder.Input("latent_down", dtype.F32, tensor.MustShape(8, 4)),
		FeedForwardLatentUp:    builder.Input("latent_up", dtype.F32, tensor.MustShape(4, 8)),
		FeedForwardUpExperts:   builder.Input("up_exps", dtype.F32, tensor.MustShape(4, 6, 4)),
		FeedForwardDownExperts: builder.Input("down_exps", dtype.F32, tensor.MustShape(6, 4, 4)),
		FeedForwardSharedUp:    builder.Input("shared_up", dtype.F32, tensor.MustShape(8, 5)),
		FeedForwardSharedDown:  builder.Input("shared_down", dtype.F32, tensor.MustShape(5, 8)),
	}
	result, err := BuildNemotronHBlockCached(builder, input, spec, weights, []uint32{0, 1}, nil, nil, 2)
	if err != nil {
		t.Fatal(err)
	}
	nodes, err := tensor.Topological(result.Output)
	if err != nil {
		t.Fatal(err)
	}
	var moe tensor.MoEAttributes
	var squaredReLU int
	for _, node := range nodes {
		switch node.Op {
		case tensor.OpMoE:
			moe = node.Attrs.(tensor.MoEAttributes)
		case tensor.OpReLUSquared:
			squaredReLU++
		case tensor.OpAttention, tensor.OpRoPENormal, tensor.OpRoPENeoX:
			t.Fatalf("Nemotron-H FFN graph contains %v", node.Op)
		}
	}
	if moe.Routing != tensor.MoERoutingSigmoid || moe.Activation != tensor.MoEActivationReLUSquared ||
		!moe.NormalizeTopKProb || !moe.HasSelectionBias || squaredReLU != 1 ||
		result.Key == nil || result.Value == nil {
		t.Fatalf("unexpected Nemotron-H MoE graph: moe=%+v squared_relu=%d", moe, squaredReLU)
	}
}
