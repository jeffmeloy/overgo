package model

import (
	"overgo/internal/tensor"

	"overgo/internal/tensor/dtype"

	"overgo/internal/tensor/reference"

	"math"

	"strings"

	"testing"
)

func TestBuildRefactExpertBlock(t *testing.T) {
	for _, gated := range []bool{false, true} {
		builder := tensor.NewBuilder()
		spec := Spec{CommonSpec: CommonSpec{Architecture: "refact", BlockCount: 1,
			EmbeddingLength: 8, FeedForwardLength: 16, RMSNormEpsilon: 1e-5},
			AttentionSpec: AttentionSpec{HeadCount: 2, HeadCountKV: 1, KeyLength: 4,
				ValueLength: 4, RopeDisabled: true, MaxALiBiBias: 8},
			MoESpec: MoESpec{ExpertCount: 4, ExpertUsedCount: 2,
				ExpertFeedForward: 16, ExpertWeightsNorm: true, ExpertWeightsScale: 1},
		}
		input := builder.Input("input", dtype.F32, tensor.MustShape(8, 2))
		weights := denseBlockInputs(builder, spec)
		weights.FeedForwardRouter = builder.Input("router", dtype.F32, tensor.MustShape(8, 4))
		weights.FeedForwardUpExperts = builder.Input("up_exps", dtype.F32, tensor.MustShape(8, 16, 4))
		weights.FeedForwardDownExperts = builder.Input("down_exps", dtype.F32, tensor.MustShape(16, 8, 4))
		if gated {
			weights.FeedForwardGateExperts = builder.Input("gate_exps", dtype.F32, tensor.MustShape(8, 16, 4))
		}
		result, err := BuildDenseBlockCachedForLayer(builder, input, spec, weights, []uint32{0, 1}, nil, nil, 0)
		if err != nil {
			t.Fatal(err)
		}
		nodes, err := tensor.Topological(result.Output)
		if err != nil {
			t.Fatal(err)
		}
		found := false
		for _, node := range nodes {
			found = found || node.Op == tensor.OpMoE
		}
		if !found {
			t.Fatalf("Refact gated=%v graph has no MoE", gated)
		}
	}
}

func TestBuildBERTMoEBlockUsesGateFreeGELUExperts(t *testing.T) {
	for _, architecture := range []string{"jina-bert-v3", "nomic-bert-moe"} {
		t.Run(architecture, func(t *testing.T) {
			builder := tensor.NewBuilder()
			spec := Spec{CommonSpec: CommonSpec{Architecture: architecture, EmbeddingLength: 8, FeedForwardLength: 16,

				LayerNormEpsilon: 1e-5, BlockCount: 2}, AttentionSpec: AttentionSpec{HeadCount: 2, HeadCountKV: 2, KeyLength: 4, ValueLength: 4,

				RopeDimensionCount: 4, RopeFrequencyBase: 10000,
				NonCausalAttention: true}, MoESpec: MoESpec{ExpertCount: 4, ExpertUsedCount: 2, ExpertFeedForward: 16, ExpertWeightsScale: 1,
				MoELayerStep: 2},
			}
			input := builder.Input("input", dtype.F32, tensor.MustShape(8, 3))
			weights := LayerGraphWeights{
				AttentionQKV:            builder.Input("qkv", dtype.F32, tensor.MustShape(8, 24)),
				AttentionOutput:         builder.Input("attn_out", dtype.F32, tensor.MustShape(8, 8)),
				AttentionPostNorm:       builder.Input("attn_post_norm", dtype.F32, tensor.MustShape(8)),
				AttentionPostNormBias:   builder.Input("attn_post_norm_bias", dtype.F32, tensor.MustShape(8)),
				FeedForwardRouter:       builder.Input("router", dtype.F32, tensor.MustShape(8, 4)),
				FeedForwardUpExperts:    builder.Input("up_exps", dtype.F32, tensor.MustShape(8, 16, 4)),
				FeedForwardDownExperts:  builder.Input("down_exps", dtype.F32, tensor.MustShape(16, 8, 4)),
				FeedForwardPostNorm:     builder.Input("ffn_post_norm", dtype.F32, tensor.MustShape(8)),
				FeedForwardPostNormBias: builder.Input("ffn_post_norm_bias", dtype.F32, tensor.MustShape(8)),
			}
			result, err := BuildDenseBlockCachedForLayer(builder, input, spec, weights, []uint32{0, 1, 2}, nil, nil, 1)
			if err != nil {
				t.Fatal(err)
			}
			nodes, err := tensor.Topological(result.Output)
			if err != nil {
				t.Fatal(err)
			}
			var moe, rope int
			for _, node := range nodes {
				if node.Op == tensor.OpMoE {
					moe++
				}
				if node.Op == tensor.OpRoPENeoX {
					rope++
				}
			}
			if moe != 1 || rope != 2 {
				t.Fatalf("%s MoE/RoPE = %d/%d", architecture, moe, rope)
			}
		})
	}
}

func TestBuildDreamBlockUsesNonCausalAttentionAndRejectsCache(t *testing.T) {
	builder := tensor.NewBuilder()
	spec := standardDecoderFixture.decoderSpec("dream")
	spec.NonCausalAttention = true
	input := builder.Input("input", dtype.F32, tensor.MustShape(8, 3))
	weights := denseBlockInputs(builder, spec)
	weights.AttentionQNorm = nil
	weights.AttentionKNorm = nil
	output, err := BuildDenseBlock(builder, input, spec, weights, []uint32{0, 1, 2})
	if err != nil {
		t.Fatal(err)
	}
	nodes, err := tensor.Topological(output)
	if err != nil {
		t.Fatal(err)
	}
	found := false
	for _, node := range nodes {
		if node.Op == tensor.OpAttention {
			found = true
			if node.Attrs.(tensor.AttentionAttributes).Causal {
				t.Fatal("Dream attention is causal")
			}
		}
	}
	if !found {
		t.Fatal("Dream block graph has no attention operation")
	}

	cacheBuilder := tensor.NewBuilder()
	cacheInput := cacheBuilder.Input("input", dtype.F32, tensor.MustShape(8, 1))
	cacheWeights := denseBlockInputs(cacheBuilder, spec)
	cacheWeights.AttentionQNorm = nil
	cacheWeights.AttentionKNorm = nil
	pastKey := cacheBuilder.Input("past_key", dtype.F32, tensor.MustShape(4, 1, 1))
	pastValue := cacheBuilder.Input("past_value", dtype.F32, tensor.MustShape(4, 1, 1))
	_, err = BuildDenseBlockCached(
		cacheBuilder, cacheInput, spec, cacheWeights, []uint32{1}, pastKey, pastValue,
	)
	if err == nil || !strings.Contains(err.Error(), "does not support a KV cache") {
		t.Fatalf("cached Dream block error = %v", err)
	}
}

func TestBuildLLaDABlocksUseNonCausalAttention(t *testing.T) {
	for _, architecture := range []string{"llada", "llada-moe"} {
		t.Run(architecture, func(t *testing.T) {
			builder := tensor.NewBuilder()
			spec := standardDecoderFixture.decoderSpec(architecture)
			spec.NonCausalAttention = true
			input := builder.Input("input", dtype.F32, tensor.MustShape(8, 3))
			weights := denseBlockInputs(builder, spec)
			if architecture == "llada-moe" {
				spec.ExpertCount, spec.ExpertUsedCount, spec.ExpertFeedForward = 4, 2, 6
				spec.ExpertWeightsScale = 1
				weights.FeedForwardGate, weights.FeedForwardUp, weights.FeedForwardDown = nil, nil, nil
				weights.FeedForwardRouter = builder.Input("router", dtype.F32, tensor.MustShape(8, 4))
				weights.FeedForwardGateExperts = builder.Input("gate_exps", dtype.F32, tensor.MustShape(8, 6, 4))
				weights.FeedForwardUpExperts = builder.Input("up_exps", dtype.F32, tensor.MustShape(8, 6, 4))
				weights.FeedForwardDownExperts = builder.Input("down_exps", dtype.F32, tensor.MustShape(6, 8, 4))
			}
			result, err := BuildDenseBlockCachedForLayer(builder, input, spec, weights, []uint32{0, 1, 2}, nil, nil, 0)
			if err != nil {
				t.Fatal(err)
			}
			nodes, err := tensor.Topological(result.Output)
			if err != nil {
				t.Fatal(err)
			}
			var attention *tensor.Tensor
			var normal, neoX int
			for _, node := range nodes {
				switch node.Op {
				case tensor.OpAttention:
					attention = node
				case tensor.OpRoPENormal:
					normal++
				case tensor.OpRoPENeoX:
					neoX++
				}
			}
			if attention == nil || attention.Attrs.(tensor.AttentionAttributes).Causal ||
				(architecture == "llada" && normal != 2) || (architecture == "llada-moe" && neoX != 2) {
				t.Fatalf("unexpected %s graph: attention=%v normal=%d neox=%d", architecture, attention, normal, neoX)
			}
			if architecture == "llada-moe" {
				for _, node := range nodes {
					if node.Op == tensor.OpMoE && node.Attrs.(tensor.MoEAttributes).NormalizeTopKProb {
						t.Fatal("LLaDA-MoE unexpectedly normalizes selected expert weights")
					}
				}
			}
		})
	}
}

func TestBuildRND1BlockUsesNonCausalMoE(t *testing.T) {
	builder := tensor.NewBuilder()
	spec := Spec{CommonSpec: CommonSpec{Architecture: "rnd1", EmbeddingLength: 8, FeedForwardLength: 24,

		RMSNormEpsilon: 1e-6}, AttentionSpec: AttentionSpec{HeadCount: 2, HeadCountKV: 1, KeyLength: 4, ValueLength: 4,
		RopeFrequencyBase: 1_000_000, NonCausalAttention: true}, MoESpec: MoESpec{ExpertCount: 4, ExpertUsedCount: 2, ExpertFeedForward: 12, ExpertWeightsScale: 1},
	}
	input := builder.Input("input", dtype.F32, tensor.MustShape(8, 3))
	weights := denseBlockInputs(builder, spec)
	weights.FeedForwardGate = nil
	weights.FeedForwardUp = nil
	weights.FeedForwardDown = nil
	weights.FeedForwardRouter = builder.Input("router", dtype.F32, tensor.MustShape(8, 4))
	weights.FeedForwardGateExperts = builder.Input("gate_exps", dtype.F32, tensor.MustShape(8, 12, 4))
	weights.FeedForwardUpExperts = builder.Input("up_exps", dtype.F32, tensor.MustShape(8, 12, 4))
	weights.FeedForwardDownExperts = builder.Input("down_exps", dtype.F32, tensor.MustShape(12, 8, 4))
	output, err := BuildDenseBlock(builder, input, spec, weights, []uint32{0, 1, 2})
	if err != nil {
		t.Fatal(err)
	}
	nodes, err := tensor.Topological(output)
	if err != nil {
		t.Fatal(err)
	}
	var nonCausal, moe bool
	for _, node := range nodes {
		if node.Op == tensor.OpAttention {
			nonCausal = !node.Attrs.(tensor.AttentionAttributes).Causal
		}
		moe = moe || node.Op == tensor.OpMoE
	}
	if !nonCausal || !moe {
		t.Fatalf("RND1 graph non-causal/MoE = %v/%v", nonCausal, moe)
	}
}

func TestBuildLagunaMoEBlockUsesYaRNGateAndSharedExpert(t *testing.T) {
	builder := tensor.NewBuilder()
	spec := Spec{CommonSpec: CommonSpec{Architecture: "laguna", BlockCount: 2, EmbeddingLength: 8,
		FeedForwardLength: 16,

		RMSNormEpsilon: 1e-6}, AttentionSpec: AttentionSpec{HeadCount: 2, HeadCountKV: 1, LayerHeadCounts: []uint32{2, 4},
		LayerKVHeadCounts: []uint32{1, 1}, KeyLength: 4, ValueLength: 4,
		RopeFrequencyBase: 500000, RopeDimensionCount: 4, RopeScalingType: "yarn",
		RopeScalingFactor: 4, OriginalContextLength: 2048, YaRNExtFactor: 1,
		YaRNAttentionFactor: 1, YaRNBetaFast: 32, YaRNBetaSlow: 1}, MoESpec: MoESpec{LeadingDenseBlocks: 1,
		ExpertCount: 4, ExpertUsedCount: 2, ExpertFeedForward: 12,
		SharedExpertFF: 10, ExpertWeightsScale: 1.25, ExpertWeightsNorm: true},
	}
	input := builder.Input("input", dtype.F32, tensor.MustShape(8, 3))
	weights := LayerGraphWeights{
		AttentionNorm:          builder.Input("attn_norm", dtype.F32, tensor.MustShape(8)),
		AttentionQ:             builder.Input("q", dtype.F32, tensor.MustShape(8, 16)),
		AttentionK:             builder.Input("k", dtype.F32, tensor.MustShape(8, 4)),
		AttentionV:             builder.Input("v", dtype.F32, tensor.MustShape(8, 4)),
		AttentionOutput:        builder.Input("o", dtype.F32, tensor.MustShape(16, 8)),
		AttentionQNorm:         builder.Input("q_norm", dtype.F32, tensor.MustShape(4)),
		AttentionKNorm:         builder.Input("k_norm", dtype.F32, tensor.MustShape(4)),
		AttentionOutputGate:    builder.Input("attn_gate", dtype.F32, tensor.MustShape(8, 16)),
		FeedForwardNorm:        builder.Input("ffn_norm", dtype.F32, tensor.MustShape(8)),
		FeedForwardRouter:      builder.Input("router", dtype.F32, tensor.MustShape(8, 4)),
		FeedForwardGateExperts: builder.Input("gate_exps", dtype.F32, tensor.MustShape(8, 12, 4)),
		FeedForwardUpExperts:   builder.Input("up_exps", dtype.F32, tensor.MustShape(8, 12, 4)),
		FeedForwardDownExperts: builder.Input("down_exps", dtype.F32, tensor.MustShape(12, 8, 4)),
		FeedForwardExpertBias:  builder.Input("correction", dtype.F32, tensor.MustShape(4)),
		FeedForwardSharedGate:  builder.Input("shared_gate", dtype.F32, tensor.MustShape(8, 10)),
		FeedForwardSharedUp:    builder.Input("shared_up", dtype.F32, tensor.MustShape(8, 10)),
		FeedForwardSharedDown:  builder.Input("shared_down", dtype.F32, tensor.MustShape(10, 8)),
	}
	result, err := BuildDenseBlockCachedForLayer(builder, input, spec, weights, []uint32{0, 1, 2}, nil, nil, 1)
	if err != nil {
		t.Fatal(err)
	}
	nodes, err := tensor.Topological(result.Output)
	if err != nil {
		t.Fatal(err)
	}
	var moe *tensor.Tensor
	var softplus, yarn bool
	for _, node := range nodes {
		if node.Op == tensor.OpMoE {
			moe = node
		}
		softplus = softplus || node.Op == tensor.OpSoftplus
		if node.Op == tensor.OpRoPENeoX {
			attrs := node.Attrs.(tensor.RoPEAttributes)
			yarn = yarn || attrs.OriginalContext == 2048 && attrs.ExtFactor == 1
		}
	}
	if moe == nil || !softplus || !yarn {
		t.Fatalf("Laguna graph missing MoE/softplus/YaRN: %v/%v/%v", moe != nil, softplus, yarn)
	}
	attrs := moe.Attrs.(tensor.MoEAttributes)
	if attrs.Routing != tensor.MoERoutingSigmoid || !attrs.NormalizeTopKProb || len(moe.Inputs) != 7 {
		t.Fatalf("unexpected Laguna MoE attributes: %+v inputs=%d", attrs, len(moe.Inputs))
	}
}

func TestBuildLagunaSlidingLayerUsesPlainRoPEAndPerHeadGate(t *testing.T) {
	builder := tensor.NewBuilder()
	spec := Spec{CommonSpec: CommonSpec{Architecture: "laguna", BlockCount: 2, EmbeddingLength: 8,
		FeedForwardLength: 12,

		RMSNormEpsilon: 1e-6}, AttentionSpec: AttentionSpec{HeadCount: 2, HeadCountKV: 1, LayerHeadCounts: []uint32{2, 4},
		LayerKVHeadCounts: []uint32{1, 1}, KeyLength: 4, ValueLength: 4,
		RopeFrequencyBase: 500000, RopeDimensionCount: 4, RopeScalingType: "yarn",
		RopeScalingFactor: 4, OriginalContextLength: 2048, YaRNExtFactor: 1,
		YaRNAttentionFactor: 1, YaRNBetaFast: 32, YaRNBetaSlow: 1,
		SlidingWindow: 64, SlidingPattern: 2, RopeFrequencySWA: 10000, RopeDimensionSWA: 4}, MoESpec: MoESpec{LeadingDenseBlocks: 1},
	}
	input := builder.Input("input", dtype.F32, tensor.MustShape(8, 2))
	weights := LayerGraphWeights{
		AttentionNorm:       builder.Input("attn_norm", dtype.F32, tensor.MustShape(8)),
		AttentionQ:          builder.Input("q", dtype.F32, tensor.MustShape(8, 16)),
		AttentionK:          builder.Input("k", dtype.F32, tensor.MustShape(8, 4)),
		AttentionV:          builder.Input("v", dtype.F32, tensor.MustShape(8, 4)),
		AttentionOutput:     builder.Input("o", dtype.F32, tensor.MustShape(16, 8)),
		AttentionQNorm:      builder.Input("q_norm", dtype.F32, tensor.MustShape(4)),
		AttentionKNorm:      builder.Input("k_norm", dtype.F32, tensor.MustShape(4)),
		AttentionOutputGate: builder.Input("attn_gate", dtype.F32, tensor.MustShape(8, 4)),
		FeedForwardNorm:     builder.Input("ffn_norm", dtype.F32, tensor.MustShape(8)),
		FeedForwardGate:     builder.Input("ffn_gate", dtype.F32, tensor.MustShape(8, 12)),
		FeedForwardUp:       builder.Input("ffn_up", dtype.F32, tensor.MustShape(8, 12)),
		FeedForwardDown:     builder.Input("ffn_down", dtype.F32, tensor.MustShape(12, 8)),
	}
	result, err := BuildDenseBlockCachedForLayer(builder, input, spec, weights, []uint32{0, 1}, nil, nil, 1)
	if err != nil {
		t.Fatal(err)
	}
	nodes, _ := tensor.Topological(result.Output)
	var window, plainRoPE, broadcastGate bool
	for _, node := range nodes {
		if node.Op == tensor.OpAttention {
			window = window || node.Attrs.(tensor.AttentionAttributes).Window == 64
		}
		if node.Op == tensor.OpRoPENeoX {
			attrs := node.Attrs.(tensor.RoPEAttributes)
			plainRoPE = plainRoPE || attrs.FrequencyBase == 10000 && attrs.OriginalContext == 0
		}
		if node.Op == tensor.OpMultiply && node.Shape.Rank == 3 && node.Shape.Dims[0] == 4 && node.Shape.Dims[1] == 4 {
			broadcastGate = true
		}
	}
	if !window || !plainRoPE || !broadcastGate {
		t.Fatalf("Laguna SWA graph missing window/plain RoPE/per-head gate: %v/%v/%v", window, plainRoPE, broadcastGate)
	}
}

func TestBuildLFM2ShortConvolutionBlock(t *testing.T) {
	builder := tensor.NewBuilder()
	spec := Spec{CommonSpec: CommonSpec{Architecture: "lfm2", EmbeddingLength: 4, FeedForwardLength: 6,

		RMSNormEpsilon: 1e-6}, AttentionSpec: AttentionSpec{HeadCount: 1, HeadCountKV: 1, KeyLength: 4, ValueLength: 4}, RecurrentSpec: RecurrentSpec{ShortConvCacheLength: 3},
	}
	input := builder.Input("input", dtype.F32, tensor.MustShape(4, 2))
	weights := LayerGraphWeights{
		AttentionNorm:   builder.Input("operator_norm", dtype.F32, tensor.MustShape(4)),
		ShortConvInput:  builder.Input("conv_in", dtype.F32, tensor.MustShape(4, 12)),
		ShortConvKernel: builder.Input("conv_kernel", dtype.F32, tensor.MustShape(3, 4)),
		ShortConvOutput: builder.Input("conv_out", dtype.F32, tensor.MustShape(4, 4)),
		FeedForwardNorm: builder.Input("ffn_norm", dtype.F32, tensor.MustShape(4)),
		FeedForwardGate: builder.Input("ffn_gate", dtype.F32, tensor.MustShape(4, 6)),
		FeedForwardUp:   builder.Input("ffn_up", dtype.F32, tensor.MustShape(4, 6)),
		FeedForwardDown: builder.Input("ffn_down", dtype.F32, tensor.MustShape(6, 4)),
	}
	state := builder.Input(string(CacheStateConvolution), dtype.F32, tensor.MustShape(2, 4))
	reserved := builder.Input("reserved", dtype.F32, tensor.MustShape(1))
	plan := spec.PlanLayer(0, true)
	result, err := BuildArchitectureBlockCached(BlockDispatchOptions{
		Context: CachedBlockContext{
			Builder: builder, Input: input, Positions: []uint32{0, 1},
			PastKey: state, PastValue: reserved, Recurrent: true,
		},
		Spec: spec, Weights: weights, Plan: &plan,
	})
	if err != nil {
		t.Fatal(err)
	}
	if !result.Output.Shape.Equal(input.Shape) || !result.Key.Shape.Equal(state.Shape) ||
		!result.Value.Shape.Equal(reserved.Shape) {
		t.Fatalf("unexpected LFM2 result: %+v", result)
	}
	nodes, _ := tensor.Topological(result.Output)
	found := false
	for _, node := range nodes {
		found = found || node.Op == tensor.OpSSMConv
	}
	if !found {
		t.Fatal("LFM2 graph has no short convolution")
	}
}

func TestBuildLFM2CenteredShortConvolutionBlock(t *testing.T) {
	builder := tensor.NewBuilder()
	spec := Spec{CommonSpec: CommonSpec{Architecture: "lfm2", EmbeddingLength: 4, FeedForwardLength: 6,

		RMSNormEpsilon: 1e-6}, AttentionSpec: AttentionSpec{HeadCount: 1, HeadCountKV: 1, KeyLength: 4, ValueLength: 4,
		NonCausalAttention: true}, RecurrentSpec: RecurrentSpec{ShortConvCacheLength: 3},
	}
	input := builder.Input("input", dtype.F32, tensor.MustShape(4, 3))
	weights := LayerGraphWeights{
		AttentionNorm:   builder.Input("operator_norm", dtype.F32, tensor.MustShape(4)),
		ShortConvInput:  builder.Input("conv_in", dtype.F32, tensor.MustShape(4, 12)),
		ShortConvKernel: builder.Input("conv_kernel", dtype.F32, tensor.MustShape(3, 4)),
		ShortConvOutput: builder.Input("conv_out", dtype.F32, tensor.MustShape(4, 4)),
		FeedForwardNorm: builder.Input("ffn_norm", dtype.F32, tensor.MustShape(4)),
		FeedForwardGate: builder.Input("ffn_gate", dtype.F32, tensor.MustShape(4, 6)),
		FeedForwardUp:   builder.Input("ffn_up", dtype.F32, tensor.MustShape(4, 6)),
		FeedForwardDown: builder.Input("ffn_down", dtype.F32, tensor.MustShape(6, 4)),
	}
	state := builder.Input(string(CacheStateConvolution), dtype.F32, tensor.MustShape(2, 4))
	reserved := builder.Input("reserved", dtype.F32, tensor.MustShape(1))
	result, err := buildLFM2BlockCachedWithPlan(
		builder, input, spec, weights, []uint32{0, 1, 2}, state, reserved,
		spec.PlanLayer(0, true),
	)
	if err != nil {
		t.Fatal(err)
	}
	if !result.Output.Shape.Equal(input.Shape) || !result.Key.Shape.Equal(state.Shape) {
		t.Fatalf("unexpected centered LFM2 result: %+v", result)
	}
	nodes, err := tensor.Topological(result.Output)
	if err != nil {
		t.Fatal(err)
	}
	var convolution *tensor.Tensor
	for _, node := range nodes {
		if node.Op == tensor.OpSSMConv {
			convolution = node
		}
	}
	if convolution == nil || !convolution.Inputs[0].Shape.Equal(tensor.MustShape(5, 4)) {
		t.Fatalf("centered LFM2 convolution input = %v", convolution)
	}
}

func TestBuildLFM2CenteredShortConvolutionSupportsEvenKernel(t *testing.T) {
	builder := tensor.NewBuilder()
	spec := Spec{CommonSpec: CommonSpec{Architecture: "lfm2", EmbeddingLength: 4, FeedForwardLength: 6,
		RMSNormEpsilon: 1e-6}, AttentionSpec: AttentionSpec{NonCausalAttention: true}, RecurrentSpec: RecurrentSpec{ShortConvCacheLength: 4},
	}
	input := builder.Input("input", dtype.F32, tensor.MustShape(4, 2))
	weights := LayerGraphWeights{
		AttentionNorm:   builder.Input("operator_norm", dtype.F32, tensor.MustShape(4)),
		ShortConvInput:  builder.Input("conv_in", dtype.F32, tensor.MustShape(4, 12)),
		ShortConvKernel: builder.Input("conv_kernel", dtype.F32, tensor.MustShape(4, 4)),
		ShortConvOutput: builder.Input("conv_out", dtype.F32, tensor.MustShape(4, 4)),
		FeedForwardNorm: builder.Input("ffn_norm", dtype.F32, tensor.MustShape(4)),
		FeedForwardGate: builder.Input("ffn_gate", dtype.F32, tensor.MustShape(4, 6)),
		FeedForwardUp:   builder.Input("ffn_up", dtype.F32, tensor.MustShape(4, 6)),
		FeedForwardDown: builder.Input("ffn_down", dtype.F32, tensor.MustShape(6, 4)),
	}
	result, err := buildLFM2BlockCachedWithPlan(
		builder, input, spec, weights, []uint32{0, 1},
		builder.Input(string(CacheStateConvolution), dtype.F32, tensor.MustShape(3, 4)),
		builder.Input("reserved", dtype.F32, tensor.MustShape(1)), spec.PlanLayer(0, true),
	)
	if err != nil {
		t.Fatal(err)
	}
	nodes, err := tensor.Topological(result.Output)
	if err != nil {
		t.Fatal(err)
	}
	var convolution *tensor.Tensor
	for _, node := range nodes {
		if node.Op == tensor.OpSSMConv {
			convolution = node
		}
	}
	if convolution == nil || !convolution.Inputs[0].Shape.Equal(tensor.MustShape(5, 4)) {
		t.Fatalf("centered even-kernel convolution input = %v", convolution)
	}
}

func TestBuildLFM2MoEShortConvolutionBlock(t *testing.T) {
	builder := tensor.NewBuilder()
	spec := Spec{CommonSpec: CommonSpec{Architecture: "lfm2moe", BlockCount: 3,
		EmbeddingLength: 4, FeedForwardLength: 6,

		RMSNormEpsilon: 1e-6}, AttentionSpec: AttentionSpec{HeadCount: 1, HeadCountKV: 1, KeyLength: 4, ValueLength: 4}, MoESpec: MoESpec{LeadingDenseBlocks: 1,

		ExpertCount: 4, ExpertUsedCount: 2, ExpertFeedForward: 6,
		ExpertWeightsScale: 1.25, ExpertGatingFunc: expertGatingSigmoid}, RecurrentSpec: RecurrentSpec{ShortConvCacheLength: 3},
	}
	input := builder.Input("input", dtype.F32, tensor.MustShape(4, 2))
	weights := LayerGraphWeights{
		AttentionNorm:          builder.Input("operator_norm", dtype.F32, tensor.MustShape(4)),
		ShortConvInput:         builder.Input("conv_in", dtype.F32, tensor.MustShape(4, 12)),
		ShortConvKernel:        builder.Input("conv_kernel", dtype.F32, tensor.MustShape(3, 4)),
		ShortConvOutput:        builder.Input("conv_out", dtype.F32, tensor.MustShape(4, 4)),
		FeedForwardNorm:        builder.Input("ffn_norm", dtype.F32, tensor.MustShape(4)),
		FeedForwardRouter:      builder.Input("router", dtype.F32, tensor.MustShape(4, 4)),
		FeedForwardGateExperts: builder.Input("gate_exps", dtype.F32, tensor.MustShape(4, 6, 4)),
		FeedForwardUpExperts:   builder.Input("up_exps", dtype.F32, tensor.MustShape(4, 6, 4)),
		FeedForwardDownExperts: builder.Input("down_exps", dtype.F32, tensor.MustShape(6, 4, 4)),
		FeedForwardExpertBias:  builder.Input("correction", dtype.F32, tensor.MustShape(4)),
	}
	state := builder.Input(string(CacheStateConvolution), dtype.F32, tensor.MustShape(2, 4))
	reserved := builder.Input("reserved", dtype.F32, tensor.MustShape(1))
	result, err := buildLFM2BlockCachedWithPlan(
		builder, input, spec, weights, []uint32{0, 1}, state, reserved,
		spec.PlanLayer(2, true),
	)
	if err != nil {
		t.Fatal(err)
	}
	nodes, err := tensor.Topological(result.Output)
	if err != nil {
		t.Fatal(err)
	}
	var moe *tensor.Tensor
	var convolution bool
	for _, node := range nodes {
		if node.Op == tensor.OpMoE {
			moe = node
		}
		convolution = convolution || node.Op == tensor.OpSSMConv
	}
	if moe == nil || !convolution {
		t.Fatalf("LFM2-MoE graph missing convolution or MoE: %v/%v", convolution, moe != nil)
	}
	attrs := moe.Attrs.(tensor.MoEAttributes)
	if attrs.Routing != tensor.MoERoutingSigmoid || !attrs.NormalizeTopKProb ||
		attrs.TopK != 2 || attrs.Scale != 1.25 || len(moe.Inputs) != 7 {
		t.Fatalf("unexpected LFM2-MoE attributes: %+v inputs=%d", attrs, len(moe.Inputs))
	}
}

func TestBuildPLMMLABlock(t *testing.T) {
	builder := tensor.NewBuilder()
	spec := Spec{CommonSpec: CommonSpec{Architecture: "plm", EmbeddingLength: 8, FeedForwardLength: 12,

		RMSNormEpsilon: 1e-6}, AttentionSpec: AttentionSpec{HeadCount: 2, HeadCountKV: 2, KeyLength: 6, ValueLength: 4,
		KVLoRARank: 3, RopeDimensionCount: 2, RopeFrequencyBase: 10000},
	}
	input := builder.Input("input", dtype.F32, tensor.MustShape(8, 2))
	weights := LayerGraphWeights{
		AttentionNorm:    builder.Input("attn_norm", dtype.F32, tensor.MustShape(8)),
		AttentionQ:       builder.Input("q", dtype.F32, tensor.MustShape(8, 12)),
		AttentionKVAMQA:  builder.Input("kv_a", dtype.F32, tensor.MustShape(8, 5)),
		AttentionKVANorm: builder.Input("kv_a_norm", dtype.F32, tensor.MustShape(3)),
		AttentionKVB:     builder.Input("kv_b", dtype.F32, tensor.MustShape(3, 16)),
		AttentionOutput:  builder.Input("attn_out", dtype.F32, tensor.MustShape(8, 8)),
		FeedForwardNorm:  builder.Input("ffn_norm", dtype.F32, tensor.MustShape(8)),
		FeedForwardUp:    builder.Input("ffn_up", dtype.F32, tensor.MustShape(8, 12)),
		FeedForwardDown:  builder.Input("ffn_down", dtype.F32, tensor.MustShape(12, 8)),
	}
	result, err := BuildPLMBlockCached(builder, input, spec, weights, []uint32{0, 1}, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	if !result.Output.Shape.Equal(input.Shape) ||
		!result.Key.Shape.Equal(tensor.MustShape(6, 2, 2)) ||
		!result.Value.Shape.Equal(tensor.MustShape(4, 2, 2)) {
		t.Fatalf("unexpected PLM result: %+v", result)
	}
	nodes, _ := tensor.Topological(result.Output)
	var attention, repeatHeads bool
	var normalRoPE, neoxRoPE int
	for _, node := range nodes {
		attention = attention || node.Op == tensor.OpAttention
		repeatHeads = repeatHeads || node.Op == tensor.OpRepeatHeads
		if node.Op == tensor.OpRoPENormal {
			normalRoPE++
		}
		if node.Op == tensor.OpRoPENeoX {
			neoxRoPE++
		}
	}
	if !attention {
		t.Fatal("PLM graph has no attention")
	}
	if !repeatHeads {
		t.Fatal("PLM graph does not repeat its shared positional key")
	}
	if normalRoPE != 2 || neoxRoPE != 0 {
		t.Fatalf("PLM RoPE ops: normal=%d NeoX=%d", normalRoPE, neoxRoPE)
	}
}

func TestBuildDeepSeek2AbsorbedMLABlock(t *testing.T) {
	testBuildDeepSeek2FamilyAbsorbedMLABlock(t, "deepseek2")
}

func TestBuildMistral4AbsorbedMLABlock(t *testing.T) {
	testBuildDeepSeek2FamilyAbsorbedMLABlock(t, "mistral4")
}

func TestBuildMambaBlock(t *testing.T) {
	builder := tensor.NewBuilder()
	spec := Spec{CommonSpec: CommonSpec{Architecture: "mamba", EmbeddingLength: 4, RMSNormEpsilon: 1e-5}, RecurrentSpec: RecurrentSpec{SSMConvKernel: 3, SSMInnerSize: 8, SSMStateSize: 2, SSMTimeStepRank: 2}}
	input := builder.Input("input", dtype.F32, tensor.MustShape(4, 2))
	weights := LayerGraphWeights{
		AttentionNorm:     builder.Input("norm", dtype.F32, tensor.MustShape(4)),
		SSMInput:          builder.Input("in", dtype.F32, tensor.MustShape(4, 16)),
		SSMConv1D:         builder.Input("conv", dtype.F32, tensor.MustShape(3, 8)),
		SSMConv1DBias:     builder.Input("conv_bias", dtype.F32, tensor.MustShape(8)),
		SSMX:              builder.Input("x", dtype.F32, tensor.MustShape(8, 6)),
		SSMTimeStepWeight: builder.Input("dt", dtype.F32, tensor.MustShape(2, 8)),
		SSMTimeStep:       builder.Input("dt_bias", dtype.F32, tensor.MustShape(8)),
		SSMA:              builder.Input("a", dtype.F32, tensor.MustShape(2, 8)),
		SSMD:              builder.Input("d", dtype.F32, tensor.MustShape(8)),
		SSMOutput:         builder.Input("out", dtype.F32, tensor.MustShape(8, 4)),
	}
	convState := builder.Input(string(CacheStateConvolution), dtype.F32, tensor.MustShape(2, 8))
	ssmState := builder.Input(string(CacheStateSSM), dtype.F32, tensor.MustShape(2, 8))
	plan := spec.PlanLayer(0, false)
	result, err := BuildArchitectureBlockCached(BlockDispatchOptions{
		Spec: spec, Weights: weights, Plan: &plan,
		Context: CachedBlockContext{
			Builder: builder, Input: input, PastKey: convState, PastValue: ssmState,
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	if !result.Output.Shape.Equal(input.Shape) || !result.Key.Shape.Equal(convState.Shape) ||
		!result.Value.Shape.Equal(ssmState.Shape) {
		t.Fatalf("unexpected Mamba result shapes: output=%v conv=%v ssm=%v", result.Output.Shape, result.Key.Shape, result.Value.Shape)
	}
	nodes, err := tensor.Topological(result.Output, result.Key, result.Value)
	if err != nil {
		t.Fatal(err)
	}
	convolution, scan := false, false
	for _, node := range nodes {
		convolution = convolution || node.Op == tensor.OpSSMConv
		scan = scan || node.Op == tensor.OpSSMScan
	}
	if !convolution || !scan {
		t.Fatalf("Mamba graph convolution=%v scan=%v", convolution, scan)
	}
}

func TestBuildMamba2Block(t *testing.T) {
	builder := tensor.NewBuilder()
	spec := Spec{CommonSpec: CommonSpec{Architecture: "mamba2", EmbeddingLength: 4, RMSNormEpsilon: 1e-5}, RecurrentSpec: RecurrentSpec{SSMConvKernel: 3, SSMInnerSize: 8, SSMStateSize: 2, SSMTimeStepRank: 4,
		SSMGroupCount: 2}}
	input := builder.Input("input", dtype.F32, tensor.MustShape(4, 2))
	weights := LayerGraphWeights{
		AttentionNorm: builder.Input("norm", dtype.F32, tensor.MustShape(4)),
		SSMInput:      builder.Input("in", dtype.F32, tensor.MustShape(4, 28)),
		SSMConv1D:     builder.Input("conv", dtype.F32, tensor.MustShape(3, 16)),
		SSMConv1DBias: builder.Input("conv_bias", dtype.F32, tensor.MustShape(16)),
		SSMTimeStep:   builder.Input("dt_bias", dtype.F32, tensor.MustShape(4)),
		SSMA:          builder.Input("a", dtype.F32, tensor.MustShape(1, 4)),
		SSMD:          builder.Input("d", dtype.F32, tensor.MustShape(1, 4)),
		SSMNorm:       builder.Input("ssm_norm", dtype.F32, tensor.MustShape(4, 2)),
		SSMOutput:     builder.Input("out", dtype.F32, tensor.MustShape(8, 4)),
	}
	convState := builder.Input(string(CacheStateConvolution), dtype.F32, tensor.MustShape(2, 16))
	ssmState := builder.Input(string(CacheStateSSM), dtype.F32, tensor.MustShape(2, 8))
	plan := spec.PlanLayer(0, false)
	result, err := BuildArchitectureBlockCached(BlockDispatchOptions{
		Spec: spec, Weights: weights, Plan: &plan,
		Context: CachedBlockContext{
			Builder: builder, Input: input, PastKey: convState, PastValue: ssmState,
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	if !result.Output.Shape.Equal(input.Shape) || !result.Key.Shape.Equal(convState.Shape) ||
		!result.Value.Shape.Equal(ssmState.Shape) {
		t.Fatalf("unexpected Mamba2 result shapes: output=%v conv=%v ssm=%v", result.Output.Shape, result.Key.Shape, result.Value.Shape)
	}
	nodes, err := tensor.Topological(result.Output, result.Key, result.Value)
	if err != nil {
		t.Fatal(err)
	}
	convolution, scan, groupedNorm := false, false, false
	for _, node := range nodes {
		convolution = convolution || node.Op == tensor.OpSSMConv
		scan = scan || node.Op == tensor.OpSSMScan
		groupedNorm = groupedNorm || node.Op == tensor.OpRMSNorm && node.Shape.Rank == 4 && node.Shape.Dims[0] == 4
	}
	if !convolution || !scan || !groupedNorm {
		t.Fatalf("Mamba2 graph convolution=%v scan=%v grouped_norm=%v", convolution, scan, groupedNorm)
	}
}

func TestBuildFalconH1Block(t *testing.T) {
	fixturePositions := []uint32{0, 1}
	builder := tensor.NewBuilder()
	spec := Spec{CommonSpec: CommonSpec{Architecture: "falcon-h1", EmbeddingLength: 4, FeedForwardLength: 6,

		RMSNormEpsilon: 1e-5}, AttentionSpec: AttentionSpec{HeadCount: 2, HeadCountKV: 1, KeyLength: 2, ValueLength: 2, RopeDimensionCount: 2,
		RopeFrequencyBase: 10000}, RecurrentSpec: RecurrentSpec{SSMConvKernel: 3,
		SSMInnerSize: 8, SSMStateSize: 2, SSMTimeStepRank: 4, SSMGroupCount: 2}}
	input := builder.Input("input", dtype.F32, tensor.MustShape(4, 2))
	weights := LayerGraphWeights{
		AttentionNorm:   builder.Input("norm", dtype.F32, tensor.MustShape(4)),
		AttentionQ:      builder.Input("q", dtype.F32, tensor.MustShape(4, 4)),
		AttentionK:      builder.Input("k", dtype.F32, tensor.MustShape(4, 2)),
		AttentionV:      builder.Input("v", dtype.F32, tensor.MustShape(4, 2)),
		AttentionOutput: builder.Input("attn_out", dtype.F32, tensor.MustShape(4, 4)),
		SSMInput:        builder.Input("ssm_in", dtype.F32, tensor.MustShape(4, 28)),
		SSMConv1D:       builder.Input("conv", dtype.F32, tensor.MustShape(3, 16)),
		SSMTimeStep:     builder.Input("dt", dtype.F32, tensor.MustShape(4)),
		SSMA:            builder.Input("a", dtype.F32, tensor.MustShape(1, 4)),
		SSMD:            builder.Input("d", dtype.F32, tensor.MustShape(1, 4)),
		SSMOutput:       builder.Input("ssm_out", dtype.F32, tensor.MustShape(8, 4)),
		FeedForwardNorm: builder.Input("ffn_norm", dtype.F32, tensor.MustShape(4)),
		FeedForwardGate: builder.Input("gate", dtype.F32, tensor.MustShape(4, 6)),
		FeedForwardUp:   builder.Input("up", dtype.F32, tensor.MustShape(4, 6)),
		FeedForwardDown: builder.Input("down", dtype.F32, tensor.MustShape(6, 4)),
	}
	convState := builder.Input(string(CacheStateConvolution), dtype.F32, tensor.MustShape(2, 16))
	ssmState := builder.Input(string(CacheStateSSM), dtype.F32, tensor.MustShape(2, 8))
	plan := spec.PlanLayer(0, false)
	result, err := BuildArchitectureBlockCached(BlockDispatchOptions{
		Spec: spec, Weights: weights, Plan: &plan,
		Context: CachedBlockContext{
			Builder: builder, Input: input, Positions: fixturePositions,
			PastStates: CacheStates[*tensor.Tensor]{
				CacheStateConvolution: {Mode: CacheStateFixed, Value: convState},
				CacheStateSSM:         {Mode: CacheStateFixed, Value: ssmState},
			},
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	convResult := result.States[CacheStateConvolution]
	ssmResult := result.States[CacheStateSSM]
	if !result.Output.Shape.Equal(input.Shape) || !result.Key.Shape.Equal(tensor.MustShape(2, 1, 2)) ||
		!result.Value.Shape.Equal(tensor.MustShape(2, 1, 2)) || len(result.States) != 2 ||
		convResult.Mode != CacheStateFixed || ssmResult.Mode != CacheStateFixed ||
		!convResult.Value.Shape.Equal(convState.Shape) || !ssmResult.Value.Shape.Equal(ssmState.Shape) {
		t.Fatalf("unexpected Falcon-H1 result: %+v", result)
	}
	nodes, err := tensor.Topological(result.Output, result.Key, result.Value, convResult.Value, ssmResult.Value)
	if err != nil {
		t.Fatal(err)
	}
	attention, convolution, scan, rope := false, false, false, false
	for _, node := range nodes {
		attention = attention || node.Op == tensor.OpAttention
		convolution = convolution || node.Op == tensor.OpSSMConv
		scan = scan || node.Op == tensor.OpSSMScan
		rope = rope || node.Op == tensor.OpRoPENeoX
	}
	if !attention || !convolution || !scan || !rope {
		t.Fatalf("Falcon-H1 graph attention=%v convolution=%v scan=%v rope=%v", attention, convolution, scan, rope)
	}
}

func TestBuildGraniteHybridRecurrentMoEBlock(t *testing.T) {
	builder := tensor.NewBuilder()
	spec := Spec{CommonSpec: CommonSpec{Architecture: "granitehybrid", EmbeddingLength: 4, FeedForwardLength: 6,
		RMSNormEpsilon: 1e-5, ResidualScale: 0.5}, MoESpec: MoESpec{ExpertCount: 4,
		ExpertUsedCount: 2, ExpertFeedForward: 6, SharedExpertFF: 5, ExpertWeightsScale: 1}, RecurrentSpec: RecurrentSpec{SSMConvKernel: 3, SSMInnerSize: 8,
		SSMStateSize: 2, SSMTimeStepRank: 4, SSMGroupCount: 2},
	}
	input := builder.Input("input", dtype.F32, tensor.MustShape(4, 2))
	weights := LayerGraphWeights{
		AttentionNorm:          builder.Input("norm", dtype.F32, tensor.MustShape(4)),
		SSMInput:               builder.Input("in", dtype.F32, tensor.MustShape(4, 28)),
		SSMConv1D:              builder.Input("conv", dtype.F32, tensor.MustShape(3, 16)),
		SSMTimeStep:            builder.Input("dt_bias", dtype.F32, tensor.MustShape(4)),
		SSMA:                   builder.Input("a", dtype.F32, tensor.MustShape(1, 4)),
		SSMD:                   builder.Input("d", dtype.F32, tensor.MustShape(1, 4)),
		SSMNorm:                builder.Input("ssm_norm", dtype.F32, tensor.MustShape(4, 2)),
		SSMOutput:              builder.Input("out", dtype.F32, tensor.MustShape(8, 4)),
		FeedForwardNorm:        builder.Input("ffn_norm", dtype.F32, tensor.MustShape(4)),
		FeedForwardRouter:      builder.Input("router", dtype.F32, tensor.MustShape(4, 4)),
		FeedForwardUpExperts:   builder.Input("up_exps", dtype.F32, tensor.MustShape(4, 6, 4)),
		FeedForwardDownExperts: builder.Input("down_exps", dtype.F32, tensor.MustShape(6, 4, 4)),
		FeedForwardSharedGate:  builder.Input("shared_gate", dtype.F32, tensor.MustShape(4, 5)),
		FeedForwardSharedUp:    builder.Input("shared_up", dtype.F32, tensor.MustShape(4, 5)),
		FeedForwardSharedDown:  builder.Input("shared_down", dtype.F32, tensor.MustShape(5, 4)),
	}
	convState := builder.Input(string(CacheStateConvolution), dtype.F32, tensor.MustShape(2, 16))
	ssmState := builder.Input(string(CacheStateSSM), dtype.F32, tensor.MustShape(2, 8))
	plan := spec.PlanLayer(0, true)
	result, err := BuildArchitectureBlockCached(BlockDispatchOptions{
		Spec: spec, Weights: weights, Plan: &plan,
		Context: CachedBlockContext{
			Builder: builder, Input: input, PastKey: convState, PastValue: ssmState,
			Recurrent: true,
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	nodes, err := tensor.Topological(result.Output, result.Key, result.Value)
	if err != nil {
		t.Fatal(err)
	}
	convolution, scan, moe, scales := false, false, false, 0
	for _, node := range nodes {
		convolution = convolution || node.Op == tensor.OpSSMConv
		scan = scan || node.Op == tensor.OpSSMScan
		moe = moe || node.Op == tensor.OpMoE
		if node.Op == tensor.OpScale {
			scales++
		}
	}
	if !convolution || !scan || !moe || scales < 2 || !result.Key.Shape.Equal(convState.Shape) ||
		!result.Value.Shape.Equal(ssmState.Shape) {
		t.Fatalf("Granite Hybrid graph convolution=%v scan=%v moe=%v scales=%d", convolution, scan, moe, scales)
	}
}

func TestBuildPLaMo2HybridBlocks(t *testing.T) {
	spec := Spec{CommonSpec: CommonSpec{Architecture: "plamo2", BlockCount: 2, EmbeddingLength: 4,
		FeedForwardLength: 6,

		RMSNormEpsilon: 1e-5}, AttentionSpec: AttentionSpec{HeadCount: 2, HeadCountKV: 1, KeyLength: 2, ValueLength: 2,
		LayerKVHeadCounts: []uint32{0, 1},
		RopeFrequencyBase: 10000, AttentionScale: 1 / float32(math.Sqrt(2))}, RecurrentSpec: RecurrentSpec{RecurrentLayers: []bool{true, false},

		SSMConvKernel: 3, SSMInnerSize: 8, SSMStateSize: 2, SSMTimeStepRank: 4}}
	builder := tensor.NewBuilder()
	input := builder.Input("input", dtype.F32, tensor.MustShape(4, 2))
	recurrentWeights := LayerGraphWeights{
		AttentionNorm:       builder.Input("norm", dtype.F32, tensor.MustShape(4)),
		AttentionPostNorm:   builder.Input("post_norm", dtype.F32, tensor.MustShape(4)),
		SSMInput:            builder.Input("in", dtype.F32, tensor.MustShape(4, 16)),
		SSMConv1D:           builder.Input("conv", dtype.F32, tensor.MustShape(3, 8)),
		SSMX:                builder.Input("x", dtype.F32, tensor.MustShape(8, 68)),
		SSMTimeStepWeight:   builder.Input("dt", dtype.F32, tensor.MustShape(64, 4)),
		SSMTimeStep:         builder.Input("dt_bias", dtype.F32, tensor.MustShape(4)),
		SSMTimeStepNorm:     builder.Input("dt_norm", dtype.F32, tensor.MustShape(64)),
		SSMA:                builder.Input("a", dtype.F32, tensor.MustShape(4)),
		SSMD:                builder.Input("d", dtype.F32, tensor.MustShape(4)),
		SSMBNorm:            builder.Input("b_norm", dtype.F32, tensor.MustShape(2)),
		SSMCNorm:            builder.Input("c_norm", dtype.F32, tensor.MustShape(2)),
		SSMOutput:           builder.Input("out", dtype.F32, tensor.MustShape(8, 4)),
		FeedForwardNorm:     builder.Input("ffn_norm", dtype.F32, tensor.MustShape(4)),
		FeedForwardUp:       builder.Input("ffn_up", dtype.F32, tensor.MustShape(4, 12)),
		FeedForwardDown:     builder.Input("ffn_down", dtype.F32, tensor.MustShape(6, 4)),
		FeedForwardPostNorm: builder.Input("ffn_post", dtype.F32, tensor.MustShape(4)),
	}
	convState := builder.Input(string(CacheStateConvolution), dtype.F32, tensor.MustShape(2, 8))
	ssmState := builder.Input(string(CacheStateSSM), dtype.F32, tensor.MustShape(2, 8))
	plan := spec.PlanLayer(0, true)
	recurrent, err := BuildArchitectureBlockCached(BlockDispatchOptions{
		Spec: spec, Weights: recurrentWeights, Plan: &plan,
		Context: CachedBlockContext{
			Builder: builder, Input: input, PastKey: convState, PastValue: ssmState,
			Recurrent: true,
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	nodes, err := tensor.Topological(recurrent.Output, recurrent.Key, recurrent.Value)
	if err != nil {
		t.Fatal(err)
	}
	convolution, scan := false, false
	for _, node := range nodes {
		convolution = convolution || node.Op == tensor.OpSSMConv
		scan = scan || node.Op == tensor.OpSSMScan
	}
	if !convolution || !scan || !recurrent.Key.Shape.Equal(convState.Shape) || !recurrent.Value.Shape.Equal(ssmState.Shape) {
		t.Fatalf("PLaMo2 recurrent graph convolution=%v scan=%v", convolution, scan)
	}

	builder = tensor.NewBuilder()
	input = builder.Input("input", dtype.F32, tensor.MustShape(4, 2))
	attentionWeights := LayerGraphWeights{
		AttentionNorm:       builder.Input("norm", dtype.F32, tensor.MustShape(4)),
		AttentionQKV:        builder.Input("qkv", dtype.F32, tensor.MustShape(4, 8)),
		AttentionQNorm:      builder.Input("q_norm", dtype.F32, tensor.MustShape(2, 2)),
		AttentionKNorm:      builder.Input("k_norm", dtype.F32, tensor.MustShape(2, 1)),
		AttentionOutput:     builder.Input("attn_out", dtype.F32, tensor.MustShape(4, 4)),
		AttentionPostNorm:   builder.Input("post_norm", dtype.F32, tensor.MustShape(4)),
		FeedForwardNorm:     builder.Input("ffn_norm", dtype.F32, tensor.MustShape(4)),
		FeedForwardUp:       builder.Input("ffn_up", dtype.F32, tensor.MustShape(4, 12)),
		FeedForwardDown:     builder.Input("ffn_down", dtype.F32, tensor.MustShape(6, 4)),
		FeedForwardPostNorm: builder.Input("ffn_post", dtype.F32, tensor.MustShape(4)),
	}
	attention, err := BuildDenseBlockCachedForLayer(builder, input, spec, attentionWeights, []uint32{0, 1}, nil, nil, 1)
	if err != nil {
		t.Fatal(err)
	}
	nodes, err = tensor.Topological(attention.Output, attention.Key, attention.Value)
	if err != nil {
		t.Fatal(err)
	}
	attentionOp, rope := false, false
	for _, node := range nodes {
		attentionOp = attentionOp || node.Op == tensor.OpAttention
		rope = rope || node.Op == tensor.OpRoPENormal
	}
	if !attentionOp || !rope {
		t.Fatalf("PLaMo2 attention graph attention=%v rope=%v", attentionOp, rope)
	}
}

func TestBuildJambaAttentionMoEBlock(t *testing.T) {
	builder := tensor.NewBuilder()
	spec := Spec{CommonSpec: CommonSpec{Architecture: "jamba", BlockCount: 2, EmbeddingLength: 4,
		FeedForwardLength: 6,

		RMSNormEpsilon: 1e-5}, AttentionSpec: AttentionSpec{HeadCount: 2, HeadCountKV: 1, KeyLength: 2, ValueLength: 2,
		LayerKVHeadCounts: []uint32{0, 1},
		RopeDisabled:      true}, MoESpec: MoESpec{ExpertCount: 4, ExpertUsedCount: 2,
		ExpertFeedForward: 6, ExpertWeightsScale: 1}, RecurrentSpec: RecurrentSpec{RecurrentLayers: []bool{true, false}},
	}
	input := builder.Input("input", dtype.F32, tensor.MustShape(4, 2))
	weights := LayerGraphWeights{
		AttentionNorm:          builder.Input("attn_norm", dtype.F32, tensor.MustShape(4)),
		AttentionQ:             builder.Input("q", dtype.F32, tensor.MustShape(4, 4)),
		AttentionK:             builder.Input("k", dtype.F32, tensor.MustShape(4, 2)),
		AttentionV:             builder.Input("v", dtype.F32, tensor.MustShape(4, 2)),
		AttentionOutput:        builder.Input("attn_out", dtype.F32, tensor.MustShape(4, 4)),
		FeedForwardNorm:        builder.Input("ffn_norm", dtype.F32, tensor.MustShape(4)),
		FeedForwardRouter:      builder.Input("router", dtype.F32, tensor.MustShape(4, 4)),
		FeedForwardGateExperts: builder.Input("gate", dtype.F32, tensor.MustShape(4, 6, 4)),
		FeedForwardUpExperts:   builder.Input("up", dtype.F32, tensor.MustShape(4, 6, 4)),
		FeedForwardDownExperts: builder.Input("down", dtype.F32, tensor.MustShape(6, 4, 4)),
	}
	result, err := BuildDenseBlockCachedForLayer(builder, input, spec, weights, []uint32{0, 1}, nil, nil, 1)
	if err != nil {
		t.Fatal(err)
	}
	nodes, err := tensor.Topological(result.Output, result.Key, result.Value)
	if err != nil {
		t.Fatal(err)
	}
	attention, moe, rope := false, false, false
	for _, node := range nodes {
		attention = attention || node.Op == tensor.OpAttention
		moe = moe || node.Op == tensor.OpMoE
		rope = rope || node.Op == tensor.OpRoPENormal || node.Op == tensor.OpRoPENeoX
	}
	if !attention || !moe || rope {
		t.Fatalf("Jamba attention graph attention=%v moe=%v rope=%v", attention, moe, rope)
	}
}

func testBuildDeepSeek2FamilyAbsorbedMLABlock(t *testing.T, architecture string) {
	builder := tensor.NewBuilder()
	spec := Spec{CommonSpec: CommonSpec{Architecture: architecture, BlockCount: 2, EmbeddingLength: 8,
		FeedForwardLength: 12,

		RMSNormEpsilon: 1e-6}, AttentionSpec: AttentionSpec{HeadCount: 2, HeadCountKV: 2, KeyLength: 6, ValueLength: 4,
		QLoRARank: 3, KVLoRARank: 3, RopeDimensionCount: 2, RopeFrequencyBase: 10000,

		RopeScalingType: "yarn", RopeScalingFactor: 4, OriginalContextLength: 4096,
		YaRNExtFactor: 1, YaRNAttentionFactor: 1 / (1 + 0.1*float32(math.Log(4))),
		YaRNBetaFast: 32, YaRNBetaSlow: 1, RopeYaRNLogMultiplier: 1}, MoESpec: MoESpec{LeadingDenseBlocks: 1, ExpertCount: 2, ExpertUsedCount: 1,
		ExpertFeedForward: 6, SharedExpertFF: 6, ExpertWeightsScale: 1, ExpertGatingFunc: expertGatingSoftmax},
	}
	input := builder.Input("input", dtype.F32, tensor.MustShape(8, 2))
	weights := LayerGraphWeights{
		AttentionNorm:             builder.Input("attn_norm", dtype.F32, tensor.MustShape(8)),
		AttentionQ:                builder.Input("q_a", dtype.F32, tensor.MustShape(8, 3)),
		AttentionQNorm:            builder.Input("q_norm", dtype.F32, tensor.MustShape(3)),
		AttentionQB:               builder.Input("q_b", dtype.F32, tensor.MustShape(3, 12)),
		AttentionKVAMQA:           builder.Input("kv_a", dtype.F32, tensor.MustShape(8, 5)),
		AttentionKVANorm:          builder.Input("kv_norm", dtype.F32, tensor.MustShape(3)),
		AttentionKB:               builder.Input("k_b", dtype.F32, tensor.MustShape(4, 3, 2)),
		AttentionVB:               builder.Input("v_b", dtype.F32, tensor.MustShape(3, 4, 2)),
		AttentionOutput:           builder.Input("attn_out", dtype.F32, tensor.MustShape(8, 8)),
		AttentionTemperatureScale: builder.Input("temperature", dtype.F32, tensor.MustShape(1, 1, 2)),
		FeedForwardNorm:           builder.Input("ffn_norm", dtype.F32, tensor.MustShape(8)),
		FeedForwardRouter:         builder.Input("router", dtype.F32, tensor.MustShape(8, 2)),
		FeedForwardGateExperts:    builder.Input("gate_exps", dtype.F32, tensor.MustShape(8, 6, 2)),
		FeedForwardUpExperts:      builder.Input("up_exps", dtype.F32, tensor.MustShape(8, 6, 2)),
		FeedForwardDownExperts:    builder.Input("down_exps", dtype.F32, tensor.MustShape(6, 8, 2)),
		FeedForwardSharedGate:     builder.Input("shared_gate", dtype.F32, tensor.MustShape(8, 6)),
		FeedForwardSharedUp:       builder.Input("shared_up", dtype.F32, tensor.MustShape(8, 6)),
		FeedForwardSharedDown:     builder.Input("shared_down", dtype.F32, tensor.MustShape(6, 8)),
	}
	result, err := BuildMLABlockCachedForLayer(builder, input, spec, weights, []uint32{0, 1}, nil, nil, 1)
	if err != nil {
		t.Fatal(err)
	}
	if !result.Key.Shape.Equal(tensor.MustShape(5, 1, 2)) || !result.Value.Shape.Equal(tensor.MustShape(3, 1, 2)) {
		t.Fatalf("unexpected absorbed cache shapes: key=%v value=%v", result.Key.Shape, result.Value.Shape)
	}
	nodes, err := tensor.Topological(result.Output, result.Key, result.Value)
	if err != nil {
		t.Fatal(err)
	}
	grouped, moe, temperature := 0, false, false
	attentionScale := float32(0)
	for _, node := range nodes {
		switch node.Op {
		case tensor.OpGroupedMulMat:
			grouped++
		case tensor.OpMoE:
			moe = true
		case tensor.OpAttention:
			attentionScale = node.Attrs.(tensor.AttentionAttributes).Scale
		case tensor.OpMultiply:
			if len(node.Inputs) == 2 && (node.Inputs[0] == weights.AttentionTemperatureScale || node.Inputs[1] == weights.AttentionTemperatureScale) {
				temperature = true
			}
		}
	}
	if grouped != 2 || !moe || !temperature {
		t.Fatalf("absorbed graph grouped=%d moe=%v temperature=%v", grouped, moe, temperature)
	}
	wantScale := float32(math.Pow(1+0.1*math.Log(4), 2) / math.Sqrt(6))
	if math.Abs(float64(attentionScale-wantScale)) > 1e-6 {
		t.Fatalf("attention scale = %v, want %v", attentionScale, wantScale)
	}
}

func TestBuildMiniCPM3MLABlock(t *testing.T) {
	builder := tensor.NewBuilder()
	spec := Spec{CommonSpec: CommonSpec{Architecture: "minicpm3", EmbeddingLength: 8, FeedForwardLength: 12,

		ResidualScale: 0.7, RMSNormEpsilon: 1e-6}, AttentionSpec: AttentionSpec{HeadCount: 2, HeadCountKV: 2, KeyLength: 6, ValueLength: 4,
		QLoRARank: 3, KVLoRARank: 3, RopeDimensionCount: 2, RopeFrequencyBase: 10000,
		RopeAttentionFactor: 1}}
	input := builder.Input("input", dtype.F32, tensor.MustShape(8, 2))
	weights := LayerGraphWeights{
		AttentionNorm:    builder.Input("attn_norm", dtype.F32, tensor.MustShape(8)),
		AttentionQ:       builder.Input("q_a", dtype.F32, tensor.MustShape(8, 3)),
		AttentionQNorm:   builder.Input("q_a_norm", dtype.F32, tensor.MustShape(3)),
		AttentionQB:      builder.Input("q_b", dtype.F32, tensor.MustShape(3, 12)),
		AttentionKVAMQA:  builder.Input("kv_a", dtype.F32, tensor.MustShape(8, 5)),
		AttentionKVANorm: builder.Input("kv_a_norm", dtype.F32, tensor.MustShape(3)),
		AttentionKVB:     builder.Input("kv_b", dtype.F32, tensor.MustShape(3, 16)),
		AttentionOutput:  builder.Input("attn_out", dtype.F32, tensor.MustShape(8, 8)),
		RopeFactors:      builder.Input("rope", dtype.F32, tensor.MustShape(1)),
		FeedForwardNorm:  builder.Input("ffn_norm", dtype.F32, tensor.MustShape(8)),
		FeedForwardGate:  builder.Input("ffn_gate", dtype.F32, tensor.MustShape(8, 12)),
		FeedForwardUp:    builder.Input("ffn_up", dtype.F32, tensor.MustShape(8, 12)),
		FeedForwardDown:  builder.Input("ffn_down", dtype.F32, tensor.MustShape(12, 8)),
	}
	result, err := BuildMLABlockCached(builder, input, spec, weights, []uint32{0, 1}, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	if !result.Output.Shape.Equal(input.Shape) || !result.Key.Shape.Equal(tensor.MustShape(6, 2, 2)) ||
		!result.Value.Shape.Equal(tensor.MustShape(4, 2, 2)) {
		t.Fatalf("unexpected MiniCPM3 result: %+v", result)
	}
	nodes, _ := tensor.Topological(result.Output)
	var rms, neox, swiglu, scale int
	for _, node := range nodes {
		switch node.Op {
		case tensor.OpRMSNorm:
			rms++
		case tensor.OpRoPENeoX:
			neox++
		case tensor.OpSiLU:
			swiglu++
		case tensor.OpScale:
			scale++
		}
	}
	if rms != 4 || neox != 2 || swiglu != 1 || scale != 2 {
		t.Fatalf("MiniCPM3 ops: RMS=%d NeoX=%d SiLU=%d Scale=%d", rms, neox, swiglu, scale)
	}
}

func TestBuildGLMDSAFullAndSharedIndexer(t *testing.T) {
	spec := Spec{CommonSpec: CommonSpec{Architecture: "glm-dsa", BlockCount: 2, EmbeddingLength: 8,
		FeedForwardLength: 12,

		RMSNormEpsilon: 1e-6}, AttentionSpec: AttentionSpec{HeadCount: 2, HeadCountKV: 1, KeyLength: 6, ValueLength: 4,
		QLoRARank: 3, KVLoRARank: 3, RopeDimensionCount: 2, RopeFrequencyBase: 10000,
		IndexerHeadCount: 2,
		IndexerKeyLength: 8, IndexerTopK: 2, IndexerFullLayers: []bool{true, false}}, MoESpec: MoESpec{LeadingDenseBlocks: 2},
	}
	builder := tensor.NewBuilder()
	input := builder.Input("input", dtype.F32, tensor.MustShape(8, 2))
	weights := LayerGraphWeights{
		AttentionNorm:      builder.Input("attn_norm", dtype.F32, tensor.MustShape(8)),
		AttentionQ:         builder.Input("q_a", dtype.F32, tensor.MustShape(8, 3)),
		AttentionQNorm:     builder.Input("q_norm", dtype.F32, tensor.MustShape(3)),
		AttentionQB:        builder.Input("q_b", dtype.F32, tensor.MustShape(3, 12)),
		AttentionKVAMQA:    builder.Input("kv_a", dtype.F32, tensor.MustShape(8, 5)),
		AttentionKVANorm:   builder.Input("kv_norm", dtype.F32, tensor.MustShape(3)),
		AttentionKB:        builder.Input("k_b", dtype.F32, tensor.MustShape(4, 3, 2)),
		AttentionVB:        builder.Input("v_b", dtype.F32, tensor.MustShape(3, 4, 2)),
		AttentionOutput:    builder.Input("attn_out", dtype.F32, tensor.MustShape(8, 8)),
		FeedForwardNorm:    builder.Input("ffn_norm", dtype.F32, tensor.MustShape(8)),
		FeedForwardGate:    builder.Input("ffn_gate", dtype.F32, tensor.MustShape(8, 12)),
		FeedForwardUp:      builder.Input("ffn_up", dtype.F32, tensor.MustShape(8, 12)),
		FeedForwardDown:    builder.Input("ffn_down", dtype.F32, tensor.MustShape(12, 8)),
		IndexerKNorm:       builder.Input("indexer_norm", dtype.F32, tensor.MustShape(8)),
		IndexerKNormBias:   builder.Input("indexer_norm_bias", dtype.F32, tensor.MustShape(8)),
		IndexerProjection:  builder.Input("indexer_proj", dtype.F32, tensor.MustShape(8, 2)),
		IndexerAttentionK:  builder.Input("indexer_k", dtype.F32, tensor.MustShape(8, 8)),
		IndexerAttentionQB: builder.Input("indexer_q", dtype.F32, tensor.MustShape(3, 16)),
	}
	full, err := BuildGLMDSABlockCached(builder, input, spec, weights, []uint32{0, 1}, nil, nil, nil, nil, 0)
	if err != nil {
		t.Fatal(err)
	}
	indexerState, hasIndexerState := full.States[CacheStateIndexerKey]
	if full.Auxiliary == nil || !full.Auxiliary.Shape.Equal(tensor.MustShape(2, 2)) ||
		!hasIndexerState || indexerState.Mode != CacheStateToken ||
		!indexerState.Value.Shape.Equal(tensor.MustShape(8, 1, 2)) {
		t.Fatalf("unexpected full indexer result: %+v", full)
	}
	nodes, err := tensor.Topological(full.Output, full.Auxiliary, indexerState.Value)
	if err != nil {
		t.Fatal(err)
	}
	var fwht, score, topK, sparse bool
	for _, node := range nodes {
		fwht = fwht || node.Op == tensor.OpFWHT
		score = score || node.Op == tensor.OpIndexerScore
		topK = topK || node.Op == tensor.OpTopK
		sparse = sparse || node.Op == tensor.OpSparseAttention
	}
	if !fwht || !score || !topK || !sparse {
		t.Fatalf("GLM-DSA ops FWHT=%v score=%v top-k=%v sparse=%v", fwht, score, topK, sparse)
	}

	sharedBuilder := tensor.NewBuilder()
	sharedInput := sharedBuilder.Input("input", dtype.F32, tensor.MustShape(8, 2))
	sharedWeights := LayerGraphWeights{
		AttentionNorm:    sharedBuilder.Input("attn_norm", dtype.F32, tensor.MustShape(8)),
		AttentionQ:       sharedBuilder.Input("q_a", dtype.F32, tensor.MustShape(8, 3)),
		AttentionQNorm:   sharedBuilder.Input("q_norm", dtype.F32, tensor.MustShape(3)),
		AttentionQB:      sharedBuilder.Input("q_b", dtype.F32, tensor.MustShape(3, 12)),
		AttentionKVAMQA:  sharedBuilder.Input("kv_a", dtype.F32, tensor.MustShape(8, 5)),
		AttentionKVANorm: sharedBuilder.Input("kv_norm", dtype.F32, tensor.MustShape(3)),
		AttentionKB:      sharedBuilder.Input("k_b", dtype.F32, tensor.MustShape(4, 3, 2)),
		AttentionVB:      sharedBuilder.Input("v_b", dtype.F32, tensor.MustShape(3, 4, 2)),
		AttentionOutput:  sharedBuilder.Input("attn_out", dtype.F32, tensor.MustShape(8, 8)),
		FeedForwardNorm:  sharedBuilder.Input("ffn_norm", dtype.F32, tensor.MustShape(8)),
		FeedForwardGate:  sharedBuilder.Input("ffn_gate", dtype.F32, tensor.MustShape(8, 12)),
		FeedForwardUp:    sharedBuilder.Input("ffn_up", dtype.F32, tensor.MustShape(8, 12)),
		FeedForwardDown:  sharedBuilder.Input("ffn_down", dtype.F32, tensor.MustShape(12, 8)),
	}
	previous := sharedBuilder.Input("top_k", dtype.F32, tensor.MustShape(2, 2))
	shared, err := BuildGLMDSABlockCached(sharedBuilder, sharedInput, spec, sharedWeights, []uint32{0, 1}, nil, nil, nil, previous, 1)
	if err != nil {
		t.Fatal(err)
	}
	if shared.Auxiliary != previous || len(shared.States) != 0 {
		t.Fatalf("unexpected shared indexer result: %+v", shared)
	}
}

func TestBuildDeepSeek32FullIndexer(t *testing.T) {
	spec := Spec{CommonSpec: CommonSpec{Architecture: "deepseek32", BlockCount: 62, EmbeddingLength: 8,
		FeedForwardLength: 12,

		RMSNormEpsilon: 1e-5, LayerNormEpsilon: 1e-6}, AttentionSpec: AttentionSpec{HeadCount: 2, HeadCountKV: 1, KeyLength: 6, ValueLength: 4,
		QLoRARank: 3, KVLoRARank: 3, RopeDimensionCount: 2, RopeFrequencyBase: 10000,
		RopeScalingType: "yarn", RopeScalingFactor: 4, OriginalContextLength: 16,
		YaRNExtFactor: 1, YaRNAttentionFactor: 1, YaRNBetaFast: 32, YaRNBetaSlow: 1,

		IndexerHeadCount: 2, IndexerKeyLength: 8, IndexerTopK: 2,
		IndexerFullLayers: make([]bool, 62)}, MoESpec: MoESpec{LeadingDenseBlocks: 62},
	}
	for index := range spec.IndexerFullLayers {
		spec.IndexerFullLayers[index] = true
	}
	builder := tensor.NewBuilder()
	input := builder.Input("input", dtype.F32, tensor.MustShape(8, 2))
	weights := LayerGraphWeights{
		AttentionNorm:      builder.Input("attn_norm", dtype.F32, tensor.MustShape(8)),
		AttentionQ:         builder.Input("q_a", dtype.F32, tensor.MustShape(8, 3)),
		AttentionQNorm:     builder.Input("q_norm", dtype.F32, tensor.MustShape(3)),
		AttentionQB:        builder.Input("q_b", dtype.F32, tensor.MustShape(3, 12)),
		AttentionKVAMQA:    builder.Input("kv_a", dtype.F32, tensor.MustShape(8, 5)),
		AttentionKVANorm:   builder.Input("kv_norm", dtype.F32, tensor.MustShape(3)),
		AttentionKB:        builder.Input("k_b", dtype.F32, tensor.MustShape(4, 3, 2)),
		AttentionVB:        builder.Input("v_b", dtype.F32, tensor.MustShape(3, 4, 2)),
		AttentionOutput:    builder.Input("attn_out", dtype.F32, tensor.MustShape(8, 8)),
		FeedForwardNorm:    builder.Input("ffn_norm", dtype.F32, tensor.MustShape(8)),
		FeedForwardGate:    builder.Input("ffn_gate", dtype.F32, tensor.MustShape(8, 12)),
		FeedForwardUp:      builder.Input("ffn_up", dtype.F32, tensor.MustShape(8, 12)),
		FeedForwardDown:    builder.Input("ffn_down", dtype.F32, tensor.MustShape(12, 8)),
		IndexerKNorm:       builder.Input("indexer_norm", dtype.F32, tensor.MustShape(8)),
		IndexerKNormBias:   builder.Input("indexer_norm_bias", dtype.F32, tensor.MustShape(8)),
		IndexerProjection:  builder.Input("indexer_proj", dtype.F32, tensor.MustShape(8, 2)),
		IndexerAttentionK:  builder.Input("indexer_k", dtype.F32, tensor.MustShape(8, 8)),
		IndexerAttentionQB: builder.Input("indexer_q", dtype.F32, tensor.MustShape(3, 16)),
	}
	result, err := BuildDSABlockCached(
		builder, input, spec, weights, []uint32{0, 1}, nil, nil, nil, nil, 0,
	)
	if err != nil {
		t.Fatal(err)
	}
	nodes, err := tensor.Topological(result.Output, result.States[CacheStateIndexerKey].Value)
	if err != nil {
		t.Fatal(err)
	}
	var neoX, affineLayerNorm, sparse int
	for _, node := range nodes {
		switch node.Op {
		case tensor.OpRoPENeoX:
			neoX++
		case tensor.OpLayerNorm:
			attributes := node.Attrs.(tensor.LayerNormAttributes)
			if attributes.Epsilon == 1e-6 {
				affineLayerNorm++
			}
		case tensor.OpSparseAttention:
			sparse++
		}
	}
	if result.Auxiliary != nil || neoX != 2 || affineLayerNorm != 1 || sparse != 1 {
		t.Fatalf("DeepSeek 3.2 ops: NeoX=%d affine-LN=%d sparse=%d", neoX, affineLayerNorm, sparse)
	}
}

func TestBuildDeepSeek4CompressedHashBlock(t *testing.T) {
	builder := tensor.NewBuilder()
	spec := Spec{CommonSpec: CommonSpec{Architecture: "deepseek4", BlockCount: 1, EmbeddingLength: 8, VocabularySize: 32,

		RMSNormEpsilon: 1e-5,

		HyperConnectionCount: 4, HyperSinkhornIters: 2, HyperConnectionEps: 1e-6,

		HashLayerCount: 1}, AttentionSpec: AttentionSpec{HeadCount: 2, HeadCountKV: 1, KeyLength: 4, ValueLength: 4, RopeDimensionCount: 2,
		RopeFrequencyBase: 10000, CompressRopeBase: 10000,
		QLoRARank: 3, SlidingWindow: 8, CompressRatios: []uint32{4},
		IndexerHeadCount: 2, IndexerKeyLength: 8, IndexerTopK: 2,
		AttentionOutputGroups: 1, AttentionOutputRank: 3}, MoESpec: MoESpec{ExpertCount: 4, ExpertUsedCount: 2, ExpertFeedForward: 8, SharedExpertFF: 8,
		ExpertWeightsNorm: true, ExpertWeightsScale: 1.25,
		LayerSwiGLUClamp: []float32{7}, LayerSharedSwiGLUClamp: []float32{6}},
	}
	input := builder.Input("input", dtype.F32, tensor.MustShape(8, 2))
	weights := LayerGraphWeights{
		AttentionNorm:           builder.Input("attn_norm", dtype.F32, tensor.MustShape(8)),
		AttentionQ:              builder.Input("q_a", dtype.F32, tensor.MustShape(8, 3)),
		AttentionQNorm:          builder.Input("q_norm", dtype.F32, tensor.MustShape(3)),
		AttentionQB:             builder.Input("q_b", dtype.F32, tensor.MustShape(3, 8)),
		AttentionK:              builder.Input("kv", dtype.F32, tensor.MustShape(8, 4)),
		AttentionKNorm:          builder.Input("kv_norm", dtype.F32, tensor.MustShape(4)),
		AttentionSinks:          builder.Input("sinks", dtype.F32, tensor.MustShape(2)),
		AttentionOutputA:        builder.Input("out_a", dtype.F32, tensor.MustShape(8, 3)),
		AttentionOutput:         builder.Input("out_b", dtype.F32, tensor.MustShape(3, 8)),
		AttentionCompressorKV:   builder.Input("comp_kv", dtype.F32, tensor.MustShape(8, 8)),
		AttentionCompressorGate: builder.Input("comp_gate", dtype.F32, tensor.MustShape(8, 8)),
		AttentionCompressorAPE:  builder.Input("comp_ape", dtype.F32, tensor.MustShape(8, 4)),
		AttentionCompressorNorm: builder.Input("comp_norm", dtype.F32, tensor.MustShape(4)),
		IndexerProjection:       builder.Input("indexer_proj", dtype.F32, tensor.MustShape(8, 2)),
		IndexerAttentionQB:      builder.Input("indexer_q", dtype.F32, tensor.MustShape(3, 16)),
		IndexerCompressorKV:     builder.Input("indexer_kv", dtype.F32, tensor.MustShape(8, 16)),
		IndexerCompressorGate:   builder.Input("indexer_gate", dtype.F32, tensor.MustShape(8, 16)),
		IndexerCompressorAPE:    builder.Input("indexer_ape", dtype.F32, tensor.MustShape(16, 4)),
		IndexerCompressorNorm:   builder.Input("indexer_norm", dtype.F32, tensor.MustShape(8)),
		HyperAttentionFN:        builder.Input("hc_attn_fn", dtype.F32, tensor.MustShape(32, 24)),
		HyperAttentionBase:      builder.Input("hc_attn_base", dtype.F32, tensor.MustShape(24)),
		HyperAttentionScale:     builder.Input("hc_attn_scale", dtype.F32, tensor.MustShape(3)),
		FeedForwardNorm:         builder.Input("ffn_norm", dtype.F32, tensor.MustShape(8)),
		FeedForwardRouter:       builder.Input("router", dtype.F32, tensor.MustShape(8, 4)),
		FeedForwardHashExperts:  builder.Input("hash", dtype.F32, tensor.MustShape(2, 32)),
		FeedForwardGateExperts:  builder.Input("gate_exps", dtype.F32, tensor.MustShape(8, 8, 4)),
		FeedForwardUpExperts:    builder.Input("up_exps", dtype.F32, tensor.MustShape(8, 8, 4)),
		FeedForwardDownExperts:  builder.Input("down_exps", dtype.F32, tensor.MustShape(8, 8, 4)),
		FeedForwardSharedGate:   builder.Input("shared_gate", dtype.F32, tensor.MustShape(8, 8)),
		FeedForwardSharedUp:     builder.Input("shared_up", dtype.F32, tensor.MustShape(8, 8)),
		FeedForwardSharedDown:   builder.Input("shared_down", dtype.F32, tensor.MustShape(8, 8)),
		HyperFeedForwardFN:      builder.Input("hc_ffn_fn", dtype.F32, tensor.MustShape(32, 24)),
		HyperFeedForwardBase:    builder.Input("hc_ffn_base", dtype.F32, tensor.MustShape(24)),
		HyperFeedForwardScale:   builder.Input("hc_ffn_scale", dtype.F32, tensor.MustShape(3)),
		HyperHeadFN:             builder.Input("hc_head_fn", dtype.F32, tensor.MustShape(32, 4)),
		HyperHeadBase:           builder.Input("hc_head_base", dtype.F32, tensor.MustShape(4)),
		HyperHeadScale:          builder.Input("hc_head_scale", dtype.F32, tensor.MustShape(1)),
	}
	positionState := builder.Input("positions", dtype.F32, tensor.MustShape(1, 1, 2))
	result, err := BuildDeepSeek4BlockCached(builder, input, spec, weights, []uint32{0, 1}, []uint32{3, 4}, nil, nil, positionState, 0)
	if err != nil {
		t.Fatal(err)
	}
	compressorState := result.States[CacheStateCompressorKV]
	indexerState := result.States[CacheStateIndexerCompressorKV]
	outputs := []*tensor.Tensor{result.Output, result.Key, compressorState.Value, indexerState.Value}
	nodes, err := tensor.Topological(outputs...)
	if err != nil {
		t.Fatal(err)
	}
	var attention, hcInit, hcPre, hcPost, hcHead int
	var moe tensor.MoEAttributes
	for _, node := range nodes {
		switch node.Op {
		case tensor.OpDeepSeek4Attention:
			attention++
		case tensor.OpDeepSeek4HCInit:
			hcInit++
		case tensor.OpDeepSeek4HCPre:
			hcPre++
		case tensor.OpDeepSeek4HCPost:
			hcPost++
		case tensor.OpDeepSeek4HCHead:
			hcHead++
		case tensor.OpMoE:
			moe = node.Attrs.(tensor.MoEAttributes)
		}
	}
	if attention != 1 || hcInit != 1 || hcPre != 2 || hcPost != 2 || hcHead != 1 ||
		moe.Routing != tensor.MoERoutingSqrtSoftplus || !moe.HasSelectedExperts || moe.SwiGLUClamp != 7 ||
		compressorState.Mode != CacheStateToken || indexerState.Mode != CacheStateToken ||
		!compressorState.Value.Shape.Equal(tensor.MustShape(8, 1, 2)) ||
		!indexerState.Value.Shape.Equal(tensor.MustShape(16, 1, 2)) {
		t.Fatalf("unexpected DeepSeek 4 graph: attention=%d HC=%d/%d/%d/%d MoE=%+v states=%v",
			attention, hcInit, hcPre, hcPost, hcHead, moe, result.States)
	}
	feeds := make(map[*tensor.Tensor]reference.Value)
	for _, node := range builder.Nodes() {
		if node.Op != tensor.OpInput {
			continue
		}
		elements, err := node.Shape.Elements()
		if err != nil {
			t.Fatal(err)
		}
		data := make([]float32, int(elements))
		for index := range data {
			data[index] = float32((index+int(node.ID))%7-3) * 0.05
		}
		feeds[node] = reference.Value{Shape: node.Shape, Data: data}
	}
	feeds[positionState] = reference.Value{Shape: positionState.Shape, Data: []float32{0, 1}}
	hash := feeds[weights.FeedForwardHashExperts]
	for row := 0; row < 32; row++ {
		hash.Data[2*row], hash.Data[2*row+1] = 0, 1
	}
	feeds[weights.FeedForwardHashExperts] = hash
	executed, err := reference.Execute(outputs, feeds)
	if err != nil {
		t.Fatal(err)
	}
	for _, output := range outputs {
		for index, value := range executed[output].Data {
			if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) {
				t.Fatalf("DeepSeek 4 output %d[%d] is non-finite", output.ID, index)
			}
		}
	}
}

func TestBuildChameleonSandwichBlock(t *testing.T) {
	builder := tensor.NewBuilder()
	spec := Spec{CommonSpec: CommonSpec{Architecture: "chameleon", EmbeddingLength: 8, FeedForwardLength: 12,

		RMSNormEpsilon: 1e-5,
		SandwichNorm:   true}, AttentionSpec: AttentionSpec{HeadCount: 2, HeadCountKV: 1, KeyLength: 4, ValueLength: 4,
		RopeFrequencyBase: 10000, QKNormEpsilon: 1e-5},
	}
	input := builder.Input("input", dtype.F32, tensor.MustShape(8, 2))
	weights := denseBlockInputs(builder, spec)
	weights.AttentionNormBias = nil
	weights.FeedForwardNormBias = nil
	weights.AttentionQNorm = builder.Input("attn_q_norm", dtype.F32, tensor.MustShape(4, 2))
	weights.AttentionKNorm = builder.Input("attn_k_norm", dtype.F32, tensor.MustShape(4, 1))
	weights.AttentionQNormBias = builder.Input("attn_q_norm_bias", dtype.F32, tensor.MustShape(4, 2))
	weights.AttentionKNormBias = builder.Input("attn_k_norm_bias", dtype.F32, tensor.MustShape(4, 1))
	result, err := BuildDenseBlockCachedForLayer(
		builder, input, spec, weights, []uint32{0, 1}, nil, nil, 0,
	)
	if err != nil {
		t.Fatal(err)
	}
	nodes, err := tensor.Topological(result.Output)
	if err != nil {
		t.Fatal(err)
	}
	var layerNorms, rmsNorms, normalRoPE int
	for _, node := range nodes {
		switch node.Op {
		case tensor.OpLayerNorm:
			layerNorms++
		case tensor.OpRMSNorm:
			rmsNorms++
		case tensor.OpRoPENormal:
			normalRoPE++
		}
	}
	if layerNorms != 2 || rmsNorms != 2 || normalRoPE != 2 {
		t.Fatalf("Chameleon norm/RoPE counts = %d/%d/%d, want 2/2/2", layerNorms, rmsNorms, normalRoPE)
	}
}

func TestBuildDenseQwen2BlockUsesNeoXAndProjectionBiases(t *testing.T) {
	builder := tensor.NewBuilder()
	spec := Spec{CommonSpec: CommonSpec{Architecture: "qwen2",
		EmbeddingLength:   8,
		FeedForwardLength: 12,

		RMSNormEpsilon: 1e-6}, AttentionSpec: AttentionSpec{HeadCount: 2,
		HeadCountKV:       1,
		KeyLength:         4,
		ValueLength:       4,
		RopeFrequencyBase: 1_000_000},
	}
	input := builder.Input("input", dtype.F32, tensor.MustShape(8, 2))
	weights := denseBlockInputs(builder, spec)
	weights.AttentionQNorm = nil
	weights.AttentionKNorm = nil
	weights.AttentionQBias = builder.Input("attn_q.bias", dtype.F32, tensor.MustShape(8))
	weights.AttentionKBias = builder.Input("attn_k.bias", dtype.F32, tensor.MustShape(4))
	weights.AttentionVBias = builder.Input("attn_v.bias", dtype.F32, tensor.MustShape(4))
	output, err := BuildDenseBlock(builder, input, spec, weights, []uint32{0, 1})
	if err != nil {
		t.Fatal(err)
	}
	nodes, err := tensor.Topological(output)
	if err != nil {
		t.Fatal(err)
	}
	var neoxCount int
	found := map[string]bool{}
	for _, node := range nodes {
		if node.Op == tensor.OpRoPENeoX {
			neoxCount++
		}
		found[node.Name] = true
	}
	if neoxCount != 2 {
		t.Fatalf("Qwen 2 NeoX RoPE count = %d, want 2", neoxCount)
	}
	for _, name := range []string{"attn_q.bias", "attn_k.bias", "attn_v.bias"} {
		if !found[name] {
			t.Fatalf("Qwen 2 projection bias %q is disconnected", name)
		}
	}
}

func TestBuildDenseInternLM2EXAONEAndXVERSEUseExpectedRoPE(t *testing.T) {
	for _, test := range []struct {
		architecture string
		want         tensor.Op
	}{
		{"internlm2", tensor.OpRoPENormal},
		{"exaone", tensor.OpRoPENeoX},
		{"mistral3", tensor.OpRoPENormal},
		{"xverse", tensor.OpRoPENormal},
	} {
		t.Run(test.architecture, func(t *testing.T) {
			builder := tensor.NewBuilder()
			spec := Spec{CommonSpec: CommonSpec{Architecture: test.architecture,
				EmbeddingLength:   8,
				FeedForwardLength: 12,

				RMSNormEpsilon: 1e-6}, AttentionSpec: AttentionSpec{HeadCount: 2,
				HeadCountKV:       1,
				KeyLength:         4,
				ValueLength:       4,
				RopeFrequencyBase: 10000},
			}
			input := builder.Input("input", dtype.F32, tensor.MustShape(8, 2))
			weights := denseBlockInputs(builder, spec)
			weights.AttentionQNorm = nil
			weights.AttentionKNorm = nil
			if test.architecture == "exaone" {
				weights.RopeFactors = builder.Input(
					"rope_factors",
					dtype.F32,
					tensor.MustShape(2),
				)
			}
			output, err := BuildDenseBlock(builder, input, spec, weights, []uint32{0, 1})
			if err != nil {
				t.Fatal(err)
			}
			nodes, err := tensor.Topological(output)
			if err != nil {
				t.Fatal(err)
			}
			var ropeCount int
			for _, node := range nodes {
				if node.Op == test.want {
					ropeCount++
					if weights.RopeFactors != nil &&
						(len(node.Inputs) != 2 || node.Inputs[1] != weights.RopeFactors) {
						t.Fatal("EXAONE RoPE factors are disconnected")
					}
				} else if node.Op == tensor.OpRoPENormal || node.Op == tensor.OpRoPENeoX {
					t.Fatalf("unexpected RoPE operation %s", node.Op)
				}
			}
			if ropeCount != 2 {
				t.Fatalf("%s RoPE count = %d, want 2", test.architecture, ropeCount)
			}
		})
	}
}
