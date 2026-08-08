package model

import (
	"overgo/internal/tensor"

	"overgo/internal/tensor/dtype"

	"math"

	"slices"

	"strings"

	"testing"
)

func TestBuildGroveMoEGroupedChunkExperts(t *testing.T) {
	builder := tensor.NewBuilder()
	spec := Spec{CommonSpec: CommonSpec{Architecture: "grovemoe", BlockCount: 1, EmbeddingLength: 8,
		FeedForwardLength: 24,

		RMSNormEpsilon: 1e-6}, AttentionSpec: AttentionSpec{HeadCount: 2, HeadCountKV: 1, KeyLength: 4, ValueLength: 4,
		RopeFrequencyBase: 10000}, MoESpec: MoESpec{ExpertCount: 4, ExpertUsedCount: 2,
		ExpertFeedForward: 6, ExpertChunkFeedForward: 3, ExpertWeightsScale: 1.25,
		ExpertGroupScale: 0.5, ExpertsPerGroup: 2},
	}
	input := builder.Input("input", dtype.F32, tensor.MustShape(8, 2))
	weights := LayerGraphWeights{
		AttentionNorm:               builder.Input("attn_norm", dtype.F32, tensor.MustShape(8)),
		AttentionQ:                  builder.Input("q", dtype.F32, tensor.MustShape(8, 8)),
		AttentionK:                  builder.Input("k", dtype.F32, tensor.MustShape(8, 4)),
		AttentionV:                  builder.Input("v", dtype.F32, tensor.MustShape(8, 4)),
		AttentionOutput:             builder.Input("attn_out", dtype.F32, tensor.MustShape(8, 8)),
		AttentionQNorm:              builder.Input("q_norm", dtype.F32, tensor.MustShape(4)),
		AttentionKNorm:              builder.Input("k_norm", dtype.F32, tensor.MustShape(4)),
		FeedForwardNorm:             builder.Input("ffn_norm", dtype.F32, tensor.MustShape(8)),
		FeedForwardRouter:           builder.Input("router", dtype.F32, tensor.MustShape(8, 4)),
		FeedForwardGateExperts:      builder.Input("gate_exps", dtype.F32, tensor.MustShape(8, 6, 4)),
		FeedForwardUpExperts:        builder.Input("up_exps", dtype.F32, tensor.MustShape(8, 6, 4)),
		FeedForwardDownExperts:      builder.Input("down_exps", dtype.F32, tensor.MustShape(6, 8, 4)),
		FeedForwardGateChunkExperts: builder.Input("gate_chexps", dtype.F32, tensor.MustShape(8, 3, 2)),
		FeedForwardUpChunkExperts:   builder.Input("up_chexps", dtype.F32, tensor.MustShape(8, 3, 2)),
		FeedForwardDownChunkExperts: builder.Input("down_chexps", dtype.F32, tensor.MustShape(3, 8, 2)),
	}
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
	var moes []*tensor.Tensor
	var rope int
	for _, node := range nodes {
		if node.Op == tensor.OpMoE {
			moes = append(moes, node)
		}
		if node.Op == tensor.OpRoPENeoX {
			rope++
		}
	}
	if len(moes) != 2 || rope != 2 {
		t.Fatalf("unexpected GroveMoE graph: moes=%d rope=%d", len(moes), rope)
	}
	grouped := moes[1].Attrs.(tensor.MoEAttributes)
	if grouped.ExpertIndexDivisor != 2 || grouped.TopK != 2 ||
		moes[1].Inputs[0] != moes[0] || moes[1].Inputs[1] == moes[1].Inputs[0] ||
		moes[1].Inputs[2] != weights.FeedForwardRouter {
		t.Fatalf("unexpected grouped GroveMoE pass: attrs=%+v inputs=%v", grouped, moes[1].Inputs)
	}
	spec.ExpertsPerGroup = 3
	if _, err := BuildDenseBlockCachedForLayer(
		builder, input, spec, weights, []uint32{0, 1}, nil, nil, 0,
	); err == nil {
		t.Fatal("GroveMoE accepted invalid expert grouping")
	}
}

func TestBuildGLM4MoEBlock(t *testing.T) {
	builder := tensor.NewBuilder()
	spec := standardDecoderFixture.decoderSpec("glm4moe")
	spec.BlockCount = 2
	spec.RopeDimensionCount = fixtureHeadWidth
	spec.RopeSections = [4]int32{1, 1, 0, 0}
	spec.MoESpec = MoESpec{LeadingDenseBlocks: 1,
		ExpertCount:     4,
		ExpertUsedCount: 2, ExpertFeedForward: 6, SharedExpertFF: 12,
		ExpertWeightsScale: 1.25, ExpertWeightsNorm: true, ExpertGatingFunc: expertGatingSigmoid}
	input := builder.Input("input", dtype.F32, tensor.MustShape(8, 2))
	weights := LayerGraphWeights{
		AttentionNorm:          builder.Input("attn_norm", dtype.F32, tensor.MustShape(8)),
		AttentionQ:             builder.Input("q", dtype.F32, tensor.MustShape(8, 8)),
		AttentionK:             builder.Input("k", dtype.F32, tensor.MustShape(8, 4)),
		AttentionV:             builder.Input("v", dtype.F32, tensor.MustShape(8, 4)),
		AttentionOutput:        builder.Input("attn_out", dtype.F32, tensor.MustShape(8, 8)),
		AttentionQNorm:         builder.Input("q_norm", dtype.F32, tensor.MustShape(4)),
		AttentionKNorm:         builder.Input("k_norm", dtype.F32, tensor.MustShape(4)),
		FeedForwardNorm:        builder.Input("attn_post_norm", dtype.F32, tensor.MustShape(8)),
		FeedForwardRouter:      builder.Input("router", dtype.F32, tensor.MustShape(8, 4)),
		FeedForwardGateExperts: builder.Input("gate_exps", dtype.F32, tensor.MustShape(8, 6, 4)),
		FeedForwardUpExperts:   builder.Input("up_exps", dtype.F32, tensor.MustShape(8, 6, 4)),
		FeedForwardDownExperts: builder.Input("down_exps", dtype.F32, tensor.MustShape(6, 8, 4)),
		FeedForwardExpertBias:  builder.Input("expert_bias", dtype.F32, tensor.MustShape(4)),
		FeedForwardSharedGate:  builder.Input("shared_gate", dtype.F32, tensor.MustShape(8, 12)),
		FeedForwardSharedUp:    builder.Input("shared_up", dtype.F32, tensor.MustShape(8, 12)),
		FeedForwardSharedDown:  builder.Input("shared_down", dtype.F32, tensor.MustShape(12, 8)),
	}
	result, err := BuildDenseBlockCachedForLayer(
		builder, input, spec, weights, []uint32{0, 1}, nil, nil, 1,
	)
	if err != nil {
		t.Fatal(err)
	}
	nodes, err := tensor.Topological(result.Output)
	if err != nil {
		t.Fatal(err)
	}
	var moe *tensor.Tensor
	var mrope, rmsNorm, sharedSiLU int
	for _, node := range nodes {
		switch node.Op {
		case tensor.OpMoE:
			moe = node
		case tensor.OpRoPEMulti:
			mrope++
		case tensor.OpRMSNorm:
			rmsNorm++
		case tensor.OpSiLU:
			sharedSiLU++
		}
	}
	if moe == nil || moe.Attrs.(tensor.MoEAttributes).Routing != tensor.MoERoutingSigmoid ||
		!moe.Attrs.(tensor.MoEAttributes).NormalizeTopKProb || len(moe.Inputs) != 7 ||
		mrope != 2 || rmsNorm != 4 || sharedSiLU < 1 {
		t.Fatalf("unexpected GLM4-MoE graph: moe=%v mrope=%d norms=%d silu=%d", moe != nil, mrope, rmsNorm, sharedSiLU)
	}
}

func TestBuildMiMo2UsesSinksValueScaleAndSigmoidMoE(t *testing.T) {
	builder := tensor.NewBuilder()
	spec := Spec{CommonSpec: CommonSpec{Architecture: "mimo2", BlockCount: 2, EmbeddingLength: 8,
		FeedForwardLength: 12,

		RMSNormEpsilon: 1e-5}, AttentionSpec: AttentionSpec{HeadCount: 2, HeadCountKV: 1,
		LayerKVHeadCounts: []uint32{1, 1}, KeyLength: 4, ValueLength: 3,
		RopeDimensionCount: 4, RopeFrequencyBase: 10000, RopeFrequencySWA: 20000,
		SlidingWindow: 128, SlidingLayers: []bool{false, true},
		AttentionValueScale: 0.5}, MoESpec: MoESpec{ExpertCount: 4, ExpertUsedCount: 2,
		ExpertFeedForward: 6, ExpertWeightsScale: 1.25, ExpertWeightsNorm: true,
		ExpertGatingFunc: expertGatingSigmoid},
	}
	input := builder.Input("input", dtype.F32, tensor.MustShape(8, 2))
	weights := LayerGraphWeights{
		AttentionNorm:          builder.Input("attn_norm", dtype.F32, tensor.MustShape(8)),
		AttentionQKV:           builder.Input("qkv", dtype.F32, tensor.MustShape(8, 15)),
		AttentionOutput:        builder.Input("attn_out", dtype.F32, tensor.MustShape(6, 8)),
		AttentionSinks:         builder.Input("sinks", dtype.F32, tensor.MustShape(2)),
		FeedForwardNorm:        builder.Input("ffn_norm", dtype.F32, tensor.MustShape(8)),
		FeedForwardRouter:      builder.Input("router", dtype.F32, tensor.MustShape(8, 4)),
		FeedForwardGateExperts: builder.Input("gate_exps", dtype.F32, tensor.MustShape(8, 6, 4)),
		FeedForwardUpExperts:   builder.Input("up_exps", dtype.F32, tensor.MustShape(8, 6, 4)),
		FeedForwardDownExperts: builder.Input("down_exps", dtype.F32, tensor.MustShape(6, 8, 4)),
		FeedForwardExpertBias:  builder.Input("expert_bias", dtype.F32, tensor.MustShape(4)),
	}
	result, err := BuildDenseBlockCachedForLayer(
		builder, input, spec, weights, []uint32{0, 1}, nil, nil, 1,
	)
	if err != nil {
		t.Fatal(err)
	}
	nodes, err := tensor.Topological(result.Output)
	if err != nil {
		t.Fatal(err)
	}
	var attention, moe *tensor.Tensor
	var valueScale, swaRope int
	for _, node := range nodes {
		switch node.Op {
		case tensor.OpAttention:
			attention = node
		case tensor.OpMoE:
			moe = node
		case tensor.OpScale:
			if node.Attrs.(tensor.ScaleAttributes).Value == 0.5 {
				valueScale++
			}
		case tensor.OpRoPENeoX:
			if node.Attrs.(tensor.RoPEAttributes).FrequencyBase == 20000 {
				swaRope++
			}
		}
	}
	if attention == nil || !attention.Attrs.(tensor.AttentionAttributes).HasSinks ||
		attention.Attrs.(tensor.AttentionAttributes).Window != 128 || attention.Inputs[3] != weights.AttentionSinks ||
		moe == nil || moe.Attrs.(tensor.MoEAttributes).Routing != tensor.MoERoutingSigmoid ||
		!moe.Attrs.(tensor.MoEAttributes).NormalizeTopKProb || len(moe.Inputs) != 7 ||
		valueScale != 1 || swaRope != 2 {
		t.Fatalf("unexpected MiMo2 graph: attention=%v moe=%v value_scale=%d swa_rope=%d", attention != nil, moe != nil, valueScale, swaRope)
	}
}

func TestBuildStep35UsesPartialRoPEHeadGateAndLimitedExperts(t *testing.T) {
	builder := tensor.NewBuilder()
	spec := Spec{CommonSpec: CommonSpec{Architecture: "step35", BlockCount: 2, EmbeddingLength: 8,
		FeedForwardLength: 12,

		RMSNormEpsilon: 1e-5}, AttentionSpec: AttentionSpec{HeadCount: 2, HeadCountKV: 1, LayerHeadCounts: []uint32{2, 4},
		LayerKVHeadCounts: []uint32{1, 2}, KeyLength: 4, ValueLength: 4,
		RopeDimensionCount: 4, RopeFrequencyBase: 10000, RopeFrequencySWA: 20000,
		SlidingWindow: 128, SlidingLayers: []bool{false, true}}, MoESpec: MoESpec{ExpertCount: 4, ExpertUsedCount: 2,
		ExpertFeedForward: 6, SharedExpertFF: 8, ExpertWeightsScale: 1.25,
		ExpertWeightsNorm: true, ExpertGatingFunc: expertGatingSigmoid,

		LayerSwiGLUClamp: []float32{2, 0}, LayerSharedSwiGLUClamp: []float32{3, 0}},
	}
	input := builder.Input("input", dtype.F32, tensor.MustShape(8, 2))
	weights := LayerGraphWeights{
		AttentionNorm:          builder.Input("attn_norm", dtype.F32, tensor.MustShape(8)),
		AttentionQ:             builder.Input("q", dtype.F32, tensor.MustShape(8, 8)),
		AttentionK:             builder.Input("k", dtype.F32, tensor.MustShape(8, 4)),
		AttentionV:             builder.Input("v", dtype.F32, tensor.MustShape(8, 4)),
		AttentionOutput:        builder.Input("attn_out", dtype.F32, tensor.MustShape(8, 8)),
		AttentionQNorm:         builder.Input("q_norm", dtype.F32, tensor.MustShape(4)),
		AttentionKNorm:         builder.Input("k_norm", dtype.F32, tensor.MustShape(4)),
		AttentionOutputGate:    builder.Input("attn_gate", dtype.F32, tensor.MustShape(8, 2)),
		RopeFactors:            builder.Input("rope_factors", dtype.F32, tensor.MustShape(2)),
		FeedForwardNorm:        builder.Input("ffn_norm", dtype.F32, tensor.MustShape(8)),
		FeedForwardRouter:      builder.Input("router", dtype.F32, tensor.MustShape(8, 4)),
		FeedForwardGateExperts: builder.Input("gate_exps", dtype.F32, tensor.MustShape(8, 6, 4)),
		FeedForwardUpExperts:   builder.Input("up_exps", dtype.F32, tensor.MustShape(8, 6, 4)),
		FeedForwardDownExperts: builder.Input("down_exps", dtype.F32, tensor.MustShape(6, 8, 4)),
		FeedForwardExpertBias:  builder.Input("expert_bias", dtype.F32, tensor.MustShape(4)),
		FeedForwardSharedGate:  builder.Input("shared_gate", dtype.F32, tensor.MustShape(8, 8)),
		FeedForwardSharedUp:    builder.Input("shared_up", dtype.F32, tensor.MustShape(8, 8)),
		FeedForwardSharedDown:  builder.Input("shared_down", dtype.F32, tensor.MustShape(8, 8)),
	}
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
	var moe *tensor.Tensor
	var rope, factorSlices, sigmoid, clamps int
	for _, node := range nodes {
		switch node.Op {
		case tensor.OpMoE:
			moe = node
		case tensor.OpRoPENeoX:
			if node.Attrs.(tensor.RoPEAttributes).RotaryDimensions == 2 && len(node.Inputs) == 2 {
				rope++
			}
		case tensor.OpFlatSlice:
			factorSlices++
		case tensor.OpSigmoid:
			sigmoid++
		case tensor.OpClamp:
			clamps++
		}
	}
	if moe == nil || moe.Attrs.(tensor.MoEAttributes).SwiGLUClamp != 2 ||
		moe.Attrs.(tensor.MoEAttributes).Routing != tensor.MoERoutingSigmoid ||
		rope != 2 || factorSlices != 1 || sigmoid != 1 || clamps != 2 {
		t.Fatalf("unexpected Step3.5 graph: moe=%v rope=%d slices=%d sigmoid=%d clamps=%d", moe != nil, rope, factorSlices, sigmoid, clamps)
	}
}

func TestBuildDOTS1MoEBlock(t *testing.T) {
	builder := tensor.NewBuilder()
	spec := Spec{CommonSpec: CommonSpec{Architecture: "dots1", BlockCount: 2,
		EmbeddingLength: 8, FeedForwardLength: 12,

		RMSNormEpsilon: 1e-6}, AttentionSpec: AttentionSpec{HeadCount: 2, HeadCountKV: 2, KeyLength: 4, ValueLength: 4,
		RopeFrequencyBase: 10000}, MoESpec: MoESpec{LeadingDenseBlocks: 1,
		ExpertCount: 4, ExpertUsedCount: 2,
		ExpertFeedForward: 6, SharedExpertCount: 2, SharedExpertFF: 12,
		ExpertWeightsScale: 1.25, ExpertWeightsNorm: true, ExpertGatingFunc: expertGatingSigmoid},
	}
	input := builder.Input("input", dtype.F32, tensor.MustShape(8, 2))
	weights := LayerGraphWeights{
		AttentionNorm:          builder.Input("attn_norm", dtype.F32, tensor.MustShape(8)),
		AttentionQKV:           builder.Input("qkv", dtype.F32, tensor.MustShape(8, 24)),
		AttentionOutput:        builder.Input("attn_out", dtype.F32, tensor.MustShape(8, 8)),
		AttentionQNorm:         builder.Input("q_norm", dtype.F32, tensor.MustShape(4)),
		AttentionKNorm:         builder.Input("k_norm", dtype.F32, tensor.MustShape(4)),
		FeedForwardNorm:        builder.Input("ffn_norm", dtype.F32, tensor.MustShape(8)),
		FeedForwardRouter:      builder.Input("router", dtype.F32, tensor.MustShape(8, 4)),
		FeedForwardGateExperts: builder.Input("gate_exps", dtype.F32, tensor.MustShape(8, 6, 4)),
		FeedForwardUpExperts:   builder.Input("up_exps", dtype.F32, tensor.MustShape(8, 6, 4)),
		FeedForwardDownExperts: builder.Input("down_exps", dtype.F32, tensor.MustShape(6, 8, 4)),
		FeedForwardExpertBias:  builder.Input("expert_bias", dtype.F32, tensor.MustShape(4)),
		FeedForwardSharedGate:  builder.Input("shared_gate", dtype.F32, tensor.MustShape(8, 12)),
		FeedForwardSharedUp:    builder.Input("shared_up", dtype.F32, tensor.MustShape(8, 12)),
		FeedForwardSharedDown:  builder.Input("shared_down", dtype.F32, tensor.MustShape(12, 8)),
	}
	result, err := BuildDenseBlockCachedForLayer(
		builder, input, spec, weights, []uint32{0, 1}, nil, nil, 1,
	)
	if err != nil {
		t.Fatal(err)
	}
	nodes, err := tensor.Topological(result.Output)
	if err != nil {
		t.Fatal(err)
	}
	var moe *tensor.Tensor
	var rope, silu int
	for _, node := range nodes {
		switch node.Op {
		case tensor.OpMoE:
			moe = node
		case tensor.OpRoPENeoX:
			rope++
		case tensor.OpSiLU:
			silu++
		}
	}
	if moe == nil {
		t.Fatal("DOTS1 graph is missing MoE")
	}
	attributes := moe.Attrs.(tensor.MoEAttributes)
	if attributes.Routing != tensor.MoERoutingSigmoid || !attributes.NormalizeTopKProb ||
		attributes.Scale != 1.25 || len(moe.Inputs) != 7 || rope != 2 || silu < 1 {
		t.Fatalf("unexpected DOTS1 graph: attrs=%+v inputs=%d rope=%d silu=%d", attributes, len(moe.Inputs), rope, silu)
	}
}

func TestBuildMiniMaxM2Block(t *testing.T) {
	builder := tensor.NewBuilder()
	spec := Spec{CommonSpec: CommonSpec{Architecture: "minimax-m2", BlockCount: 1, EmbeddingLength: 8,
		FeedForwardLength: 6,

		RMSNormEpsilon: 1e-6}, AttentionSpec: AttentionSpec{HeadCount: 2, HeadCountKV: 1, KeyLength: 4, ValueLength: 4,
		RopeDimensionCount: 2, RopeFrequencyBase: 10000}, MoESpec: MoESpec{ExpertCount: 4, ExpertUsedCount: 2,
		ExpertFeedForward: 6, ExpertWeightsScale: 1.25, ExpertGatingFunc: expertGatingSigmoid},
	}
	input := builder.Input("input", dtype.F32, tensor.MustShape(8, 2))
	weights := LayerGraphWeights{
		AttentionNorm:          builder.Input("attn_norm", dtype.F32, tensor.MustShape(8)),
		AttentionQKV:           builder.Input("qkv", dtype.F32, tensor.MustShape(8, 16)),
		AttentionOutput:        builder.Input("attn_out", dtype.F32, tensor.MustShape(8, 8)),
		AttentionQNorm:         builder.Input("q_norm", dtype.F32, tensor.MustShape(8)),
		AttentionKNorm:         builder.Input("k_norm", dtype.F32, tensor.MustShape(4)),
		FeedForwardNorm:        builder.Input("ffn_norm", dtype.F32, tensor.MustShape(8)),
		FeedForwardRouter:      builder.Input("router", dtype.F32, tensor.MustShape(8, 4)),
		FeedForwardGateExperts: builder.Input("gate_exps", dtype.F32, tensor.MustShape(8, 6, 4)),
		FeedForwardUpExperts:   builder.Input("up_exps", dtype.F32, tensor.MustShape(8, 6, 4)),
		FeedForwardDownExperts: builder.Input("down_exps", dtype.F32, tensor.MustShape(6, 8, 4)),
		FeedForwardExpertBias:  builder.Input("expert_bias", dtype.F32, tensor.MustShape(4)),
	}
	output, err := BuildDenseBlock(builder, input, spec, weights, []uint32{0, 1})
	if err != nil {
		t.Fatal(err)
	}
	nodes, _ := tensor.Topological(output)
	var moe *tensor.Tensor
	var rope int
	for _, node := range nodes {
		if node.Op == tensor.OpMoE {
			moe = node
		}
		if node.Op == tensor.OpRoPENeoX && node.Attrs.(tensor.RoPEAttributes).RotaryDimensions == 2 {
			rope++
		}
	}
	if moe == nil || moe.Attrs.(tensor.MoEAttributes).Routing != tensor.MoERoutingSigmoid ||
		!moe.Attrs.(tensor.MoEAttributes).NormalizeTopKProb || len(moe.Inputs) != 7 || rope != 2 {
		t.Fatalf("unexpected MiniMax-M2 graph: moe=%v rope=%d", moe != nil, rope)
	}
}

func TestBuildDenseApertus(t *testing.T) {
	builder := tensor.NewBuilder()
	spec := Spec{CommonSpec: CommonSpec{Architecture: "apertus", BlockCount: 1, ContextLength: 128,
		EmbeddingLength: 8, FeedForwardLength: 12,

		RMSNormEpsilon: 1e-5}, AttentionSpec: AttentionSpec{OriginalContextLength: 32,
		HeadCount: 2, HeadCountKV: 1, KeyLength: 4, ValueLength: 4,
		RopeDimensionCount: 4, RopeFrequencyBase: 10000,
		RopeScalingType: "longrope", RopeAttentionFactor: 1.25,
		AttentionScale: 0.3}, MoESpec: MoESpec{XIELUAlphaN: []float32{0.8}, XIELUAlphaP: []float32{0.2},
		XIELUBeta: []float32{0.5}, XIELUEpsilon: []float32{-0.1}},
	}
	input := builder.Input("input", dtype.F32, tensor.MustShape(8, 2))
	weights := denseBlockInputs(builder, spec)
	weights.AttentionQ = nil
	weights.AttentionK = nil
	weights.AttentionV = nil
	weights.AttentionQKV = builder.Input("attn_qkv", dtype.F32, tensor.MustShape(8, 16))
	weights.AttentionQKVBias = builder.Input("attn_qkv_bias", dtype.F32, tensor.MustShape(16))
	weights.AttentionOutputBias = builder.Input("attn_output_bias", dtype.F32, tensor.MustShape(8))
	weights.FeedForwardGate = nil
	weights.RopeFactors = builder.Input("rope_long", dtype.F32, tensor.MustShape(2))
	output, err := BuildDenseBlock(builder, input, spec, weights, []uint32{0, 1})
	if err != nil {
		t.Fatal(err)
	}
	nodes, err := tensor.Topological(output)
	if err != nil {
		t.Fatal(err)
	}
	var slices, xielu, rmsNorms, ropeWithFactors, ropeScale int
	var attentionScale float32
	for _, node := range nodes {
		switch node.Op {
		case tensor.OpGroupSlice:
			slices++
		case tensor.OpXIELU:
			xielu++
			attributes := node.Attrs.(tensor.XIELUAttributes)
			if attributes.AlphaN != 0.8 || attributes.AlphaP != 0.2 ||
				attributes.Beta != 0.5 || attributes.Epsilon != -0.1 {
				t.Fatalf("unexpected xIELU attributes: %+v", attributes)
			}
		case tensor.OpRMSNorm:
			rmsNorms++
		case tensor.OpRoPENeoX:
			if len(node.Inputs) == 2 && node.Inputs[1] == weights.RopeFactors {
				ropeWithFactors++
			}
		case tensor.OpScale:
			if node.Attrs.(tensor.ScaleAttributes).Value == 1.25 {
				ropeScale++
			}
		case tensor.OpAttention:
			attentionScale = node.Attrs.(tensor.AttentionAttributes).Scale
		}
	}
	if slices != 3 || xielu != 1 || rmsNorms != 4 || ropeWithFactors != 2 ||
		ropeScale != 2 || attentionScale != 0.3 {
		t.Fatalf("unexpected Apertus graph: slices=%d xIELU=%d RMSNorm=%d rope=%d scale=%d attention=%v", slices, xielu, rmsNorms, ropeWithFactors, ropeScale, attentionScale)
	}
}

func TestBuildDenseGPTNeoXResidualModes(t *testing.T) {
	for _, parallel := range []bool{false, true} {
		name := "sequential"
		if parallel {
			name = "parallel"
		}
		t.Run(name, func(t *testing.T) {
			builder := tensor.NewBuilder()
			spec := Spec{CommonSpec: CommonSpec{Architecture: "gptneox", BlockCount: 1, EmbeddingLength: 8,
				FeedForwardLength: 12,

				LayerNormEpsilon: 1e-5,
				ParallelResidual: parallel}, AttentionSpec: AttentionSpec{HeadCount: 2, HeadCountKV: 2,
				KeyLength: 4, ValueLength: 4, RopeDimensionCount: 2,
				RopeFrequencyBase: 10000},
			}
			input := builder.Input("input", dtype.F32, tensor.MustShape(8, 2))
			weights := denseBlockInputs(builder, spec)
			weights.AttentionQ = nil
			weights.AttentionK = nil
			weights.AttentionV = nil
			weights.AttentionQKV = builder.Input("attn_qkv", dtype.F32, tensor.MustShape(8, 24))
			weights.AttentionQKVBias = builder.Input("attn_qkv_bias", dtype.F32, tensor.MustShape(24))
			weights.AttentionOutputBias = builder.Input("attn_output_bias", dtype.F32, tensor.MustShape(8))
			weights.FeedForwardGate = nil
			weights.FeedForwardUpBias = builder.Input("ffn_up_bias", dtype.F32, tensor.MustShape(12))
			weights.FeedForwardDownBias = builder.Input("ffn_down_bias", dtype.F32, tensor.MustShape(8))
			output, err := BuildDenseBlock(builder, input, spec, weights, []uint32{0, 1})
			if err != nil {
				t.Fatal(err)
			}
			nodes, err := tensor.Topological(output)
			if err != nil {
				t.Fatal(err)
			}
			var layerNorms, normsOverInput, slices, gelu int
			for _, node := range nodes {
				switch node.Op {
				case tensor.OpLayerNorm:
					layerNorms++
					if node.Inputs[0] == input {
						normsOverInput++
					}
				case tensor.OpGroupSlice:
					slices++
				case tensor.OpGELU:
					gelu++
				}
			}
			wantNormsOverInput := 1
			if parallel {
				wantNormsOverInput = 2
			}
			if layerNorms != 2 || normsOverInput != wantNormsOverInput || slices != 3 || gelu != 1 {
				t.Fatalf(
					"unexpected GPT-NeoX graph: layernorms=%d roots=%d slices=%d GELU=%d",
					layerNorms, normsOverInput, slices, gelu,
				)
			}
		})
	}
}

func TestBuildDenseGPTJParallelResidualNormalRoPEGELU(t *testing.T) {
	builder := tensor.NewBuilder()
	spec := Spec{CommonSpec: CommonSpec{Architecture: "gptj", BlockCount: 1, EmbeddingLength: 8,
		FeedForwardLength: 12, LayerNormEpsilon: 1e-5}, AttentionSpec: AttentionSpec{
		HeadCount: 2, HeadCountKV: 2, KeyLength: 4, ValueLength: 4,
		RopeDimensionCount: 2, RopeFrequencyBase: 10000}}
	input := builder.Input("input", dtype.F32, tensor.MustShape(8, 2))
	weights := denseBlockInputs(builder, spec)
	weights.FeedForwardGate = nil
	weights.FeedForwardUpBias = builder.Input("ffn_up_bias", dtype.F32, tensor.MustShape(12))
	weights.FeedForwardDownBias = builder.Input("ffn_down_bias", dtype.F32, tensor.MustShape(8))
	output, err := BuildDenseBlock(builder, input, spec, weights, []uint32{0, 1})
	if err != nil {
		t.Fatal(err)
	}
	nodes, err := tensor.Topological(output)
	if err != nil {
		t.Fatal(err)
	}
	var layerNorms, normalRoPE, neoXRoPE, gelu int
	for _, node := range nodes {
		switch node.Op {
		case tensor.OpLayerNorm:
			layerNorms++
			if node.Inputs[0] != input {
				t.Fatal("GPT-J FFN does not share attention LayerNorm input")
			}
		case tensor.OpRoPENormal:
			normalRoPE++
		case tensor.OpRoPENeoX:
			neoXRoPE++
		case tensor.OpGELU:
			gelu++
		}
	}
	if layerNorms != 1 || normalRoPE != 2 || neoXRoPE != 0 || gelu != 1 {
		t.Fatalf("unexpected GPT-J graph: LayerNorm=%d normalRoPE=%d NeoXRoPE=%d GELU=%d",
			layerNorms, normalRoPE, neoXRoPE, gelu)
	}
}

func TestBuildDenseGLM4(t *testing.T) {
	builder := tensor.NewBuilder()
	spec := Spec{CommonSpec: CommonSpec{Architecture: "glm4", BlockCount: 1, EmbeddingLength: 8,
		FeedForwardLength: 12,

		RMSNormEpsilon: 1e-5}, AttentionSpec: AttentionSpec{HeadCount: 2, HeadCountKV: 1,
		KeyLength: 4, ValueLength: 4, RopeDimensionCount: 4,
		RopeFrequencyBase: 10000},
	}
	input := builder.Input("input", dtype.F32, tensor.MustShape(8, 2))
	weights := denseBlockInputs(builder, spec)
	weights.AttentionQ = nil
	weights.AttentionK = nil
	weights.AttentionV = nil
	weights.AttentionQKV = builder.Input("attn_qkv", dtype.F32, tensor.MustShape(8, 16))
	weights.AttentionQKVBias = builder.Input("attn_qkv_bias", dtype.F32, tensor.MustShape(16))
	weights.AttentionPostNorm = builder.Input("post_attention_norm", dtype.F32, tensor.MustShape(8))
	weights.FeedForwardPostNorm = builder.Input("post_ffw_norm", dtype.F32, tensor.MustShape(8))
	weights.FeedForwardGate = nil
	weights.FeedForwardUp = builder.Input("ffn_up", dtype.F32, tensor.MustShape(8, 24))
	output, err := BuildDenseBlock(builder, input, spec, weights, []uint32{0, 1})
	if err != nil {
		t.Fatal(err)
	}
	nodes, err := tensor.Topological(output)
	if err != nil {
		t.Fatal(err)
	}
	var slices, rmsNorms, rope, silu int
	var attentionScale float32
	for _, node := range nodes {
		switch node.Op {
		case tensor.OpGroupSlice:
			slices++
		case tensor.OpRMSNorm:
			rmsNorms++
		case tensor.OpRoPENormal:
			rope++
		case tensor.OpSiLU:
			silu++
		case tensor.OpAttention:
			attentionScale = node.Attrs.(tensor.AttentionAttributes).Scale
		}
	}
	if slices != 5 || rmsNorms != 4 || rope != 2 || silu != 1 || attentionScale != 0.5 {
		t.Fatalf("unexpected GLM4 graph: slices=%d RMSNorm=%d RoPE=%d SiLU=%d attention=%v", slices, rmsNorms, rope, silu, attentionScale)
	}
}

func TestBuildDenseGLM4UsesDistinctMRoPEPositions(t *testing.T) {
	builder := tensor.NewBuilder()
	spec := Spec{CommonSpec: CommonSpec{Architecture: "glm4", BlockCount: 1, EmbeddingLength: 8,
		FeedForwardLength: 12,

		RMSNormEpsilon: 1e-5}, AttentionSpec: AttentionSpec{HeadCount: 2, HeadCountKV: 1,
		KeyLength: 4, ValueLength: 4, RopeDimensionCount: 4,
		RopeSections:      [4]int32{2, 2, 0, 0},
		RopeFrequencyBase: 10000},
	}
	input := builder.Input("input", dtype.F32, tensor.MustShape(8, 2))
	weights := denseBlockInputs(builder, spec)
	weights.AttentionQ, weights.AttentionK, weights.AttentionV = nil, nil, nil
	weights.AttentionQKV = builder.Input("attn_qkv", dtype.F32, tensor.MustShape(8, 16))
	weights.AttentionQKVBias = builder.Input("attn_qkv_bias", dtype.F32, tensor.MustShape(16))
	weights.AttentionPostNorm = builder.Input("post_attention_norm", dtype.F32, tensor.MustShape(8))
	weights.FeedForwardPostNorm = builder.Input("post_ffw_norm", dtype.F32, tensor.MustShape(8))
	weights.FeedForwardGate = nil
	weights.FeedForwardUp = builder.Input("ffn_up", dtype.F32, tensor.MustShape(8, 24))
	positions := [4][]uint32{{10, 11}, {20, 21}, {30, 31}, {40, 41}}
	result, err := BuildDenseBlockCachedForLayerWithMultiPositions(
		builder, input, spec, weights, positions, nil, nil, 0,
	)
	if err != nil {
		t.Fatal(err)
	}
	nodes, err := tensor.Topological(result.Output)
	if err != nil {
		t.Fatal(err)
	}
	count := 0
	for _, node := range nodes {
		if node.Op != tensor.OpRoPEMulti {
			continue
		}
		count++
		attributes := node.Attrs.(tensor.RoPEMultiAttributes)
		if attributes.Sections != spec.RopeSections {
			t.Fatalf("MRoPE sections = %v, want %v", attributes.Sections, spec.RopeSections)
		}
		for axis := range positions {
			if !slices.Equal(attributes.Positions[axis], positions[axis]) {
				t.Fatalf("MRoPE axis %d = %v, want %v", axis, attributes.Positions[axis], positions[axis])
			}
		}
	}
	if count != 2 {
		t.Fatalf("MRoPE nodes = %d, want 2", count)
	}
}

func TestBuildDenseEXAONE4SlidingPattern(t *testing.T) {
	for _, test := range []struct {
		layer      uint32
		wantRoPE   int
		wantWindow uint32
	}{
		{layer: 0, wantRoPE: 2, wantWindow: 4096},
		{layer: 3, wantRoPE: 0, wantWindow: 0},
	} {
		builder := tensor.NewBuilder()
		spec := Spec{CommonSpec: CommonSpec{Architecture: "exaone4", BlockCount: 64, EmbeddingLength: 8,
			FeedForwardLength: 12,

			RMSNormEpsilon: 1e-5}, AttentionSpec: AttentionSpec{HeadCount: 2, HeadCountKV: 1,
			KeyLength: 4, ValueLength: 4, RopeDimensionCount: 4,
			RopeFrequencyBase: 1_000_000,
			SlidingWindow:     4096, SlidingPattern: 4, NoRopeLayerStep: 4},
		}
		input := builder.Input("input", dtype.F32, tensor.MustShape(8, 2))
		weights := denseBlockInputs(builder, spec)
		weights.AttentionNorm = nil
		weights.AttentionNormBias = nil
		weights.FeedForwardNorm = nil
		weights.FeedForwardNormBias = nil
		weights.AttentionQ = nil
		weights.AttentionK = nil
		weights.AttentionV = nil
		weights.AttentionQKV = builder.Input("attn_qkv", dtype.F32, tensor.MustShape(8, 16))
		weights.AttentionQKVBias = builder.Input("attn_qkv_bias", dtype.F32, tensor.MustShape(16))
		weights.AttentionPostNorm = builder.Input("post_attention_norm", dtype.F32, tensor.MustShape(8))
		weights.FeedForwardPostNorm = builder.Input("post_ffw_norm", dtype.F32, tensor.MustShape(8))
		result, err := BuildDenseBlockCachedForLayer(
			builder, input, spec, weights, []uint32{0, 1}, nil, nil, test.layer,
		)
		if err != nil {
			t.Fatal(err)
		}
		nodes, err := tensor.Topological(result.Output)
		if err != nil {
			t.Fatal(err)
		}
		var rmsNorms, rope int
		var window uint32
		for _, node := range nodes {
			switch node.Op {
			case tensor.OpRMSNorm:
				rmsNorms++
			case tensor.OpRoPENeoX:
				rope++
			case tensor.OpAttention:
				window = node.Attrs.(tensor.AttentionAttributes).Window
			}
		}
		if rmsNorms != 4 || rope != test.wantRoPE || window != test.wantWindow {
			t.Fatalf("EXAONE 4 layer %d graph: RMSNorm=%d RoPE=%d window=%d", test.layer, rmsNorms, rope, window)
		}
	}
}

func TestBuildDenseFalconNormLayouts(t *testing.T) {
	for _, secondNorm := range []bool{false, true} {
		name := "7b-shared-norm"
		if secondNorm {
			name = "40b-dual-norm"
		}
		t.Run(name, func(t *testing.T) {
			builder := tensor.NewBuilder()
			spec := Spec{CommonSpec: CommonSpec{Architecture: "falcon", BlockCount: 1, EmbeddingLength: 8,
				FeedForwardLength: 12,

				LayerNormEpsilon: 1e-5}, AttentionSpec: AttentionSpec{HeadCount: 2, HeadCountKV: 1,
				KeyLength: 4, ValueLength: 4, RopeDimensionCount: 4,
				RopeFrequencyBase: 10000},
			}
			input := builder.Input("input", dtype.F32, tensor.MustShape(8, 2))
			weights := denseBlockInputs(builder, spec)
			weights.AttentionQ = nil
			weights.AttentionK = nil
			weights.AttentionV = nil
			weights.AttentionQKV = builder.Input("attn_qkv", dtype.F32, tensor.MustShape(8, 16))
			weights.FeedForwardNorm = nil
			weights.FeedForwardNormBias = nil
			weights.FeedForwardGate = nil
			if secondNorm {
				weights.AttentionNorm2 = builder.Input("attn_norm_2", dtype.F32, tensor.MustShape(8))
				weights.AttentionNorm2Bias = builder.Input("attn_norm_2_bias", dtype.F32, tensor.MustShape(8))
			}
			output, err := BuildDenseBlock(builder, input, spec, weights, []uint32{0, 1})
			if err != nil {
				t.Fatal(err)
			}
			nodes, err := tensor.Topological(output)
			if err != nil {
				t.Fatal(err)
			}
			var layerNorms, normsOverInput, slices, gelu int
			var attentionInput, feedForwardInput *tensor.Tensor
			for _, node := range nodes {
				switch node.Op {
				case tensor.OpLayerNorm:
					layerNorms++
					if node.Inputs[0] == input {
						normsOverInput++
					}
				case tensor.OpGroupSlice:
					slices++
				case tensor.OpGELU:
					gelu++
				case tensor.OpMulMat:
					if node.Inputs[0].Name == "attn_qkv" {
						attentionInput = node.Inputs[1]
					}
					if node.Inputs[0].Name == "ffn_up" {
						feedForwardInput = node.Inputs[1]
					}
				}
			}
			wantNorms := 1
			if secondNorm {
				wantNorms = 2
			}
			if layerNorms != wantNorms || normsOverInput != wantNorms || slices != 3 || gelu != 1 {
				t.Fatalf("unexpected Falcon graph: norms=%d roots=%d slices=%d GELU=%d", layerNorms, normsOverInput, slices, gelu)
			}
			if secondNorm && attentionInput == feedForwardInput {
				t.Fatal("Falcon-40B attention and FFN unexpectedly share a norm")
			}
		})
	}
}

func TestBuildDenseBitNetSubNormsAndScales(t *testing.T) {
	builder := tensor.NewBuilder()
	spec := Spec{CommonSpec: CommonSpec{Architecture: "bitnet", BlockCount: 1, EmbeddingLength: 8,
		FeedForwardLength: 12,
		RMSNormEpsilon:    1e-5}, AttentionSpec: AttentionSpec{HeadCount: 2, HeadCountKV: 1,
		KeyLength: 4, ValueLength: 4, RopeFrequencyBase: 10000},
	}
	input := builder.Input("input", dtype.F32, tensor.MustShape(8, 2))
	weights := denseBlockInputs(builder, spec)
	weights.AttentionSubNorm = builder.Input("attn_sub_norm", dtype.F32, tensor.MustShape(8))
	weights.FeedForwardSubNorm = builder.Input("ffn_sub_norm", dtype.F32, tensor.MustShape(12))
	scale := func(name string) *tensor.Tensor {
		return builder.Input(name, dtype.F32, tensor.MustShape(1))
	}
	weights.AttentionQScale = scale("attn_q_scale")
	weights.AttentionKScale = scale("attn_k_scale")
	weights.AttentionVScale = scale("attn_v_scale")
	weights.AttentionOutputScale = scale("attn_output_scale")
	weights.FeedForwardGateScale = scale("ffn_gate_scale")
	weights.FeedForwardUpScale = scale("ffn_up_scale")
	weights.FeedForwardDownScale = scale("ffn_down_scale")
	output, err := BuildDenseBlock(builder, input, spec, weights, []uint32{0, 1})
	if err != nil {
		t.Fatal(err)
	}
	nodes, err := tensor.Topological(output)
	if err != nil {
		t.Fatal(err)
	}
	var rmsNorms, scaleMultiplies int
	for _, node := range nodes {
		if node.Op == tensor.OpRMSNorm {
			rmsNorms++
		}
		if node.Op == tensor.OpMultiply && len(node.Inputs) == 2 &&
			strings.HasSuffix(node.Inputs[1].Name, "_scale") {
			scaleMultiplies++
		}
	}
	if rmsNorms != 4 || scaleMultiplies != 7 {
		t.Fatalf("unexpected BitNet graph: RMSNorms=%d scale-multiplies=%d", rmsNorms, scaleMultiplies)
	}
}

func TestBuildDenseQwen3BlockWithCache(t *testing.T) {
	builder := tensor.NewBuilder()
	spec := Spec{CommonSpec: CommonSpec{Architecture: "qwen3",
		EmbeddingLength:   8,
		FeedForwardLength: 12,

		RMSNormEpsilon: 1e-6}, AttentionSpec: AttentionSpec{HeadCount: 2,
		HeadCountKV:       1,
		KeyLength:         4,
		ValueLength:       4,
		RopeFrequencyBase: 1_000_000},
	}
	input := builder.Input("input", dtype.F32, tensor.MustShape(8, 1))
	pastKey := builder.Input("past_key", dtype.F32, tensor.MustShape(4, 1, 3))
	pastValue := builder.Input("past_value", dtype.F32, tensor.MustShape(4, 1, 3))
	result, err := BuildDenseBlockCached(
		builder,
		input,
		spec,
		denseBlockInputs(builder, spec),
		[]uint32{3},
		pastKey,
		pastValue,
	)
	if err != nil {
		t.Fatal(err)
	}
	if result.Key.Shape.Dims[2] != 4 || result.Value.Shape.Dims[2] != 4 {
		t.Fatalf("cache token counts = %d/%d, want 4/4", result.Key.Shape.Dims[2], result.Value.Shape.Dims[2])
	}
	if !result.Output.Shape.Equal(input.Shape) {
		t.Fatalf("output shape = %v, want %v", result.Output.Shape.Slice(), input.Shape.Slice())
	}
}

func TestBuildDenseLlamaBlockUsesNormalRoPE(t *testing.T) {
	builder := tensor.NewBuilder()
	spec := Spec{CommonSpec: CommonSpec{Architecture: "llama",
		EmbeddingLength:   8,
		FeedForwardLength: 12,

		RMSNormEpsilon: 1e-5}, AttentionSpec: AttentionSpec{HeadCount: 2,
		HeadCountKV:       1,
		KeyLength:         4,
		ValueLength:       4,
		RopeFrequencyBase: 10000},
	}
	input := builder.Input("input", dtype.F32, tensor.MustShape(8, 2))
	weights := denseBlockInputs(builder, spec)
	weights.AttentionQNorm = nil
	weights.AttentionKNorm = nil
	weights.RopeFactors = builder.Input("rope_factors", dtype.F32, tensor.MustShape(2))
	output, err := BuildDenseBlock(builder, input, spec, weights, []uint32{0, 1})
	if err != nil {
		t.Fatal(err)
	}
	nodes, err := tensor.Topological(output)
	if err != nil {
		t.Fatal(err)
	}
	normalCount := 0
	for _, node := range nodes {
		if node.Op == tensor.OpRoPENormal {
			normalCount++
			if len(node.Inputs) != 2 || node.Inputs[1] != weights.RopeFactors {
				t.Fatal("Llama RoPE node does not consume frequency factors")
			}
		}
		if node.Op == tensor.OpRoPENeoX {
			t.Fatal("Llama block unexpectedly uses NeoX RoPE")
		}
	}
	if normalCount != 2 {
		t.Fatalf("normal RoPE count = %d, want 2", normalCount)
	}
}

func TestBuildLlamaEmbedBlocksUseBidirectionalNormalRoPE(t *testing.T) {
	for _, moe := range []bool{false, true} {
		name := "dense"
		if moe {
			name = "moe"
		}
		t.Run(name, func(t *testing.T) {
			builder := tensor.NewBuilder()
			spec := Spec{CommonSpec: CommonSpec{Architecture: "llama-embed", EmbeddingLength: 8, FeedForwardLength: 12,

				RMSNormEpsilon: 1e-5}, AttentionSpec: AttentionSpec{HeadCount: 2, HeadCountKV: 1, KeyLength: 4, ValueLength: 4,
				RopeFrequencyBase: 10000, NonCausalAttention: true},
			}
			input := builder.Input("input", dtype.F32, tensor.MustShape(8, 3))
			weights := denseBlockInputs(builder, spec)
			weights.AttentionQNorm = nil
			weights.AttentionKNorm = nil
			if moe {
				spec.ExpertCount, spec.ExpertUsedCount, spec.ExpertFeedForward = 4, 2, 12
				spec.ExpertWeightsScale = 1
				weights.FeedForwardGate, weights.FeedForwardUp, weights.FeedForwardDown = nil, nil, nil
				weights.FeedForwardRouter = builder.Input("router", dtype.F32, tensor.MustShape(8, 4))
				weights.FeedForwardGateExperts = builder.Input("gate_exps", dtype.F32, tensor.MustShape(8, 12, 4))
				weights.FeedForwardUpExperts = builder.Input("up_exps", dtype.F32, tensor.MustShape(8, 12, 4))
				weights.FeedForwardDownExperts = builder.Input("down_exps", dtype.F32, tensor.MustShape(12, 8, 4))
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
			var normal, experts int
			for _, node := range nodes {
				switch node.Op {
				case tensor.OpAttention:
					attention = node
				case tensor.OpRoPENormal:
					normal++
				case tensor.OpMoE:
					experts++
				}
			}
			wantExperts := 0
			if moe {
				wantExperts = 1
			}
			if attention == nil || attention.Attrs.(tensor.AttentionAttributes).Causal ||
				normal != 2 || experts != wantExperts {
				t.Fatalf("unexpected Llama Embed graph: attention=%v normal=%d experts=%d", attention, normal, experts)
			}

			cacheBuilder := tensor.NewBuilder()
			cacheInput := cacheBuilder.Input("input", dtype.F32, tensor.MustShape(8, 1))
			cacheWeights := denseBlockInputs(cacheBuilder, spec)
			cacheWeights.AttentionQNorm = nil
			cacheWeights.AttentionKNorm = nil
			pastKey := cacheBuilder.Input("past_key", dtype.F32, tensor.MustShape(4, 1, 1))
			pastValue := cacheBuilder.Input("past_value", dtype.F32, tensor.MustShape(4, 1, 1))
			_, err = BuildDenseBlockCached(cacheBuilder, cacheInput, spec, cacheWeights, []uint32{1}, pastKey, pastValue)
			if err == nil || !strings.Contains(err.Error(), "does not support a KV cache") {
				t.Fatalf("cached Llama Embed block error = %v", err)
			}
		})
	}
}

func TestBuildPanguEmbeddedBlockUsesCausalNeoXRoPEAndOutputBias(t *testing.T) {
	builder := tensor.NewBuilder()
	spec := Spec{CommonSpec: CommonSpec{Architecture: "pangu-embedded", EmbeddingLength: 8, FeedForwardLength: 12,

		RMSNormEpsilon: 1e-6}, AttentionSpec: AttentionSpec{HeadCount: 2, HeadCountKV: 1, KeyLength: 4, ValueLength: 4,
		RopeDimensionCount: 4, RopeFrequencyBase: 10000},
	}
	input := builder.Input("input", dtype.F32, tensor.MustShape(8, 3))
	weights := denseBlockInputs(builder, spec)
	weights.AttentionQNorm = nil
	weights.AttentionKNorm = nil
	weights.AttentionOutputBias = builder.Input("attn_out_bias", dtype.F32, tensor.MustShape(8))
	weights.RopeFactors = builder.Input("rope_factors", dtype.F32, tensor.MustShape(2))
	result, err := BuildDenseBlockCachedForLayer(builder, input, spec, weights, []uint32{0, 1, 2}, nil, nil, 0)
	if err != nil {
		t.Fatal(err)
	}
	nodes, err := tensor.Topological(result.Output)
	if err != nil {
		t.Fatal(err)
	}
	var attention *tensor.Tensor
	var neoX, biasAdds int
	for _, node := range nodes {
		switch node.Op {
		case tensor.OpAttention:
			attention = node
		case tensor.OpRoPENeoX:
			neoX++
			if len(node.Inputs) != 2 || node.Inputs[1] != weights.RopeFactors {
				t.Fatal("Pangu Embedded RoPE factors are disconnected")
			}
		case tensor.OpAdd:
			if len(node.Inputs) == 2 && (node.Inputs[0] == weights.AttentionOutputBias || node.Inputs[1] == weights.AttentionOutputBias) {
				biasAdds++
			}
		}
	}
	if attention == nil || !attention.Attrs.(tensor.AttentionAttributes).Causal || neoX != 2 || biasAdds != 1 {
		t.Fatalf("unexpected Pangu Embedded graph: attention=%v NeoX=%d output-bias=%d", attention, neoX, biasAdds)
	}
}

func TestBuildModernBERTBlocksUseDenseFirstSymmetricWindows(t *testing.T) {
	for _, test := range []struct {
		name       string
		layer      uint32
		activation string
		wantWindow bool
		wantOp     tensor.Op
	}{
		{name: "first GEGLU", layer: 0, activation: "geglu", wantOp: tensor.OpGELU},
		{name: "local SwiGLU", layer: 1, activation: "swiglu", wantWindow: true, wantOp: tensor.OpSiLU},
		{name: "local ReGLU", layer: 2, activation: "reglu", wantWindow: true, wantOp: tensor.OpReLU},
	} {
		t.Run(test.name, func(t *testing.T) {
			builder := tensor.NewBuilder()
			spec := Spec{CommonSpec: CommonSpec{Architecture: "modern-bert", BlockCount: 3, EmbeddingLength: 8,
				FeedForwardLength: 16,

				LayerNormEpsilon: 1e-5,
				HiddenActivation: test.activation}, AttentionSpec: AttentionSpec{HeadCount: 2, HeadCountKV: 2,
				KeyLength: 4, ValueLength: 4, RopeDimensionCount: 4,
				RopeFrequencyBase: 10000, RopeFrequencySWA: 50000,
				SlidingWindow: 4, SlidingPattern: 3,
				NonCausalAttention: true},
			}
			input := builder.Input("input", dtype.F32, tensor.MustShape(8, 5))
			weights := LayerGraphWeights{
				AttentionQKV:    builder.Input("qkv", dtype.F32, tensor.MustShape(8, 24)),
				AttentionOutput: builder.Input("attn_out", dtype.F32, tensor.MustShape(8, 8)),
				FeedForwardNorm: builder.Input("ffn_norm", dtype.F32, tensor.MustShape(8)),
				FeedForwardUp:   builder.Input("ffn_up", dtype.F32, tensor.MustShape(8, 32)),
				FeedForwardDown: builder.Input("ffn_down", dtype.F32, tensor.MustShape(16, 8)),
			}
			if test.layer > 0 {
				weights.AttentionNorm = builder.Input("attn_norm", dtype.F32, tensor.MustShape(8))
			}
			result, err := BuildDenseBlockCachedForLayer(builder, input, spec, weights, []uint32{0, 1, 2, 3, 4}, nil, nil, test.layer)
			if err != nil {
				t.Fatal(err)
			}
			nodes, err := tensor.Topological(result.Output)
			if err != nil {
				t.Fatal(err)
			}
			var attention *tensor.Tensor
			var layerNorm, neoX, gelu, silu, relu int
			var ropeBase float32
			for _, node := range nodes {
				switch node.Op {
				case tensor.OpAttention:
					attention = node
				case tensor.OpLayerNorm:
					layerNorm++
				case tensor.OpRoPENeoX:
					neoX++
					ropeBase = node.Attrs.(tensor.RoPEAttributes).FrequencyBase
				case tensor.OpGELU:
					gelu++
				case tensor.OpSiLU:
					silu++
				case tensor.OpReLU:
					relu++
				}
			}
			wantNorms := 1
			if test.layer > 0 {
				wantNorms = 2
			}
			if attention == nil || attention.Attrs.(tensor.AttentionAttributes).Causal ||
				attention.Attrs.(tensor.AttentionAttributes).SymmetricWindow != test.wantWindow ||
				neoX != 2 || layerNorm != wantNorms ||
				(test.wantOp == tensor.OpGELU && (gelu != 1 || silu != 0 || relu != 0)) ||
				(test.wantOp == tensor.OpSiLU && (silu != 1 || gelu != 0 || relu != 0)) ||
				(test.wantOp == tensor.OpReLU && (relu != 1 || gelu != 0 || silu != 0)) {
				t.Fatalf("unexpected ModernBERT graph: attention=%v norms=%d NeoX=%d GELU=%d SiLU=%d ReLU=%d", attention, layerNorm, neoX, gelu, silu, relu)
			}
			wantBase := float32(10000)
			if test.wantWindow {
				wantBase = 50000
			}
			if ropeBase != wantBase {
				t.Fatalf("ModernBERT RoPE base = %v, want %v", ropeBase, wantBase)
			}
		})
	}
}

func TestBuildGemmaEmbeddingBlocksUseQKNormAndPeriodicSymmetricWindows(t *testing.T) {
	for _, test := range []struct {
		name       string
		layer      uint32
		wantWindow bool
		wantBase   float32
	}{
		{name: "local", layer: 0, wantWindow: true, wantBase: 50000},
		{name: "periodic full", layer: 5, wantBase: 10000},
	} {
		t.Run(test.name, func(t *testing.T) {
			builder := tensor.NewBuilder()
			spec := Spec{CommonSpec: CommonSpec{Architecture: "gemma-embedding", BlockCount: 6, EmbeddingLength: 8,
				FeedForwardLength: 16,

				RMSNormEpsilon: 1e-6}, AttentionSpec: AttentionSpec{HeadCount: 2, HeadCountKV: 1,
				KeyLength: 4, ValueLength: 4, RopeDimensionCount: 4,
				RopeFrequencyBase: 10000, RopeFrequencySWA: 50000,
				SlidingWindow: 4, SlidingPattern: 6,
				NonCausalAttention: true},
			}
			input := builder.Input("input", dtype.F32, tensor.MustShape(8, 5))
			weights := LayerGraphWeights{
				AttentionNorm:       builder.Input("attn_norm", dtype.F32, tensor.MustShape(8)),
				AttentionQKV:        builder.Input("qkv", dtype.F32, tensor.MustShape(8, 16)),
				AttentionQKVBias:    builder.Input("qkv_bias", dtype.F32, tensor.MustShape(16)),
				AttentionOutput:     builder.Input("attn_out", dtype.F32, tensor.MustShape(8, 8)),
				AttentionQNorm:      builder.Input("q_norm", dtype.F32, tensor.MustShape(4)),
				AttentionKNorm:      builder.Input("k_norm", dtype.F32, tensor.MustShape(4)),
				AttentionPostNorm:   builder.Input("attn_post_norm", dtype.F32, tensor.MustShape(8)),
				FeedForwardNorm:     builder.Input("ffn_norm", dtype.F32, tensor.MustShape(8)),
				FeedForwardGate:     builder.Input("ffn_gate", dtype.F32, tensor.MustShape(8, 16)),
				FeedForwardUp:       builder.Input("ffn_up", dtype.F32, tensor.MustShape(8, 16)),
				FeedForwardDown:     builder.Input("ffn_down", dtype.F32, tensor.MustShape(16, 8)),
				FeedForwardPostNorm: builder.Input("ffn_post_norm", dtype.F32, tensor.MustShape(8)),
			}
			result, err := BuildDenseBlockCachedForLayer(
				builder, input, spec, weights, []uint32{0, 1, 2, 3, 4}, nil, nil, test.layer,
			)
			if err != nil {
				t.Fatal(err)
			}
			nodes, err := tensor.Topological(result.Output)
			if err != nil {
				t.Fatal(err)
			}
			var attention *tensor.Tensor
			var neoX, rmsNorm, gelu int
			var ropeBase float32
			queryScale := float32(0)
			for _, node := range nodes {
				switch node.Op {
				case tensor.OpAttention:
					attention = node
				case tensor.OpRoPENeoX:
					neoX++
					ropeBase = node.Attrs.(tensor.RoPEAttributes).FrequencyBase
				case tensor.OpRMSNorm:
					rmsNorm++
				case tensor.OpGELU:
					gelu++
				case tensor.OpScale:
					queryScale = node.Attrs.(tensor.ScaleAttributes).Value
				}
			}
			if attention == nil {
				t.Fatal("Gemma embedding attention node is missing")
			}
			attributes := attention.Attrs.(tensor.AttentionAttributes)
			if attributes.Causal || attributes.Scale != 1 ||
				attributes.SymmetricWindow != test.wantWindow || neoX != 2 || rmsNorm != 6 || gelu != 1 ||
				math.Abs(float64(queryScale-0.5)) > 1e-6 || ropeBase != test.wantBase {
				t.Fatalf("unexpected Gemma embedding graph: attention=%v RMS=%d NeoX=%d GELU=%d query-scale=%v base=%v", attention, rmsNorm, neoX, gelu, queryScale, ropeBase)
			}
		})
	}
}

func TestBuildTalkieBlockUsesPostRoPEQueryGainAndEmbeddingSkip(t *testing.T) {
	builder := tensor.NewBuilder()
	spec := Spec{CommonSpec: CommonSpec{Architecture: "talkie", EmbeddingLength: 8, FeedForwardLength: 12,

		RMSNormEpsilon: 1e-6}, AttentionSpec: AttentionSpec{HeadCount: 2, HeadCountKV: 1, KeyLength: 4, ValueLength: 4,
		RopeDimensionCount: 4, RopeFrequencyBase: 10000},
	}
	input := builder.Input("input", dtype.F32, tensor.MustShape(8, 2))
	skip := builder.Input("embedding_skip", dtype.F32, tensor.MustShape(8, 2))
	weights := LayerGraphWeights{
		AttentionQ:       builder.Input("q", dtype.F32, tensor.MustShape(8, 8)),
		AttentionK:       builder.Input("k", dtype.F32, tensor.MustShape(8, 4)),
		AttentionV:       builder.Input("v", dtype.F32, tensor.MustShape(8, 4)),
		AttentionOutput:  builder.Input("attn_out", dtype.F32, tensor.MustShape(8, 8)),
		AttentionQNorm:   builder.Input("q_norm", dtype.F32, tensor.MustShape(1, 2)),
		FeedForwardGate:  builder.Input("ffn_gate", dtype.F32, tensor.MustShape(8, 12)),
		FeedForwardUp:    builder.Input("ffn_up", dtype.F32, tensor.MustShape(8, 12)),
		FeedForwardDown:  builder.Input("ffn_down", dtype.F32, tensor.MustShape(12, 8)),
		LayerOutputScale: builder.Input("layer_scale", dtype.F32, tensor.MustShape(1)),
		EmbeddingSkip:    skip,
	}
	pastKey := builder.Input("past_key", dtype.F32, tensor.MustShape(4, 1, 1))
	pastValue := builder.Input("past_value", dtype.F32, tensor.MustShape(4, 1, 1))
	result, err := BuildDenseBlockCachedForLayer(
		builder, input, spec, weights, []uint32{1, 2}, pastKey, pastValue, 0,
	)
	if err != nil {
		t.Fatal(err)
	}
	if result.Key.Shape.Dims[2] != 3 || result.Value.Shape.Dims[2] != 3 {
		t.Fatalf("Talkie cache shapes = %v/%v", result.Key.Shape.Slice(), result.Value.Shape.Slice())
	}
	nodes, err := tensor.Topological(result.Output)
	if err != nil {
		t.Fatal(err)
	}
	var attention *tensor.Tensor
	var neoX, rmsNorm, silu, skipProducts int
	for _, node := range nodes {
		switch node.Op {
		case tensor.OpAttention:
			attention = node
		case tensor.OpRoPENeoX:
			neoX++
		case tensor.OpRMSNorm:
			rmsNorm++
		case tensor.OpSiLU:
			silu++
		case tensor.OpMultiply:
			if len(node.Inputs) == 2 &&
				((node.Inputs[0] == skip && node.Inputs[1] == weights.LayerOutputScale) ||
					(node.Inputs[1] == skip && node.Inputs[0] == weights.LayerOutputScale)) {
				skipProducts++
			}
		}
	}
	if attention == nil {
		t.Fatal("Talkie attention node is missing")
	}
	attributes := attention.Attrs.(tensor.AttentionAttributes)
	if !attributes.Causal || attributes.QueryStart != 1 ||
		math.Abs(float64(attributes.Scale-0.5)) > 1e-6 ||
		neoX != 2 || rmsNorm != 4 || silu != 1 || skipProducts != 1 {
		t.Fatalf("unexpected Talkie graph: attention=%v RMS=%d NeoX=%d SiLU=%d skip=%d", attention, rmsNorm, neoX, silu, skipProducts)
	}
}

func TestBuildDenseLlamaBlockUsesLinearRoPEScale(t *testing.T) {
	builder := tensor.NewBuilder()
	spec := Spec{CommonSpec: CommonSpec{Architecture: "llama",
		EmbeddingLength:   8,
		FeedForwardLength: 12,

		RMSNormEpsilon: 1e-5}, AttentionSpec: AttentionSpec{HeadCount: 2,
		HeadCountKV:       1,
		KeyLength:         4,
		ValueLength:       4,
		RopeFrequencyBase: 10000,
		RopeScalingType:   "linear",
		RopeScalingFactor: 8},
	}
	input := builder.Input("input", dtype.F32, tensor.MustShape(8, 2))
	weights := denseBlockInputs(builder, spec)
	weights.AttentionQNorm = nil
	weights.AttentionKNorm = nil
	output, err := BuildDenseBlock(builder, input, spec, weights, []uint32{0, 1})
	if err != nil {
		t.Fatal(err)
	}
	nodes, err := tensor.Topological(output)
	if err != nil {
		t.Fatal(err)
	}
	ropeCount := 0
	for _, node := range nodes {
		if node.Op != tensor.OpRoPENormal {
			continue
		}
		ropeCount++
		attributes := node.Attrs.(tensor.RoPEAttributes)
		if attributes.FrequencyScale != 0.125 {
			t.Fatalf("RoPE frequency scale = %v, want 0.125", attributes.FrequencyScale)
		}
	}
	if ropeCount != 2 {
		t.Fatalf("normal RoPE count = %d, want 2", ropeCount)
	}
}

func TestBuildDenseGemma2BlockUsesSoftcappedSlidingAttention(t *testing.T) {
	builder := tensor.NewBuilder()
	spec := Spec{CommonSpec: CommonSpec{Architecture: "gemma2",
		BlockCount:        26,
		EmbeddingLength:   8,
		FeedForwardLength: 12,

		RMSNormEpsilon: 1e-6}, AttentionSpec: AttentionSpec{HeadCount: 2,
		HeadCountKV:       1,
		KeyLength:         4,
		ValueLength:       4,
		RopeFrequencyBase: 10000,
		RopeFrequencySWA:  10000,
		RopeScalingType:   "linear",
		RopeScalingFactor: 4,
		AttentionSoftcap:  50,

		SlidingWindow:  4096,
		SlidingPattern: 2},
	}
	input := builder.Input("input", dtype.F32, tensor.MustShape(8, 2))
	weights := denseBlockInputs(builder, spec)
	weights.AttentionQNorm = nil
	weights.AttentionKNorm = nil
	weights.AttentionPostNorm = builder.Input(
		"post_attention_norm", dtype.F32, tensor.MustShape(8),
	)
	weights.FeedForwardPostNorm = builder.Input(
		"post_ffw_norm", dtype.F32, tensor.MustShape(8),
	)
	output, err := BuildDenseBlockCachedForLayer(
		builder, input, spec, weights, []uint32{0, 1}, nil, nil, 0,
	)
	if err != nil {
		t.Fatal(err)
	}
	nodes, err := tensor.Topological(output.Output)
	if err != nil {
		t.Fatal(err)
	}
	var attentionCount, neoxCount int
	for _, node := range nodes {
		switch node.Op {
		case tensor.OpAttention:
			attentionCount++
			attributes := node.Attrs.(tensor.AttentionAttributes)
			if attributes.Softcap != 50 || attributes.Window != 4096 || attributes.Scale != 1 {
				t.Fatalf("unexpected Gemma 2 attention attributes: %+v", attributes)
			}
		case tensor.OpRoPENeoX:
			neoxCount++
			attributes := node.Attrs.(tensor.RoPEAttributes)
			if attributes.FrequencyScale != 0.25 {
				t.Fatalf("Gemma 2 sliding RoPE scale = %v, want 0.25", attributes.FrequencyScale)
			}
		}
	}
	if attentionCount != 1 || neoxCount != 2 {
		t.Fatalf("Gemma 2 graph has attention=%d NeoX RoPE=%d, want 1/2", attentionCount, neoxCount)
	}
}

func TestBuildDenseGemmaBlockUsesScaledNeoXGEGLU(t *testing.T) {
	builder := tensor.NewBuilder()
	spec := Spec{CommonSpec: CommonSpec{Architecture: "gemma",
		BlockCount:        18,
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
	output, err := BuildDenseBlock(builder, input, spec, weights, []uint32{0, 1})
	if err != nil {
		t.Fatal(err)
	}
	nodes, err := tensor.Topological(output)
	if err != nil {
		t.Fatal(err)
	}
	var neoxCount, scaleCount, geluCount int
	for _, node := range nodes {
		switch node.Op {
		case tensor.OpRoPENeoX:
			neoxCount++
		case tensor.OpScale:
			scaleCount++
			if value := node.Attrs.(tensor.ScaleAttributes).Value; value != 0.5 {
				t.Fatalf("Gemma query scale = %v, want 0.5", value)
			}
		case tensor.OpGELU:
			geluCount++
		}
	}
	if neoxCount != 2 || scaleCount != 1 || geluCount != 1 {
		t.Fatalf(
			"Gemma graph has NeoX=%d scale=%d GELU=%d, want 2/1/1",
			neoxCount,
			scaleCount,
			geluCount,
		)
	}
}

func TestBuildDenseBlockConsumesProjectionBiases(t *testing.T) {
	builder := tensor.NewBuilder()
	spec := Spec{CommonSpec: CommonSpec{Architecture: "llama",
		EmbeddingLength:   8,
		FeedForwardLength: 12,

		RMSNormEpsilon: 1e-5}, AttentionSpec: AttentionSpec{HeadCount: 2,
		HeadCountKV:       1,
		KeyLength:         4,
		ValueLength:       4,
		RopeFrequencyBase: 10000},
	}
	input := builder.Input("input", dtype.F32, tensor.MustShape(8, 2))
	weights := denseBlockInputs(builder, spec)
	weights.AttentionQNorm = nil
	weights.AttentionKNorm = nil
	weights.AttentionQBias = builder.Input("attn_q.bias", dtype.F32, tensor.MustShape(8))
	weights.AttentionKBias = builder.Input("attn_k.bias", dtype.F32, tensor.MustShape(4))
	weights.AttentionVBias = builder.Input("attn_v.bias", dtype.F32, tensor.MustShape(4))
	weights.AttentionOutputBias = builder.Input("attn_output.bias", dtype.F32, tensor.MustShape(8))
	weights.FeedForwardGateBias = builder.Input("ffn_gate.bias", dtype.F32, tensor.MustShape(12))
	weights.FeedForwardUpBias = builder.Input("ffn_up.bias", dtype.F32, tensor.MustShape(12))
	weights.FeedForwardDownBias = builder.Input("ffn_down.bias", dtype.F32, tensor.MustShape(8))
	output, err := BuildDenseBlock(builder, input, spec, weights, []uint32{0, 1})
	if err != nil {
		t.Fatal(err)
	}
	nodes, err := tensor.Topological(output)
	if err != nil {
		t.Fatal(err)
	}
	found := make(map[string]bool)
	for _, node := range nodes {
		found[node.Name] = true
	}
	for _, name := range []string{
		"attn_q.bias",
		"attn_k.bias",
		"attn_v.bias",
		"attn_output.bias",
		"ffn_gate.bias",
		"ffn_up.bias",
		"ffn_down.bias",
	} {
		if !found[name] {
			t.Fatalf("projection bias %q is not connected to the graph", name)
		}
	}
}

func TestBuildGPT2DenseBlockUsesLearnedPositionsWithoutRoPE(t *testing.T) {
	builder := tensor.NewBuilder()
	spec := Spec{CommonSpec: CommonSpec{Architecture: "gpt2", EmbeddingLength: 8, FeedForwardLength: 16,

		LayerNormEpsilon: 1e-5}, AttentionSpec: AttentionSpec{HeadCount: 2, HeadCountKV: 2, KeyLength: 4, ValueLength: 4,
		RopeDisabled: true},
	}
	input := builder.Input("input", dtype.F32, tensor.MustShape(8, 2))
	weights := denseBlockInputs(builder, spec)
	weights.AttentionQ = nil
	weights.AttentionK = nil
	weights.AttentionV = nil
	weights.AttentionQNorm = nil
	weights.AttentionKNorm = nil
	weights.FeedForwardGate = nil
	weights.AttentionQKV = builder.Input("attn_qkv", dtype.F32, tensor.MustShape(8, 24))
	weights.AttentionQKVBias = builder.Input("attn_qkv.bias", dtype.F32, tensor.MustShape(24))
	weights.AttentionOutputBias = builder.Input("attn_output.bias", dtype.F32, tensor.MustShape(8))
	weights.FeedForwardUpBias = builder.Input("ffn_up.bias", dtype.F32, tensor.MustShape(16))
	weights.FeedForwardDownBias = builder.Input("ffn_down.bias", dtype.F32, tensor.MustShape(8))
	output, err := BuildDenseBlock(builder, input, spec, weights, []uint32{3, 4})
	if err != nil {
		t.Fatal(err)
	}
	nodes, err := tensor.Topological(output)
	if err != nil {
		t.Fatal(err)
	}
	var rope, gelu int
	for _, node := range nodes {
		switch node.Op {
		case tensor.OpRoPENormal, tensor.OpRoPENeoX:
			rope++
		case tensor.OpGELU:
			gelu++
		}
	}
	if rope != 0 || gelu != 1 {
		t.Fatalf("GPT-2 graph has RoPE=%d GELU=%d, want 0/1", rope, gelu)
	}
}

func TestBuildBloomDenseBlockUsesALiBiWithoutRoPE(t *testing.T) {
	builder := tensor.NewBuilder()
	spec := Spec{CommonSpec: CommonSpec{Architecture: "bloom", EmbeddingLength: 8, FeedForwardLength: 16,

		LayerNormEpsilon: 1e-5}, AttentionSpec: AttentionSpec{HeadCount: 2, HeadCountKV: 2, KeyLength: 4, ValueLength: 4,
		RopeDisabled: true, MaxALiBiBias: 8},
	}
	input := builder.Input("input", dtype.F32, tensor.MustShape(8, 2))
	weights := denseBlockInputs(builder, spec)
	weights.AttentionQ = nil
	weights.AttentionK = nil
	weights.AttentionV = nil
	weights.AttentionQNorm = nil
	weights.AttentionKNorm = nil
	weights.FeedForwardGate = nil
	weights.AttentionQKV = builder.Input("attn_qkv", dtype.F32, tensor.MustShape(8, 24))
	weights.AttentionQKVBias = builder.Input("attn_qkv.bias", dtype.F32, tensor.MustShape(24))
	weights.AttentionOutputBias = builder.Input("attn_output.bias", dtype.F32, tensor.MustShape(8))
	weights.FeedForwardUpBias = builder.Input("ffn_up.bias", dtype.F32, tensor.MustShape(16))
	weights.FeedForwardDownBias = builder.Input("ffn_down.bias", dtype.F32, tensor.MustShape(8))
	output, err := BuildDenseBlock(builder, input, spec, weights, []uint32{0, 1})
	if err != nil {
		t.Fatal(err)
	}
	nodes, err := tensor.Topological(output)
	if err != nil {
		t.Fatal(err)
	}
	var attentionCount int
	for _, node := range nodes {
		if node.Op == tensor.OpRoPENormal || node.Op == tensor.OpRoPENeoX {
			t.Fatal("Bloom graph unexpectedly contains RoPE")
		}
		if node.Op == tensor.OpAttention {
			attentionCount++
			attributes := node.Attrs.(tensor.AttentionAttributes)
			if attributes.MaxALiBiBias != 8 || !attributes.Causal {
				t.Fatalf("unexpected Bloom attention attributes: %+v", attributes)
			}
		}
	}
	if attentionCount != 1 {
		t.Fatalf("Bloom graph has %d attention nodes, want 1", attentionCount)
	}
}

func TestBuildMPTBiasFreeBlockUsesALiBi(t *testing.T) {
	builder := tensor.NewBuilder()
	spec := Spec{CommonSpec: CommonSpec{Architecture: "mpt", EmbeddingLength: 8, FeedForwardLength: 16,

		LayerNormEpsilon: 1e-5}, AttentionSpec: AttentionSpec{HeadCount: 2, HeadCountKV: 2, KeyLength: 4, ValueLength: 4,
		RopeDisabled: true, MaxALiBiBias: 8, AttentionClamp: 2},
	}
	input := builder.Input("input", dtype.F32, tensor.MustShape(8, 2))
	weights := denseBlockInputs(builder, spec)
	weights.AttentionNormBias = nil
	weights.FeedForwardNormBias = nil
	weights.AttentionQ = nil
	weights.AttentionK = nil
	weights.AttentionV = nil
	weights.AttentionQNorm = nil
	weights.AttentionKNorm = nil
	weights.FeedForwardGate = nil
	weights.AttentionQKV = builder.Input("attn_qkv", dtype.F32, tensor.MustShape(8, 24))
	output, err := BuildDenseBlock(builder, input, spec, weights, []uint32{0, 1})
	if err != nil {
		t.Fatal(err)
	}
	nodes, err := tensor.Topological(output)
	if err != nil {
		t.Fatal(err)
	}
	var gelu, alibi, clamp int
	for _, node := range nodes {
		if node.Op == tensor.OpGELU {
			gelu++
		}
		if node.Op == tensor.OpAttention &&
			node.Attrs.(tensor.AttentionAttributes).MaxALiBiBias == 8 {
			alibi++
		}
		if node.Op == tensor.OpClamp &&
			node.Attrs.(tensor.ClampAttributes) ==
				(tensor.ClampAttributes{Minimum: -2, Maximum: 2}) {
			clamp++
		}
	}
	if gelu != 1 || alibi != 1 || clamp != 1 {
		t.Fatalf("MPT graph has GELU=%d ALiBi=%d clamp=%d, want 1/1/1", gelu, alibi, clamp)
	}
}

func TestBuildMPTQKLayerNormUsesFullProjections(t *testing.T) {
	builder := tensor.NewBuilder()
	spec := Spec{CommonSpec: CommonSpec{Architecture: "mpt", EmbeddingLength: 8, FeedForwardLength: 16,

		LayerNormEpsilon: 1e-5}, AttentionSpec: AttentionSpec{HeadCount: 2, HeadCountKV: 2, KeyLength: 4, ValueLength: 4,
		RopeDisabled: true, MaxALiBiBias: 8},
	}
	input := builder.Input("input", dtype.F32, tensor.MustShape(8, 2))
	weights := denseBlockInputs(builder, spec)
	weights.AttentionNormBias = nil
	weights.FeedForwardNormBias = nil
	weights.AttentionQ = nil
	weights.AttentionK = nil
	weights.AttentionV = nil
	weights.FeedForwardGate = nil
	weights.AttentionQKV = builder.Input("attn_qkv", dtype.F32, tensor.MustShape(8, 24))
	weights.AttentionQNorm = builder.Input("attn_q_norm", dtype.F32, tensor.MustShape(8))
	weights.AttentionKNorm = builder.Input("attn_k_norm", dtype.F32, tensor.MustShape(8))
	weights.AttentionQNormBias = builder.Input("attn_q_norm_bias", dtype.F32, tensor.MustShape(8))
	weights.AttentionKNormBias = builder.Input("attn_k_norm_bias", dtype.F32, tensor.MustShape(8))
	weights.FeedForwardActivationScale = builder.Input("ffn_act_scales", dtype.F32, tensor.MustShape(16))
	output, err := BuildDenseBlock(builder, input, spec, weights, []uint32{0, 1})
	if err != nil {
		t.Fatal(err)
	}
	nodes, err := tensor.Topological(output)
	if err != nil {
		t.Fatal(err)
	}
	var fullProjectionNorms, activationDivides int
	for _, node := range nodes {
		if node.Op == tensor.OpLayerNorm && node.Shape.Equal(tensor.MustShape(8, 2)) {
			fullProjectionNorms++
		}
		if node.Op == tensor.OpDivide && len(node.Inputs) == 2 &&
			node.Inputs[0].Op == tensor.OpGELU && node.Inputs[1] == weights.FeedForwardActivationScale {
			activationDivides++
		}
	}
	if fullProjectionNorms != 4 || activationDivides != 1 {
		t.Fatalf(
			"MPT full-width LayerNorm/divide count = %d/%d, want 4/1",
			fullProjectionNorms, activationDivides,
		)
	}
}

func TestBuildOLMoClampsSeparateQKV(t *testing.T) {
	builder := tensor.NewBuilder()
	spec := Spec{CommonSpec: CommonSpec{Architecture: "olmo", EmbeddingLength: 8, FeedForwardLength: 16,

		LayerNormEpsilon: 1e-5}, AttentionSpec: AttentionSpec{HeadCount: 2, HeadCountKV: 1, KeyLength: 4, ValueLength: 4,
		RopeFrequencyBase: 10000, AttentionClamp: 3},
	}
	input := builder.Input("input", dtype.F32, tensor.MustShape(8, 2))
	weights := denseBlockInputs(builder, spec)
	weights.AttentionNorm = nil
	weights.AttentionNormBias = nil
	weights.FeedForwardNorm = nil
	weights.FeedForwardNormBias = nil
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
	var clamp int
	for _, node := range nodes {
		if node.Op == tensor.OpClamp &&
			node.Attrs.(tensor.ClampAttributes) ==
				(tensor.ClampAttributes{Minimum: -3, Maximum: 3}) {
			clamp++
		}
	}
	if clamp != 3 {
		t.Fatalf("OLMo graph has %d QKV clamps, want 3", clamp)
	}
}

func TestBuildQwen35AttentionBlock(t *testing.T) {
	builder := tensor.NewBuilder()
	spec := qwen35TestSpec()
	input := builder.Input("input", dtype.F32, tensor.MustShape(8, 2))
	result, err := BuildQwen35BlockWithOptions(Qwen35BlockOptions{
		Builder: builder, Input: input, Spec: spec, Weights: qwen35AttentionInputs(builder, spec),
		Positions: []uint32{0, 1}, Sequences: 1, CacheWrite: tensor.CacheWriteConcat,
	})
	if err != nil {
		t.Fatal(err)
	}
	if !result.Output.Shape.Equal(input.Shape) ||
		!result.Key.Shape.Equal(tensor.MustShape(4, 1, 2)) ||
		!result.Value.Shape.Equal(tensor.MustShape(4, 1, 2)) {
		t.Fatalf("unexpected Qwen3.5 attention result: %+v", result)
	}
	nodes, err := tensor.Topological(result.Output, result.Key, result.Value)
	if err != nil {
		t.Fatal(err)
	}
	var groupSlices, multiRoPE, sigmoid int
	for _, node := range nodes {
		switch node.Op {
		case tensor.OpGroupSlice:
			groupSlices++
		case tensor.OpRoPEMulti:
			multiRoPE++
		case tensor.OpSigmoid:
			sigmoid++
		}
	}
	if groupSlices != 2 || multiRoPE != 2 || sigmoid != 1 {
		t.Fatalf(
			"Qwen3.5 attention ops: group slices=%d RoPE=%d sigmoid=%d",
			groupSlices,
			multiRoPE,
			sigmoid,
		)
	}
}

func TestBuildQwen35PackedAttentionBlock(t *testing.T) {
	const (
		embeddingWidth = 8
		keyWidth       = 4
		keyHeads       = 1
		pastTokens     = 3
		newTokens      = 1
		sequenceCount  = 2
		position       = pastTokens
	)
	builder := tensor.NewBuilder()
	spec := qwen35TestSpec()
	input := builder.Input("input", dtype.F32, tensor.MustShape(embeddingWidth, sequenceCount))
	pastKey := builder.Input(
		"past_key", dtype.F32, tensor.MustShape(keyWidth, keyHeads, pastTokens, sequenceCount),
	)
	pastValue := builder.Input(
		"past_value", dtype.F32, tensor.MustShape(keyWidth, keyHeads, pastTokens, sequenceCount),
	)
	result, err := BuildQwen35BlockWithOptions(Qwen35BlockOptions{
		Builder: builder, Input: input, Spec: spec, Weights: qwen35AttentionInputs(builder, spec),
		Positions: []uint32{position}, Sequences: sequenceCount,
		PastKey: pastKey, PastValue: pastValue, CacheWrite: tensor.CacheWriteConcat,
	})
	if err != nil {
		t.Fatal(err)
	}
	wantCache := tensor.MustShape(keyWidth, keyHeads, pastTokens+newTokens, sequenceCount)
	if !result.Output.Shape.Equal(input.Shape) ||
		!result.Key.Shape.Equal(wantCache) || !result.Value.Shape.Equal(wantCache) {
		t.Fatalf("unexpected packed Qwen3.5 attention result: %+v", result)
	}
}

func TestBuildQwen35AttentionBlockUsesDistinctMRoPEPositions(t *testing.T) {
	builder := tensor.NewBuilder()
	spec := qwen35TestSpec()
	input := builder.Input("input", dtype.F32, tensor.MustShape(8, 2))
	positions := [4][]uint32{{10, 11}, {20, 21}, {30, 31}, {40, 41}}
	result, err := BuildQwen35BlockWithOptions(Qwen35BlockOptions{
		Builder: builder, Input: input, Spec: spec, Weights: qwen35AttentionInputs(builder, spec),
		Positions: positions[0], MultiPositions: &positions, Sequences: 1,
		CacheWrite: tensor.CacheWriteConcat,
	})
	if err != nil {
		t.Fatal(err)
	}
	nodes, err := tensor.Topological(result.Output)
	if err != nil {
		t.Fatal(err)
	}
	count := 0
	for _, node := range nodes {
		if node.Op != tensor.OpRoPEMulti {
			continue
		}
		count++
		attrs := node.Attrs.(tensor.RoPEMultiAttributes)
		for axis := range positions {
			if !slices.Equal(attrs.Positions[axis], positions[axis]) {
				t.Fatalf("Qwen3.5 MRoPE axis %d = %v, want %v", axis, attrs.Positions[axis], positions[axis])
			}
		}
	}
	if count != 2 {
		t.Fatalf("Qwen3.5 MRoPE nodes = %d, want 2", count)
	}
}
