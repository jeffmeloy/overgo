package model

import (
	"fmt"
	"math"
	"strings"
	"testing"

	"llamacpp2go/internal/tensor"
	"llamacpp2go/internal/tensor/dtype"
)

func TestBuildDenseQwen3Block(t *testing.T) {
	builder := tensor.NewBuilder()
	spec := Spec{
		Architecture:      "qwen3",
		EmbeddingLength:   8,
		FeedForwardLength: 12,
		HeadCount:         2,
		HeadCountKV:       1,
		KeyLength:         4,
		ValueLength:       4,
		RopeFrequencyBase: 1_000_000,
		RMSNormEpsilon:    1e-6,
	}
	input := builder.Input("input", dtype.F32, tensor.MustShape(8, 3))
	weights := denseBlockInputs(builder, spec)
	output, err := BuildDenseBlock(builder, input, spec, weights, []uint32{0, 1, 2})
	if err != nil {
		t.Fatal(err)
	}
	if !output.Shape.Equal(input.Shape) {
		t.Fatalf("output shape = %v, want %v", output.Shape.Slice(), input.Shape.Slice())
	}
	nodes, err := tensor.Topological(output)
	if err != nil {
		t.Fatal(err)
	}
	var foundAttention bool
	for _, node := range nodes {
		foundAttention = foundAttention || node.Op == tensor.OpAttention
	}
	if !foundAttention {
		t.Fatal("dense block graph has no attention operation")
	}
}

func TestBuildDeciSparseLayerModes(t *testing.T) {
	spec := Spec{
		Architecture: "deci", BlockCount: 4, EmbeddingLength: 8,
		FeedForwardLength: 12, LayerFeedForward: []uint32{12, 12, 12, 0},
		HeadCount: 2, HeadCountKV: 1, LayerHeadCounts: []uint32{2, 2, 0, 0},
		LayerKVHeadCounts: []uint32{1, 0, 0, 0}, KeyLength: 4, ValueLength: 4,
		RopeDimensionCount: 4, RopeFrequencyBase: 10000, RMSNormEpsilon: 1e-6,
	}
	for _, test := range []struct {
		name       string
		layer      uint32
		mulMatWant int
		dummy      bool
	}{
		{name: "linear", layer: 1, mulMatWant: 4},
		{name: "attention-free", layer: 2, mulMatWant: 3},
		{name: "dummy", layer: 3, dummy: true},
	} {
		t.Run(test.name, func(t *testing.T) {
			builder := tensor.NewBuilder()
			input := builder.Input("input", dtype.F32, tensor.MustShape(8, 3))
			weights := LayerGraphWeights{}
			if !test.dummy {
				weights.FeedForwardNorm = builder.Input("ffn_norm", dtype.F32, tensor.MustShape(8))
				weights.FeedForwardGate = builder.Input("ffn_gate", dtype.F32, tensor.MustShape(8, 12))
				weights.FeedForwardUp = builder.Input("ffn_up", dtype.F32, tensor.MustShape(8, 12))
				weights.FeedForwardDown = builder.Input("ffn_down", dtype.F32, tensor.MustShape(12, 8))
			}
			if test.layer == 1 {
				weights.AttentionNorm = builder.Input("attn_norm", dtype.F32, tensor.MustShape(8))
				weights.AttentionOutput = builder.Input("attn_output", dtype.F32, tensor.MustShape(8, 8))
			}
			pastKey := builder.Input("past_key", dtype.F32, tensor.MustShape(1, 1, 2))
			pastValue := builder.Input("past_value", dtype.F32, tensor.MustShape(1, 1, 2))
			result, err := BuildDenseBlockCachedForLayer(
				builder, input, spec, weights, []uint32{2, 3, 4}, pastKey, pastValue, test.layer,
			)
			if err != nil {
				t.Fatal(err)
			}
			if test.dummy && result.Output != input {
				t.Fatal("dummy Deci layer changed activation")
			}
			if !result.Key.Shape.Equal(tensor.MustShape(1, 1, 5)) ||
				!result.Value.Shape.Equal(tensor.MustShape(1, 1, 5)) {
				t.Fatalf("sentinel cache shapes = %v/%v", result.Key.Shape.Slice(), result.Value.Shape.Slice())
			}
			nodes, err := tensor.Topological(result.Output, result.Key, result.Value)
			if err != nil {
				t.Fatal(err)
			}
			var mulMats int
			for _, node := range nodes {
				if node.Op == tensor.OpAttention {
					t.Fatal("sparse Deci layer built attention")
				}
				if node.Op == tensor.OpMulMat {
					mulMats++
				}
			}
			if mulMats != test.mulMatWant {
				t.Fatalf("MulMat count = %d, want %d", mulMats, test.mulMatWant)
			}
		})
	}
}

func TestBuildDenseQwen3MoEBlock(t *testing.T) {
	builder := tensor.NewBuilder()
	spec := Spec{
		Architecture:       "qwen3moe",
		EmbeddingLength:    8,
		FeedForwardLength:  24,
		ExpertCount:        4,
		ExpertUsedCount:    2,
		ExpertFeedForward:  12,
		ExpertWeightsScale: 1.25,
		HeadCount:          2,
		HeadCountKV:        1,
		KeyLength:          4,
		ValueLength:        4,
		RopeFrequencyBase:  1_000_000,
		RMSNormEpsilon:     1e-6,
	}
	input := builder.Input("input", dtype.F32, tensor.MustShape(8, 3))
	weights := denseBlockInputs(builder, spec)
	weights.FeedForwardGate = nil
	weights.FeedForwardUp = nil
	weights.FeedForwardDown = nil
	weights.FeedForwardRouter = builder.Input("ffn_router", dtype.F32, tensor.MustShape(8, 4))
	weights.FeedForwardGateExperts = builder.Input("ffn_gate_exps", dtype.F32, tensor.MustShape(8, 12, 4))
	weights.FeedForwardUpExperts = builder.Input("ffn_up_exps", dtype.F32, tensor.MustShape(8, 12, 4))
	weights.FeedForwardDownExperts = builder.Input("ffn_down_exps", dtype.F32, tensor.MustShape(12, 8, 4))
	output, err := BuildDenseBlock(builder, input, spec, weights, []uint32{0, 1, 2})
	if err != nil {
		t.Fatal(err)
	}
	nodes, err := tensor.Topological(output)
	if err != nil {
		t.Fatal(err)
	}
	var moe *tensor.Tensor
	for _, node := range nodes {
		if node.Op == tensor.OpMoE {
			moe = node
		}
	}
	if moe == nil {
		t.Fatalf("Qwen3-MoE graph is missing configured MoE operation: %+v", moe)
	}
	attributes := moe.Attrs.(tensor.MoEAttributes)
	if attributes.TopK != 2 || attributes.Scale != 1.25 || !attributes.NormalizeTopKProb {
		t.Fatalf("unexpected Qwen3-MoE attributes: %+v", attributes)
	}
}

func TestBuildDBRXBlockUsesClampedFusedQKVAndNormalizedMoE(t *testing.T) {
	builder := tensor.NewBuilder()
	spec := Spec{
		Architecture: "dbrx", EmbeddingLength: 8, FeedForwardLength: 6,
		ExpertCount: 4, ExpertUsedCount: 2, ExpertFeedForward: 6,
		ExpertWeightsScale: 1.25, ExpertWeightsNorm: true,
		HeadCount: 2, HeadCountKV: 1, KeyLength: 4, ValueLength: 4,
		AttentionClamp: 2, RopeFrequencyBase: 10000, LayerNormEpsilon: 1e-5,
	}
	input := builder.Input("input", dtype.F32, tensor.MustShape(8, 2))
	weights := denseBlockInputs(builder, spec)
	weights.AttentionQ, weights.AttentionK, weights.AttentionV = nil, nil, nil
	weights.AttentionQKV = builder.Input("qkv", dtype.F32, tensor.MustShape(8, 16))
	weights.FeedForwardGate, weights.FeedForwardUp, weights.FeedForwardDown = nil, nil, nil
	weights.FeedForwardRouter = builder.Input("router", dtype.F32, tensor.MustShape(8, 4))
	weights.FeedForwardGateExperts = builder.Input("gate_exps", dtype.F32, tensor.MustShape(8, 6, 4))
	weights.FeedForwardUpExperts = builder.Input("up_exps", dtype.F32, tensor.MustShape(8, 6, 4))
	weights.FeedForwardDownExperts = builder.Input("down_exps", dtype.F32, tensor.MustShape(6, 8, 4))
	result, err := BuildDenseBlockCachedForLayer(builder, input, spec, weights, []uint32{0, 1}, nil, nil, 0)
	if err != nil {
		t.Fatal(err)
	}
	nodes, err := tensor.Topological(result.Output)
	if err != nil {
		t.Fatal(err)
	}
	var clamp, moe *tensor.Tensor
	var rope, layerNorm int
	for _, node := range nodes {
		switch node.Op {
		case tensor.OpClamp:
			clamp = node
		case tensor.OpMoE:
			moe = node
		case tensor.OpRoPENeoX:
			rope++
		case tensor.OpLayerNorm:
			layerNorm++
		}
	}
	if clamp == nil || clamp.Attrs.(tensor.ClampAttributes) != (tensor.ClampAttributes{Minimum: -2, Maximum: 2}) ||
		moe == nil || !moe.Attrs.(tensor.MoEAttributes).NormalizeTopKProb || rope != 2 || layerNorm != 2 {
		t.Fatalf("unexpected DBRX graph: clamp=%+v moe=%+v rope=%d layer_norm=%d", clamp, moe, rope, layerNorm)
	}
}

func TestBuildGrokBlockUsesYaRNPostNormAndGELUMoE(t *testing.T) {
	builder := tensor.NewBuilder()
	spec := Spec{
		Architecture: "grok", BlockCount: 1, EmbeddingLength: 8, FeedForwardLength: 12,
		ExpertCount: 4, ExpertUsedCount: 2, ExpertFeedForward: 6,
		ExpertWeightsScale: 1.25, ExpertWeightsNorm: true,
		HeadCount: 2, HeadCountKV: 1, KeyLength: 4, ValueLength: 4,
		RopeDimensionCount: 4, RopeFrequencyBase: 10000, RopeScalingType: "yarn",
		RopeScalingFactor: 4, OriginalContextLength: 2048, YaRNExtFactor: 1,
		YaRNAttentionFactor: 1.25, YaRNBetaFast: 8, YaRNBetaSlow: 1,
		AttentionScale: 0.25, AttentionSoftcap: 30, RMSNormEpsilon: 1e-6,
	}
	input := builder.Input("input", dtype.F32, tensor.MustShape(8, 2))
	weights := LayerGraphWeights{
		AttentionNorm:          builder.Input("attn_norm", dtype.F32, tensor.MustShape(8)),
		AttentionQKV:           builder.Input("qkv", dtype.F32, tensor.MustShape(8, 16)),
		AttentionOutput:        builder.Input("attn_out", dtype.F32, tensor.MustShape(8, 8)),
		AttentionPostNorm:      builder.Input("attn_post_norm", dtype.F32, tensor.MustShape(8)),
		FeedForwardNorm:        builder.Input("ffn_norm", dtype.F32, tensor.MustShape(8)),
		FeedForwardRouter:      builder.Input("router", dtype.F32, tensor.MustShape(8, 4)),
		FeedForwardGateExperts: builder.Input("gate_exps", dtype.F32, tensor.MustShape(8, 6, 4)),
		FeedForwardUpExperts:   builder.Input("up_exps", dtype.F32, tensor.MustShape(8, 6, 4)),
		FeedForwardDownExperts: builder.Input("down_exps", dtype.F32, tensor.MustShape(6, 8, 4)),
		FeedForwardGate:        builder.Input("dense_gate", dtype.F32, tensor.MustShape(8, 12)),
		FeedForwardUp:          builder.Input("dense_up", dtype.F32, tensor.MustShape(8, 12)),
		FeedForwardDown:        builder.Input("dense_down", dtype.F32, tensor.MustShape(12, 8)),
		FeedForwardPostNorm:    builder.Input("ffn_post_norm", dtype.F32, tensor.MustShape(8)),
	}
	result, err := BuildDenseBlockCachedForLayer(
		builder, input, spec, weights, []uint32{0, 1}, nil, nil, 0,
	)
	if err != nil {
		t.Fatal(err)
	}
	if !result.Output.Shape.Equal(input.Shape) ||
		!result.Key.Shape.Equal(tensor.MustShape(4, 1, 2)) ||
		!result.Value.Shape.Equal(tensor.MustShape(4, 1, 2)) {
		t.Fatalf("unexpected Grok result: %+v", result)
	}
	nodes, err := tensor.Topological(result.Output, result.Key, result.Value)
	if err != nil {
		t.Fatal(err)
	}
	var moe *tensor.Tensor
	var yarn, gelu, rmsNorm int
	var attention tensor.AttentionAttributes
	for _, node := range nodes {
		switch node.Op {
		case tensor.OpMoE:
			moe = node
		case tensor.OpRoPENeoX:
			attrs := node.Attrs.(tensor.RoPEAttributes)
			if attrs.OriginalContext == 2048 && attrs.AttentionFactor == 1.25 &&
				attrs.BetaFast == 8 && attrs.BetaSlow == 1 {
				yarn++
			}
		case tensor.OpAttention:
			attention = node.Attrs.(tensor.AttentionAttributes)
		case tensor.OpGELU:
			gelu++
		case tensor.OpRMSNorm:
			rmsNorm++
		}
	}
	if moe == nil {
		t.Fatal("Grok graph is missing MoE")
	}
	attrs := moe.Attrs.(tensor.MoEAttributes)
	if attrs.Activation != tensor.MoEActivationGELU || !attrs.Gated ||
		!attrs.NormalizeTopKProb || attrs.TopK != 2 || attrs.Scale != 1.25 ||
		yarn != 2 || attention.Scale != 0.25 || attention.Softcap != 30 ||
		gelu != 1 || rmsNorm != 4 {
		t.Fatalf(
			"unexpected Grok graph: moe=%+v yarn=%d attention=%+v GELU=%d RMSNorm=%d",
			attrs, yarn, attention, gelu, rmsNorm,
		)
	}
}

func TestBuildMellumSlidingAndYaRNBlocks(t *testing.T) {
	for _, layer := range []uint32{0, 3} {
		t.Run(fmt.Sprintf("layer=%d", layer), func(t *testing.T) {
			builder := tensor.NewBuilder()
			spec := Spec{
				Architecture: "mellum", BlockCount: 4, EmbeddingLength: 8, FeedForwardLength: 12,
				ExpertCount: 4, ExpertUsedCount: 2, ExpertFeedForward: 6,
				ExpertWeightsScale: 1, ExpertWeightsNorm: true,
				HeadCount: 2, HeadCountKV: 1, KeyLength: 4, ValueLength: 4,
				RopeDimensionCount: 4, RopeFrequencyBase: 10000, RopeFrequencySWA: 20000,
				RopeScalingType: "yarn", RopeScalingFactor: 4, OriginalContextLength: 2048,
				YaRNExtFactor: 1, YaRNAttentionFactor: 1.25, YaRNBetaFast: 32, YaRNBetaSlow: 1,
				SlidingWindow: 128, SlidingPattern: 4, RMSNormEpsilon: 1e-6,
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
				FeedForwardNorm:        builder.Input("ffn_norm", dtype.F32, tensor.MustShape(8)),
				FeedForwardRouter:      builder.Input("router", dtype.F32, tensor.MustShape(8, 4)),
				FeedForwardGateExperts: builder.Input("gate_exps", dtype.F32, tensor.MustShape(8, 6, 4)),
				FeedForwardUpExperts:   builder.Input("up_exps", dtype.F32, tensor.MustShape(8, 6, 4)),
				FeedForwardDownExperts: builder.Input("down_exps", dtype.F32, tensor.MustShape(6, 8, 4)),
			}
			result, err := BuildDenseBlockCachedForLayer(builder, input, spec, weights, []uint32{0, 1}, nil, nil, layer)
			if err != nil {
				t.Fatal(err)
			}
			nodes, _ := tensor.Topological(result.Output)
			var attention tensor.AttentionAttributes
			var rope tensor.RoPEAttributes
			var moe tensor.MoEAttributes
			for _, node := range nodes {
				switch node.Op {
				case tensor.OpAttention:
					attention = node.Attrs.(tensor.AttentionAttributes)
				case tensor.OpRoPENeoX:
					rope = node.Attrs.(tensor.RoPEAttributes)
				case tensor.OpMoE:
					moe = node.Attrs.(tensor.MoEAttributes)
				}
			}
			if moe.TopK != 2 || !moe.NormalizeTopKProb || moe.Activation != tensor.MoEActivationSiLU {
				t.Fatalf("unexpected Mellum MoE: %+v", moe)
			}
			if layer == 0 && (attention.Window != 128 || rope.FrequencyBase != 20000 || rope.OriginalContext != 0) {
				t.Fatalf("unexpected Mellum sliding graph: attention=%+v rope=%+v", attention, rope)
			}
			if layer == 3 && (attention.Window != 0 || rope.OriginalContext != 2048 || rope.AttentionFactor != 1.25) {
				t.Fatalf("unexpected Mellum full graph: attention=%+v rope=%+v", attention, rope)
			}
		})
	}
}

func TestBuildHunyuanMoEBlock(t *testing.T) {
	b := tensor.NewBuilder()
	s := Spec{Architecture: "hunyuan-moe", EmbeddingLength: 8, FeedForwardLength: 12,
		ExpertCount: 4, ExpertUsedCount: 2, ExpertFeedForward: 6, SharedExpertFF: 10,
		ExpertWeightsScale: 1, ExpertWeightsNorm: true, HeadCount: 2, HeadCountKV: 1,
		KeyLength: 4, ValueLength: 4, RopeDimensionCount: 4, RopeFrequencyBase: 10000, RMSNormEpsilon: 1e-6}
	in := b.Input("input", dtype.F32, tensor.MustShape(8, 2))
	w := LayerGraphWeights{
		AttentionNorm: b.Input("an", dtype.F32, tensor.MustShape(8)), AttentionQ: b.Input("q", dtype.F32, tensor.MustShape(8, 8)), AttentionK: b.Input("k", dtype.F32, tensor.MustShape(8, 4)), AttentionV: b.Input("v", dtype.F32, tensor.MustShape(8, 4)), AttentionOutput: b.Input("o", dtype.F32, tensor.MustShape(8, 8)), AttentionQNorm: b.Input("qn", dtype.F32, tensor.MustShape(4)), AttentionKNorm: b.Input("kn", dtype.F32, tensor.MustShape(4)), FeedForwardNorm: b.Input("fn", dtype.F32, tensor.MustShape(8)),
		FeedForwardRouter: b.Input("r", dtype.F32, tensor.MustShape(8, 4)), FeedForwardGateExperts: b.Input("ge", dtype.F32, tensor.MustShape(8, 6, 4)), FeedForwardUpExperts: b.Input("ue", dtype.F32, tensor.MustShape(8, 6, 4)), FeedForwardDownExperts: b.Input("de", dtype.F32, tensor.MustShape(6, 8, 4)), FeedForwardSharedGate: b.Input("sg", dtype.F32, tensor.MustShape(8, 10)), FeedForwardSharedUp: b.Input("su", dtype.F32, tensor.MustShape(8, 10)), FeedForwardSharedDown: b.Input("sd", dtype.F32, tensor.MustShape(10, 8)),
	}
	r, err := BuildDenseBlockCachedForLayer(b, in, s, w, []uint32{0, 1}, nil, nil, 0)
	if err != nil {
		t.Fatal(err)
	}
	nodes, _ := tensor.Topological(r.Output)
	var moe, silu int
	for _, n := range nodes {
		if n.Op == tensor.OpMoE {
			moe++
		}
		if n.Op == tensor.OpSiLU {
			silu++
		}
	}
	if moe != 1 || silu != 1 {
		t.Fatalf("Hunyuan-MoE ops: MoE=%d SiLU=%d", moe, silu)
	}
}

func TestBuildQwenBlock(t *testing.T) {
	b := tensor.NewBuilder()
	s := Spec{Architecture: "qwen", EmbeddingLength: 8, FeedForwardLength: 12,
		HeadCount: 2, HeadCountKV: 2, KeyLength: 4, ValueLength: 4,
		RopeDimensionCount: 4, RopeFrequencyBase: 10000, RMSNormEpsilon: 1e-6}
	in := b.Input("input", dtype.F32, tensor.MustShape(8, 2))
	w := LayerGraphWeights{
		AttentionNorm:    b.Input("an", dtype.F32, tensor.MustShape(8)),
		AttentionQKV:     b.Input("qkv", dtype.F32, tensor.MustShape(8, 24)),
		AttentionQKVBias: b.Input("qkvb", dtype.F32, tensor.MustShape(24)),
		AttentionOutput:  b.Input("o", dtype.F32, tensor.MustShape(8, 8)),
		FeedForwardNorm:  b.Input("fn", dtype.F32, tensor.MustShape(8)),
		FeedForwardGate:  b.Input("fg", dtype.F32, tensor.MustShape(8, 12)),
		FeedForwardUp:    b.Input("fu", dtype.F32, tensor.MustShape(8, 12)),
		FeedForwardDown:  b.Input("fd", dtype.F32, tensor.MustShape(12, 8)),
	}
	r, err := BuildDenseBlockCachedForLayer(b, in, s, w, []uint32{0, 1}, nil, nil, 0)
	if err != nil {
		t.Fatal(err)
	}
	nodes, _ := tensor.Topological(r.Output)
	var slices, rope, silu int
	for _, n := range nodes {
		if n.Op == tensor.OpGroupSlice {
			slices++
		}
		if n.Op == tensor.OpRoPENeoX {
			rope++
		}
		if n.Op == tensor.OpSiLU {
			silu++
		}
	}
	if slices != 3 || rope != 2 || silu != 1 {
		t.Fatalf("Qwen ops: slices=%d rope=%d SiLU=%d", slices, rope, silu)
	}
}

func TestBuildChatGLMBlock(t *testing.T) {
	b := tensor.NewBuilder()
	s := Spec{Architecture: "chatglm", EmbeddingLength: 8, FeedForwardLength: 12, HeadCount: 2, HeadCountKV: 1, KeyLength: 4, ValueLength: 4, RopeDimensionCount: 4, RopeFrequencyBase: 10000, RMSNormEpsilon: 1e-6}
	in := b.Input("input", dtype.F32, tensor.MustShape(8, 2))
	w := LayerGraphWeights{AttentionNorm: b.Input("an", dtype.F32, tensor.MustShape(8)), AttentionQKV: b.Input("qkv", dtype.F32, tensor.MustShape(8, 16)), AttentionOutput: b.Input("o", dtype.F32, tensor.MustShape(8, 8)), FeedForwardNorm: b.Input("fn", dtype.F32, tensor.MustShape(8)), FeedForwardUp: b.Input("fu", dtype.F32, tensor.MustShape(8, 24)), FeedForwardDown: b.Input("fd", dtype.F32, tensor.MustShape(12, 8))}
	r, err := BuildDenseBlockCachedForLayer(b, in, s, w, []uint32{0, 1}, nil, nil, 0)
	if err != nil {
		t.Fatal(err)
	}
	nodes, _ := tensor.Topological(r.Output)
	var slices, rope, silu int
	for _, n := range nodes {
		if n.Op == tensor.OpGroupSlice {
			slices++
		}
		if n.Op == tensor.OpRoPENormal {
			rope++
		}
		if n.Op == tensor.OpSiLU {
			silu++
		}
	}
	if slices != 5 || rope != 2 || silu != 1 {
		t.Fatalf("ChatGLM ops: slices=%d rope=%d SiLU=%d", slices, rope, silu)
	}
}

func TestBuildHunyuanDenseBlock(t *testing.T) {
	b := tensor.NewBuilder()
	s := Spec{Architecture: "hunyuan-dense", EmbeddingLength: 8, FeedForwardLength: 12, HeadCount: 2, HeadCountKV: 1, KeyLength: 4, ValueLength: 4, RopeDimensionCount: 4, RopeFrequencyBase: 40000, RMSNormEpsilon: 1e-6}
	in := b.Input("input", dtype.F32, tensor.MustShape(8, 2))
	w := LayerGraphWeights{AttentionNorm: b.Input("an", dtype.F32, tensor.MustShape(8)), AttentionQKV: b.Input("qkv", dtype.F32, tensor.MustShape(8, 16)), AttentionOutput: b.Input("o", dtype.F32, tensor.MustShape(8, 8)), AttentionQNorm: b.Input("qn", dtype.F32, tensor.MustShape(4)), AttentionKNorm: b.Input("kn", dtype.F32, tensor.MustShape(4)), FeedForwardNorm: b.Input("fn", dtype.F32, tensor.MustShape(8)), FeedForwardGate: b.Input("fg", dtype.F32, tensor.MustShape(8, 12)), FeedForwardUp: b.Input("fu", dtype.F32, tensor.MustShape(8, 12)), FeedForwardDown: b.Input("fd", dtype.F32, tensor.MustShape(12, 8))}
	r, err := BuildDenseBlockCachedForLayer(b, in, s, w, []uint32{0, 1}, nil, nil, 0)
	if err != nil {
		t.Fatal(err)
	}
	nodes, _ := tensor.Topological(r.Output)
	var rope, rms, silu int
	for _, n := range nodes {
		if n.Op == tensor.OpRoPENormal {
			rope++
		}
		if n.Op == tensor.OpRMSNorm {
			rms++
		}
		if n.Op == tensor.OpSiLU {
			silu++
		}
	}
	if rope != 2 || rms != 4 || silu != 1 {
		t.Fatalf("Hunyuan-Dense ops: rope=%d RMS=%d SiLU=%d", rope, rms, silu)
	}
}

func TestBuildHunyuanVLBlockUsesPostMRoPEQKNorm(t *testing.T) {
	b := tensor.NewBuilder()
	s := Spec{Architecture: "hunyuan-vl", EmbeddingLength: 8, FeedForwardLength: 12, HeadCount: 2, HeadCountKV: 1, KeyLength: 4, ValueLength: 4, RopeDimensionCount: 4, RopeSections: [4]int32{1, 1, 0, 0}, RopeFrequencyBase: 40000, RMSNormEpsilon: 1e-6}
	in := b.Input("input", dtype.F32, tensor.MustShape(8, 2))
	w := LayerGraphWeights{AttentionNorm: b.Input("an", dtype.F32, tensor.MustShape(8)), AttentionQKV: b.Input("qkv", dtype.F32, tensor.MustShape(8, 16)), AttentionOutput: b.Input("o", dtype.F32, tensor.MustShape(8, 8)), AttentionQNorm: b.Input("qn", dtype.F32, tensor.MustShape(4)), AttentionKNorm: b.Input("kn", dtype.F32, tensor.MustShape(4)), FeedForwardNorm: b.Input("fn", dtype.F32, tensor.MustShape(8)), FeedForwardGate: b.Input("fg", dtype.F32, tensor.MustShape(8, 12)), FeedForwardUp: b.Input("fu", dtype.F32, tensor.MustShape(8, 12)), FeedForwardDown: b.Input("fd", dtype.F32, tensor.MustShape(12, 8))}
	r, err := BuildDenseBlockCachedForLayer(b, in, s, w, []uint32{0, 1}, nil, nil, 0)
	if err != nil {
		t.Fatal(err)
	}
	nodes, err := tensor.Topological(r.Output)
	if err != nil {
		t.Fatal(err)
	}
	var multiRoPE, postRoPENorm int
	for _, node := range nodes {
		if node.Op == tensor.OpRoPEMulti {
			multiRoPE++
		}
		if node.Op == tensor.OpRMSNorm && len(node.Inputs) == 1 && node.Inputs[0].Op == tensor.OpRoPEMulti {
			postRoPENorm++
		}
	}
	if multiRoPE != 2 || postRoPENorm != 2 {
		t.Fatalf("Hunyuan-VL graph has MRoPE=%d post-MRoPE norms=%d", multiRoPE, postRoPENorm)
	}
}

func TestBuildCogVLMTokenBlockUsesFusedQKVAndNormalRoPE(t *testing.T) {
	b := tensor.NewBuilder()
	s := Spec{Architecture: "cogvlm", EmbeddingLength: 8, FeedForwardLength: 12, HeadCount: 2, HeadCountKV: 2, KeyLength: 4, ValueLength: 4, RopeDimensionCount: 4, RopeFrequencyBase: 10000, RMSNormEpsilon: 1e-6}
	in := b.Input("input", dtype.F32, tensor.MustShape(8, 2))
	w := LayerGraphWeights{AttentionNorm: b.Input("an", dtype.F32, tensor.MustShape(8)), AttentionQKV: b.Input("qkv", dtype.F32, tensor.MustShape(8, 24)), AttentionOutput: b.Input("o", dtype.F32, tensor.MustShape(8, 8)), FeedForwardNorm: b.Input("fn", dtype.F32, tensor.MustShape(8)), FeedForwardGate: b.Input("fg", dtype.F32, tensor.MustShape(8, 12)), FeedForwardUp: b.Input("fu", dtype.F32, tensor.MustShape(8, 12)), FeedForwardDown: b.Input("fd", dtype.F32, tensor.MustShape(12, 8))}
	r, err := BuildDenseBlockCachedForLayer(b, in, s, w, []uint32{0, 1}, nil, nil, 0)
	if err != nil {
		t.Fatal(err)
	}
	nodes, err := tensor.Topological(r.Output)
	if err != nil {
		t.Fatal(err)
	}
	var rope, rms, silu int
	for _, node := range nodes {
		switch node.Op {
		case tensor.OpRoPENormal:
			rope++
		case tensor.OpRMSNorm:
			rms++
		case tensor.OpSiLU:
			silu++
		}
	}
	if rope != 2 || rms != 2 || silu != 1 {
		t.Fatalf("CogVLM token graph has RoPE=%d RMS=%d SiLU=%d", rope, rms, silu)
	}
}

func TestBuildArcticParallelDenseAndMoEBlock(t *testing.T) {
	builder := tensor.NewBuilder()
	spec := Spec{
		Architecture: "arctic", EmbeddingLength: 8, FeedForwardLength: 12,
		ExpertCount: 4, ExpertUsedCount: 2, ExpertFeedForward: 12,
		ExpertWeightsScale: 1, HeadCount: 2, HeadCountKV: 1,
		KeyLength: 4, ValueLength: 4, RopeFrequencyBase: 10000, RMSNormEpsilon: 1e-5,
	}
	input := builder.Input("input", dtype.F32, tensor.MustShape(8, 2))
	weights := denseBlockInputs(builder, spec)
	weights.FeedForwardGate = builder.Input("dense_gate", dtype.F32, tensor.MustShape(8, 8))
	weights.FeedForwardUp = builder.Input("dense_up", dtype.F32, tensor.MustShape(8, 8))
	weights.FeedForwardDown = builder.Input("dense_down", dtype.F32, tensor.MustShape(8, 8))
	weights.FeedForwardExpertNorm = builder.Input("expert_norm", dtype.F32, tensor.MustShape(8))
	weights.FeedForwardRouter = builder.Input("router", dtype.F32, tensor.MustShape(8, 4))
	weights.FeedForwardGateExperts = builder.Input("gate_exps", dtype.F32, tensor.MustShape(8, 12, 4))
	weights.FeedForwardUpExperts = builder.Input("up_exps", dtype.F32, tensor.MustShape(8, 12, 4))
	weights.FeedForwardDownExperts = builder.Input("down_exps", dtype.F32, tensor.MustShape(12, 8, 4))
	result, err := BuildDenseBlockCachedForLayer(builder, input, spec, weights, []uint32{0, 1}, nil, nil, 0)
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
		case tensor.OpRoPENormal:
			rope++
		case tensor.OpSiLU:
			silu++
		}
	}
	if moe == nil {
		t.Fatal("Arctic graph is missing MoE operation")
	}
	attrs := moe.Attrs.(tensor.MoEAttributes)
	if attrs.Routing != tensor.MoERoutingSoftmax || !attrs.NormalizeTopKProb ||
		attrs.TopK != 2 || attrs.Scale != 1 || rope != 2 || silu != 1 {
		t.Fatalf("unexpected Arctic graph: attrs=%+v rope=%d silu=%d", attrs, rope, silu)
	}
	if len(moe.Inputs) == 0 || len(moe.Inputs[0].Inputs) == 0 ||
		len(moe.Inputs[0].Inputs[0].Inputs) == 0 || moe.Inputs[0].Inputs[0].Inputs[0] != input {
		t.Fatal("Arctic expert norm is not rooted at pre-attention input")
	}
}

func TestBuildOpenELMBlockUsesPerLayerHeadsAndQKNorm(t *testing.T) {
	builder := tensor.NewBuilder()
	spec := Spec{
		Architecture: "openelm", BlockCount: 2, EmbeddingLength: 8,
		FeedForwardLength: 12, LayerFeedForward: []uint32{12, 16},
		HeadCount: 2, HeadCountKV: 1, LayerHeadCounts: []uint32{2, 4},
		LayerKVHeadCounts: []uint32{1, 2}, KeyLength: 4, ValueLength: 4,
		RopeFrequencyBase: 10000, RMSNormEpsilon: 1e-6,
	}
	input := builder.Input("input", dtype.F32, tensor.MustShape(8, 2))
	weights := LayerGraphWeights{
		AttentionNorm:   builder.Input("attn_norm", dtype.F32, tensor.MustShape(8)),
		AttentionQKV:    builder.Input("qkv", dtype.F32, tensor.MustShape(8, 32)),
		AttentionQNorm:  builder.Input("q_norm", dtype.F32, tensor.MustShape(4)),
		AttentionKNorm:  builder.Input("k_norm", dtype.F32, tensor.MustShape(4)),
		AttentionOutput: builder.Input("attn_output", dtype.F32, tensor.MustShape(16, 8)),
		FeedForwardNorm: builder.Input("ffn_norm", dtype.F32, tensor.MustShape(8)),
		FeedForwardGate: builder.Input("ffn_gate", dtype.F32, tensor.MustShape(8, 16)),
		FeedForwardUp:   builder.Input("ffn_up", dtype.F32, tensor.MustShape(8, 16)),
		FeedForwardDown: builder.Input("ffn_down", dtype.F32, tensor.MustShape(16, 8)),
	}
	result, err := BuildDenseBlockCachedForLayer(builder, input, spec, weights, []uint32{0, 1}, nil, nil, 1)
	if err != nil {
		t.Fatal(err)
	}
	nodes, err := tensor.Topological(result.Output)
	if err != nil {
		t.Fatal(err)
	}
	var rms, neoX int
	for _, node := range nodes {
		switch node.Op {
		case tensor.OpRMSNorm:
			rms++
		case tensor.OpRoPENeoX:
			neoX++
		}
	}
	if rms != 4 || neoX != 2 || result.Key.Shape.Dims[1] != 2 || result.Output.Shape.Dims[0] != 8 {
		t.Fatalf("unexpected OpenELM graph: rms=%d neox=%d key=%v output=%v", rms, neoX, result.Key.Shape.Slice(), result.Output.Shape.Slice())
	}
}

func TestBuildBailingMoEBlockUsesNormalizedSoftmaxAndSharedExpert(t *testing.T) {
	builder := tensor.NewBuilder()
	spec := Spec{
		Architecture: "bailingmoe", EmbeddingLength: 8, FeedForwardLength: 16,
		ExpertCount: 4, ExpertUsedCount: 2, ExpertFeedForward: 6,
		SharedExpertCount: 2, SharedExpertFF: 12, ExpertWeightsScale: 1.25,
		ExpertWeightsNorm: true, HeadCount: 2, HeadCountKV: 1,
		KeyLength: 4, ValueLength: 4, RopeFrequencyBase: 10000, RMSNormEpsilon: 1e-6,
	}
	input := builder.Input("input", dtype.F32, tensor.MustShape(8, 2))
	weights := denseBlockInputs(builder, spec)
	weights.FeedForwardGate, weights.FeedForwardUp, weights.FeedForwardDown = nil, nil, nil
	weights.FeedForwardRouter = builder.Input("router", dtype.F32, tensor.MustShape(8, 4))
	weights.FeedForwardGateExperts = builder.Input("gate_exps", dtype.F32, tensor.MustShape(8, 6, 4))
	weights.FeedForwardUpExperts = builder.Input("up_exps", dtype.F32, tensor.MustShape(8, 6, 4))
	weights.FeedForwardDownExperts = builder.Input("down_exps", dtype.F32, tensor.MustShape(6, 8, 4))
	weights.FeedForwardSharedGate = builder.Input("shared_gate", dtype.F32, tensor.MustShape(8, 12))
	weights.FeedForwardSharedUp = builder.Input("shared_up", dtype.F32, tensor.MustShape(8, 12))
	weights.FeedForwardSharedDown = builder.Input("shared_down", dtype.F32, tensor.MustShape(12, 8))
	output, err := BuildDenseBlock(builder, input, spec, weights, []uint32{0, 1})
	if err != nil {
		t.Fatal(err)
	}
	nodes, err := tensor.Topological(output)
	if err != nil {
		t.Fatal(err)
	}
	var moe *tensor.Tensor
	var rope, silu int
	for _, node := range nodes {
		switch node.Op {
		case tensor.OpMoE:
			moe = node
		case tensor.OpRoPENormal:
			rope++
		case tensor.OpSiLU:
			silu++
		}
	}
	if moe == nil {
		t.Fatal("BailingMoE graph is missing MoE operation")
	}
	attrs := moe.Attrs.(tensor.MoEAttributes)
	if attrs.Routing != tensor.MoERoutingSoftmax || !attrs.NormalizeTopKProb ||
		attrs.TopK != 2 || attrs.Scale != 1.25 || rope != 2 || silu < 1 {
		t.Fatalf("unexpected BailingMoE graph: attrs=%+v rope=%d silu=%d", attrs, rope, silu)
	}
}

func TestBuildDeepSeekMoEBlockUsesUnnormalizedSoftmaxAndSharedExpert(t *testing.T) {
	builder := tensor.NewBuilder()
	spec := Spec{
		Architecture: "deepseek", EmbeddingLength: 8, FeedForwardLength: 16,
		ExpertCount: 4, ExpertUsedCount: 2, ExpertFeedForward: 6,
		SharedExpertCount: 2, SharedExpertFF: 12, ExpertWeightsScale: 1.3,
		HeadCount: 2, HeadCountKV: 1, KeyLength: 4, ValueLength: 4,
		RopeFrequencyBase: 10000, RMSNormEpsilon: 1e-6,
	}
	input := builder.Input("input", dtype.F32, tensor.MustShape(8, 2))
	weights := denseBlockInputs(builder, spec)
	weights.FeedForwardGate, weights.FeedForwardUp, weights.FeedForwardDown = nil, nil, nil
	weights.FeedForwardRouter = builder.Input("router", dtype.F32, tensor.MustShape(8, 4))
	weights.FeedForwardGateExperts = builder.Input("gate_exps", dtype.F32, tensor.MustShape(8, 6, 4))
	weights.FeedForwardUpExperts = builder.Input("up_exps", dtype.F32, tensor.MustShape(8, 6, 4))
	weights.FeedForwardDownExperts = builder.Input("down_exps", dtype.F32, tensor.MustShape(6, 8, 4))
	weights.FeedForwardSharedGate = builder.Input("shared_gate", dtype.F32, tensor.MustShape(8, 12))
	weights.FeedForwardSharedUp = builder.Input("shared_up", dtype.F32, tensor.MustShape(8, 12))
	weights.FeedForwardSharedDown = builder.Input("shared_down", dtype.F32, tensor.MustShape(12, 8))
	output, err := BuildDenseBlock(builder, input, spec, weights, []uint32{0, 1})
	if err != nil {
		t.Fatal(err)
	}
	nodes, err := tensor.Topological(output)
	if err != nil {
		t.Fatal(err)
	}
	var moe *tensor.Tensor
	var rope, silu int
	for _, node := range nodes {
		switch node.Op {
		case tensor.OpMoE:
			moe = node
		case tensor.OpRoPENormal:
			rope++
		case tensor.OpSiLU:
			silu++
		}
	}
	if moe == nil {
		t.Fatal("DeepSeek graph is missing MoE operation")
	}
	attrs := moe.Attrs.(tensor.MoEAttributes)
	if attrs.Routing != tensor.MoERoutingSoftmax || attrs.NormalizeTopKProb ||
		attrs.TopK != 2 || attrs.Scale != 1.3 || rope != 2 || silu < 1 {
		t.Fatalf("unexpected DeepSeek graph: attrs=%+v rope=%d silu=%d", attrs, rope, silu)
	}
}

func TestBuildGraniteMoEBlockUsesUngatedExpertsAndSharedSwiGLU(t *testing.T) {
	builder := tensor.NewBuilder()
	spec := Spec{
		Architecture: "granitemoe", EmbeddingLength: 8, FeedForwardLength: 6,
		ExpertCount: 4, ExpertUsedCount: 2, ExpertFeedForward: 6,
		SharedExpertFF: 5, ExpertWeightsScale: 1, ExpertWeightsNorm: true,
		HeadCount: 2, HeadCountKV: 1, KeyLength: 4, ValueLength: 4,
		RopeFrequencyBase: 10000, RopeAttentionFactor: 1,
		RMSNormEpsilon: 1e-6, ResidualScale: 0.5,
	}
	input := builder.Input("input", dtype.F32, tensor.MustShape(8, 2))
	weights := denseBlockInputs(builder, spec)
	weights.FeedForwardGate, weights.FeedForwardUp, weights.FeedForwardDown = nil, nil, nil
	weights.FeedForwardRouter = builder.Input("router", dtype.F32, tensor.MustShape(8, 4))
	weights.FeedForwardGateExperts = nil
	weights.FeedForwardUpExperts = builder.Input("up_exps", dtype.F32, tensor.MustShape(8, 6, 4))
	weights.FeedForwardDownExperts = builder.Input("down_exps", dtype.F32, tensor.MustShape(6, 8, 4))
	weights.FeedForwardSharedGate = builder.Input("shared_gate", dtype.F32, tensor.MustShape(8, 5))
	weights.FeedForwardSharedUp = builder.Input("shared_up", dtype.F32, tensor.MustShape(8, 5))
	weights.FeedForwardSharedDown = builder.Input("shared_down", dtype.F32, tensor.MustShape(5, 8))
	output, err := BuildDenseBlock(builder, input, spec, weights, []uint32{0, 1})
	if err != nil {
		t.Fatal(err)
	}
	nodes, err := tensor.Topological(output)
	if err != nil {
		t.Fatal(err)
	}
	var moe *tensor.Tensor
	var rope, silu, scale int
	for _, node := range nodes {
		switch node.Op {
		case tensor.OpMoE:
			moe = node
		case tensor.OpRoPENormal:
			rope++
		case tensor.OpSiLU:
			silu++
		case tensor.OpScale:
			scale++
		}
	}
	if moe == nil {
		t.Fatal("GraniteMoE graph is missing MoE operation")
	}
	attrs := moe.Attrs.(tensor.MoEAttributes)
	if attrs.Gated || !attrs.NormalizeTopKProb || attrs.TopK != 2 ||
		attrs.Scale != 1 || rope != 2 || silu < 1 || scale != 2 {
		t.Fatalf("unexpected GraniteMoE graph: attrs=%+v rope=%d silu=%d scale=%d", attrs, rope, silu, scale)
	}
}

func TestBuildBailingMoE2BlockUsesFusedQKVNeoXAndSharedExpert(t *testing.T) {
	for _, test := range []struct {
		name    string
		gating  uint32
		routing tensor.MoERouting
	}{
		{name: "softmax", gating: 1, routing: tensor.MoERoutingSoftmax},
		{name: "sigmoid", gating: 2, routing: tensor.MoERoutingSigmoid},
	} {
		t.Run(test.name, func(t *testing.T) {
			builder := tensor.NewBuilder()
			spec := Spec{
				Architecture: "bailingmoe2", BlockCount: 2, LeadingDenseBlocks: 1,
				EmbeddingLength: 8, FeedForwardLength: 16,
				ExpertCount: 4, ExpertUsedCount: 2, ExpertFeedForward: 6,
				SharedExpertCount: 2, SharedExpertFF: 10, ExpertWeightsScale: 1.25,
				ExpertWeightsNorm: true, ExpertGatingFunc: test.gating,
				HeadCount: 2, HeadCountKV: 1, KeyLength: 4, ValueLength: 4,
				RopeDimensionCount: 4, RopeFrequencyBase: 10000, RMSNormEpsilon: 1e-6,
			}
			input := builder.Input("input", dtype.F32, tensor.MustShape(8, 2))
			weights := LayerGraphWeights{
				AttentionNorm:          builder.Input("attn_norm", dtype.F32, tensor.MustShape(8)),
				AttentionQKV:           builder.Input("qkv", dtype.F32, tensor.MustShape(8, 16)),
				AttentionOutput:        builder.Input("attn_out", dtype.F32, tensor.MustShape(8, 8)),
				AttentionQNorm:         builder.Input("q_norm", dtype.F32, tensor.MustShape(4)),
				AttentionKNorm:         builder.Input("k_norm", dtype.F32, tensor.MustShape(4)),
				FeedForwardNorm:        builder.Input("ffn_norm", dtype.F32, tensor.MustShape(8)),
				FeedForwardRouter:      builder.Input("router", dtype.F32, tensor.MustShape(8, 4)),
				FeedForwardGateExperts: builder.Input("gate_exps", dtype.F32, tensor.MustShape(8, 6, 4)),
				FeedForwardUpExperts:   builder.Input("up_exps", dtype.F32, tensor.MustShape(8, 6, 4)),
				FeedForwardDownExperts: builder.Input("down_exps", dtype.F32, tensor.MustShape(6, 8, 4)),
				FeedForwardExpertBias:  builder.Input("expert_bias", dtype.F32, tensor.MustShape(4)),
				FeedForwardSharedGate:  builder.Input("shared_gate", dtype.F32, tensor.MustShape(8, 10)),
				FeedForwardSharedUp:    builder.Input("shared_up", dtype.F32, tensor.MustShape(8, 10)),
				FeedForwardSharedDown:  builder.Input("shared_down", dtype.F32, tensor.MustShape(10, 8)),
			}
			result, err := BuildDenseBlockCachedForLayer(builder, input, spec, weights, []uint32{0, 1}, nil, nil, 1)
			if err != nil {
				t.Fatal(err)
			}
			nodes, err := tensor.Topological(result.Output)
			if err != nil {
				t.Fatal(err)
			}
			var moe *tensor.Tensor
			var neox, silu int
			for _, node := range nodes {
				switch node.Op {
				case tensor.OpMoE:
					moe = node
				case tensor.OpRoPENeoX:
					neox++
				case tensor.OpSiLU:
					silu++
				}
			}
			if moe == nil {
				t.Fatal("BailingMoE2 graph is missing MoE operation")
			}
			attrs := moe.Attrs.(tensor.MoEAttributes)
			if attrs.Routing != test.routing || !attrs.NormalizeTopKProb || len(moe.Inputs) != 7 ||
				attrs.TopK != 2 || attrs.Scale != 1.25 || neox != 2 || silu < 1 {
				t.Fatalf("unexpected BailingMoE2 graph: attrs=%+v inputs=%d neox=%d silu=%d", attrs, len(moe.Inputs), neox, silu)
			}
		})
	}
}

func TestBuildQwen2MoEBlockUsesGatedSharedExpert(t *testing.T) {
	builder := tensor.NewBuilder()
	spec := Spec{
		Architecture: "qwen2moe", EmbeddingLength: 8, FeedForwardLength: 16,
		ExpertCount: 4, ExpertUsedCount: 2, ExpertFeedForward: 6,
		SharedExpertCount: 1, SharedExpertFF: 10, ExpertWeightsScale: 1,
		HeadCount: 2, HeadCountKV: 1, KeyLength: 4, ValueLength: 4,
		RopeFrequencyBase: 1_000_000, RMSNormEpsilon: 1e-6,
	}
	input := builder.Input("input", dtype.F32, tensor.MustShape(8, 2))
	weights := denseBlockInputs(builder, spec)
	weights.FeedForwardGate, weights.FeedForwardUp, weights.FeedForwardDown = nil, nil, nil
	weights.FeedForwardRouter = builder.Input("router", dtype.F32, tensor.MustShape(8, 4))
	weights.FeedForwardGateExperts = builder.Input("gate_exps", dtype.F32, tensor.MustShape(8, 6, 4))
	weights.FeedForwardUpExperts = builder.Input("up_exps", dtype.F32, tensor.MustShape(8, 6, 4))
	weights.FeedForwardDownExperts = builder.Input("down_exps", dtype.F32, tensor.MustShape(6, 8, 4))
	weights.FeedForwardSharedRouter = builder.Input("shared_router", dtype.F32, tensor.MustShape(8))
	weights.FeedForwardSharedGate = builder.Input("shared_gate", dtype.F32, tensor.MustShape(8, 10))
	weights.FeedForwardSharedUp = builder.Input("shared_up", dtype.F32, tensor.MustShape(8, 10))
	weights.FeedForwardSharedDown = builder.Input("shared_down", dtype.F32, tensor.MustShape(10, 8))
	output, err := BuildDenseBlock(builder, input, spec, weights, []uint32{0, 1})
	if err != nil {
		t.Fatal(err)
	}
	nodes, _ := tensor.Topological(output)
	var moe *tensor.Tensor
	var sigmoid bool
	for _, node := range nodes {
		if node.Op == tensor.OpMoE {
			moe = node
		}
		sigmoid = sigmoid || node.Op == tensor.OpSigmoid
	}
	if moe == nil || moe.Attrs.(tensor.MoEAttributes).NormalizeTopKProb || !sigmoid {
		t.Fatalf("Qwen2-MoE graph missing unnormalized MoE/shared gate")
	}
}

func TestBuildOLMoEBlockNormalizesFullQKProjections(t *testing.T) {
	builder := tensor.NewBuilder()
	spec := Spec{
		Architecture: "olmoe", EmbeddingLength: 8, FeedForwardLength: 12,
		ExpertCount: 4, ExpertUsedCount: 2, ExpertFeedForward: 12, ExpertWeightsScale: 1,
		HeadCount: 2, HeadCountKV: 2, KeyLength: 4, ValueLength: 4,
		RopeFrequencyBase: 10000, RMSNormEpsilon: 1e-6,
	}
	input := builder.Input("input", dtype.F32, tensor.MustShape(8, 2))
	weights := LayerGraphWeights{
		AttentionNorm:          builder.Input("attn_norm", dtype.F32, tensor.MustShape(8)),
		AttentionQ:             builder.Input("q", dtype.F32, tensor.MustShape(8, 8)),
		AttentionK:             builder.Input("k", dtype.F32, tensor.MustShape(8, 8)),
		AttentionV:             builder.Input("v", dtype.F32, tensor.MustShape(8, 8)),
		AttentionOutput:        builder.Input("o", dtype.F32, tensor.MustShape(8, 8)),
		AttentionQNorm:         builder.Input("q_norm", dtype.F32, tensor.MustShape(8)),
		AttentionKNorm:         builder.Input("k_norm", dtype.F32, tensor.MustShape(8)),
		FeedForwardNorm:        builder.Input("ffn_norm", dtype.F32, tensor.MustShape(8)),
		FeedForwardRouter:      builder.Input("router", dtype.F32, tensor.MustShape(8, 4)),
		FeedForwardGateExperts: builder.Input("gate_exps", dtype.F32, tensor.MustShape(8, 12, 4)),
		FeedForwardUpExperts:   builder.Input("up_exps", dtype.F32, tensor.MustShape(8, 12, 4)),
		FeedForwardDownExperts: builder.Input("down_exps", dtype.F32, tensor.MustShape(12, 8, 4)),
	}
	output, err := BuildDenseBlock(builder, input, spec, weights, []uint32{0, 1})
	if err != nil {
		t.Fatal(err)
	}
	nodes, _ := tensor.Topological(output)
	var projectionNorms int
	var moe *tensor.Tensor
	for _, node := range nodes {
		if node.Op == tensor.OpRMSNorm && len(node.Inputs) == 1 && node.Inputs[0].Op == tensor.OpMulMat &&
			node.Shape.Rank == 2 && node.Shape.Dims[0] == 8 {
			projectionNorms++
		}
		if node.Op == tensor.OpMoE {
			moe = node
		}
	}
	if projectionNorms < 2 || moe == nil || moe.Attrs.(tensor.MoEAttributes).NormalizeTopKProb {
		t.Fatalf("OLMoE graph projection norms/MoE = %d/%v", projectionNorms, moe != nil)
	}
}

func TestBuildEuroBERTBlockUsesNonCausalNeoXAttention(t *testing.T) {
	builder := tensor.NewBuilder()
	spec := Spec{
		Architecture: "eurobert", EmbeddingLength: 8, FeedForwardLength: 16,
		HeadCount: 2, HeadCountKV: 1, KeyLength: 4, ValueLength: 4,
		RopeFrequencyBase: 10000, RMSNormEpsilon: 1e-6, NonCausalAttention: true,
	}
	input := builder.Input("input", dtype.F32, tensor.MustShape(8, 3))
	weights := denseBlockInputs(builder, spec)
	weights.AttentionQNorm = nil
	weights.AttentionKNorm = nil
	result, err := BuildDenseBlockCachedForLayer(
		builder, input, spec, weights, []uint32{0, 1, 2}, nil, nil, 0,
	)
	if err != nil {
		t.Fatal(err)
	}
	nodes, err := tensor.Topological(result.Output)
	if err != nil {
		t.Fatal(err)
	}
	var attention *tensor.Tensor
	var rope int
	for _, node := range nodes {
		if node.Op == tensor.OpAttention {
			attention = node
		}
		if node.Op == tensor.OpRoPENeoX {
			rope++
		}
	}
	if attention == nil || attention.Attrs.(tensor.AttentionAttributes).Causal || rope != 2 {
		t.Fatalf("unexpected EuroBERT graph: attention=%+v RoPE=%d", attention, rope)
	}
}

func TestBuildBERTBlockUsesPostNormNonCausalAttention(t *testing.T) {
	builder := tensor.NewBuilder()
	spec := Spec{
		Architecture: "bert", EmbeddingLength: 8, FeedForwardLength: 16,
		HeadCount: 2, HeadCountKV: 2, KeyLength: 4, ValueLength: 4,
		LayerNormEpsilon: 1e-5, NonCausalAttention: true, RopeDisabled: true,
	}
	input := builder.Input("input", dtype.F32, tensor.MustShape(8, 3))
	weights := LayerGraphWeights{
		AttentionQKV:            builder.Input("qkv", dtype.F32, tensor.MustShape(8, 24)),
		AttentionQKVBias:        builder.Input("qkv_bias", dtype.F32, tensor.MustShape(24)),
		AttentionOutput:         builder.Input("attn_out", dtype.F32, tensor.MustShape(8, 8)),
		AttentionOutputBias:     builder.Input("attn_out_bias", dtype.F32, tensor.MustShape(8)),
		AttentionPostNorm:       builder.Input("attn_post_norm", dtype.F32, tensor.MustShape(8)),
		AttentionPostNormBias:   builder.Input("attn_post_norm_bias", dtype.F32, tensor.MustShape(8)),
		FeedForwardUp:           builder.Input("ffn_up", dtype.F32, tensor.MustShape(8, 16)),
		FeedForwardUpBias:       builder.Input("ffn_up_bias", dtype.F32, tensor.MustShape(16)),
		FeedForwardDown:         builder.Input("ffn_down", dtype.F32, tensor.MustShape(16, 8)),
		FeedForwardDownBias:     builder.Input("ffn_down_bias", dtype.F32, tensor.MustShape(8)),
		FeedForwardPostNorm:     builder.Input("ffn_post_norm", dtype.F32, tensor.MustShape(8)),
		FeedForwardPostNormBias: builder.Input("ffn_post_norm_bias", dtype.F32, tensor.MustShape(8)),
	}
	result, err := BuildDenseBlockCachedForLayer(
		builder, input, spec, weights, []uint32{0, 1, 2}, nil, nil, 0,
	)
	if err != nil {
		t.Fatal(err)
	}
	nodes, err := tensor.Topological(result.Output)
	if err != nil {
		t.Fatal(err)
	}
	var attention *tensor.Tensor
	var layerNorm, gelu, rope int
	for _, node := range nodes {
		switch node.Op {
		case tensor.OpAttention:
			attention = node
		case tensor.OpLayerNorm:
			layerNorm++
		case tensor.OpGELU:
			gelu++
		case tensor.OpRoPENormal, tensor.OpRoPENeoX:
			rope++
		}
	}
	if attention == nil || attention.Attrs.(tensor.AttentionAttributes).Causal ||
		layerNorm != 2 || gelu != 1 || rope != 0 {
		t.Fatalf("unexpected BERT graph: attention=%+v LayerNorm=%d GELU=%d RoPE=%d", attention, layerNorm, gelu, rope)
	}
}

func TestBuildNeoBERTBlockUsesNormalRoPEAndFusedSwiGLU(t *testing.T) {
	builder := tensor.NewBuilder()
	spec := Spec{
		Architecture: "neo-bert", EmbeddingLength: 8, FeedForwardLength: 16,
		HeadCount: 2, HeadCountKV: 2, KeyLength: 4, ValueLength: 4,
		RopeDimensionCount: 4, RopeFrequencyBase: 10000,
		RMSNormEpsilon: 1e-6, NonCausalAttention: true,
	}
	input := builder.Input("input", dtype.F32, tensor.MustShape(8, 3))
	weights := LayerGraphWeights{
		AttentionNorm:   builder.Input("attn_norm", dtype.F32, tensor.MustShape(8)),
		AttentionQKV:    builder.Input("qkv", dtype.F32, tensor.MustShape(8, 24)),
		AttentionOutput: builder.Input("attn_out", dtype.F32, tensor.MustShape(8, 8)),
		FeedForwardNorm: builder.Input("ffn_norm", dtype.F32, tensor.MustShape(8)),
		FeedForwardUp:   builder.Input("ffn_up", dtype.F32, tensor.MustShape(8, 32)),
		FeedForwardDown: builder.Input("ffn_down", dtype.F32, tensor.MustShape(16, 8)),
	}
	result, err := BuildDenseBlockCachedForLayer(
		builder, input, spec, weights, []uint32{0, 1, 2}, nil, nil, 0,
	)
	if err != nil {
		t.Fatal(err)
	}
	nodes, err := tensor.Topological(result.Output)
	if err != nil {
		t.Fatal(err)
	}
	var attention *tensor.Tensor
	var normalRoPE, silu int
	for _, node := range nodes {
		switch node.Op {
		case tensor.OpAttention:
			attention = node
		case tensor.OpRoPENormal:
			normalRoPE++
		case tensor.OpSiLU:
			silu++
		}
	}
	if attention == nil || attention.Attrs.(tensor.AttentionAttributes).Causal ||
		normalRoPE != 2 || silu != 1 {
		t.Fatalf("unexpected NeoBERT graph: attention=%+v RoPE=%d SiLU=%d", attention, normalRoPE, silu)
	}
}

func TestBuildNomicBERTBlockUsesNeoXRoPEAndSwiGLU(t *testing.T) {
	builder := tensor.NewBuilder()
	spec := Spec{
		Architecture: "nomic-bert", EmbeddingLength: 8, FeedForwardLength: 16,
		HeadCount: 2, HeadCountKV: 2, KeyLength: 4, ValueLength: 4,
		RopeDimensionCount: 4, RopeFrequencyBase: 10000,
		LayerNormEpsilon: 1e-5, NonCausalAttention: true,
	}
	input := builder.Input("input", dtype.F32, tensor.MustShape(8, 3))
	weights := LayerGraphWeights{
		AttentionQKV:            builder.Input("qkv", dtype.F32, tensor.MustShape(8, 24)),
		AttentionOutput:         builder.Input("attn_out", dtype.F32, tensor.MustShape(8, 8)),
		AttentionPostNorm:       builder.Input("attn_post_norm", dtype.F32, tensor.MustShape(8)),
		AttentionPostNormBias:   builder.Input("attn_post_norm_bias", dtype.F32, tensor.MustShape(8)),
		FeedForwardGate:         builder.Input("ffn_gate", dtype.F32, tensor.MustShape(8, 16)),
		FeedForwardUp:           builder.Input("ffn_up", dtype.F32, tensor.MustShape(8, 16)),
		FeedForwardDown:         builder.Input("ffn_down", dtype.F32, tensor.MustShape(16, 8)),
		FeedForwardPostNorm:     builder.Input("ffn_post_norm", dtype.F32, tensor.MustShape(8)),
		FeedForwardPostNormBias: builder.Input("ffn_post_norm_bias", dtype.F32, tensor.MustShape(8)),
	}
	result, err := BuildDenseBlockCachedForLayer(
		builder, input, spec, weights, []uint32{0, 1, 2}, nil, nil, 0,
	)
	if err != nil {
		t.Fatal(err)
	}
	nodes, err := tensor.Topological(result.Output)
	if err != nil {
		t.Fatal(err)
	}
	var attention *tensor.Tensor
	var neoXRoPE, layerNorm, silu, gelu int
	for _, node := range nodes {
		switch node.Op {
		case tensor.OpAttention:
			attention = node
		case tensor.OpRoPENeoX:
			neoXRoPE++
		case tensor.OpLayerNorm:
			layerNorm++
		case tensor.OpSiLU:
			silu++
		case tensor.OpGELU:
			gelu++
		}
	}
	if attention == nil || attention.Attrs.(tensor.AttentionAttributes).Causal ||
		neoXRoPE != 2 || layerNorm != 2 || silu != 1 || gelu != 0 {
		t.Fatalf(
			"unexpected NomicBERT graph: attention=%+v RoPE=%d LayerNorm=%d SiLU=%d GELU=%d",
			attention, neoXRoPE, layerNorm, silu, gelu,
		)
	}
}

func TestBuildJinaBERTV2BlockUsesALiBiNormsAndFusedGEGLU(t *testing.T) {
	builder := tensor.NewBuilder()
	spec := Spec{
		Architecture: "jina-bert-v2", EmbeddingLength: 8, FeedForwardLength: 16,
		HeadCount: 2, HeadCountKV: 2, KeyLength: 4, ValueLength: 4,
		LayerNormEpsilon: 1e-5, NonCausalAttention: true, RopeDisabled: true, MaxALiBiBias: 8,
	}
	input := builder.Input("input", dtype.F32, tensor.MustShape(8, 3))
	weights := LayerGraphWeights{
		AttentionQKV:            builder.Input("qkv", dtype.F32, tensor.MustShape(8, 24)),
		AttentionQNorm:          builder.Input("q_norm", dtype.F32, tensor.MustShape(8)),
		AttentionQNormBias:      builder.Input("q_norm_bias", dtype.F32, tensor.MustShape(8)),
		AttentionKNorm:          builder.Input("k_norm", dtype.F32, tensor.MustShape(8)),
		AttentionKNormBias:      builder.Input("k_norm_bias", dtype.F32, tensor.MustShape(8)),
		AttentionOutput:         builder.Input("attn_out", dtype.F32, tensor.MustShape(8, 8)),
		AttentionPostNorm:       builder.Input("attn_post_norm", dtype.F32, tensor.MustShape(8)),
		AttentionPostNormBias:   builder.Input("attn_post_norm_bias", dtype.F32, tensor.MustShape(8)),
		AttentionNorm2:          builder.Input("attn_norm_2", dtype.F32, tensor.MustShape(8)),
		AttentionNorm2Bias:      builder.Input("attn_norm_2_bias", dtype.F32, tensor.MustShape(8)),
		FeedForwardUp:           builder.Input("ffn_up", dtype.F32, tensor.MustShape(8, 32)),
		FeedForwardDown:         builder.Input("ffn_down", dtype.F32, tensor.MustShape(16, 8)),
		FeedForwardPostNorm:     builder.Input("ffn_post_norm", dtype.F32, tensor.MustShape(8)),
		FeedForwardPostNormBias: builder.Input("ffn_post_norm_bias", dtype.F32, tensor.MustShape(8)),
	}
	result, err := BuildDenseBlockCachedForLayer(
		builder, input, spec, weights, []uint32{0, 1, 2}, nil, nil, 0,
	)
	if err != nil {
		t.Fatal(err)
	}
	nodes, err := tensor.Topological(result.Output)
	if err != nil {
		t.Fatal(err)
	}
	var attention *tensor.Tensor
	var layerNorm, gelu, multiply, rope int
	for _, node := range nodes {
		switch node.Op {
		case tensor.OpAttention:
			attention = node
		case tensor.OpLayerNorm:
			layerNorm++
		case tensor.OpGELU:
			gelu++
		case tensor.OpMultiply:
			multiply++
		case tensor.OpRoPENormal, tensor.OpRoPENeoX:
			rope++
		}
	}
	if attention == nil {
		t.Fatal("JinaBERT v2 attention node is missing")
	}
	attrs := attention.Attrs.(tensor.AttentionAttributes)
	if attrs.Causal || attrs.MaxALiBiBias != 8 ||
		layerNorm != 5 || gelu != 1 || multiply < 6 || rope != 0 {
		t.Fatalf(
			"unexpected JinaBERT v2 graph: attention=%+v LayerNorm=%d GELU=%d Multiply=%d RoPE=%d",
			attention, layerNorm, gelu, multiply, rope,
		)
	}
}

func TestBuildJinaBERTV2BlockSelectsPlainOrSeparateGateFFN(t *testing.T) {
	for _, test := range []struct {
		name      string
		withGate  bool
		wantGEGLU bool
	}{
		{name: "plain GELU"},
		{name: "separate GEGLU", withGate: true, wantGEGLU: true},
	} {
		t.Run(test.name, func(t *testing.T) {
			builder := tensor.NewBuilder()
			spec := Spec{
				Architecture: "jina-bert-v2", EmbeddingLength: 8, FeedForwardLength: 16,
				HeadCount: 2, HeadCountKV: 2, KeyLength: 4, ValueLength: 4,
				LayerNormEpsilon: 1e-5, NonCausalAttention: true, RopeDisabled: true, MaxALiBiBias: 8,
			}
			input := builder.Input("input", dtype.F32, tensor.MustShape(8, 2))
			weights := LayerGraphWeights{
				AttentionQKV:            builder.Input("qkv", dtype.F32, tensor.MustShape(8, 24)),
				AttentionOutput:         builder.Input("attn_out", dtype.F32, tensor.MustShape(8, 8)),
				AttentionPostNorm:       builder.Input("attn_post_norm", dtype.F32, tensor.MustShape(8)),
				AttentionPostNormBias:   builder.Input("attn_post_norm_bias", dtype.F32, tensor.MustShape(8)),
				FeedForwardUp:           builder.Input("ffn_up", dtype.F32, tensor.MustShape(8, 16)),
				FeedForwardDown:         builder.Input("ffn_down", dtype.F32, tensor.MustShape(16, 8)),
				FeedForwardPostNorm:     builder.Input("ffn_post_norm", dtype.F32, tensor.MustShape(8)),
				FeedForwardPostNormBias: builder.Input("ffn_post_norm_bias", dtype.F32, tensor.MustShape(8)),
			}
			if test.withGate {
				weights.FeedForwardGate = builder.Input("ffn_gate", dtype.F32, tensor.MustShape(8, 16))
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
			var geglu bool
			for _, node := range nodes {
				if node.Op == tensor.OpMultiply && len(node.Inputs) == 2 && node.Inputs[0].Op == tensor.OpGELU {
					geglu = true
				}
			}
			if geglu != test.wantGEGLU {
				t.Fatalf("GEGLU=%v, want %v", geglu, test.wantGEGLU)
			}
		})
	}
}

func TestBuildJinaBERTV3BlockUsesNeoXRoPEAndGELU(t *testing.T) {
	builder := tensor.NewBuilder()
	spec := Spec{
		Architecture: "jina-bert-v3", EmbeddingLength: 8, FeedForwardLength: 16,
		HeadCount: 2, HeadCountKV: 2, KeyLength: 4, ValueLength: 4,
		RopeDimensionCount: 4, RopeFrequencyBase: 10000,
		LayerNormEpsilon: 1e-5, NonCausalAttention: true,
	}
	input := builder.Input("input", dtype.F32, tensor.MustShape(8, 3))
	weights := LayerGraphWeights{
		AttentionQKV:            builder.Input("qkv", dtype.F32, tensor.MustShape(8, 24)),
		AttentionOutput:         builder.Input("attn_out", dtype.F32, tensor.MustShape(8, 8)),
		AttentionPostNorm:       builder.Input("attn_post_norm", dtype.F32, tensor.MustShape(8)),
		AttentionPostNormBias:   builder.Input("attn_post_norm_bias", dtype.F32, tensor.MustShape(8)),
		FeedForwardUp:           builder.Input("ffn_up", dtype.F32, tensor.MustShape(8, 16)),
		FeedForwardDown:         builder.Input("ffn_down", dtype.F32, tensor.MustShape(16, 8)),
		FeedForwardPostNorm:     builder.Input("ffn_post_norm", dtype.F32, tensor.MustShape(8)),
		FeedForwardPostNormBias: builder.Input("ffn_post_norm_bias", dtype.F32, tensor.MustShape(8)),
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
	var neoXRoPE, layerNorm, gelu int
	for _, node := range nodes {
		switch node.Op {
		case tensor.OpAttention:
			attention = node
		case tensor.OpRoPENeoX:
			neoXRoPE++
		case tensor.OpLayerNorm:
			layerNorm++
		case tensor.OpGELU:
			gelu++
		}
	}
	if attention == nil || attention.Attrs.(tensor.AttentionAttributes).Causal ||
		neoXRoPE != 2 || layerNorm != 2 || gelu != 1 {
		t.Fatalf("unexpected JinaBERT v3 graph: attention=%+v RoPE=%d LayerNorm=%d GELU=%d", attention, neoXRoPE, layerNorm, gelu)
	}
}

func TestBuildNomicBERTMoEBlockUsesGateFreeGELUExperts(t *testing.T) {
	builder := tensor.NewBuilder()
	spec := Spec{
		Architecture: "nomic-bert-moe", EmbeddingLength: 8, FeedForwardLength: 16,
		HeadCount: 2, HeadCountKV: 2, KeyLength: 4, ValueLength: 4,
		ExpertCount: 4, ExpertUsedCount: 2, ExpertFeedForward: 16, ExpertWeightsScale: 1,
		MoELayerStep: 2, RopeDimensionCount: 4, RopeFrequencyBase: 10000,
		LayerNormEpsilon: 1e-5, NonCausalAttention: true, BlockCount: 2,
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
		t.Fatalf("NomicBERT-MoE graph MoE/RoPE = %d/%d", moe, rope)
	}
}

func TestBuildDreamBlockUsesNonCausalAttentionAndRejectsCache(t *testing.T) {
	builder := tensor.NewBuilder()
	spec := Spec{
		Architecture:       "dream",
		EmbeddingLength:    8,
		FeedForwardLength:  12,
		HeadCount:          2,
		HeadCountKV:        1,
		KeyLength:          4,
		ValueLength:        4,
		RopeFrequencyBase:  10000,
		RMSNormEpsilon:     1e-6,
		NonCausalAttention: true,
	}
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
			spec := Spec{
				Architecture: architecture, EmbeddingLength: 8, FeedForwardLength: 12,
				HeadCount: 2, HeadCountKV: 1, KeyLength: 4, ValueLength: 4,
				RopeFrequencyBase: 10000, RMSNormEpsilon: 1e-6, NonCausalAttention: true,
			}
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
	spec := Spec{
		Architecture: "rnd1", EmbeddingLength: 8, FeedForwardLength: 24,
		ExpertCount: 4, ExpertUsedCount: 2, ExpertFeedForward: 12, ExpertWeightsScale: 1,
		HeadCount: 2, HeadCountKV: 1, KeyLength: 4, ValueLength: 4,
		RopeFrequencyBase: 1_000_000, RMSNormEpsilon: 1e-6, NonCausalAttention: true,
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
	spec := Spec{
		Architecture: "laguna", BlockCount: 2, EmbeddingLength: 8,
		FeedForwardLength: 16, LeadingDenseBlocks: 1,
		ExpertCount: 4, ExpertUsedCount: 2, ExpertFeedForward: 12,
		SharedExpertFF: 10, ExpertWeightsScale: 1.25, ExpertWeightsNorm: true,
		HeadCount: 2, HeadCountKV: 1, LayerHeadCounts: []uint32{2, 4},
		LayerKVHeadCounts: []uint32{1, 1}, KeyLength: 4, ValueLength: 4,
		RopeFrequencyBase: 500000, RopeDimensionCount: 4, RopeScalingType: "yarn",
		RopeScalingFactor: 4, OriginalContextLength: 2048, YaRNExtFactor: 1,
		YaRNAttentionFactor: 1, YaRNBetaFast: 32, YaRNBetaSlow: 1,
		RMSNormEpsilon: 1e-6,
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
	spec := Spec{
		Architecture: "laguna", BlockCount: 2, EmbeddingLength: 8,
		FeedForwardLength: 12, LeadingDenseBlocks: 1,
		HeadCount: 2, HeadCountKV: 1, LayerHeadCounts: []uint32{2, 4},
		LayerKVHeadCounts: []uint32{1, 1}, KeyLength: 4, ValueLength: 4,
		RopeFrequencyBase: 500000, RopeDimensionCount: 4, RopeScalingType: "yarn",
		RopeScalingFactor: 4, OriginalContextLength: 2048, YaRNExtFactor: 1,
		YaRNAttentionFactor: 1, YaRNBetaFast: 32, YaRNBetaSlow: 1,
		SlidingWindow: 64, SlidingPattern: 2, RopeFrequencySWA: 10000, RopeDimensionSWA: 4,
		RMSNormEpsilon: 1e-6,
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
	spec := Spec{
		Architecture: "lfm2", EmbeddingLength: 4, FeedForwardLength: 6,
		HeadCount: 1, HeadCountKV: 1, KeyLength: 4, ValueLength: 4,
		RMSNormEpsilon: 1e-6, ShortConvCacheLength: 3,
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
	state := builder.Input("conv_state", dtype.F32, tensor.MustShape(2, 4))
	reserved := builder.Input("reserved", dtype.F32, tensor.MustShape(1))
	result, err := BuildLFM2BlockCached(
		builder, input, spec, weights, []uint32{0, 1}, true, state, reserved, 0,
	)
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

func TestBuildLFM2MoEShortConvolutionBlock(t *testing.T) {
	builder := tensor.NewBuilder()
	spec := Spec{
		Architecture: "lfm2moe", BlockCount: 3, LeadingDenseBlocks: 1,
		EmbeddingLength: 4, FeedForwardLength: 6,
		ExpertCount: 4, ExpertUsedCount: 2, ExpertFeedForward: 6,
		ExpertWeightsScale: 1.25, ExpertGatingFunc: 2,
		HeadCount: 1, HeadCountKV: 1, KeyLength: 4, ValueLength: 4,
		RMSNormEpsilon: 1e-6, ShortConvCacheLength: 3,
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
	state := builder.Input("conv_state", dtype.F32, tensor.MustShape(2, 4))
	reserved := builder.Input("reserved", dtype.F32, tensor.MustShape(1))
	result, err := BuildLFM2BlockCached(
		builder, input, spec, weights, []uint32{0, 1}, true, state, reserved, 2,
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
	spec := Spec{
		Architecture: "plm", EmbeddingLength: 8, FeedForwardLength: 12,
		HeadCount: 2, HeadCountKV: 2, KeyLength: 6, ValueLength: 4,
		KVLoRARank: 3, RopeDimensionCount: 2, RopeFrequencyBase: 10000,
		RMSNormEpsilon: 1e-6,
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

func TestBuildMiniCPM3MLABlock(t *testing.T) {
	builder := tensor.NewBuilder()
	spec := Spec{Architecture: "minicpm3", EmbeddingLength: 8, FeedForwardLength: 12,
		HeadCount: 2, HeadCountKV: 2, KeyLength: 6, ValueLength: 4,
		QLoRARank: 3, KVLoRARank: 3, RopeDimensionCount: 2, RopeFrequencyBase: 10000,
		RopeAttentionFactor: 1, ResidualScale: 0.7, RMSNormEpsilon: 1e-6}
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

func TestBuildChameleonSandwichBlock(t *testing.T) {
	builder := tensor.NewBuilder()
	spec := Spec{
		Architecture: "chameleon", EmbeddingLength: 8, FeedForwardLength: 12,
		HeadCount: 2, HeadCountKV: 1, KeyLength: 4, ValueLength: 4,
		RopeFrequencyBase: 10000, RMSNormEpsilon: 1e-5, QKNormEpsilon: 1e-5,
		SandwichNorm: true,
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
	spec := Spec{
		Architecture:      "qwen2",
		EmbeddingLength:   8,
		FeedForwardLength: 12,
		HeadCount:         2,
		HeadCountKV:       1,
		KeyLength:         4,
		ValueLength:       4,
		RopeFrequencyBase: 1_000_000,
		RMSNormEpsilon:    1e-6,
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
			spec := Spec{
				Architecture:      test.architecture,
				EmbeddingLength:   8,
				FeedForwardLength: 12,
				HeadCount:         2,
				HeadCountKV:       1,
				KeyLength:         4,
				ValueLength:       4,
				RopeFrequencyBase: 10000,
				RMSNormEpsilon:    1e-6,
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

func TestBuildDenseOLMo2PostNormalizedSlidingBlock(t *testing.T) {
	builder := tensor.NewBuilder()
	spec := Spec{
		Architecture:      "olmo2",
		BlockCount:        4,
		EmbeddingLength:   8,
		FeedForwardLength: 12,
		HeadCount:         2,
		HeadCountKV:       1,
		KeyLength:         4,
		ValueLength:       4,
		RopeFrequencyBase: 10000,
		RopeFrequencySWA:  10000,
		RopeScalingType:   "linear",
		RopeScalingFactor: 4,
		RMSNormEpsilon:    1e-6,
		SlidingWindow:     1024,
		SlidingPattern:    4,
	}
	input := builder.Input("input", dtype.F32, tensor.MustShape(8, 2))
	weights := denseBlockInputs(builder, spec)
	weights.AttentionNorm = nil
	weights.FeedForwardNorm = nil
	weights.AttentionQNorm = builder.Input(
		"attn_q_norm",
		dtype.F32,
		tensor.MustShape(8),
	)
	weights.AttentionKNorm = builder.Input(
		"attn_k_norm",
		dtype.F32,
		tensor.MustShape(4),
	)
	weights.AttentionPostNorm = builder.Input(
		"post_attention_norm",
		dtype.F32,
		tensor.MustShape(8),
	)
	weights.FeedForwardPostNorm = builder.Input(
		"post_ffw_norm",
		dtype.F32,
		tensor.MustShape(8),
	)
	result, err := BuildDenseBlockCachedForLayer(
		builder,
		input,
		spec,
		weights,
		[]uint32{0, 1},
		nil,
		nil,
		0,
	)
	if err != nil {
		t.Fatal(err)
	}
	nodes, err := tensor.Topological(result.Output)
	if err != nil {
		t.Fatal(err)
	}
	found := map[string]bool{}
	var attentionCount, ropeCount int
	for _, node := range nodes {
		found[node.Name] = true
		switch node.Op {
		case tensor.OpAttention:
			attentionCount++
			attributes := node.Attrs.(tensor.AttentionAttributes)
			if attributes.Window != 1024 {
				t.Fatalf("OLMo2 attention window = %d, want 1024", attributes.Window)
			}
		case tensor.OpRoPENeoX:
			ropeCount++
			attributes := node.Attrs.(tensor.RoPEAttributes)
			if attributes.FrequencyScale != 1 {
				t.Fatalf("OLMo2 sliding RoPE scale = %v, want 1", attributes.FrequencyScale)
			}
		}
	}
	for _, name := range []string{
		"attn_q_norm",
		"attn_k_norm",
		"post_attention_norm",
		"post_ffw_norm",
	} {
		if !found[name] {
			t.Fatalf("OLMo2 norm %q is disconnected", name)
		}
	}
	if attentionCount != 1 || ropeCount != 2 {
		t.Fatalf("OLMo2 graph has attention=%d RoPE=%d, want 1/2", attentionCount, ropeCount)
	}
}

func TestBuildDenseSmolLM3SkipsPeriodicRoPE(t *testing.T) {
	for _, test := range []struct {
		layer    uint32
		wantRoPE int
	}{
		{2, 2},
		{3, 0},
	} {
		builder := tensor.NewBuilder()
		spec := Spec{
			Architecture:      "smollm3",
			BlockCount:        36,
			EmbeddingLength:   8,
			FeedForwardLength: 12,
			HeadCount:         2,
			HeadCountKV:       1,
			KeyLength:         4,
			ValueLength:       4,
			RopeFrequencyBase: 10000,
			AttentionScale:    0.25,
			RMSNormEpsilon:    1e-6,
			NoRopeLayerStep:   4,
		}
		input := builder.Input("input", dtype.F32, tensor.MustShape(8, 2))
		weights := denseBlockInputs(builder, spec)
		weights.AttentionQNorm = nil
		weights.AttentionKNorm = nil
		result, err := BuildDenseBlockCachedForLayer(
			builder,
			input,
			spec,
			weights,
			[]uint32{0, 1},
			nil,
			nil,
			test.layer,
		)
		if err != nil {
			t.Fatal(err)
		}
		nodes, err := tensor.Topological(result.Output)
		if err != nil {
			t.Fatal(err)
		}
		var ropeCount int
		for _, node := range nodes {
			if node.Op == tensor.OpRoPENormal {
				ropeCount++
			}
			if node.Op == tensor.OpAttention {
				attributes := node.Attrs.(tensor.AttentionAttributes)
				if attributes.Scale != 0.25 {
					t.Fatalf("SmolLM3 attention scale = %v, want 0.25", attributes.Scale)
				}
			}
		}
		if ropeCount != test.wantRoPE {
			t.Fatalf("SmolLM3 layer %d RoPE count = %d, want %d", test.layer, ropeCount, test.wantRoPE)
		}
	}
}

func TestBuildDenseMiniCPMScalesResidualBranches(t *testing.T) {
	builder := tensor.NewBuilder()
	spec := Spec{
		Architecture:      "minicpm",
		EmbeddingLength:   8,
		FeedForwardLength: 12,
		HeadCount:         2,
		HeadCountKV:       1,
		KeyLength:         4,
		ValueLength:       4,
		RopeFrequencyBase: 10000,
		ResidualScale:     0.25,
		RMSNormEpsilon:    1e-6,
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
	var residualScales int
	for _, node := range nodes {
		if node.Op == tensor.OpScale &&
			node.Attrs.(tensor.ScaleAttributes).Value == 0.25 {
			residualScales++
		}
	}
	if residualScales != 2 {
		t.Fatalf("MiniCPM residual scale count = %d, want 2", residualScales)
	}
}

func TestBuildDenseGraniteHonorsRoPESwitchAndScales(t *testing.T) {
	builder := tensor.NewBuilder()
	spec := Spec{
		Architecture:      "granite",
		EmbeddingLength:   8,
		FeedForwardLength: 12,
		HeadCount:         2,
		HeadCountKV:       1,
		KeyLength:         4,
		ValueLength:       4,
		RopeFrequencyBase: 10000,
		AttentionScale:    0.25,
		ResidualScale:     0.5,
		RMSNormEpsilon:    1e-6,
		RopeDisabled:      true,
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
	var ropeCount, residualScales int
	for _, node := range nodes {
		if node.Op == tensor.OpRoPENormal {
			ropeCount++
		}
		if node.Op == tensor.OpScale &&
			node.Attrs.(tensor.ScaleAttributes).Value == 0.5 {
			residualScales++
		}
		if node.Op == tensor.OpAttention &&
			node.Attrs.(tensor.AttentionAttributes).Scale != 0.25 {
			t.Fatalf("Granite attention scale was not honored")
		}
	}
	if ropeCount != 0 || residualScales != 2 {
		t.Fatalf("Granite graph has RoPE=%d residual scales=%d, want 0/2", ropeCount, residualScales)
	}
}

func TestBuildDenseMaincoderNormalizesQKAfterRoPE(t *testing.T) {
	builder := tensor.NewBuilder()
	spec := Spec{
		Architecture:      "maincoder",
		EmbeddingLength:   8,
		FeedForwardLength: 12,
		HeadCount:         2,
		HeadCountKV:       1,
		KeyLength:         4,
		ValueLength:       4,
		RopeFrequencyBase: 10000,
		RMSNormEpsilon:    1e-6,
	}
	input := builder.Input("input", dtype.F32, tensor.MustShape(8, 2))
	weights := denseBlockInputs(builder, spec)
	output, err := BuildDenseBlock(builder, input, spec, weights, []uint32{0, 1})
	if err != nil {
		t.Fatal(err)
	}
	nodes, err := tensor.Topological(output)
	if err != nil {
		t.Fatal(err)
	}
	for _, node := range nodes {
		if node.Op != tensor.OpAttention {
			continue
		}
		for index, qk := range node.Inputs[:2] {
			if qk.Op != tensor.OpMultiply ||
				len(qk.Inputs) != 2 ||
				qk.Inputs[0].Op != tensor.OpRMSNorm ||
				qk.Inputs[0].Inputs[0].Op != tensor.OpRoPENormal {
				t.Fatalf("Maincoder attention input %d is not post-RoPE normalized", index)
			}
		}
		return
	}
	t.Fatal("Maincoder attention node was not found")
}

func TestBuildDenseOrionUsesAffineLayerNorm(t *testing.T) {
	builder := tensor.NewBuilder()
	spec := Spec{
		Architecture:      "orion",
		EmbeddingLength:   8,
		FeedForwardLength: 12,
		HeadCount:         2,
		HeadCountKV:       1,
		KeyLength:         4,
		ValueLength:       4,
		RopeFrequencyBase: 10000,
		LayerNormEpsilon:  1e-5,
	}
	input := builder.Input("input", dtype.F32, tensor.MustShape(8, 2))
	weights := denseBlockInputs(builder, spec)
	output, err := BuildDenseBlock(builder, input, spec, weights, []uint32{0, 1})
	if err != nil {
		t.Fatal(err)
	}
	nodes, err := tensor.Topological(output)
	if err != nil {
		t.Fatal(err)
	}
	var layerNormCount, ropeCount int
	for _, node := range nodes {
		switch node.Op {
		case tensor.OpLayerNorm:
			layerNormCount++
		case tensor.OpRoPENeoX:
			ropeCount++
		}
	}
	if layerNormCount != 2 || ropeCount != 2 {
		t.Fatalf(
			"Orion graph has LayerNorm=%d RoPE=%d, want 2/2",
			layerNormCount,
			ropeCount,
		)
	}
}

func TestBuildDenseStarCoder2UsesSequentialGELU(t *testing.T) {
	builder := tensor.NewBuilder()
	spec := Spec{
		Architecture:      "starcoder2",
		EmbeddingLength:   8,
		FeedForwardLength: 12,
		HeadCount:         2,
		HeadCountKV:       1,
		KeyLength:         4,
		ValueLength:       4,
		RopeFrequencyBase: 10000,
		LayerNormEpsilon:  1e-5,
	}
	input := builder.Input("input", dtype.F32, tensor.MustShape(8, 2))
	weights := denseBlockInputs(builder, spec)
	weights.AttentionOutputBias = builder.Input(
		"attn_output_bias", dtype.F32, tensor.MustShape(8),
	)
	weights.FeedForwardUpBias = builder.Input(
		"ffn_up_bias", dtype.F32, tensor.MustShape(12),
	)
	weights.FeedForwardDownBias = builder.Input(
		"ffn_down_bias", dtype.F32, tensor.MustShape(8),
	)
	output, err := BuildDenseBlock(builder, input, spec, weights, []uint32{0, 1})
	if err != nil {
		t.Fatal(err)
	}
	nodes, err := tensor.Topological(output)
	if err != nil {
		t.Fatal(err)
	}
	var layerNormCount, ropeCount, geluCount, siluCount int
	for _, node := range nodes {
		switch node.Op {
		case tensor.OpLayerNorm:
			layerNormCount++
		case tensor.OpRoPENeoX:
			ropeCount++
		case tensor.OpGELU:
			geluCount++
		case tensor.OpSiLU:
			siluCount++
		}
		if node == weights.FeedForwardGate {
			t.Fatal("StarCoder2 graph unexpectedly consumes an FFN gate")
		}
	}
	if layerNormCount != 2 || ropeCount != 2 || geluCount != 1 || siluCount != 0 {
		t.Fatalf(
			"StarCoder2 graph has LayerNorm=%d RoPE=%d GELU=%d SiLU=%d, want 2/2/1/0",
			layerNormCount,
			ropeCount,
			geluCount,
			siluCount,
		)
	}
}

func TestBuildDenseCodeShellUsesSequentialGELU(t *testing.T) {
	builder := tensor.NewBuilder()
	spec := Spec{
		Architecture:      "codeshell",
		EmbeddingLength:   8,
		FeedForwardLength: 12,
		HeadCount:         2,
		HeadCountKV:       1,
		KeyLength:         4,
		ValueLength:       4,
		RopeFrequencyBase: 10000,
		LayerNormEpsilon:  1e-5,
	}
	input := builder.Input("input", dtype.F32, tensor.MustShape(8, 1))
	weights := denseBlockInputs(builder, spec)
	weights.AttentionOutputBias = builder.Input("attn_output_bias", dtype.F32, tensor.MustShape(8))
	weights.FeedForwardUpBias = builder.Input("ffn_up_bias", dtype.F32, tensor.MustShape(12))
	weights.FeedForwardDownBias = builder.Input("ffn_down_bias", dtype.F32, tensor.MustShape(8))
	output, err := BuildDenseBlock(builder, input, spec, weights, []uint32{0})
	if err != nil {
		t.Fatal(err)
	}
	nodes, err := tensor.Topological(output)
	if err != nil {
		t.Fatal(err)
	}
	var geluCount int
	for _, node := range nodes {
		if node == weights.FeedForwardGate || node.Op == tensor.OpSiLU {
			t.Fatal("CodeShell graph unexpectedly consumes a gated SwiGLU path")
		}
		if node.Op == tensor.OpGELU {
			geluCount++
		}
	}
	if geluCount != 1 {
		t.Fatalf("CodeShell GELU count = %d, want 1", geluCount)
	}
}

func TestBuildDenseBaichuan7BUsesNormalRoPE(t *testing.T) {
	builder := tensor.NewBuilder()
	spec := Spec{
		Architecture:      "baichuan",
		EmbeddingLength:   8,
		FeedForwardLength: 12,
		HeadCount:         2,
		HeadCountKV:       2,
		KeyLength:         4,
		ValueLength:       4,
		RopeFrequencyBase: 10000,
		RMSNormEpsilon:    1e-6,
	}
	input := builder.Input("input", dtype.F32, tensor.MustShape(8, 2))
	output, err := BuildDenseBlock(
		builder,
		input,
		spec,
		denseBlockInputs(builder, spec),
		[]uint32{0, 1},
	)
	if err != nil {
		t.Fatal(err)
	}
	nodes, err := tensor.Topological(output)
	if err != nil {
		t.Fatal(err)
	}
	var normalRoPE, neoXRoPE int
	for _, node := range nodes {
		switch node.Op {
		case tensor.OpRoPENormal:
			normalRoPE++
		case tensor.OpRoPENeoX:
			neoXRoPE++
		}
	}
	if normalRoPE != 2 || neoXRoPE != 0 {
		t.Fatalf("Baichuan RoPE count normal/NeoX = %d/%d, want 2/0", normalRoPE, neoXRoPE)
	}
}

func TestBuildDenseArceeUsesSquaredReLU(t *testing.T) {
	builder := tensor.NewBuilder()
	spec := Spec{
		Architecture:      "arcee",
		EmbeddingLength:   8,
		FeedForwardLength: 12,
		HeadCount:         2,
		HeadCountKV:       1,
		KeyLength:         4,
		ValueLength:       4,
		RopeFrequencyBase: 10000,
		RMSNormEpsilon:    1e-5,
		AttentionScale:    0.25,
	}
	input := builder.Input("input", dtype.F32, tensor.MustShape(8, 2))
	weights := denseBlockInputs(builder, spec)
	output, err := BuildDenseBlock(builder, input, spec, weights, []uint32{0, 1})
	if err != nil {
		t.Fatal(err)
	}
	nodes, err := tensor.Topological(output)
	if err != nil {
		t.Fatal(err)
	}
	var squaredReLU, normalRoPE int
	for _, node := range nodes {
		if node == weights.FeedForwardGate {
			t.Fatal("Arcee graph unexpectedly consumes an FFN gate")
		}
		switch node.Op {
		case tensor.OpReLUSquared:
			squaredReLU++
		case tensor.OpRoPENormal:
			normalRoPE++
		case tensor.OpAttention:
			if node.Attrs.(tensor.AttentionAttributes).Scale != 0.25 {
				t.Fatal("Arcee metadata attention scale was not honored")
			}
		}
	}
	if squaredReLU != 1 || normalRoPE != 2 {
		t.Fatalf("Arcee graph squared-ReLU/RoPE = %d/%d, want 1/2", squaredReLU, normalRoPE)
	}
}

func TestBuildDenseNemotronUsesAffineNormAndSquaredReLU(t *testing.T) {
	builder := tensor.NewBuilder()
	spec := Spec{
		Architecture:      "nemotron",
		EmbeddingLength:   8,
		FeedForwardLength: 12,
		HeadCount:         2,
		HeadCountKV:       1,
		KeyLength:         4,
		ValueLength:       4,
		RopeFrequencyBase: 10000,
		LayerNormEpsilon:  1e-5,
	}
	input := builder.Input("input", dtype.F32, tensor.MustShape(8, 2))
	weights := denseBlockInputs(builder, spec)
	output, err := BuildDenseBlock(builder, input, spec, weights, []uint32{0, 1})
	if err != nil {
		t.Fatal(err)
	}
	nodes, err := tensor.Topological(output)
	if err != nil {
		t.Fatal(err)
	}
	var layerNorm, squaredReLU, neoXRoPE int
	for _, node := range nodes {
		if node == weights.FeedForwardGate {
			t.Fatal("Nemotron graph unexpectedly consumes an FFN gate")
		}
		switch node.Op {
		case tensor.OpLayerNorm:
			layerNorm++
		case tensor.OpReLUSquared:
			squaredReLU++
		case tensor.OpRoPENeoX:
			neoXRoPE++
		}
	}
	if layerNorm != 2 || squaredReLU != 1 || neoXRoPE != 2 {
		t.Fatalf(
			"Nemotron graph LayerNorm/squared-ReLU/NeoX = %d/%d/%d, want 2/1/2",
			layerNorm,
			squaredReLU,
			neoXRoPE,
		)
	}
}

func TestBuildDenseJais2UsesBiasesAndSquaredReLU(t *testing.T) {
	builder := tensor.NewBuilder()
	spec := Spec{
		Architecture: "jais2", EmbeddingLength: 8, FeedForwardLength: 12,
		HeadCount: 2, HeadCountKV: 2, KeyLength: 4, ValueLength: 4,
		RopeFrequencyBase: 10000, LayerNormEpsilon: 1e-5,
	}
	input := builder.Input("input", dtype.F32, tensor.MustShape(8, 2))
	weights := denseBlockInputs(builder, spec)
	weights.AttentionQBias = builder.Input("q_bias", dtype.F32, tensor.MustShape(8))
	weights.AttentionKBias = builder.Input("k_bias", dtype.F32, tensor.MustShape(8))
	weights.AttentionVBias = builder.Input("v_bias", dtype.F32, tensor.MustShape(8))
	weights.AttentionOutputBias = builder.Input("o_bias", dtype.F32, tensor.MustShape(8))
	weights.FeedForwardUpBias = builder.Input("up_bias", dtype.F32, tensor.MustShape(12))
	weights.FeedForwardDownBias = builder.Input("down_bias", dtype.F32, tensor.MustShape(8))
	output, err := BuildDenseBlock(builder, input, spec, weights, []uint32{0, 1})
	if err != nil {
		t.Fatal(err)
	}
	nodes, err := tensor.Topological(output)
	if err != nil {
		t.Fatal(err)
	}
	var layerNorm, squaredReLU, neoX int
	for _, node := range nodes {
		if node == weights.FeedForwardGate {
			t.Fatal("Jais2 graph unexpectedly consumes an FFN gate")
		}
		switch node.Op {
		case tensor.OpLayerNorm:
			layerNorm++
		case tensor.OpReLUSquared:
			squaredReLU++
		case tensor.OpRoPENeoX:
			neoX++
		}
	}
	if layerNorm != 2 || squaredReLU != 1 || neoX != 2 {
		t.Fatalf("Jais2 graph LayerNorm/squared-ReLU/NeoX = %d/%d/%d", layerNorm, squaredReLU, neoX)
	}
}

func TestBuildDenseJaisUsesBiasedFusedQKVAndALiBi(t *testing.T) {
	builder := tensor.NewBuilder()
	spec := Spec{
		Architecture: "jais", EmbeddingLength: 8, FeedForwardLength: 12,
		HeadCount: 2, HeadCountKV: 2, KeyLength: 4, ValueLength: 4,
		RopeDisabled: true, AttentionScale: 0.25, MaxALiBiBias: 8, LayerNormEpsilon: 1e-5,
	}
	input := builder.Input("input", dtype.F32, tensor.MustShape(8, 2))
	weights := denseBlockInputs(builder, spec)
	weights.AttentionQ, weights.AttentionK, weights.AttentionV = nil, nil, nil
	weights.AttentionQKV = builder.Input("qkv", dtype.F32, tensor.MustShape(8, 24))
	weights.AttentionQKVBias = builder.Input("qkv_bias", dtype.F32, tensor.MustShape(24))
	weights.AttentionOutputBias = builder.Input("output_bias", dtype.F32, tensor.MustShape(8))
	weights.FeedForwardGateBias = builder.Input("gate_bias", dtype.F32, tensor.MustShape(12))
	weights.FeedForwardUpBias = builder.Input("up_bias", dtype.F32, tensor.MustShape(12))
	weights.FeedForwardDownBias = builder.Input("down_bias", dtype.F32, tensor.MustShape(8))
	output, err := BuildDenseBlock(builder, input, spec, weights, []uint32{0, 1})
	if err != nil {
		t.Fatal(err)
	}
	nodes, err := tensor.Topological(output)
	if err != nil {
		t.Fatal(err)
	}
	var attention *tensor.Tensor
	var layerNorm, rope, silu int
	for _, node := range nodes {
		switch node.Op {
		case tensor.OpAttention:
			attention = node
		case tensor.OpLayerNorm:
			layerNorm++
		case tensor.OpRoPENormal, tensor.OpRoPENeoX:
			rope++
		case tensor.OpSiLU:
			silu++
		}
	}
	if attention == nil {
		t.Fatal("Jais graph is missing attention")
	}
	attrs := attention.Attrs.(tensor.AttentionAttributes)
	if attrs.Scale != 0.25 || attrs.MaxALiBiBias != 8 || !attrs.Causal ||
		layerNorm != 2 || rope != 0 || silu != 1 {
		t.Fatalf("unexpected Jais graph: attention=%+v norm=%d rope=%d silu=%d", attrs, layerNorm, rope, silu)
	}
}

func TestBuildDenseOLMoUsesUnweightedLayerNorm(t *testing.T) {
	builder := tensor.NewBuilder()
	spec := Spec{
		Architecture: "olmo", EmbeddingLength: 8, FeedForwardLength: 12,
		HeadCount: 2, HeadCountKV: 1, KeyLength: 4, ValueLength: 4,
		RopeFrequencyBase: 10000, LayerNormEpsilon: 1e-5,
	}
	input := builder.Input("input", dtype.F32, tensor.MustShape(8, 2))
	weights := denseBlockInputs(builder, spec)
	weights.AttentionNorm = nil
	weights.AttentionNormBias = nil
	weights.FeedForwardNorm = nil
	weights.FeedForwardNormBias = nil
	output, err := BuildDenseBlock(builder, input, spec, weights, []uint32{0, 1})
	if err != nil {
		t.Fatal(err)
	}
	nodes, err := tensor.Topological(output)
	if err != nil {
		t.Fatal(err)
	}
	var layerNorm, multiplyNorm, normalRoPE int
	for _, node := range nodes {
		switch node.Op {
		case tensor.OpLayerNorm:
			layerNorm++
		case tensor.OpRoPENormal:
			normalRoPE++
		case tensor.OpMultiply:
			for _, inputNode := range node.Inputs {
				if inputNode.Op == tensor.OpLayerNorm {
					multiplyNorm++
				}
			}
		}
	}
	if layerNorm != 2 || multiplyNorm != 0 || normalRoPE != 2 {
		t.Fatalf("OLMo graph LayerNorm/weighted/normal-RoPE = %d/%d/%d", layerNorm, multiplyNorm, normalRoPE)
	}
}

func TestBuildDenseSeedOSSUsesNeoXAndAttentionScale(t *testing.T) {
	builder := tensor.NewBuilder()
	spec := Spec{
		Architecture: "seed_oss", EmbeddingLength: 8, FeedForwardLength: 12,
		HeadCount: 2, HeadCountKV: 1, KeyLength: 4, ValueLength: 4,
		RopeFrequencyBase: 10000, RMSNormEpsilon: 1e-5, AttentionScale: 0.25,
	}
	input := builder.Input("input", dtype.F32, tensor.MustShape(8, 2))
	output, err := BuildDenseBlock(builder, input, spec, denseBlockInputs(builder, spec), []uint32{0, 1})
	if err != nil {
		t.Fatal(err)
	}
	nodes, err := tensor.Topological(output)
	if err != nil {
		t.Fatal(err)
	}
	var neoX int
	for _, node := range nodes {
		if node.Op == tensor.OpRoPENeoX {
			neoX++
		}
		if node.Op == tensor.OpAttention && node.Attrs.(tensor.AttentionAttributes).Scale != 0.25 {
			t.Fatal("Seed-OSS attention scale was not honored")
		}
	}
	if neoX != 2 {
		t.Fatalf("Seed-OSS NeoX count = %d, want 2", neoX)
	}
}

func TestBuildDenseCohere2UsesParallelResidualAndPartialSlidingRoPE(t *testing.T) {
	for _, test := range []struct {
		layer      uint32
		wantRoPE   int
		wantWindow uint32
	}{
		{layer: 0, wantRoPE: 2, wantWindow: 128},
		{layer: 3, wantRoPE: 0, wantWindow: 0},
	} {
		builder := tensor.NewBuilder()
		spec := Spec{
			Architecture: "cohere2", BlockCount: 4, EmbeddingLength: 8,
			FeedForwardLength: 12, HeadCount: 2, HeadCountKV: 1,
			KeyLength: 4, ValueLength: 4, RopeDimensionCount: 2,
			RopeFrequencyBase: 10000, RopeFrequencySWA: 20000,
			LayerNormEpsilon: 1e-5, SlidingWindow: 128,
			SlidingPattern: 4, NoRopeLayerStep: 4,
		}
		input := builder.Input("input", dtype.F32, tensor.MustShape(8, 2))
		weights := denseBlockInputs(builder, spec)
		weights.FeedForwardNorm = nil
		weights.FeedForwardNormBias = nil
		output, err := BuildDenseBlockCachedForLayer(
			builder, input, spec, weights, []uint32{0, 1}, nil, nil, test.layer,
		)
		if err != nil {
			t.Fatal(err)
		}
		nodes, err := tensor.Topological(output.Output)
		if err != nil {
			t.Fatal(err)
		}
		var ropeCount, layerNormCount int
		var window uint32
		var attentionInput, feedForwardInput *tensor.Tensor
		for _, node := range nodes {
			switch node.Op {
			case tensor.OpLayerNorm:
				layerNormCount++
			case tensor.OpRoPENormal:
				ropeCount++
				attributes := node.Attrs.(tensor.RoPEAttributes)
				if attributes.RotaryDimensions != 2 || attributes.FrequencyBase != 20000 {
					t.Fatalf("unexpected Cohere2 RoPE attributes: %+v", attributes)
				}
			case tensor.OpAttention:
				window = node.Attrs.(tensor.AttentionAttributes).Window
			case tensor.OpMulMat:
				if node.Inputs[0].Name == "attn_q" {
					attentionInput = node.Inputs[1]
				}
				if node.Inputs[0].Name == "ffn_up" {
					feedForwardInput = node.Inputs[1]
				}
			}
		}
		if ropeCount != test.wantRoPE || window != test.wantWindow || layerNormCount != 1 {
			t.Fatalf(
				"Cohere2 layer %d has RoPE=%d window=%d LayerNorm=%d",
				test.layer, ropeCount, window, layerNormCount,
			)
		}
		if attentionInput == nil || attentionInput != feedForwardInput {
			t.Fatal("Cohere2 attention and FFN do not share the normalized block input")
		}
	}
}

func TestBuildCohere2MoEBlockFusedExpertsAndSharedBranch(t *testing.T) {
	builder := tensor.NewBuilder()
	spec := Spec{
		Architecture: "cohere2moe", BlockCount: 2, EmbeddingLength: 8,
		FeedForwardLength: 12, HeadCount: 2, HeadCountKV: 1,
		KeyLength: 4, ValueLength: 4, RopeDimensionCount: 4,
		RopeFrequencyBase: 10000, RopeFrequencySWA: 20000,
		RMSNormEpsilon: 1e-5, SlidingWindow: 128,
		SlidingLayers: []bool{false, true}, LeadingDenseBlocks: 1,
		ExpertCount: 4, ExpertUsedCount: 2, ExpertFeedForward: 6,
		ExpertGatingFunc: 2, ExpertWeightsNorm: true, ExpertWeightsScale: 1.25,
		SharedExpertCount: 1, SharedExpertFF: 6,
	}
	input := builder.Input("input", dtype.F32, tensor.MustShape(8, 2))
	weights := LayerGraphWeights{
		AttentionNorm:            builder.Input("attn_norm", dtype.F32, tensor.MustShape(8)),
		AttentionQKV:             builder.Input("attn_qkv", dtype.F32, tensor.MustShape(8, 16)),
		AttentionOutput:          builder.Input("attn_output", dtype.F32, tensor.MustShape(8, 8)),
		FeedForwardRouter:        builder.Input("router", dtype.F32, tensor.MustShape(8, 4)),
		FeedForwardGateUpExperts: builder.Input("gate_up_exps", dtype.F32, tensor.MustShape(8, 12, 4)),
		FeedForwardDownExperts:   builder.Input("down_exps", dtype.F32, tensor.MustShape(6, 8, 4)),
		FeedForwardSharedGate:    builder.Input("shared_gate", dtype.F32, tensor.MustShape(8, 6)),
		FeedForwardSharedUp:      builder.Input("shared_up", dtype.F32, tensor.MustShape(8, 6)),
		FeedForwardSharedDown:    builder.Input("shared_down", dtype.F32, tensor.MustShape(6, 8)),
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
	var normalized, attentionInput *tensor.Tensor
	var rope, silu, halfScale int
	var window uint32
	for _, node := range nodes {
		switch node.Op {
		case tensor.OpMoE:
			moe = node
			normalized = node.Inputs[0]
		case tensor.OpRoPENormal:
			rope++
			if node.Attrs.(tensor.RoPEAttributes).FrequencyBase != 20000 {
				t.Fatalf("unexpected Cohere2-MoE RoPE: %+v", node.Attrs)
			}
		case tensor.OpAttention:
			window = node.Attrs.(tensor.AttentionAttributes).Window
		case tensor.OpSiLU:
			silu++
		case tensor.OpScale:
			if node.Attrs.(tensor.ScaleAttributes).Value == 0.5 {
				halfScale++
			}
		case tensor.OpMulMat:
			if node.Inputs[0].Name == "attn_qkv" {
				attentionInput = node.Inputs[1]
			}
		}
	}
	if moe == nil || !moe.Attrs.(tensor.MoEAttributes).FusedGateUp ||
		moe.Attrs.(tensor.MoEAttributes).Routing != tensor.MoERoutingSigmoid ||
		!moe.Attrs.(tensor.MoEAttributes).NormalizeTopKProb ||
		moe.Attrs.(tensor.MoEAttributes).Scale != 1.25 || rope != 2 || window != 128 ||
		silu != 1 || halfScale != 1 || normalized == nil || normalized != attentionInput {
		t.Fatalf("unexpected Cohere2-MoE graph: MoE=%+v RoPE=%d window=%d SiLU=%d half=%d", moe, rope, window, silu, halfScale)
	}
}

func TestBuildHYV3MoEBlockFusedExpertsAndPreRoPEQKNorm(t *testing.T) {
	builder := tensor.NewBuilder()
	spec := Spec{
		Architecture: "hy_v3", BlockCount: 2, EmbeddingLength: 8,
		FeedForwardLength: 12, HeadCount: 2, HeadCountKV: 1,
		KeyLength: 4, ValueLength: 4, RopeDimensionCount: 4,
		RopeFrequencyBase: 10000, RMSNormEpsilon: 1e-5,
		ExpertCount: 4, ExpertUsedCount: 2, ExpertFeedForward: 6,
		ExpertGatingFunc: 2, ExpertWeightsNorm: true, ExpertWeightsScale: 1.25,
		SharedExpertFF: 6,
	}
	input := builder.Input("input", dtype.F32, tensor.MustShape(8, 2))
	weights := LayerGraphWeights{
		AttentionNorm:            builder.Input("attn_norm", dtype.F32, tensor.MustShape(8)),
		AttentionQKV:             builder.Input("attn_qkv", dtype.F32, tensor.MustShape(8, 16)),
		AttentionQNorm:           builder.Input("attn_q_norm", dtype.F32, tensor.MustShape(4)),
		AttentionKNorm:           builder.Input("attn_k_norm", dtype.F32, tensor.MustShape(4)),
		AttentionOutput:          builder.Input("attn_output", dtype.F32, tensor.MustShape(8, 8)),
		FeedForwardNorm:          builder.Input("ffn_norm", dtype.F32, tensor.MustShape(8)),
		FeedForwardRouter:        builder.Input("router", dtype.F32, tensor.MustShape(8, 4)),
		FeedForwardGateUpExperts: builder.Input("gate_up_exps", dtype.F32, tensor.MustShape(8, 12, 4)),
		FeedForwardDownExperts:   builder.Input("down_exps", dtype.F32, tensor.MustShape(6, 8, 4)),
		FeedForwardExpertBias:    builder.Input("expert_bias", dtype.F32, tensor.MustShape(4)),
		FeedForwardSharedGate:    builder.Input("shared_gate", dtype.F32, tensor.MustShape(8, 6)),
		FeedForwardSharedUp:      builder.Input("shared_up", dtype.F32, tensor.MustShape(8, 6)),
		FeedForwardSharedDown:    builder.Input("shared_down", dtype.F32, tensor.MustShape(6, 8)),
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
	var rope, preNormRoPE, silu int
	for _, node := range nodes {
		switch node.Op {
		case tensor.OpMoE:
			moe = node
		case tensor.OpRoPENeoX:
			rope++
			if node.Inputs[0].Op == tensor.OpMultiply && node.Inputs[0].Inputs[0].Op == tensor.OpRMSNorm {
				preNormRoPE++
			}
		case tensor.OpSiLU:
			silu++
		}
	}
	if moe == nil || !moe.Attrs.(tensor.MoEAttributes).FusedGateUp ||
		moe.Attrs.(tensor.MoEAttributes).Routing != tensor.MoERoutingSigmoid ||
		!moe.Attrs.(tensor.MoEAttributes).NormalizeTopKProb ||
		moe.Attrs.(tensor.MoEAttributes).Scale != 1.25 || len(moe.Inputs) != 6 ||
		rope != 2 || preNormRoPE != 2 || silu != 1 {
		t.Fatalf("unexpected HY-V3 graph: MoE=%+v RoPE=%d pre-norm=%d SiLU=%d", moe, rope, preNormRoPE, silu)
	}
}

func TestBuildDeepSeek2OCRMoEBlockUsesSplitQKVAndSharedExpert(t *testing.T) {
	builder := tensor.NewBuilder()
	spec := Spec{
		Architecture: "deepseek2-ocr", BlockCount: 2, EmbeddingLength: 8,
		FeedForwardLength: 12, HeadCount: 2, HeadCountKV: 2,
		KeyLength: 4, ValueLength: 4, RopeDimensionCount: 4,
		RopeFrequencyBase: 10000, RMSNormEpsilon: 1e-6,
		ExpertCount: 4, ExpertUsedCount: 2, ExpertFeedForward: 6,
		ExpertGatingFunc: 1, ExpertWeightsScale: 1, SharedExpertFF: 12,
	}
	input := builder.Input("input", dtype.F32, tensor.MustShape(8, 2))
	weights := LayerGraphWeights{
		AttentionNorm:            builder.Input("attn_norm", dtype.F32, tensor.MustShape(8)),
		AttentionQ:               builder.Input("attn_q", dtype.F32, tensor.MustShape(8, 8)),
		AttentionK:               builder.Input("attn_k", dtype.F32, tensor.MustShape(8, 8)),
		AttentionV:               builder.Input("attn_v", dtype.F32, tensor.MustShape(8, 8)),
		AttentionOutput:          builder.Input("attn_output", dtype.F32, tensor.MustShape(8, 8)),
		FeedForwardNorm:          builder.Input("ffn_norm", dtype.F32, tensor.MustShape(8)),
		FeedForwardRouter:        builder.Input("router", dtype.F32, tensor.MustShape(8, 4)),
		FeedForwardGateUpExperts: builder.Input("gate_up_exps", dtype.F32, tensor.MustShape(8, 12, 4)),
		FeedForwardDownExperts:   builder.Input("down_exps", dtype.F32, tensor.MustShape(6, 8, 4)),
		FeedForwardExpertBias:    builder.Input("expert_bias", dtype.F32, tensor.MustShape(4)),
		FeedForwardSharedGate:    builder.Input("shared_gate", dtype.F32, tensor.MustShape(8, 12)),
		FeedForwardSharedUp:      builder.Input("shared_up", dtype.F32, tensor.MustShape(8, 12)),
		FeedForwardSharedDown:    builder.Input("shared_down", dtype.F32, tensor.MustShape(12, 8)),
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
	var rope, qkNorm, silu int
	for _, node := range nodes {
		switch node.Op {
		case tensor.OpMoE:
			moe = node
		case tensor.OpRoPENeoX:
			rope++
		case tensor.OpRMSNorm:
			if node.Name == "q_norm" || node.Name == "k_norm" {
				qkNorm++
			}
		case tensor.OpSiLU:
			silu++
		}
	}
	if moe == nil || !moe.Attrs.(tensor.MoEAttributes).FusedGateUp ||
		moe.Attrs.(tensor.MoEAttributes).Routing != tensor.MoERoutingSoftmax ||
		moe.Attrs.(tensor.MoEAttributes).NormalizeTopKProb ||
		moe.Attrs.(tensor.MoEAttributes).Scale != 1 || len(moe.Inputs) != 6 ||
		rope != 2 || qkNorm != 0 || silu != 1 {
		t.Fatalf("unexpected DeepSeek2-OCR graph: MoE=%+v RoPE=%d Q/K norm=%d SiLU=%d", moe, rope, qkNorm, silu)
	}
}

func TestBuildErnie45MoEBlockUsesBiasedUngatedExpertsAndSharedBranch(t *testing.T) {
	builder := tensor.NewBuilder()
	spec := Spec{
		Architecture: "ernie4_5-moe", BlockCount: 4, EmbeddingLength: 8,
		FeedForwardLength: 12, HeadCount: 2, HeadCountKV: 1,
		KeyLength: 4, ValueLength: 4, RopeDimensionCount: 4,
		RopeFrequencyBase: 10000, RMSNormEpsilon: 1e-6,
		ExpertCount: 4, ExpertUsedCount: 2, ExpertFeedForward: 6,
		ExpertWeightsScale: 1.25, ExpertWeightsNorm: true,
		LeadingDenseBlocks: 1, MoELayerStep: 2, SharedExpertFF: 5,
	}
	input := builder.Input("input", dtype.F32, tensor.MustShape(8, 2))
	weights := LayerGraphWeights{
		AttentionNorm:          builder.Input("attn_norm", dtype.F32, tensor.MustShape(8)),
		AttentionQKV:           builder.Input("attn_qkv", dtype.F32, tensor.MustShape(8, 16)),
		AttentionOutput:        builder.Input("attn_output", dtype.F32, tensor.MustShape(8, 8)),
		FeedForwardNorm:        builder.Input("ffn_norm", dtype.F32, tensor.MustShape(8)),
		FeedForwardRouter:      builder.Input("router", dtype.F32, tensor.MustShape(8, 4)),
		FeedForwardUpExperts:   builder.Input("up_exps", dtype.F32, tensor.MustShape(8, 6, 4)),
		FeedForwardDownExperts: builder.Input("down_exps", dtype.F32, tensor.MustShape(6, 8, 4)),
		FeedForwardExpertBias:  builder.Input("expert_bias", dtype.F32, tensor.MustShape(4)),
		FeedForwardSharedGate:  builder.Input("shared_gate", dtype.F32, tensor.MustShape(8, 5)),
		FeedForwardSharedUp:    builder.Input("shared_up", dtype.F32, tensor.MustShape(8, 5)),
		FeedForwardSharedDown:  builder.Input("shared_down", dtype.F32, tensor.MustShape(5, 8)),
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
	var normalRoPE, silu int
	for _, node := range nodes {
		switch node.Op {
		case tensor.OpMoE:
			moe = node
		case tensor.OpRoPENormal:
			normalRoPE++
		case tensor.OpSiLU:
			silu++
		}
	}
	if moe == nil {
		t.Fatal("ERNIE MoE node is missing")
	}
	attributes := moe.Attrs.(tensor.MoEAttributes)
	if attributes.Gated || attributes.Routing != tensor.MoERoutingSoftmax ||
		!attributes.NormalizeTopKProb || attributes.Scale != 1.25 ||
		len(moe.Inputs) != 6 || normalRoPE != 2 || silu != 1 {
		t.Fatalf("unexpected ERNIE MoE graph: MoE=%+v RoPE=%d SiLU=%d", moe, normalRoPE, silu)
	}
}

func TestBuildPaddleOCRBlockUsesMRoPEAndOutputBias(t *testing.T) {
	testBuildMRoPETextDecoderBlock(t, "paddleocr")
}

func TestBuildQwen2VLBlockUsesMRoPEAndOutputBias(t *testing.T) {
	testBuildMRoPETextDecoderBlock(t, "qwen2vl")
}

func TestBuildQwen3VLBlockUsesQKNormAndMRoPE(t *testing.T) {
	testBuildMRoPETextDecoderBlock(t, "qwen3vl")
}

func TestBuildQwen3VLMoEBlockUsesQKNormMRoPEAndNormalizedExperts(t *testing.T) {
	builder := tensor.NewBuilder()
	spec := Spec{
		Architecture: "qwen3vlmoe", EmbeddingLength: 8, FeedForwardLength: 24,
		ExpertCount: 4, ExpertUsedCount: 2, ExpertFeedForward: 12, ExpertWeightsScale: 1.25,
		HeadCount: 2, HeadCountKV: 1, KeyLength: 4, ValueLength: 4,
		RopeDimensionCount: 4, RopeSections: [4]int32{1, 1, 0, 0},
		RopeFrequencyBase: 10000, RopeScalingType: "linear", RopeScalingFactor: 4,
		RMSNormEpsilon: 1e-6,
	}
	input := builder.Input("input", dtype.F32, tensor.MustShape(8, 2))
	weights := denseBlockInputs(builder, spec)
	weights.AttentionQ, weights.AttentionK, weights.AttentionV = nil, nil, nil
	weights.AttentionQKV = builder.Input("attn_qkv", dtype.F32, tensor.MustShape(8, 16))
	weights.FeedForwardGate, weights.FeedForwardUp, weights.FeedForwardDown = nil, nil, nil
	weights.FeedForwardRouter = builder.Input("ffn_router", dtype.F32, tensor.MustShape(8, 4))
	weights.FeedForwardGateExperts = builder.Input("ffn_gate_exps", dtype.F32, tensor.MustShape(8, 12, 4))
	weights.FeedForwardUpExperts = builder.Input("ffn_up_exps", dtype.F32, tensor.MustShape(8, 12, 4))
	weights.FeedForwardDownExperts = builder.Input("ffn_down_exps", dtype.F32, tensor.MustShape(12, 8, 4))
	result, err := BuildDenseBlockCachedForLayer(builder, input, spec, weights, []uint32{0, 1}, nil, nil, 0)
	if err != nil {
		t.Fatal(err)
	}
	nodes, err := tensor.Topological(result.Output)
	if err != nil {
		t.Fatal(err)
	}
	var moe *tensor.Tensor
	var multiRoPE, rmsNorm int
	for _, node := range nodes {
		switch node.Op {
		case tensor.OpMoE:
			moe = node
		case tensor.OpRoPEMulti:
			multiRoPE++
		case tensor.OpRMSNorm:
			rmsNorm++
		}
	}
	if moe == nil {
		t.Fatal("Qwen3-VL-MoE graph is missing MoE")
	}
	attrs := moe.Attrs.(tensor.MoEAttributes)
	if !attrs.Gated || attrs.Routing != tensor.MoERoutingSoftmax || !attrs.NormalizeTopKProb ||
		attrs.TopK != 2 || attrs.Scale != 1.25 || multiRoPE != 2 || rmsNorm != 4 {
		t.Fatalf("unexpected Qwen3-VL-MoE graph: MoE=%+v MRoPE=%d RMS=%d", attrs, multiRoPE, rmsNorm)
	}
}

func testBuildMRoPETextDecoderBlock(t *testing.T, architecture string) {
	builder := tensor.NewBuilder()
	spec := Spec{
		Architecture: architecture, EmbeddingLength: 8, FeedForwardLength: 12,
		HeadCount: 2, HeadCountKV: 1, KeyLength: 4, ValueLength: 4,
		RopeDimensionCount: 4, RopeSections: [4]int32{1, 1, 0, 0},
		RopeFrequencyBase: 10000, RopeScalingType: "linear", RopeScalingFactor: 4,
		RMSNormEpsilon: 1e-6,
	}
	input := builder.Input("input", dtype.F32, tensor.MustShape(8, 2))
	weights := denseBlockInputs(builder, spec)
	weights.AttentionQ = nil
	weights.AttentionK = nil
	weights.AttentionV = nil
	weights.AttentionQKV = builder.Input("attn_qkv", dtype.F32, tensor.MustShape(8, 16))
	weights.AttentionOutputBias = builder.Input("attn_output_bias", dtype.F32, tensor.MustShape(8))
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
	var multiRoPE, outputBias, rmsNorm int
	for _, node := range nodes {
		if node.Op == tensor.OpRoPEMulti {
			multiRoPE++
			attributes := node.Attrs.(tensor.RoPEMultiAttributes)
			if attributes.Sections != spec.RopeSections || attributes.FrequencyScale != 0.25 {
				t.Fatalf("unexpected %s MRoPE: %+v", architecture, attributes)
			}
		}
		if node.Op == tensor.OpAdd && len(node.Inputs) == 2 && node.Inputs[1].Name == "attn_output_bias" {
			outputBias++
		}
		if node.Op == tensor.OpRMSNorm {
			rmsNorm++
		}
	}
	if multiRoPE != 2 || outputBias != 1 || (architecture == "qwen3vl" && rmsNorm != 4) {
		t.Fatalf("%s graph has MRoPE=%d output-bias=%d RMS=%d", architecture, multiRoPE, outputBias, rmsNorm)
	}
}

func TestBuildDenseCommandR64UsesParallelResidualAndQKNorms(t *testing.T) {
	builder := tensor.NewBuilder()
	spec := Spec{
		Architecture: "command-r", BlockCount: 64, EmbeddingLength: 8,
		FeedForwardLength: 12, HeadCount: 2, HeadCountKV: 1,
		KeyLength: 4, ValueLength: 4, RopeFrequencyBase: 8000000,
		LayerNormEpsilon: 1e-5,
	}
	input := builder.Input("input", dtype.F32, tensor.MustShape(8, 2))
	weights := denseBlockInputs(builder, spec)
	weights.FeedForwardNorm = nil
	weights.FeedForwardNormBias = nil
	weights.AttentionQNorm = builder.Input("attn_q_norm", dtype.F32, tensor.MustShape(4, 2))
	weights.AttentionKNorm = builder.Input("attn_k_norm", dtype.F32, tensor.MustShape(4, 1))
	output, err := BuildDenseBlock(builder, input, spec, weights, []uint32{0, 1})
	if err != nil {
		t.Fatal(err)
	}
	nodes, err := tensor.Topological(output)
	if err != nil {
		t.Fatal(err)
	}
	var layerNormCount, ropeCount int
	var attentionInput, feedForwardInput *tensor.Tensor
	for _, node := range nodes {
		switch node.Op {
		case tensor.OpLayerNorm:
			layerNormCount++
		case tensor.OpRoPENormal:
			ropeCount++
		case tensor.OpMulMat:
			if node.Inputs[0].Name == "attn_q" {
				attentionInput = node.Inputs[1]
			}
			if node.Inputs[0].Name == "ffn_up" {
				feedForwardInput = node.Inputs[1]
			}
		}
	}
	if layerNormCount != 3 || ropeCount != 2 {
		t.Fatalf("Command R graph has LayerNorm=%d RoPE=%d, want 3/2", layerNormCount, ropeCount)
	}
	if attentionInput == nil || attentionInput != feedForwardInput {
		t.Fatal("Command R attention and FFN do not share the normalized block input")
	}
}

func TestBuildDensePLaMoUsesParallelResidualAndNeoX(t *testing.T) {
	builder := tensor.NewBuilder()
	spec := Spec{
		Architecture: "plamo", BlockCount: 40, EmbeddingLength: 8,
		FeedForwardLength: 12, HeadCount: 2, HeadCountKV: 1,
		KeyLength: 4, ValueLength: 4, RopeFrequencyBase: 10000,
		RMSNormEpsilon: 1e-5,
	}
	input := builder.Input("input", dtype.F32, tensor.MustShape(8, 2))
	weights := denseBlockInputs(builder, spec)
	weights.FeedForwardNorm = nil
	output, err := BuildDenseBlock(builder, input, spec, weights, []uint32{0, 1})
	if err != nil {
		t.Fatal(err)
	}
	nodes, err := tensor.Topological(output)
	if err != nil {
		t.Fatal(err)
	}
	var rmsNormCount, ropeCount int
	var attentionInput, feedForwardInput *tensor.Tensor
	for _, node := range nodes {
		switch node.Op {
		case tensor.OpRMSNorm:
			rmsNormCount++
		case tensor.OpRoPENeoX:
			ropeCount++
		case tensor.OpMulMat:
			if node.Inputs[0].Name == "attn_q" {
				attentionInput = node.Inputs[1]
			}
			if node.Inputs[0].Name == "ffn_up" {
				feedForwardInput = node.Inputs[1]
			}
		}
	}
	if rmsNormCount != 1 || ropeCount != 2 {
		t.Fatalf("PLaMo graph has RMSNorm=%d RoPE=%d, want 1/2", rmsNormCount, ropeCount)
	}
	if attentionInput == nil || attentionInput != feedForwardInput {
		t.Fatal("PLaMo attention and FFN do not share the normalized block input")
	}
}

func TestBuildDensePLaMo3UsesPreRoPEQKNormPostNormAndFusedSwiGLU(t *testing.T) {
	builder := tensor.NewBuilder()
	spec := Spec{
		Architecture: "plamo3", BlockCount: 2, EmbeddingLength: 8,
		FeedForwardLength: 12, HeadCount: 2, HeadCountKV: 1,
		KeyLength: 2, ValueLength: 2, RopeDimensionCount: 2,
		RopeFrequencyBase: 10000, RopeFrequencySWA: 20000,
		RMSNormEpsilon: 1e-6, SlidingWindow: 128, SlidingPattern: 2,
		LayerHeadCounts: []uint32{2, 4}, LayerKVHeadCounts: []uint32{1, 2},
		LayerFeedForward: []uint32{12, 16},
	}
	input := builder.Input("input", dtype.F32, tensor.MustShape(8, 2))
	weights := LayerGraphWeights{
		AttentionNorm:       builder.Input("attn_norm", dtype.F32, tensor.MustShape(8)),
		AttentionQKV:        builder.Input("attn_qkv", dtype.F32, tensor.MustShape(8, 8)),
		AttentionQNorm:      builder.Input("attn_q_norm", dtype.F32, tensor.MustShape(2)),
		AttentionKNorm:      builder.Input("attn_k_norm", dtype.F32, tensor.MustShape(2)),
		AttentionOutput:     builder.Input("attn_output", dtype.F32, tensor.MustShape(4, 8)),
		AttentionPostNorm:   builder.Input("attn_post_norm", dtype.F32, tensor.MustShape(8)),
		FeedForwardNorm:     builder.Input("ffn_norm", dtype.F32, tensor.MustShape(8)),
		FeedForwardUp:       builder.Input("ffn_up", dtype.F32, tensor.MustShape(8, 24)),
		FeedForwardDown:     builder.Input("ffn_down", dtype.F32, tensor.MustShape(12, 8)),
		FeedForwardPostNorm: builder.Input("ffn_post_norm", dtype.F32, tensor.MustShape(8)),
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
	var rmsNorms, groupSlices int
	var attention *tensor.Tensor
	for _, node := range nodes {
		switch node.Op {
		case tensor.OpRMSNorm:
			rmsNorms++
		case tensor.OpGroupSlice:
			groupSlices++
		case tensor.OpAttention:
			attention = node
		}
		if node.Op == tensor.OpRoPENeoX {
			attributes := node.Attrs.(tensor.RoPEAttributes)
			if attributes.FrequencyBase != 20000 || node.Inputs[0].Op != tensor.OpMultiply ||
				node.Inputs[0].Inputs[0].Op != tensor.OpRMSNorm {
				t.Fatalf("PLaMo 3 NeoX RoPE does not follow Q/K norm: %+v", node)
			}
		}
	}
	if attention == nil || attention.Attrs.(tensor.AttentionAttributes).Window != 128 ||
		rmsNorms != 6 || groupSlices != 5 {
		t.Fatalf("PLaMo 3 graph: attention=%+v RMSNorm=%d slices=%d", attention, rmsNorms, groupSlices)
	}
}

func TestBuildDenseStableLMOptionalNormLayouts(t *testing.T) {
	for _, sequential := range []bool{false, true} {
		builder := tensor.NewBuilder()
		spec := Spec{
			Architecture: "stablelm", BlockCount: 40, EmbeddingLength: 8,
			FeedForwardLength: 12, HeadCount: 2, HeadCountKV: 1,
			KeyLength: 4, ValueLength: 4, RopeDimensionCount: 2,
			RopeFrequencyBase: 10000, LayerNormEpsilon: 1e-5,
		}
		input := builder.Input("input", dtype.F32, tensor.MustShape(8, 2))
		weights := denseBlockInputs(builder, spec)
		if sequential {
			weights.AttentionQNorm = nil
			weights.AttentionKNorm = nil
		} else {
			weights.FeedForwardNorm = nil
			weights.FeedForwardNormBias = nil
			weights.AttentionQNorm = builder.Input("attn_q_norm", dtype.F32, tensor.MustShape(4, 2))
			weights.AttentionKNorm = builder.Input("attn_k_norm", dtype.F32, tensor.MustShape(4, 1))
		}
		output, err := BuildDenseBlock(builder, input, spec, weights, []uint32{0, 1})
		if err != nil {
			t.Fatal(err)
		}
		nodes, err := tensor.Topological(output)
		if err != nil {
			t.Fatal(err)
		}
		var layerNormCount, ropeCount int
		var attentionInput, feedForwardInput *tensor.Tensor
		for _, node := range nodes {
			switch node.Op {
			case tensor.OpLayerNorm:
				layerNormCount++
			case tensor.OpRoPENeoX:
				ropeCount++
				if node.Attrs.(tensor.RoPEAttributes).RotaryDimensions != 2 {
					t.Fatal("StableLM partial rotary dimension was not honored")
				}
			case tensor.OpMulMat:
				if node.Inputs[0].Name == "attn_q" {
					attentionInput = node.Inputs[1]
				}
				if node.Inputs[0].Name == "ffn_up" {
					feedForwardInput = node.Inputs[1]
				}
			}
		}
		wantNorms := 3
		if sequential {
			wantNorms = 2
		}
		if layerNormCount != wantNorms || ropeCount != 2 {
			t.Fatalf("StableLM graph has LayerNorm=%d RoPE=%d", layerNormCount, ropeCount)
		}
		if (attentionInput == feedForwardInput) == sequential {
			t.Fatal("StableLM residual topology does not match its FFN norm layout")
		}
	}
}

func TestBuildDensePhi2FusedQKVAndQueryScale(t *testing.T) {
	builder := tensor.NewBuilder()
	spec := Spec{
		Architecture: "phi2", BlockCount: 1, EmbeddingLength: 8,
		FeedForwardLength: 12, HeadCount: 2, HeadCountKV: 2,
		KeyLength: 4, ValueLength: 4, RopeDimensionCount: 2,
		RopeFrequencyBase: 10000, LayerNormEpsilon: 1e-5,
	}
	input := builder.Input("input", dtype.F32, tensor.MustShape(8, 2))
	weights := denseBlockInputs(builder, spec)
	weights.AttentionQ = nil
	weights.AttentionK = nil
	weights.AttentionV = nil
	weights.AttentionQKV = builder.Input("attn_qkv", dtype.F32, tensor.MustShape(8, 24))
	weights.AttentionQKVBias = builder.Input("attn_qkv_bias", dtype.F32, tensor.MustShape(24))
	weights.AttentionOutputBias = builder.Input("attn_output_bias", dtype.F32, tensor.MustShape(8))
	weights.FeedForwardNorm = nil
	weights.FeedForwardNormBias = nil
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
	var slices, ropeCount, geluCount int
	var attentionScale, queryScale float32
	var attentionInput, feedForwardInput *tensor.Tensor
	offsets := map[uint64]bool{}
	for _, node := range nodes {
		switch node.Op {
		case tensor.OpGroupSlice:
			slices++
			offsets[node.Attrs.(tensor.GroupSliceAttributes).Offset] = true
		case tensor.OpRoPENeoX:
			ropeCount++
			if node.Attrs.(tensor.RoPEAttributes).RotaryDimensions != 2 {
				t.Fatal("Phi-2 partial rotary dimension was not honored")
			}
		case tensor.OpGELU:
			geluCount++
		case tensor.OpScale:
			queryScale = node.Attrs.(tensor.ScaleAttributes).Value
		case tensor.OpAttention:
			attentionScale = node.Attrs.(tensor.AttentionAttributes).Scale
		case tensor.OpMulMat:
			if node.Inputs[0].Name == "attn_qkv" {
				attentionInput = node.Inputs[1]
			}
			if node.Inputs[0].Name == "ffn_up" {
				feedForwardInput = node.Inputs[1]
			}
		}
	}
	if slices != 3 || !offsets[0] || !offsets[8] || !offsets[16] ||
		ropeCount != 2 || geluCount != 1 || queryScale != 0.5 || attentionScale != 1 {
		t.Fatalf(
			"unexpected Phi-2 graph: slices=%d offsets=%v rope=%d GELU=%d qscale=%v ascale=%v",
			slices, offsets, ropeCount, geluCount, queryScale, attentionScale,
		)
	}
	if attentionInput == nil || attentionInput != feedForwardInput {
		t.Fatal("Phi-2 attention and FFN do not share the normalized block input")
	}
}

func TestBuildDensePhi3FusedProjectionsAndLongRoPE(t *testing.T) {
	builder := tensor.NewBuilder()
	spec := Spec{
		Architecture: "phi3", BlockCount: 1, ContextLength: 128,
		OriginalContextLength: 32, EmbeddingLength: 8, FeedForwardLength: 12,
		HeadCount: 2, HeadCountKV: 2, KeyLength: 4, ValueLength: 4,
		RopeDimensionCount: 4, RopeFrequencyBase: 10000,
		RopeScalingType: "longrope", RopeAttentionFactor: 1.25, RMSNormEpsilon: 1e-5,
	}
	input := builder.Input("input", dtype.F32, tensor.MustShape(8, 2))
	weights := denseBlockInputs(builder, spec)
	weights.AttentionQ = nil
	weights.AttentionK = nil
	weights.AttentionV = nil
	weights.AttentionQKV = builder.Input("attn_qkv", dtype.F32, tensor.MustShape(8, 24))
	weights.AttentionQKVBias = builder.Input("attn_qkv_bias", dtype.F32, tensor.MustShape(24))
	weights.FeedForwardGate = nil
	weights.FeedForwardUp = builder.Input("ffn_up", dtype.F32, tensor.MustShape(8, 24))
	weights.RopeFactors = builder.Input("rope_long", dtype.F32, tensor.MustShape(2))
	output, err := BuildDenseBlock(builder, input, spec, weights, []uint32{0, 1})
	if err != nil {
		t.Fatal(err)
	}
	nodes, err := tensor.Topological(output)
	if err != nil {
		t.Fatal(err)
	}
	var slices, silu, ropeWithFactors, queryPrescale, ropeScale int
	for _, node := range nodes {
		switch node.Op {
		case tensor.OpGroupSlice:
			slices++
		case tensor.OpSiLU:
			silu++
		case tensor.OpRoPENeoX:
			if len(node.Inputs) == 2 && node.Inputs[1] == weights.RopeFactors {
				ropeWithFactors++
			}
		case tensor.OpScale:
			value := node.Attrs.(tensor.ScaleAttributes).Value
			if value == 0.5 {
				queryPrescale++
			}
			if value == 1.25 {
				ropeScale++
			}
		}
	}
	if slices != 5 || silu != 1 || ropeWithFactors != 2 ||
		queryPrescale != 1 || ropeScale != 2 {
		t.Fatalf("unexpected Phi-3 graph: slices=%d SiLU=%d rope=%d qscale=%d rope-scale=%d", slices, silu, ropeWithFactors, queryPrescale, ropeScale)
	}
}

func TestBuildDensePhiMoEBlock(t *testing.T) {
	builder := tensor.NewBuilder()
	spec := Spec{
		Architecture: "phimoe", BlockCount: 1, ContextLength: 128,
		OriginalContextLength: 32, EmbeddingLength: 8, FeedForwardLength: 12,
		ExpertCount: 4, ExpertUsedCount: 2, ExpertFeedForward: 12, ExpertWeightsScale: 1,
		HeadCount: 2, HeadCountKV: 2, KeyLength: 4, ValueLength: 4,
		RopeDimensionCount: 4, RopeFrequencyBase: 10000,
		RopeScalingType: "longrope", RopeAttentionFactor: 1.25, RMSNormEpsilon: 1e-5,
	}
	input := builder.Input("input", dtype.F32, tensor.MustShape(8, 2))
	attentionNormBias := builder.Input("attn_norm_bias", dtype.F32, tensor.MustShape(8))
	feedForwardNormBias := builder.Input("ffn_norm_bias", dtype.F32, tensor.MustShape(8))
	weights := LayerGraphWeights{
		AttentionNorm:          builder.Input("attn_norm", dtype.F32, tensor.MustShape(8)),
		AttentionNormBias:      attentionNormBias,
		AttentionQ:             builder.Input("q", dtype.F32, tensor.MustShape(8, 8)),
		AttentionK:             builder.Input("k", dtype.F32, tensor.MustShape(8, 8)),
		AttentionV:             builder.Input("v", dtype.F32, tensor.MustShape(8, 8)),
		AttentionOutput:        builder.Input("attn_out", dtype.F32, tensor.MustShape(8, 8)),
		AttentionOutputBias:    builder.Input("attn_out_bias", dtype.F32, tensor.MustShape(8)),
		FeedForwardNorm:        builder.Input("ffn_norm", dtype.F32, tensor.MustShape(8)),
		FeedForwardNormBias:    feedForwardNormBias,
		FeedForwardRouter:      builder.Input("router", dtype.F32, tensor.MustShape(8, 4)),
		FeedForwardGateExperts: builder.Input("gate_exps", dtype.F32, tensor.MustShape(8, 12, 4)),
		FeedForwardUpExperts:   builder.Input("up_exps", dtype.F32, tensor.MustShape(8, 12, 4)),
		FeedForwardDownExperts: builder.Input("down_exps", dtype.F32, tensor.MustShape(12, 8, 4)),
		RopeFactors:            builder.Input("rope_long", dtype.F32, tensor.MustShape(2)),
	}
	output, err := BuildDenseBlock(builder, input, spec, weights, []uint32{0, 1})
	if err != nil {
		t.Fatal(err)
	}
	nodes, err := tensor.Topological(output)
	if err != nil {
		t.Fatal(err)
	}
	var moe *tensor.Tensor
	var normBiasAdds, ropeWithFactors, queryPrescale int
	for _, node := range nodes {
		if node.Op == tensor.OpMoE {
			moe = node
		}
		if node.Op == tensor.OpAdd && len(node.Inputs) == 2 &&
			(node.Inputs[1] == attentionNormBias || node.Inputs[1] == feedForwardNormBias) {
			normBiasAdds++
		}
		if node.Op == tensor.OpRoPENeoX && len(node.Inputs) == 2 && node.Inputs[1] == weights.RopeFactors {
			ropeWithFactors++
		}
		if node.Op == tensor.OpScale && node.Attrs.(tensor.ScaleAttributes).Value == 0.5 {
			queryPrescale++
		}
	}
	if moe == nil || !moe.Attrs.(tensor.MoEAttributes).NormalizeTopKProb ||
		normBiasAdds != 2 || ropeWithFactors != 2 || queryPrescale != 1 {
		t.Fatalf("unexpected PhiMoE graph: moe=%v norm-bias=%d rope=%d qscale=%d", moe != nil, normBiasAdds, ropeWithFactors, queryPrescale)
	}
}

func TestBuildDenseEXAOneMoEBlock(t *testing.T) {
	builder := tensor.NewBuilder()
	spec := Spec{
		Architecture: "exaone-moe", BlockCount: 4, EmbeddingLength: 8, FeedForwardLength: 16,
		ExpertCount: 4, ExpertUsedCount: 2, ExpertFeedForward: 6, ExpertWeightsScale: 1.5,
		SharedExpertFF: 12, ExpertGatingFunc: 2, ExpertWeightsNorm: true,
		HeadCount: 2, HeadCountKV: 1, KeyLength: 4, ValueLength: 4,
		RopeDimensionCount: 4, RopeFrequencyBase: 10000, RopeFrequencySWA: 500000,
		SlidingWindow: 128, SlidingPattern: 4, RMSNormEpsilon: 1e-6,
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
	local, err := BuildDenseBlockCachedForLayer(builder, input, spec, weights, []uint32{0, 1}, nil, nil, 0)
	if err != nil {
		t.Fatal(err)
	}
	nodes, _ := tensor.Topological(local.Output)
	var moe *tensor.Tensor
	var rope, window, silu int
	for _, node := range nodes {
		switch node.Op {
		case tensor.OpMoE:
			moe = node
		case tensor.OpRoPENeoX:
			rope++
		case tensor.OpAttention:
			if node.Attrs.(tensor.AttentionAttributes).Window > 0 {
				window++
			}
		case tensor.OpSiLU:
			silu++
		}
	}
	if moe == nil || moe.Attrs.(tensor.MoEAttributes).Routing != tensor.MoERoutingSigmoid ||
		len(moe.Inputs) != 7 || rope != 2 || window != 1 || silu < 1 {
		t.Fatalf("unexpected EXAONE-MoE graph: moe=%v rope=%d window=%d silu=%d", moe != nil, rope, window, silu)
	}
}

func TestBuildSmallThinkerUsesSplitRouterReGLUAndSlidingRoPE(t *testing.T) {
	builder := tensor.NewBuilder()
	spec := Spec{
		Architecture: "smallthinker", BlockCount: 4, EmbeddingLength: 8,
		FeedForwardLength: 6, ExpertCount: 4, ExpertUsedCount: 2,
		ExpertFeedForward: 6, ExpertWeightsScale: 1.25, ExpertGatingFunc: 2,
		ExpertWeightsNorm: true, HeadCount: 2, HeadCountKV: 1,
		KeyLength: 4, ValueLength: 4, RopeDimensionCount: 4,
		RopeFrequencyBase: 10000, RopeFrequencySWA: 20000,
		SlidingWindow: 128, SlidingPattern: 4, NoRopeLayerStep: 4,
		RMSNormEpsilon: 1e-6,
	}
	input := builder.Input("input", dtype.F32, tensor.MustShape(8, 2))
	weights := LayerGraphWeights{
		AttentionNorm:          builder.Input("attn_norm", dtype.F32, tensor.MustShape(8)),
		AttentionQKV:           builder.Input("qkv", dtype.F32, tensor.MustShape(8, 16)),
		AttentionOutput:        builder.Input("attn_out", dtype.F32, tensor.MustShape(8, 8)),
		FeedForwardNorm:        builder.Input("ffn_norm", dtype.F32, tensor.MustShape(8)),
		FeedForwardRouter:      builder.Input("router", dtype.F32, tensor.MustShape(8, 4)),
		FeedForwardGateExperts: builder.Input("gate_exps", dtype.F32, tensor.MustShape(8, 6, 4)),
		FeedForwardUpExperts:   builder.Input("up_exps", dtype.F32, tensor.MustShape(8, 6, 4)),
		FeedForwardDownExperts: builder.Input("down_exps", dtype.F32, tensor.MustShape(6, 8, 4)),
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
	var rope, sliding int
	for _, node := range nodes {
		switch node.Op {
		case tensor.OpMoE:
			moe = node
		case tensor.OpRoPENeoX:
			rope++
		case tensor.OpAttention:
			if node.Attrs.(tensor.AttentionAttributes).Window == 128 {
				sliding++
			}
		}
	}
	if moe == nil {
		t.Fatal("SmallThinker graph is missing MoE")
	}
	attributes := moe.Attrs.(tensor.MoEAttributes)
	if attributes.Activation != tensor.MoEActivationReLU ||
		attributes.Routing != tensor.MoERoutingSigmoid || !attributes.NormalizeTopKProb ||
		len(moe.Inputs) != 6 || moe.Inputs[1] != input || moe.Inputs[0] == input ||
		rope != 2 || sliding != 1 {
		t.Fatalf("unexpected SmallThinker graph: attrs=%+v inputs=%d rope=%d sliding=%d", attributes, len(moe.Inputs), rope, sliding)
	}
}

func TestBuildGroveMoEGroupedChunkExperts(t *testing.T) {
	builder := tensor.NewBuilder()
	spec := Spec{
		Architecture: "grovemoe", BlockCount: 1, EmbeddingLength: 8,
		FeedForwardLength: 24, ExpertCount: 4, ExpertUsedCount: 2,
		ExpertFeedForward: 6, ExpertChunkFeedForward: 3, ExpertWeightsScale: 1.25,
		ExpertGroupScale: 0.5, ExpertsPerGroup: 2,
		HeadCount: 2, HeadCountKV: 1, KeyLength: 4, ValueLength: 4,
		RopeFrequencyBase: 10000, RMSNormEpsilon: 1e-6,
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
}

func TestBuildDOTS1MoEBlock(t *testing.T) {
	builder := tensor.NewBuilder()
	spec := Spec{
		Architecture: "dots1", BlockCount: 2, LeadingDenseBlocks: 1,
		EmbeddingLength: 8, FeedForwardLength: 12, ExpertCount: 4, ExpertUsedCount: 2,
		ExpertFeedForward: 6, SharedExpertCount: 2, SharedExpertFF: 12,
		ExpertWeightsScale: 1.25, ExpertWeightsNorm: true, ExpertGatingFunc: 2,
		HeadCount: 2, HeadCountKV: 2, KeyLength: 4, ValueLength: 4,
		RopeFrequencyBase: 10000, RMSNormEpsilon: 1e-6,
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
	spec := Spec{
		Architecture: "minimax-m2", BlockCount: 1, EmbeddingLength: 8,
		FeedForwardLength: 6, ExpertCount: 4, ExpertUsedCount: 2,
		ExpertFeedForward: 6, ExpertWeightsScale: 1.25, ExpertGatingFunc: 2,
		HeadCount: 2, HeadCountKV: 1, KeyLength: 4, ValueLength: 4,
		RopeDimensionCount: 2, RopeFrequencyBase: 10000, RMSNormEpsilon: 1e-6,
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
	spec := Spec{
		Architecture: "apertus", BlockCount: 1, ContextLength: 128,
		OriginalContextLength: 32, EmbeddingLength: 8, FeedForwardLength: 12,
		HeadCount: 2, HeadCountKV: 1, KeyLength: 4, ValueLength: 4,
		RopeDimensionCount: 4, RopeFrequencyBase: 10000,
		RopeScalingType: "longrope", RopeAttentionFactor: 1.25,
		AttentionScale: 0.3, RMSNormEpsilon: 1e-5,
		XIELUAlphaN: []float32{0.8}, XIELUAlphaP: []float32{0.2},
		XIELUBeta: []float32{0.5}, XIELUEpsilon: []float32{-0.1},
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
			spec := Spec{
				Architecture: "gptneox", BlockCount: 1, EmbeddingLength: 8,
				FeedForwardLength: 12, HeadCount: 2, HeadCountKV: 2,
				KeyLength: 4, ValueLength: 4, RopeDimensionCount: 2,
				RopeFrequencyBase: 10000, LayerNormEpsilon: 1e-5,
				ParallelResidual: parallel,
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

func TestBuildDenseGLM4(t *testing.T) {
	builder := tensor.NewBuilder()
	spec := Spec{
		Architecture: "glm4", BlockCount: 1, EmbeddingLength: 8,
		FeedForwardLength: 12, HeadCount: 2, HeadCountKV: 1,
		KeyLength: 4, ValueLength: 4, RopeDimensionCount: 4,
		RopeFrequencyBase: 10000, RMSNormEpsilon: 1e-5,
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
		spec := Spec{
			Architecture: "exaone4", BlockCount: 64, EmbeddingLength: 8,
			FeedForwardLength: 12, HeadCount: 2, HeadCountKV: 1,
			KeyLength: 4, ValueLength: 4, RopeDimensionCount: 4,
			RopeFrequencyBase: 1_000_000, RMSNormEpsilon: 1e-5,
			SlidingWindow: 4096, SlidingPattern: 4, NoRopeLayerStep: 4,
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
			spec := Spec{
				Architecture: "falcon", BlockCount: 1, EmbeddingLength: 8,
				FeedForwardLength: 12, HeadCount: 2, HeadCountKV: 1,
				KeyLength: 4, ValueLength: 4, RopeDimensionCount: 4,
				RopeFrequencyBase: 10000, LayerNormEpsilon: 1e-5,
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
	spec := Spec{
		Architecture: "bitnet", BlockCount: 1, EmbeddingLength: 8,
		FeedForwardLength: 12, HeadCount: 2, HeadCountKV: 1,
		KeyLength: 4, ValueLength: 4, RopeFrequencyBase: 10000, RMSNormEpsilon: 1e-5,
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
	spec := Spec{
		Architecture:      "qwen3",
		EmbeddingLength:   8,
		FeedForwardLength: 12,
		HeadCount:         2,
		HeadCountKV:       1,
		KeyLength:         4,
		ValueLength:       4,
		RopeFrequencyBase: 1_000_000,
		RMSNormEpsilon:    1e-6,
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
	spec := Spec{
		Architecture:      "llama",
		EmbeddingLength:   8,
		FeedForwardLength: 12,
		HeadCount:         2,
		HeadCountKV:       1,
		KeyLength:         4,
		ValueLength:       4,
		RopeFrequencyBase: 10000,
		RMSNormEpsilon:    1e-5,
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
			spec := Spec{
				Architecture: "llama-embed", EmbeddingLength: 8, FeedForwardLength: 12,
				HeadCount: 2, HeadCountKV: 1, KeyLength: 4, ValueLength: 4,
				RopeFrequencyBase: 10000, RMSNormEpsilon: 1e-5, NonCausalAttention: true,
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
	spec := Spec{
		Architecture: "pangu-embedded", EmbeddingLength: 8, FeedForwardLength: 12,
		HeadCount: 2, HeadCountKV: 1, KeyLength: 4, ValueLength: 4,
		RopeDimensionCount: 4, RopeFrequencyBase: 10000, RMSNormEpsilon: 1e-6,
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
	}{
		{name: "first GEGLU", layer: 0, activation: "gelu"},
		{name: "local SwiGLU", layer: 1, activation: "silu", wantWindow: true},
	} {
		t.Run(test.name, func(t *testing.T) {
			builder := tensor.NewBuilder()
			spec := Spec{
				Architecture: "modern-bert", BlockCount: 3, EmbeddingLength: 8,
				FeedForwardLength: 16, HeadCount: 2, HeadCountKV: 2,
				KeyLength: 4, ValueLength: 4, RopeDimensionCount: 4,
				RopeFrequencyBase: 10000, RopeFrequencySWA: 50000,
				SlidingWindow: 4, SlidingPattern: 3, LayerNormEpsilon: 1e-5,
				NonCausalAttention: true, HiddenActivation: test.activation,
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
			var layerNorm, neoX, gelu, silu int
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
				}
			}
			if attention == nil || attention.Attrs.(tensor.AttentionAttributes).Causal ||
				attention.Attrs.(tensor.AttentionAttributes).SymmetricWindow != test.wantWindow ||
				neoX != 2 || layerNorm != int(test.layer)+1 ||
				(test.activation == "gelu" && (gelu != 1 || silu != 0)) ||
				(test.activation == "silu" && (silu != 1 || gelu != 0)) {
				t.Fatalf("unexpected ModernBERT graph: attention=%v norms=%d NeoX=%d GELU=%d SiLU=%d", attention, layerNorm, neoX, gelu, silu)
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
			spec := Spec{
				Architecture: "gemma-embedding", BlockCount: 6, EmbeddingLength: 8,
				FeedForwardLength: 16, HeadCount: 2, HeadCountKV: 1,
				KeyLength: 4, ValueLength: 4, RopeDimensionCount: 4,
				RopeFrequencyBase: 10000, RopeFrequencySWA: 50000,
				SlidingWindow: 4, SlidingPattern: 6, RMSNormEpsilon: 1e-6,
				NonCausalAttention: true,
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
	spec := Spec{
		Architecture: "talkie", EmbeddingLength: 8, FeedForwardLength: 12,
		HeadCount: 2, HeadCountKV: 1, KeyLength: 4, ValueLength: 4,
		RopeDimensionCount: 4, RopeFrequencyBase: 10000, RMSNormEpsilon: 1e-6,
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
	spec := Spec{
		Architecture:      "llama",
		EmbeddingLength:   8,
		FeedForwardLength: 12,
		HeadCount:         2,
		HeadCountKV:       1,
		KeyLength:         4,
		ValueLength:       4,
		RopeFrequencyBase: 10000,
		RopeScalingType:   "linear",
		RopeScalingFactor: 8,
		RMSNormEpsilon:    1e-5,
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
	spec := Spec{
		Architecture:      "gemma2",
		BlockCount:        26,
		EmbeddingLength:   8,
		FeedForwardLength: 12,
		HeadCount:         2,
		HeadCountKV:       1,
		KeyLength:         4,
		ValueLength:       4,
		RopeFrequencyBase: 10000,
		RopeFrequencySWA:  10000,
		RopeScalingType:   "linear",
		RopeScalingFactor: 4,
		AttentionSoftcap:  50,
		RMSNormEpsilon:    1e-6,
		SlidingWindow:     4096,
		SlidingPattern:    2,
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
	spec := Spec{
		Architecture:      "gemma",
		BlockCount:        18,
		EmbeddingLength:   8,
		FeedForwardLength: 12,
		HeadCount:         2,
		HeadCountKV:       1,
		KeyLength:         4,
		ValueLength:       4,
		RopeFrequencyBase: 10000,
		RMSNormEpsilon:    1e-6,
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
	spec := Spec{
		Architecture:      "llama",
		EmbeddingLength:   8,
		FeedForwardLength: 12,
		HeadCount:         2,
		HeadCountKV:       1,
		KeyLength:         4,
		ValueLength:       4,
		RopeFrequencyBase: 10000,
		RMSNormEpsilon:    1e-5,
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
	spec := Spec{
		Architecture: "gpt2", EmbeddingLength: 8, FeedForwardLength: 16,
		HeadCount: 2, HeadCountKV: 2, KeyLength: 4, ValueLength: 4,
		LayerNormEpsilon: 1e-5, RopeDisabled: true,
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
	spec := Spec{
		Architecture: "bloom", EmbeddingLength: 8, FeedForwardLength: 16,
		HeadCount: 2, HeadCountKV: 2, KeyLength: 4, ValueLength: 4,
		LayerNormEpsilon: 1e-5, RopeDisabled: true, MaxALiBiBias: 8,
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
	spec := Spec{
		Architecture: "mpt", EmbeddingLength: 8, FeedForwardLength: 16,
		HeadCount: 2, HeadCountKV: 2, KeyLength: 4, ValueLength: 4,
		LayerNormEpsilon: 1e-5, RopeDisabled: true, MaxALiBiBias: 8,
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
	var gelu, alibi int
	for _, node := range nodes {
		if node.Op == tensor.OpGELU {
			gelu++
		}
		if node.Op == tensor.OpAttention &&
			node.Attrs.(tensor.AttentionAttributes).MaxALiBiBias == 8 {
			alibi++
		}
	}
	if gelu != 1 || alibi != 1 {
		t.Fatalf("MPT graph has GELU=%d ALiBi=%d, want 1/1", gelu, alibi)
	}
}

func TestBuildQwen35AttentionBlock(t *testing.T) {
	builder := tensor.NewBuilder()
	spec := qwen35TestSpec()
	input := builder.Input("input", dtype.F32, tensor.MustShape(8, 2))
	result, err := BuildQwen35BlockCached(
		builder,
		input,
		spec,
		qwen35AttentionInputs(builder, spec),
		[]uint32{0, 1},
		false,
		nil,
		nil,
		nil,
		nil,
	)
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

func TestBuildT5EncoderBlock(t *testing.T) {
	builder := tensor.NewBuilder()
	spec := Spec{
		Architecture:      "t5encoder",
		EmbeddingLength:   8,
		FeedForwardLength: 16,
		HeadCount:         2,
		HeadCountKV:       2,
		KeyLength:         4,
		ValueLength:       4,
		RMSNormEpsilon:    1e-6,
		RelativeBuckets:   4,
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
	return Spec{
		Architecture:          "qwen35",
		EmbeddingLength:       8,
		FeedForwardLength:     12,
		HeadCount:             2,
		HeadCountKV:           1,
		KeyLength:             4,
		ValueLength:           4,
		RopeFrequencyBase:     10000,
		RMSNormEpsilon:        1e-6,
		RopeDimensionCount:    4,
		RopeSections:          [4]int32{1, 1, 0, 0},
		SSMConvKernel:         3,
		SSMInnerSize:          4,
		SSMStateSize:          2,
		SSMTimeStepRank:       2,
		SSMGroupCount:         1,
		FullAttentionInterval: 4,
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
