package model

import (
	"fmt"

	"overgo/internal/tensor"

	"overgo/internal/tensor/dtype"

	"overgo/internal/tensor/reference"

	"math"

	"slices"

	"testing"
)

func TestBuildGemma4PerLayerInputs(t *testing.T) {
	builder := tensor.NewBuilder()
	spec := Spec{CommonSpec: CommonSpec{Architecture: "gemma4", BlockCount: 2, EmbeddingLength: 2,
		RMSNormEpsilon: 1e-6}, MultimodalSpec: MultimodalSpec{EmbeddingPerLayer: 1},
	}
	input := builder.Input("input", dtype.F32, tensor.MustShape(2, 1))
	selected := builder.Input("selected", dtype.F32, tensor.MustShape(2, 1))
	projection := builder.Input("projection", dtype.F32, tensor.MustShape(2, 2))
	norm := builder.Input("norm", dtype.F32, tensor.MustShape(1))
	program := fixtureModelPlan(t, spec, Weights{})
	outputs, err := program.BuildPerLayerInputs(builder, input, selected, projection, norm)
	if err != nil {
		t.Fatal(err)
	}
	results, err := reference.Execute(outputs, map[*tensor.Tensor]reference.Value{
		input:      {Shape: input.Shape, Data: []float32{3, 4}},
		selected:   {Shape: selected.Shape, Data: []float32{2, 3}},
		projection: {Shape: projection.Shape, Data: make([]float32, 4)},
		norm:       {Shape: norm.Shape, Data: []float32{1}},
	})
	if err != nil {
		t.Fatal(err)
	}
	for layer, want := range []float32{float32(math.Sqrt(2)), 3 / float32(math.Sqrt(2))} {
		got := results[outputs[layer]]
		if !got.Shape.Equal(tensor.MustShape(1, 1)) || math.Abs(float64(got.Data[0]-want)) > 1e-6 {
			t.Fatalf("layer %d = %v %v, want [1 1] %g", layer, got.Shape.Slice(), got.Data, want)
		}
	}
}

func TestBuildGemma4SharedKVMoEBlock(t *testing.T) {
	builder := tensor.NewBuilder()
	spec := Spec{CommonSpec: CommonSpec{Architecture: "gemma4", BlockCount: 4, EmbeddingLength: 4,
		FeedForwardLength: 6,

		RMSNormEpsilon: 1e-6}, AttentionSpec: AttentionSpec{HeadCount: 2, HeadCountKV: 1,
		KeyLength: 2, ValueLength: 2, KeyLengthSWA: 2, ValueLengthSWA: 2,
		RopeDimensionCount: 2, RopeDimensionSWA: 2,
		RopeFrequencyBase: 10000, RopeFrequencySWA: 10000,
		AttentionScale: 1, SlidingWindow: 4,
		SlidingLayers: []bool{true, false, true, false}}, MoESpec: MoESpec{ExpertCount: 2, ExpertUsedCount: 1,
		ExpertFeedForward: 2, ExpertWeightsScale: 1}, MultimodalSpec: MultimodalSpec{SharedKVLayers: 2,
		EmbeddingPerLayer: 1},
	}
	input := builder.Input("input", dtype.F32, tensor.MustShape(4, 2))
	owned := gemma4BlockInputs(builder, spec, true, false)
	first, err := buildFixtureDenseBlockCachedForLayer(
		builder, input, spec, owned, []uint32{0, 1}, nil, nil, 0,
	)
	if err != nil {
		t.Fatal(err)
	}
	shared := gemma4BlockInputs(builder, spec, false, true)
	second, err := buildFixtureDenseBlockCachedForLayer(
		builder, first.Output, spec, shared, []uint32{0, 1}, first.Key, first.Value, 2,
	)
	if err != nil {
		t.Fatal(err)
	}
	if second.Key != first.Key || second.Value != first.Value || !second.Output.Shape.Equal(input.Shape) {
		t.Fatalf("Gemma 4 shared result shapes = %v/%v/%v", second.Output.Shape.Slice(), second.Key.Shape.Slice(), second.Value.Shape.Slice())
	}
	nodes, err := tensor.Topological(second.Output, second.Key, second.Value)
	if err != nil {
		t.Fatal(err)
	}
	var foundMoE, foundExpertScale bool
	for _, node := range nodes {
		if node.Op == tensor.OpMoE {
			foundMoE = true
			foundExpertScale = node.Attrs.(tensor.MoEAttributes).HasExpertScale
		}
	}
	if !foundMoE || !foundExpertScale {
		t.Fatal("Gemma 4 graph lacks scaled MoE execution")
	}
}

func TestBuildGemma4SlidingBlockUsesVisionBlockMask(t *testing.T) {
	builder := tensor.NewBuilder()
	spec := Spec{CommonSpec: CommonSpec{Architecture: "gemma4", BlockCount: 2, EmbeddingLength: 4,
		FeedForwardLength: 6,

		RMSNormEpsilon: 1e-6}, AttentionSpec: AttentionSpec{HeadCount: 2, HeadCountKV: 1,
		KeyLength: 2, ValueLength: 2, KeyLengthSWA: 2, ValueLengthSWA: 2,
		RopeDimensionCount: 2, RopeDimensionSWA: 2,
		RopeFrequencyBase: 10000, RopeFrequencySWA: 10000,
		AttentionScale: 1, SlidingWindow: 4,
		SlidingLayers: []bool{true, false}}, MultimodalSpec: MultimodalSpec{EmbeddingPerLayer: 1},
	}
	input := builder.Input("input", dtype.F32, tensor.MustShape(4, 2))
	weights := gemma4BlockInputs(builder, spec, true, false)
	weights.AttentionBlockIDs = builder.Input("attention_blocks", dtype.F32, tensor.MustShape(2))
	result, err := buildFixtureDenseBlockCachedForLayer(
		builder, input, spec, weights, []uint32{0, 1}, nil, nil, 0,
	)
	if err != nil {
		t.Fatal(err)
	}
	nodes, err := tensor.Topological(result.Output)
	if err != nil {
		t.Fatal(err)
	}
	for _, node := range nodes {
		if node.Op == tensor.OpAttention {
			attributes := node.Attrs.(tensor.AttentionAttributes)
			if !attributes.Causal || !attributes.HasBlockMask || attributes.Window != 4 {
				t.Fatalf("Gemma 4 attention attributes = %+v", attributes)
			}
			return
		}
	}
	t.Fatal("Gemma 4 sliding attention node missing")
}

// TestBuildGemma4BlockHonorsAppendCacheWrite pins the capacity-session contract:
// under a fixed-capacity append plan the Gemma 4 block must emit an OpCacheAppend
// (bounded write into the source-capacity buffer) rather than an unbounded Concat.
// A raw Concat produced keyShape = sourceCapacity+tokens (e.g. 257) while the cache
// plan sized keyCapacity at the page capacity (256), OOMing capacity-eligible models
// (gemma-4-12B). Offset/KeyValueTokens must come from the append plan, not the
// buffer's token axis.
func TestBuildGemma4BlockHonorsAppendCacheWrite(t *testing.T) {
	builder := tensor.NewBuilder()
	spec := Spec{CommonSpec: CommonSpec{Architecture: "gemma4", BlockCount: 2, EmbeddingLength: 4,
		FeedForwardLength: 6, RMSNormEpsilon: 1e-6}, AttentionSpec: AttentionSpec{HeadCount: 2, HeadCountKV: 1,
		KeyLength: 2, ValueLength: 2, KeyLengthSWA: 2, ValueLengthSWA: 2,
		RopeDimensionCount: 2, RopeDimensionSWA: 2, RopeFrequencyBase: 10000, RopeFrequencySWA: 10000,
		AttentionScale: 1, SlidingWindow: 4,
		SlidingLayers: []bool{true, false}}, MultimodalSpec: MultimodalSpec{EmbeddingPerLayer: 1},
	}
	const (
		activeTokens   = uint32(2)
		sourceCapacity = uint32(4)
		capacityTokens = uint32(4)
	)
	// Mirror the capacity path: SetCacheAppendPlan + past buffer whose token axis
	// is the SOURCE CAPACITY (not the active-token count).
	builder.SetCacheAppendPlan(tensor.CacheAppendPlan{
		ActiveTokens: activeTokens, SourceCapacityTokens: sourceCapacity, CapacityTokens: capacityTokens,
	})
	input := builder.Input("input", dtype.F32, tensor.MustShape(4, 1))
	weights := gemma4BlockInputs(builder, spec, true, false)
	pastKey := builder.Input("past_key", dtype.F32, tensor.MustShape(2, 1, uint64(sourceCapacity)))
	pastValue := builder.Input("past_value", dtype.F32, tensor.MustShape(2, 1, uint64(sourceCapacity)))
	// Layer 1 is a full (non-sliding) KV-owning layer.
	plan := bindFixtureSpec(spec).PlanLayer(1, false)
	result, err := executeCompiledLayer(blockDispatchOptions{
		Context: CachedBlockContext{
			Builder: builder, Input: input, Positions: []uint32{activeTokens},
			PastKey: pastKey, PastValue: pastValue, Layer: 1, CacheWrite: tensor.CacheWriteAppend,
		},
		Spec: spec, Weights: weights, Plan: &plan,
	})
	if err != nil {
		t.Fatal(err)
	}
	if result.Key.Op != tensor.OpCacheAppend || result.Value.Op != tensor.OpCacheAppend {
		t.Fatalf("Gemma 4 append cache ops = %v/%v, want OpCacheAppend", result.Key.Op, result.Value.Op)
	}
	if result.Key.Shape.Dims[2] != uint64(capacityTokens) {
		t.Fatalf("Gemma 4 append key token axis = %d, want capacity %d", result.Key.Shape.Dims[2], capacityTokens)
	}
	nodes, err := tensor.Topological(result.Output, result.Key, result.Value)
	if err != nil {
		t.Fatal(err)
	}
	var foundAttention bool
	for _, node := range nodes {
		if node.Op == tensor.OpAttention {
			foundAttention = true
			attributes := node.Attrs.(tensor.AttentionAttributes)
			if attributes.QueryStart != activeTokens || attributes.KeyValueTokens != activeTokens+1 {
				t.Fatalf("Gemma 4 append attention offsets = %+v, want QueryStart=%d KeyValueTokens=%d",
					attributes, activeTokens, activeTokens+1)
			}
		}
	}
	if !foundAttention {
		t.Fatal("Gemma 4 attention node missing")
	}
}

func gemma4BlockInputs(builder *tensor.Builder, spec Spec, hasKV, moe bool) LayerGraphWeights {
	embedding := uint64(spec.EmbeddingLength)
	key := uint64(spec.KeyLengthSWA)
	ff := uint64(spec.FeedForwardLength)
	weights := LayerGraphWeights{
		AttentionNorm:       builder.Input("attn_norm", dtype.F32, tensor.MustShape(embedding)),
		AttentionQ:          builder.Input("attn_q", dtype.F32, tensor.MustShape(embedding, uint64(spec.HeadCount)*key)),
		AttentionOutput:     builder.Input("attn_output", dtype.F32, tensor.MustShape(uint64(spec.HeadCount)*key, embedding)),
		AttentionQNorm:      builder.Input("attn_q_norm", dtype.F32, tensor.MustShape(key)),
		AttentionPostNorm:   builder.Input("post_attention_norm", dtype.F32, tensor.MustShape(embedding)),
		FeedForwardNorm:     builder.Input("ffn_norm", dtype.F32, tensor.MustShape(embedding)),
		FeedForwardGate:     builder.Input("ffn_gate", dtype.F32, tensor.MustShape(embedding, ff)),
		FeedForwardUp:       builder.Input("ffn_up", dtype.F32, tensor.MustShape(embedding, ff)),
		FeedForwardDown:     builder.Input("ffn_down", dtype.F32, tensor.MustShape(ff, embedding)),
		FeedForwardPostNorm: builder.Input("post_ffw_norm", dtype.F32, tensor.MustShape(embedding)),
		PerLayerInput:       builder.Input("per_layer_input", dtype.F32, tensor.MustShape(1, 2)),
		PerLayerInputGate:   builder.Input("per_layer_gate", dtype.F32, tensor.MustShape(embedding, 1)),
		PerLayerProjection:  builder.Input("per_layer_proj", dtype.F32, tensor.MustShape(1, embedding)),
		PerLayerPostNorm:    builder.Input("per_layer_post_norm", dtype.F32, tensor.MustShape(embedding)),
	}
	if hasKV {
		weights.AttentionK = builder.Input("attn_k", dtype.F32, tensor.MustShape(embedding, key))
		weights.AttentionKNorm = builder.Input("attn_k_norm", dtype.F32, tensor.MustShape(key))
	}
	if moe {
		experts := uint64(spec.ExpertCount)
		expertFF := uint64(spec.ExpertFeedForward)
		weights.FeedForwardRouter = builder.Input("ffn_router", dtype.F32, tensor.MustShape(embedding, experts))
		weights.FeedForwardRouterScale = builder.Input("ffn_router_scale", dtype.F32, tensor.MustShape(embedding))
		weights.FeedForwardGateExperts = builder.Input("ffn_gate_exps", dtype.F32, tensor.MustShape(embedding, expertFF, experts))
		weights.FeedForwardUpExperts = builder.Input("ffn_up_exps", dtype.F32, tensor.MustShape(embedding, expertFF, experts))
		weights.FeedForwardDownExperts = builder.Input("ffn_down_exps", dtype.F32, tensor.MustShape(expertFF, embedding, experts))
		weights.FeedForwardDownExpertsScale = builder.Input("ffn_down_exps_scale", dtype.F32, tensor.MustShape(experts))
		weights.FeedForwardPreNorm2 = builder.Input("pre_ffw_norm_2", dtype.F32, tensor.MustShape(embedding))
		weights.FeedForwardPostNorm1 = builder.Input("post_ffw_norm_1", dtype.F32, tensor.MustShape(embedding))
		weights.FeedForwardPostNorm2 = builder.Input("post_ffw_norm_2", dtype.F32, tensor.MustShape(embedding))
	}
	return weights
}

func TestBuildGemma3nAttentionAndFeedForwardStages(t *testing.T) {
	builder := tensor.NewBuilder()
	spec := Spec{CommonSpec: CommonSpec{Architecture: "gemma3n", BlockCount: 21, EmbeddingLength: 4,
		FeedForwardLength: 6,

		RMSNormEpsilon: 1e-6}, AttentionSpec: AttentionSpec{HeadCount: 2, HeadCountKV: 1,
		KeyLength: 2, ValueLength: 2, RopeDimensionCount: 2,
		RopeFrequencyBase: 10000, RopeFrequencySWA: 10000,
		SlidingWindow: 4}, MultimodalSpec: MultimodalSpec{KVFromStart: 20, SharedKVLayers: 1},
	}
	input := builder.Input("input", dtype.F32, tensor.MustShape(4, 2))
	ownedProgram := fixtureLayerProgram(t, spec, Weights{}, 0)
	owned, err := ownedProgram.BuildActivationProjection(CachedBlockContext{
		Builder: builder, Input: input, Positions: []uint32{0, 1}, Layer: 0,
	}, gemma3nBlockInputs(builder, spec, true))
	if err != nil {
		t.Fatal(err)
	}
	activated := builder.Input("activated", dtype.F32, tensor.MustShape(6, 2))
	output, err := ownedProgram.BuildActivatedOutput(
		builder, owned.Residual, activated, gemma3nBlockInputs(builder, spec, true),
	)
	if err != nil {
		t.Fatal(err)
	}
	sharedProgram := fixtureLayerProgram(t, spec, Weights{}, 20)
	shared, err := sharedProgram.BuildActivationProjection(CachedBlockContext{
		Builder: builder, Input: output, Positions: []uint32{0, 1},
		PastKey: owned.Key, PastValue: owned.Value, Layer: 20,
	}, gemma3nBlockInputs(builder, spec, false))
	if err != nil {
		t.Fatal(err)
	}
	if !output.Shape.Equal(input.Shape) || shared.Key != owned.Key || shared.Value != owned.Value ||
		!shared.Gate.Shape.Equal(tensor.MustShape(6, 2)) ||
		!shared.Up.Shape.Equal(tensor.MustShape(6, 2)) {
		t.Fatalf("Gemma 3n stage shapes = %v/%v/%v/%v", output.Shape.Slice(), shared.Key.Shape.Slice(), shared.Gate.Shape.Slice(), shared.Up.Shape.Slice())
	}
}

func gemma3nBlockInputs(builder *tensor.Builder, spec Spec, hasKV bool) LayerGraphWeights {
	embedding := uint64(spec.EmbeddingLength)
	key := uint64(spec.KeyLength)
	value := uint64(spec.ValueLength)
	ff := uint64(spec.FeedForwardLength)
	weights := LayerGraphWeights{
		AttentionNorm:       builder.Input("g3n_attn_norm", dtype.F32, tensor.MustShape(embedding)),
		AttentionQ:          builder.Input("g3n_attn_q", dtype.F32, tensor.MustShape(embedding, uint64(spec.HeadCount)*key)),
		AttentionOutput:     builder.Input("g3n_attn_output", dtype.F32, tensor.MustShape(uint64(spec.HeadCount)*value, embedding)),
		AttentionQNorm:      builder.Input("g3n_attn_q_norm", dtype.F32, tensor.MustShape(key)),
		AttentionPostNorm:   builder.Input("g3n_post_attention_norm", dtype.F32, tensor.MustShape(embedding)),
		FeedForwardNorm:     builder.Input("g3n_ffn_norm", dtype.F32, tensor.MustShape(embedding)),
		FeedForwardGate:     builder.Input("g3n_ffn_gate", dtype.F32, tensor.MustShape(embedding, ff)),
		FeedForwardUp:       builder.Input("g3n_ffn_up", dtype.F32, tensor.MustShape(embedding, ff)),
		FeedForwardDown:     builder.Input("g3n_ffn_down", dtype.F32, tensor.MustShape(ff, embedding)),
		FeedForwardPostNorm: builder.Input("g3n_post_ffw_norm", dtype.F32, tensor.MustShape(embedding)),
		LaurelLeft:          builder.Input("g3n_laurel_l", dtype.F32, tensor.MustShape(embedding, 2)),
		LaurelRight:         builder.Input("g3n_laurel_r", dtype.F32, tensor.MustShape(2, embedding)),
		LaurelPostNorm:      builder.Input("g3n_laurel_post_norm", dtype.F32, tensor.MustShape(embedding)),
	}
	if hasKV {
		weights.AttentionK = builder.Input("g3n_attn_k", dtype.F32, tensor.MustShape(embedding, key))
		weights.AttentionV = builder.Input("g3n_attn_v", dtype.F32, tensor.MustShape(embedding, value))
		weights.AttentionKNorm = builder.Input("g3n_attn_k_norm", dtype.F32, tensor.MustShape(key))
	}
	return weights
}

func TestBuildDenseQwen3Block(t *testing.T) {
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
	input := builder.Input("input", dtype.F32, tensor.MustShape(8, 3))
	weights := denseBlockInputs(builder, spec)
	output, err := buildFixtureDenseBlock(builder, input, spec, weights, []uint32{0, 1, 2})
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
	spec := Spec{CommonSpec: CommonSpec{Architecture: "deci", BlockCount: 4, EmbeddingLength: 8,
		FeedForwardLength: 12,

		RMSNormEpsilon: 1e-6}, AttentionSpec: AttentionSpec{HeadCount: 2, HeadCountKV: 1, LayerHeadCounts: []uint32{2, 2, 0, 0},
		LayerKVHeadCounts: []uint32{1, 0, 0, 0}, KeyLength: 4, ValueLength: 4,
		RopeDimensionCount: 4, RopeFrequencyBase: 10000}, MoESpec: MoESpec{LayerFeedForward: []uint32{12, 12, 12, 0}},
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
			result, err := buildFixtureDenseBlockCachedForLayer(
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
	spec := Spec{CommonSpec: CommonSpec{Architecture: "qwen3moe",
		EmbeddingLength:   8,
		FeedForwardLength: 24,

		RMSNormEpsilon: 1e-6}, AttentionSpec: AttentionSpec{HeadCount: 2,
		HeadCountKV:       1,
		KeyLength:         4,
		ValueLength:       4,
		RopeFrequencyBase: 1_000_000}, MoESpec: MoESpec{ExpertCount: 4,
		ExpertUsedCount:    2,
		ExpertFeedForward:  12,
		ExpertWeightsScale: 1.25},
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
	output, err := buildFixtureDenseBlock(builder, input, spec, weights, []uint32{0, 1, 2})
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
	spec := Spec{CommonSpec: CommonSpec{Architecture: "dbrx", EmbeddingLength: 8, FeedForwardLength: 6,

		LayerNormEpsilon: 1e-5}, AttentionSpec: AttentionSpec{HeadCount: 2, HeadCountKV: 1, KeyLength: 4, ValueLength: 4,
		AttentionClamp: 2, RopeFrequencyBase: 10000}, MoESpec: MoESpec{ExpertCount: 4, ExpertUsedCount: 2, ExpertFeedForward: 6,
		ExpertWeightsScale: 1.25, ExpertWeightsNorm: true},
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
	result, err := buildFixtureDenseBlockCachedForLayer(builder, input, spec, weights, []uint32{0, 1}, nil, nil, 0)
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
	spec := Spec{CommonSpec: CommonSpec{Architecture: "grok", BlockCount: 1, EmbeddingLength: 8, FeedForwardLength: 12,

		RMSNormEpsilon: 1e-6}, AttentionSpec: AttentionSpec{HeadCount: 2, HeadCountKV: 1, KeyLength: 4, ValueLength: 4,
		RopeDimensionCount: 4, RopeFrequencyBase: 10000, RopeScalingType: "yarn",
		RopeScalingFactor: 4, OriginalContextLength: 2048, YaRNExtFactor: 1,
		YaRNAttentionFactor: 1.25, YaRNBetaFast: 8, YaRNBetaSlow: 1,
		AttentionScale: 0.25, AttentionSoftcap: 30}, MoESpec: MoESpec{ExpertCount: 4, ExpertUsedCount: 2, ExpertFeedForward: 6,
		ExpertWeightsScale: 1.25, ExpertWeightsNorm: true},
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
	result, err := buildFixtureDenseBlockCachedForLayer(
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
			spec := Spec{CommonSpec: CommonSpec{Architecture: "mellum", BlockCount: 4, EmbeddingLength: 8, FeedForwardLength: 12,

				RMSNormEpsilon: 1e-6}, AttentionSpec: AttentionSpec{HeadCount: 2, HeadCountKV: 1, KeyLength: 4, ValueLength: 4,
				RopeDimensionCount: 4, RopeFrequencyBase: 10000, RopeFrequencySWA: 20000,
				RopeScalingType: "yarn", RopeScalingFactor: 4, OriginalContextLength: 2048,
				YaRNExtFactor: 1, YaRNAttentionFactor: 1.25, YaRNBetaFast: 32, YaRNBetaSlow: 1,
				SlidingWindow: 128, SlidingPattern: 4}, MoESpec: MoESpec{ExpertCount: 4, ExpertUsedCount: 2, ExpertFeedForward: 6,
				ExpertWeightsScale: 1, ExpertWeightsNorm: true},
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
			result, err := buildFixtureDenseBlockCachedForLayer(builder, input, spec, weights, []uint32{0, 1}, nil, nil, layer)
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
	s := Spec{CommonSpec: CommonSpec{Architecture: "hunyuan-moe", EmbeddingLength: 8, FeedForwardLength: 12,

		RMSNormEpsilon: 1e-6}, AttentionSpec: AttentionSpec{HeadCount: 2, HeadCountKV: 1,
		KeyLength: 4, ValueLength: 4, RopeDimensionCount: 4, RopeFrequencyBase: 10000}, MoESpec: MoESpec{ExpertCount: 4, ExpertUsedCount: 2, ExpertFeedForward: 6, SharedExpertFF: 10,
		ExpertWeightsScale: 1, ExpertWeightsNorm: true},
	}
	in := b.Input("input", dtype.F32, tensor.MustShape(8, 2))
	w := LayerGraphWeights{
		AttentionNorm: b.Input("an", dtype.F32, tensor.MustShape(8)), AttentionQ: b.Input("q", dtype.F32, tensor.MustShape(8, 8)), AttentionK: b.Input("k", dtype.F32, tensor.MustShape(8, 4)), AttentionV: b.Input("v", dtype.F32, tensor.MustShape(8, 4)), AttentionOutput: b.Input("o", dtype.F32, tensor.MustShape(8, 8)), AttentionQNorm: b.Input("qn", dtype.F32, tensor.MustShape(4)), AttentionKNorm: b.Input("kn", dtype.F32, tensor.MustShape(4)), FeedForwardNorm: b.Input("fn", dtype.F32, tensor.MustShape(8)),
		FeedForwardRouter: b.Input("r", dtype.F32, tensor.MustShape(8, 4)), FeedForwardGateExperts: b.Input("ge", dtype.F32, tensor.MustShape(8, 6, 4)), FeedForwardUpExperts: b.Input("ue", dtype.F32, tensor.MustShape(8, 6, 4)), FeedForwardDownExperts: b.Input("de", dtype.F32, tensor.MustShape(6, 8, 4)), FeedForwardSharedGate: b.Input("sg", dtype.F32, tensor.MustShape(8, 10)), FeedForwardSharedUp: b.Input("su", dtype.F32, tensor.MustShape(8, 10)), FeedForwardSharedDown: b.Input("sd", dtype.F32, tensor.MustShape(10, 8)),
	}
	r, err := buildFixtureDenseBlockCachedForLayer(b, in, s, w, []uint32{0, 1}, nil, nil, 0)
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
	s := Spec{CommonSpec: CommonSpec{Architecture: "qwen", EmbeddingLength: 8, FeedForwardLength: 12,

		RMSNormEpsilon: 1e-6}, AttentionSpec: AttentionSpec{HeadCount: 2, HeadCountKV: 2, KeyLength: 4, ValueLength: 4,
		RopeDimensionCount: 4, RopeFrequencyBase: 10000}}
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
	r, err := buildFixtureDenseBlockCachedForLayer(b, in, s, w, []uint32{0, 1}, nil, nil, 0)
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
	s := Spec{CommonSpec: CommonSpec{Architecture: "chatglm", EmbeddingLength: 8, FeedForwardLength: 12, RMSNormEpsilon: 1e-6}, AttentionSpec: AttentionSpec{HeadCount: 2, HeadCountKV: 1, KeyLength: 4, ValueLength: 4, RopeDimensionCount: 4, RopeFrequencyBase: 10000}}
	in := b.Input("input", dtype.F32, tensor.MustShape(8, 2))
	w := LayerGraphWeights{AttentionNorm: b.Input("an", dtype.F32, tensor.MustShape(8)), AttentionQKV: b.Input("qkv", dtype.F32, tensor.MustShape(8, 16)), AttentionOutput: b.Input("o", dtype.F32, tensor.MustShape(8, 8)), FeedForwardNorm: b.Input("fn", dtype.F32, tensor.MustShape(8)), FeedForwardUp: b.Input("fu", dtype.F32, tensor.MustShape(8, 24)), FeedForwardDown: b.Input("fd", dtype.F32, tensor.MustShape(12, 8))}
	r, err := buildFixtureDenseBlockCachedForLayer(b, in, s, w, []uint32{0, 1}, nil, nil, 0)
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
	s := Spec{CommonSpec: CommonSpec{Architecture: "hunyuan-dense", EmbeddingLength: 8, FeedForwardLength: 12, RMSNormEpsilon: 1e-6}, AttentionSpec: AttentionSpec{HeadCount: 2, HeadCountKV: 1, KeyLength: 4, ValueLength: 4, RopeDimensionCount: 4, RopeFrequencyBase: 40000}}
	in := b.Input("input", dtype.F32, tensor.MustShape(8, 2))
	w := LayerGraphWeights{AttentionNorm: b.Input("an", dtype.F32, tensor.MustShape(8)), AttentionQKV: b.Input("qkv", dtype.F32, tensor.MustShape(8, 16)), AttentionOutput: b.Input("o", dtype.F32, tensor.MustShape(8, 8)), AttentionQNorm: b.Input("qn", dtype.F32, tensor.MustShape(4)), AttentionKNorm: b.Input("kn", dtype.F32, tensor.MustShape(4)), FeedForwardNorm: b.Input("fn", dtype.F32, tensor.MustShape(8)), FeedForwardGate: b.Input("fg", dtype.F32, tensor.MustShape(8, 12)), FeedForwardUp: b.Input("fu", dtype.F32, tensor.MustShape(8, 12)), FeedForwardDown: b.Input("fd", dtype.F32, tensor.MustShape(12, 8))}
	r, err := buildFixtureDenseBlockCachedForLayer(b, in, s, w, []uint32{0, 1}, nil, nil, 0)
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

func TestBuildHunyuanBlocksUsePostMRoPEQKNorm(t *testing.T) {
	for _, architecture := range []string{"hunyuan-dense", "hunyuan_vl"} {
		t.Run(architecture, func(t *testing.T) {
			b := tensor.NewBuilder()
			s := Spec{CommonSpec: CommonSpec{Architecture: architecture, EmbeddingLength: 8, FeedForwardLength: 12, RMSNormEpsilon: 1e-6}, AttentionSpec: AttentionSpec{HeadCount: 2, HeadCountKV: 1, KeyLength: 4, ValueLength: 4, RopeDimensionCount: 4, RopeSections: [4]int32{1, 1, 0, 0}, RopeFrequencyBase: 40000}}
			in := b.Input("input", dtype.F32, tensor.MustShape(8, 2))
			w := LayerGraphWeights{AttentionNorm: b.Input("an", dtype.F32, tensor.MustShape(8)), AttentionQKV: b.Input("qkv", dtype.F32, tensor.MustShape(8, 16)), AttentionOutput: b.Input("o", dtype.F32, tensor.MustShape(8, 8)), AttentionQNorm: b.Input("qn", dtype.F32, tensor.MustShape(4)), AttentionKNorm: b.Input("kn", dtype.F32, tensor.MustShape(4)), FeedForwardNorm: b.Input("fn", dtype.F32, tensor.MustShape(8)), FeedForwardGate: b.Input("fg", dtype.F32, tensor.MustShape(8, 12)), FeedForwardUp: b.Input("fu", dtype.F32, tensor.MustShape(8, 12)), FeedForwardDown: b.Input("fd", dtype.F32, tensor.MustShape(12, 8))}
			positions := [4][]uint32{{10, 11}, {20, 21}, {30, 31}, {40, 41}}
			r, err := buildFixtureDenseBlockCachedWithMultiPositions(b, in, s, w, positions, nil, nil, 0)
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
					attributes := node.Attrs.(tensor.RoPEMultiAttributes)
					for axis := range positions {
						if !slices.Equal(attributes.Positions[axis], positions[axis]) {
							t.Fatalf("MRoPE axis %d = %v, want %v", axis, attributes.Positions[axis], positions[axis])
						}
					}
				}
				if node.Op == tensor.OpRMSNorm && len(node.Inputs) == 1 && node.Inputs[0].Op == tensor.OpRoPEMulti {
					postRoPENorm++
				}
			}
			if multiRoPE != 2 || postRoPENorm != 2 {
				t.Fatalf("Hunyuan graph has MRoPE=%d post-MRoPE norms=%d", multiRoPE, postRoPENorm)
			}
		})
	}
}

func TestBuildCogVLMTokenBlockUsesFusedQKVAndNormalRoPE(t *testing.T) {
	b := tensor.NewBuilder()
	s := Spec{CommonSpec: CommonSpec{Architecture: "cogvlm", EmbeddingLength: 8, FeedForwardLength: 12, RMSNormEpsilon: 1e-6}, AttentionSpec: AttentionSpec{HeadCount: 2, HeadCountKV: 2, KeyLength: 4, ValueLength: 4, RopeDimensionCount: 4, RopeFrequencyBase: 10000}}
	in := b.Input("input", dtype.F32, tensor.MustShape(8, 2))
	w := LayerGraphWeights{AttentionNorm: b.Input("an", dtype.F32, tensor.MustShape(8)), AttentionQKV: b.Input("qkv", dtype.F32, tensor.MustShape(8, 24)), AttentionOutput: b.Input("o", dtype.F32, tensor.MustShape(8, 8)), FeedForwardNorm: b.Input("fn", dtype.F32, tensor.MustShape(8)), FeedForwardGate: b.Input("fg", dtype.F32, tensor.MustShape(8, 12)), FeedForwardUp: b.Input("fu", dtype.F32, tensor.MustShape(8, 12)), FeedForwardDown: b.Input("fd", dtype.F32, tensor.MustShape(12, 8))}
	r, err := buildFixtureDenseBlockCachedForLayer(b, in, s, w, []uint32{0, 1}, nil, nil, 0)
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
	spec := Spec{CommonSpec: CommonSpec{Architecture: "arctic", EmbeddingLength: 8, FeedForwardLength: 12,

		RMSNormEpsilon: 1e-5}, AttentionSpec: AttentionSpec{HeadCount: 2, HeadCountKV: 1,
		KeyLength: 4, ValueLength: 4, RopeFrequencyBase: 10000}, MoESpec: MoESpec{ExpertCount: 4, ExpertUsedCount: 2, ExpertFeedForward: 12,
		ExpertWeightsScale: 1},
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
	result, err := buildFixtureDenseBlockCachedForLayer(builder, input, spec, weights, []uint32{0, 1}, nil, nil, 0)
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
	spec := Spec{CommonSpec: CommonSpec{Architecture: "openelm", BlockCount: 2, EmbeddingLength: 8,
		FeedForwardLength: 12,

		RMSNormEpsilon: 1e-6}, AttentionSpec: AttentionSpec{HeadCount: 2, HeadCountKV: 1, LayerHeadCounts: []uint32{2, 4},
		LayerKVHeadCounts: []uint32{1, 2}, KeyLength: 4, ValueLength: 4,
		RopeFrequencyBase: 10000}, MoESpec: MoESpec{LayerFeedForward: []uint32{12, 16}},
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
	result, err := buildFixtureDenseBlockCachedForLayer(builder, input, spec, weights, []uint32{0, 1}, nil, nil, 1)
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
	spec := Spec{CommonSpec: CommonSpec{Architecture: "bailingmoe", EmbeddingLength: 8, FeedForwardLength: 16,

		RMSNormEpsilon: 1e-6}, AttentionSpec: AttentionSpec{HeadCount: 2, HeadCountKV: 1,
		KeyLength: 4, ValueLength: 4, RopeFrequencyBase: 10000}, MoESpec: MoESpec{ExpertCount: 4, ExpertUsedCount: 2, ExpertFeedForward: 6,
		SharedExpertCount: 2, SharedExpertFF: 12, ExpertWeightsScale: 1.25,
		ExpertWeightsNorm: true},
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
	output, err := buildFixtureDenseBlock(builder, input, spec, weights, []uint32{0, 1})
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
	spec := Spec{CommonSpec: CommonSpec{Architecture: "deepseek", EmbeddingLength: 8, FeedForwardLength: 16,

		RMSNormEpsilon: 1e-6}, AttentionSpec: AttentionSpec{HeadCount: 2, HeadCountKV: 1, KeyLength: 4, ValueLength: 4,
		RopeFrequencyBase: 10000}, MoESpec: MoESpec{ExpertCount: 4, ExpertUsedCount: 2, ExpertFeedForward: 6,
		SharedExpertCount: 2, SharedExpertFF: 12, ExpertWeightsScale: 1.3},
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
	output, err := buildFixtureDenseBlock(builder, input, spec, weights, []uint32{0, 1})
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
	for _, architecture := range []string{"granitemoe", "granite"} {
		t.Run(architecture, func(t *testing.T) {
			builder := tensor.NewBuilder()
			spec := Spec{CommonSpec: CommonSpec{Architecture: architecture, EmbeddingLength: 8, FeedForwardLength: 6,

				RMSNormEpsilon: 1e-6, ResidualScale: 0.5}, AttentionSpec: AttentionSpec{HeadCount: 2, HeadCountKV: 1, KeyLength: 4, ValueLength: 4,
				RopeFrequencyBase: 10000, RopeAttentionFactor: 1}, MoESpec: MoESpec{ExpertCount: 4, ExpertUsedCount: 2, ExpertFeedForward: 6,
				SharedExpertFF: 5, ExpertWeightsScale: 1, ExpertWeightsNorm: true},
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
			output, err := buildFixtureDenseBlock(builder, input, spec, weights, []uint32{0, 1})
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
			weights.FeedForwardSharedGate = nil
			if _, err := buildFixtureDenseBlock(builder, input, spec, weights, []uint32{0, 1}); err == nil {
				t.Fatal("Granite MoE accepted an incomplete shared expert")
			}
		})
	}
}

func TestBuildBailingMoE2BlockUsesFusedQKVNeoXAndSharedExpert(t *testing.T) {
	for _, test := range []struct {
		name    string
		gating  uint32
		routing tensor.MoERouting
	}{
		{name: "softmax", gating: expertGatingSoftmax, routing: tensor.MoERoutingSoftmax},
		{name: "sigmoid", gating: expertGatingSigmoid, routing: tensor.MoERoutingSigmoid},
	} {
		t.Run(test.name, func(t *testing.T) {
			builder := tensor.NewBuilder()
			spec := Spec{CommonSpec: CommonSpec{Architecture: "bailingmoe2", BlockCount: 2,
				EmbeddingLength: 8, FeedForwardLength: 16,

				RMSNormEpsilon: 1e-6}, AttentionSpec: AttentionSpec{HeadCount: 2, HeadCountKV: 1, KeyLength: 4, ValueLength: 4,
				RopeDimensionCount: 4, RopeFrequencyBase: 10000}, MoESpec: MoESpec{LeadingDenseBlocks: 1,

				ExpertCount: 4, ExpertUsedCount: 2, ExpertFeedForward: 6,
				SharedExpertCount: 2, SharedExpertFF: 10, ExpertWeightsScale: 1.25,
				ExpertWeightsNorm: true, ExpertGatingFunc: test.gating},
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
			result, err := buildFixtureDenseBlockCachedForLayer(builder, input, spec, weights, []uint32{0, 1}, nil, nil, 1)
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
	spec := Spec{CommonSpec: CommonSpec{Architecture: "qwen2moe", EmbeddingLength: 8, FeedForwardLength: 16,

		RMSNormEpsilon: 1e-6}, AttentionSpec: AttentionSpec{HeadCount: 2, HeadCountKV: 1, KeyLength: 4, ValueLength: 4,
		RopeFrequencyBase: 1_000_000}, MoESpec: MoESpec{ExpertCount: 4, ExpertUsedCount: 2, ExpertFeedForward: 6,
		SharedExpertCount: 1, SharedExpertFF: 10, ExpertWeightsScale: 1},
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
	output, err := buildFixtureDenseBlock(builder, input, spec, weights, []uint32{0, 1})
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
	spec := Spec{CommonSpec: CommonSpec{Architecture: "olmoe", EmbeddingLength: 8, FeedForwardLength: 12,

		RMSNormEpsilon: 1e-6}, AttentionSpec: AttentionSpec{HeadCount: 2, HeadCountKV: 2, KeyLength: 4, ValueLength: 4,
		RopeFrequencyBase: 10000}, MoESpec: MoESpec{ExpertCount: 4, ExpertUsedCount: 2, ExpertFeedForward: 12, ExpertWeightsScale: 1},
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
	output, err := buildFixtureDenseBlock(builder, input, spec, weights, []uint32{0, 1})
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
	spec := Spec{CommonSpec: CommonSpec{Architecture: "eurobert", EmbeddingLength: 8, FeedForwardLength: 16,

		RMSNormEpsilon: 1e-6}, AttentionSpec: AttentionSpec{HeadCount: 2, HeadCountKV: 1, KeyLength: 4, ValueLength: 4,
		RopeFrequencyBase: 10000, NonCausalAttention: true},
	}
	input := builder.Input("input", dtype.F32, tensor.MustShape(8, 3))
	weights := denseBlockInputs(builder, spec)
	weights.AttentionQNorm = nil
	weights.AttentionKNorm = nil
	result, err := buildFixtureDenseBlockCachedForLayer(
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
	spec := Spec{CommonSpec: CommonSpec{Architecture: "bert", EmbeddingLength: 8, FeedForwardLength: 16,

		LayerNormEpsilon: 1e-5}, AttentionSpec: AttentionSpec{HeadCount: 2, HeadCountKV: 2, KeyLength: 4, ValueLength: 4,
		NonCausalAttention: true, RopeDisabled: true},
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
	result, err := buildFixtureDenseBlockCachedForLayer(
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
	spec := Spec{CommonSpec: CommonSpec{Architecture: "neo-bert", EmbeddingLength: 8, FeedForwardLength: 16,

		RMSNormEpsilon: 1e-6}, AttentionSpec: AttentionSpec{HeadCount: 2, HeadCountKV: 2, KeyLength: 4, ValueLength: 4,
		RopeDimensionCount: 4, RopeFrequencyBase: 10000,
		NonCausalAttention: true},
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
	result, err := buildFixtureDenseBlockCachedForLayer(
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
	spec := Spec{CommonSpec: CommonSpec{Architecture: "nomic-bert", EmbeddingLength: 8, FeedForwardLength: 16,

		LayerNormEpsilon: 1e-5}, AttentionSpec: AttentionSpec{HeadCount: 2, HeadCountKV: 2, KeyLength: 4, ValueLength: 4,
		RopeDimensionCount: 4, RopeFrequencyBase: 10000,
		NonCausalAttention: true},
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
	result, err := buildFixtureDenseBlockCachedForLayer(
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
	spec := Spec{CommonSpec: CommonSpec{Architecture: "jina-bert-v2", EmbeddingLength: 8, FeedForwardLength: 16,

		LayerNormEpsilon: 1e-5}, AttentionSpec: AttentionSpec{HeadCount: 2, HeadCountKV: 2, KeyLength: 4, ValueLength: 4,
		NonCausalAttention: true, RopeDisabled: true, MaxALiBiBias: 8},
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
	result, err := buildFixtureDenseBlockCachedForLayer(
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
			spec := Spec{CommonSpec: CommonSpec{Architecture: "jina-bert-v2", EmbeddingLength: 8, FeedForwardLength: 16,

				LayerNormEpsilon: 1e-5}, AttentionSpec: AttentionSpec{HeadCount: 2, HeadCountKV: 2, KeyLength: 4, ValueLength: 4,
				NonCausalAttention: true, RopeDisabled: true, MaxALiBiBias: 8},
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
			result, err := buildFixtureDenseBlockCachedForLayer(
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
	spec := Spec{CommonSpec: CommonSpec{Architecture: "jina-bert-v3", EmbeddingLength: 8, FeedForwardLength: 16,

		LayerNormEpsilon: 1e-5}, AttentionSpec: AttentionSpec{HeadCount: 2, HeadCountKV: 2, KeyLength: 4, ValueLength: 4,
		RopeDimensionCount: 4, RopeFrequencyBase: 10000,
		NonCausalAttention: true},
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
	result, err := buildFixtureDenseBlockCachedForLayer(builder, input, spec, weights, []uint32{0, 1, 2}, nil, nil, 0)
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
