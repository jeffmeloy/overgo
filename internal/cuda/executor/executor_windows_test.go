//go:build windows

package executor

import (
	"context"
	"errors"

	"overgo/internal/cuda/device"

	"overgo/internal/cuda/driver"

	"overgo/internal/model"

	"overgo/internal/quant"

	"overgo/internal/tensor"

	"overgo/internal/tensor/dtype"

	"overgo/internal/tensor/reference"

	"math"

	"testing"

	cudatest "overgo/internal/cuda/testutil"
)

func TestExecutorImplicitZeroFeed(t *testing.T) {
	cudatest.Require(t)
	builder := tensor.NewBuilder()
	input := builder.Input("zero", dtype.F32, tensor.MustShape(4))
	output := builder.Scale(input, 3)
	cuda, err := New(0)
	if err != nil {
		t.Fatal(err)
	}
	defer cuda.Close()
	got, err := cuda.Execute(context.Background(), []*tensor.Tensor{output}, map[*tensor.Tensor]reference.Value{
		input: reference.ZeroValue(input.Shape),
	})
	if err != nil {
		t.Fatal(err)
	}
	for index, value := range got[output].Data {
		if value != 0 {
			t.Fatalf("implicit zero[%d] = %v", index, value)
		}
	}
}

func TestExecutorBF16RoundMatchesReference(t *testing.T) {
	cudatest.Require(t)
	builder := tensor.NewBuilder()
	input := builder.Input("input", dtype.F32, tensor.MustShape(6))
	output := builder.BF16Round(input)
	value := reference.Value{Shape: input.Shape, Data: []float32{
		1, 1.00390625, 1.01171875, -1.01171875, float32(math.Inf(1)), float32(math.NaN()),
	}}
	feeds := map[*tensor.Tensor]reference.Value{input: value}
	want, err := reference.Execute([]*tensor.Tensor{output.Inputs[0], output}, feeds)
	if err != nil {
		t.Fatal(err)
	}
	cuda, err := New(0)
	if err != nil {
		t.Fatal(err)
	}
	defer cuda.Close()
	got, err := cuda.Execute(context.Background(), []*tensor.Tensor{output}, feeds)
	if err != nil {
		t.Fatal(err)
	}
	for index, item := range got[output].Data {
		if math.Float32bits(item) != math.Float32bits(want[output].Data[index]) {
			t.Fatalf("BF16 round[%d] = %08x, want %08x", index, math.Float32bits(item), math.Float32bits(want[output].Data[index]))
		}
	}
}

func TestExecutorLoRAMatchesReference(t *testing.T) {
	cudatest.Require(t)
	builder := tensor.NewBuilder()
	builder.SetLoRA(map[string][]tensor.LoRADefinition{
		"projection": {{
			AName: "adapter.projection.lora_a", BName: "adapter.projection.lora_b",
			AShape: tensor.MustShape(2, 1), BShape: tensor.MustShape(1, 2),
			AData: []float32{2, 3}, BData: []float32{4, 5}, Scale: 0.5,
		}},
		"token_embd.weight": {{
			AName: "adapter.token.lora_a", BName: "adapter.token.lora_b",
			AShape: tensor.MustShape(1, 3), BShape: tensor.MustShape(1, 2),
			AData: []float32{7, 11, 13}, BData: []float32{2, 3}, Scale: 0.25, Embedding: true,
		}},
		"up": {
			{
				AName: "adapter.0.up.lora_a", BName: "adapter.0.up.lora_b",
				AShape: tensor.MustShape(2, 1, 1), BShape: tensor.MustShape(1, 1, 1),
				AData: []float32{1, 1}, BData: []float32{2}, Scale: 1,
			},
			{
				AName: "adapter.1.up.lora_a", BName: "adapter.1.up.lora_b",
				AShape: tensor.MustShape(2, 1, 1), BShape: tensor.MustShape(1, 1, 1),
				AData: []float32{1, 0}, BData: []float32{1}, Scale: 1,
			},
		},
	})
	projection := builder.Input("projection", dtype.F32, tensor.MustShape(2, 2))
	input := builder.Input("input", dtype.F32, tensor.MustShape(2, 1))
	table := builder.Input("token_embd.weight", dtype.F32, tensor.MustShape(2, 3))
	outputs := []*tensor.Tensor{
		builder.MulMat(projection, input),
		builder.GetRows(table, []uint32{1}),
	}
	moeRouter := builder.Input("router", dtype.F32, tensor.MustShape(2, 1))
	moeGate := builder.Input("gate", dtype.F32, tensor.MustShape(2, 1, 1))
	moeUp := builder.Input("up", dtype.F32, tensor.MustShape(2, 1, 1))
	moeDown := builder.Input("down", dtype.F32, tensor.MustShape(1, 2, 1))
	outputs = append(outputs, builder.MoE(input, moeRouter, moeGate, moeUp, moeDown, 1, false, 1))
	feeds := map[*tensor.Tensor]reference.Value{
		projection: {Shape: projection.Shape, Data: []float32{1, 0, 0, 1}},
		input:      {Shape: input.Shape, Data: []float32{1, 2}},
		table:      {Shape: table.Shape, Data: []float32{1, 2, 3, 4, 5, 6}},
		moeRouter:  {Shape: moeRouter.Shape, Data: []float32{0, 0}},
		moeGate:    {Shape: moeGate.Shape, Data: []float32{1, 0}},
		moeUp:      {Shape: moeUp.Shape, Data: []float32{0, 0}},
		moeDown:    {Shape: moeDown.Shape, Data: []float32{1, 2}},
	}
	checkCUDAGraph(t, feeds, uniformGraphChecks(outputs, 1e-5)...)
}

func TestExecutorSparsePrimitivesMatchReference(t *testing.T) {
	cudatest.Require(t)
	builder := tensor.NewBuilder()
	input := builder.Input("input", dtype.F32, tensor.MustShape(4, 2))
	transformed := builder.FWHT(input)
	argmax := builder.TopK(transformed, 1)
	indices := builder.TopK(transformed, 2)
	pairs := builder.TopKPairs(transformed, 2)
	table := builder.Input("table", dtype.F32, tensor.MustShape(2, 4))
	gathered := builder.GatherLast(table, indices)
	query := builder.Input("query", dtype.F32, tensor.MustShape(1, 1, 2))
	key := builder.Input("key", dtype.F32, tensor.MustShape(1, 1, 4))
	value := builder.Input("value", dtype.F32, tensor.MustShape(1, 1, 4))
	attention := builder.SparseAttention(query, key, value, indices, 1)
	causalAttention := builder.SparseAttentionWithOffset(query, key, value, indices, 1, true, 0)
	indexerQuery := builder.Input("indexer_query", dtype.F32, tensor.MustShape(2, 2, 2))
	indexerKey := builder.Input("indexer_key", dtype.F32, tensor.MustShape(2, 1, 4))
	indexerWeights := builder.Input("indexer_weights", dtype.F32, tensor.MustShape(2, 2))
	indexerScores := builder.IndexerScore(indexerQuery, indexerKey, indexerWeights, 0.5, 2)
	if err := builder.Err(); err != nil {
		t.Fatal(err)
	}
	feeds := map[*tensor.Tensor]reference.Value{
		input:          {Shape: input.Shape, Data: []float32{1, 2, 3, 4, 4, 3, 2, 1}},
		table:          {Shape: table.Shape, Data: []float32{10, 11, 20, 21, 30, 31, 40, 41}},
		query:          {Shape: query.Shape, Data: []float32{1, 1}},
		key:            {Shape: key.Shape, Data: []float32{0, 1, 2, 3}},
		value:          {Shape: value.Shape, Data: []float32{10, 20, 30, 40}},
		indexerQuery:   {Shape: indexerQuery.Shape, Data: []float32{1, 0, 0, 1, 1, 1, -1, 1}},
		indexerKey:     {Shape: indexerKey.Shape, Data: []float32{1, 0, 0, 1, 1, 1, -1, 1}},
		indexerWeights: {Shape: indexerWeights.Shape, Data: []float32{2, 1, 1, 3}},
	}
	outputs := []*tensor.Tensor{transformed, argmax, indices, pairs, gathered, attention, causalAttention, indexerScores}
	checkCUDAGraph(t, feeds, uniformGraphChecks(outputs, 1e-5)...)
}

func TestExecutorTopKPairsMultiChunkMatchesReference(t *testing.T) {
	cudatest.Require(t)
	const (
		fixtureWidth = 4099
		fixtureRows  = 2
		fixtureTopK  = 40
	)
	builder := tensor.NewBuilder()
	input := builder.Input("input", dtype.F32, tensor.MustShape(fixtureWidth, fixtureRows))
	output := builder.TopKPairs(input, fixtureTopK)
	feeds := map[*tensor.Tensor]reference.Value{
		input: patternedValue(input.Shape, 17, 0.013, -0.2),
	}
	checkCUDAGraph(t, feeds, uniformGraphChecks([]*tensor.Tensor{output}, 0)...)
}

func TestExecutorDeepSeek4PrimitivesMatchReference(t *testing.T) {
	cudatest.Require(t)
	builder := tensor.NewBuilder()
	input := builder.Input("input", dtype.F32, tensor.MustShape(2, 1))
	hcFN := builder.Input("hc_fn", dtype.F32, tensor.MustShape(4, 8))
	hcScale := builder.Input("hc_scale", dtype.F32, tensor.MustShape(3))
	hcBase := builder.Input("hc_base", dtype.F32, tensor.MustShape(8))
	hc := builder.DeepSeek4HCInit(input, 2)
	branch := builder.DeepSeek4HCPre(hc, hcFN, hcScale, hcBase, 2, 2, 1e-5, 1e-6)
	hc = builder.DeepSeek4HCPost(branch, hc, hcFN, hcScale, hcBase, 2, 2, 1e-5, 1e-6)
	headFN := builder.Input("head_fn", dtype.F32, tensor.MustShape(4, 2))
	headScale := builder.Input("head_scale", dtype.F32, tensor.MustShape(1))
	headBase := builder.Input("head_base", dtype.F32, tensor.MustShape(2))
	head := builder.DeepSeek4HCHead(hc, headFN, headScale, headBase, 2, 1e-5, 1e-6)
	query := builder.Input("query", dtype.F32, tensor.MustShape(2, 1, 2))
	cache := builder.Input("cache", dtype.F32, tensor.MustShape(2, 1, 3))
	positions := builder.Input("positions", dtype.F32, tensor.MustShape(1, 1, 3))
	sinks := builder.Input("sinks", dtype.F32, tensor.MustShape(1))
	attention := builder.DeepSeek4Attention(query, cache, positions, sinks, nil, nil, nil, nil, nil, nil, nil, nil,
		tensor.DeepSeek4AttentionAttributes{
			Positions: []uint32{1, 2}, Window: 3, Heads: 1, RotaryDimensions: 2,
			FrequencyBase: 10000, FrequencyScale: 1, AttentionFactor: 1, NormEpsilon: 1e-5,
		})
	router := builder.Input("router", dtype.F32, tensor.MustShape(2, 2))
	gate := builder.Input("gate", dtype.F32, tensor.MustShape(2, 2, 2))
	up := builder.Input("up", dtype.F32, tensor.MustShape(2, 2, 2))
	down := builder.Input("down", dtype.F32, tensor.MustShape(2, 2, 2))
	selected := builder.Input("selected", dtype.F32, tensor.MustShape(1, 1))
	moe := builder.MoESqrtSoftplusLimited(input, router, gate, up, down, nil, selected, 1, true, 1, 0.15)
	if err := builder.Err(); err != nil {
		t.Fatal(err)
	}
	feeds := map[*tensor.Tensor]reference.Value{
		input: {Shape: input.Shape, Data: []float32{2, 4}},
		hcFN:  patternedValue(hcFN.Shape, 3, 0.1, 0), hcScale: patternedValue(hcScale.Shape, 5, 0.1, 0),
		hcBase: patternedValue(hcBase.Shape, 7, 0.1, 0), headFN: patternedValue(headFN.Shape, 11, 0.1, 0),
		headScale: patternedValue(headScale.Shape, 13, 0.1, 0), headBase: patternedValue(headBase.Shape, 17, 0.1, 0),
		query:     {Shape: query.Shape, Data: []float32{1, 0, 0, 1}},
		cache:     {Shape: cache.Shape, Data: []float32{1, 0, 0, 1, 1, 1}},
		positions: {Shape: positions.Shape, Data: []float32{0, 1, 2}},
		sinks:     {Shape: sinks.Shape, Data: []float32{-2}},
		router:    patternedValue(router.Shape, 19, 0.2, 0), gate: patternedValue(gate.Shape, 23, 0.2, 0),
		up: patternedValue(up.Shape, 29, 0.2, 0), down: patternedValue(down.Shape, 31, 0.2, 0),
		selected: {Shape: selected.Shape, Data: []float32{1}},
	}
	outputs := []*tensor.Tensor{head, attention, moe}
	checkCUDAGraph(t, feeds, uniformGraphChecks(outputs, 2e-4)...)
}

func TestExecutorGLMDSABlockMatchesReference(t *testing.T) {
	cudatest.Require(t)
	builder := tensor.NewBuilder()
	spec := model.Spec{CommonSpec: model.CommonSpec{Architecture: "glm-dsa", BlockCount: 1, EmbeddingLength: 8,
		FeedForwardLength: 12,

		RMSNormEpsilon: 1e-6}, AttentionSpec: model.AttentionSpec{HeadCount: 2, HeadCountKV: 1, KeyLength: 6, ValueLength: 4,
		QLoRARank: 3, KVLoRARank: 3, RopeDimensionCount: 2, RopeFrequencyBase: 10000,
		IndexerHeadCount: 2,
		IndexerKeyLength: 8, IndexerTopK: 2, IndexerFullLayers: []bool{true}}, MoESpec: model.MoESpec{LeadingDenseBlocks: 1},
	}
	input := builder.Input("input", dtype.F32, tensor.MustShape(8, 3))
	weights := model.LayerGraphWeights{
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
	result, err := model.BuildGLMDSABlockCached(builder, input, spec, weights, []uint32{0, 1, 2}, nil, nil, nil, nil, 0)
	if err != nil {
		t.Fatal(err)
	}
	outputs := []*tensor.Tensor{
		result.Output, result.Key, result.Value, result.Auxiliary,
		result.States[model.CacheStateIndexerKey].Value,
	}
	feeds := make(map[*tensor.Tensor]reference.Value)
	for _, node := range builder.Nodes() {
		if node.Op == tensor.OpInput {
			feeds[node] = patternedValue(node.Shape, int(node.ID%13)+3, 0.2, 0.1)
		}
	}
	checkCUDAGraph(t, feeds, uniformGraphChecks(outputs, 2e-4)...)
}

func TestExecutorDeepSeek32BlockMatchesReference(t *testing.T) {
	cudatest.Require(t)
	builder := tensor.NewBuilder()
	spec := model.Spec{CommonSpec: model.CommonSpec{Architecture: "deepseek32", BlockCount: 62, EmbeddingLength: 8,
		FeedForwardLength: 12,

		RMSNormEpsilon: 1e-5, LayerNormEpsilon: 1e-6}, AttentionSpec: model.AttentionSpec{HeadCount: 2, HeadCountKV: 1, KeyLength: 6, ValueLength: 4,
		QLoRARank: 3, KVLoRARank: 3, RopeDimensionCount: 2, RopeFrequencyBase: 10000,
		RopeScalingType: "yarn", RopeScalingFactor: 4, OriginalContextLength: 16,
		YaRNExtFactor: 1, YaRNAttentionFactor: 1, YaRNBetaFast: 32, YaRNBetaSlow: 1,

		IndexerHeadCount: 2, IndexerKeyLength: 8, IndexerTopK: 2,
		IndexerFullLayers: make([]bool, 62)}, MoESpec: model.MoESpec{LeadingDenseBlocks: 62},
	}
	for index := range spec.IndexerFullLayers {
		spec.IndexerFullLayers[index] = true
	}
	input := builder.Input("input", dtype.F32, tensor.MustShape(8, 3))
	weights := model.LayerGraphWeights{
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
	result, err := model.BuildDSABlockCached(
		builder, input, spec, weights, []uint32{0, 1, 2}, nil, nil, nil, nil, 0,
	)
	if err != nil {
		t.Fatal(err)
	}
	outputs := []*tensor.Tensor{
		result.Output, result.Key, result.Value,
		result.States[model.CacheStateIndexerKey].Value,
	}
	feeds := make(map[*tensor.Tensor]reference.Value)
	for _, node := range builder.Nodes() {
		if node.Op == tensor.OpInput {
			feeds[node] = patternedValue(node.Shape, int(node.ID%13)+3, 0.2, 0.1)
		}
	}
	checkCUDAGraph(t, feeds, uniformGraphChecks(outputs, 3e-4)...)
}

func TestExecutorNemotronHRecurrentBlockMatchesReference(t *testing.T) {
	cudatest.Require(t)
	builder := tensor.NewBuilder()
	spec := model.Spec{CommonSpec: model.CommonSpec{Architecture: "nemotron_h", BlockCount: 1, EmbeddingLength: 4,
		FeedForwardLength: 1,

		RMSNormEpsilon: 1e-5}, AttentionSpec: model.AttentionSpec{HeadCount: 1, HeadCountKV: 1, LayerHeadCounts: []uint32{0}, LayerKVHeadCounts: []uint32{0},
		KeyLength: 4, ValueLength: 4}, MoESpec: model.MoESpec{LayerFeedForward: []uint32{0}}, RecurrentSpec: model.RecurrentSpec{RecurrentLayers: []bool{true},
		SSMConvKernel: 3, SSMInnerSize: 8, SSMStateSize: 2, SSMTimeStepRank: 4, SSMGroupCount: 2},
	}
	input := builder.Input("input", dtype.F32, tensor.MustShape(4, 2))
	convState := builder.Input("conv_state", dtype.F32, tensor.MustShape(2, 16))
	ssmState := builder.Input("ssm_state", dtype.F32, tensor.MustShape(2, 8))
	weights := model.LayerGraphWeights{
		AttentionNorm: builder.Input("norm", dtype.F32, tensor.MustShape(4)),
		SSMInput:      builder.Input("ssm_in", dtype.F32, tensor.MustShape(4, 28)),
		SSMConv1D:     builder.Input("conv", dtype.F32, tensor.MustShape(3, 16)),
		SSMConv1DBias: builder.Input("conv_bias", dtype.F32, tensor.MustShape(16)),
		SSMTimeStep:   builder.Input("dt_bias", dtype.F32, tensor.MustShape(4)),
		SSMA:          builder.Input("a", dtype.F32, tensor.MustShape(1, 4)),
		SSMD:          builder.Input("d", dtype.F32, tensor.MustShape(1, 4)),
		SSMNorm:       builder.Input("ssm_norm", dtype.F32, tensor.MustShape(4, 2)),
		SSMOutput:     builder.Input("ssm_out", dtype.F32, tensor.MustShape(8, 4)),
	}
	result, err := model.BuildNemotronHBlockCached(
		builder, input, spec, weights, []uint32{0, 1}, convState, ssmState, 0,
	)
	if err != nil {
		t.Fatal(err)
	}
	feeds := map[*tensor.Tensor]reference.Value{}
	for index, node := range []*tensor.Tensor{
		input, convState, ssmState, weights.AttentionNorm, weights.SSMInput, weights.SSMConv1D,
		weights.SSMConv1DBias, weights.SSMTimeStep, weights.SSMA, weights.SSMD, weights.SSMNorm, weights.SSMOutput,
	} {
		offset := float32(0)
		if node == weights.AttentionNorm || node == weights.SSMNorm {
			offset = 1
		}
		feeds[node] = patternedValue(node.Shape, index+3, 0.04, offset)
	}
	outputs := []*tensor.Tensor{result.Output, result.Key, result.Value}
	want, err := reference.Execute(outputs, feeds)
	if err != nil {
		t.Fatal(err)
	}
	cuda, err := New(0)
	if err != nil {
		t.Fatal(err)
	}
	defer cuda.Close()
	got, err := cuda.Execute(context.Background(), outputs, feeds)
	if err != nil {
		t.Fatal(err)
	}
	compare(t, got[result.Output].Data, want[result.Output].Data, 8e-4)
	compare(t, got[result.Key].Data, want[result.Key].Data, 5e-5)
	compare(t, got[result.Value].Data, want[result.Value].Data, 8e-4)
}

func TestExecutorNemotronHMoEBlockMatchesReference(t *testing.T) {
	cudatest.Require(t)
	builder := tensor.NewBuilder()
	spec := model.Spec{CommonSpec: model.CommonSpec{Architecture: "nemotron_h_moe", BlockCount: 1, EmbeddingLength: 8,
		FeedForwardLength: 6,

		RMSNormEpsilon: 1e-5}, AttentionSpec: model.AttentionSpec{HeadCount: 2, HeadCountKV: 1,
		LayerHeadCounts: []uint32{2}, LayerKVHeadCounts: []uint32{1},
		KeyLength: 4, ValueLength: 4}, MoESpec: model.MoESpec{LayerFeedForward: []uint32{6},

		ExpertCount: 4, ExpertUsedCount: 2, ExpertFeedForward: 6, SharedExpertFF: 5,
		ExpertWeightsNorm: true, ExpertWeightsScale: 1.25, MoELatentSize: 4}, RecurrentSpec: model.RecurrentSpec{RecurrentLayers: []bool{false}},
	}
	input := builder.Input("input", dtype.F32, tensor.MustShape(8, 2))
	weights := model.LayerGraphWeights{
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
	result, err := model.BuildNemotronHBlockCached(builder, input, spec, weights, []uint32{0, 1}, nil, nil, 0)
	if err != nil {
		t.Fatal(err)
	}
	feeds := map[*tensor.Tensor]reference.Value{
		input:                         patternedValue(input.Shape, 3, 0.15, 0),
		weights.AttentionNorm:         patternedValue(weights.AttentionNorm.Shape, 5, 0.03, 1),
		weights.FeedForwardExpertBias: {Shape: weights.FeedForwardExpertBias.Shape, Data: []float32{0.1, -0.2, 0.3, -0.1}},
	}
	for index, node := range []*tensor.Tensor{
		weights.FeedForwardRouter, weights.FeedForwardLatentDown, weights.FeedForwardLatentUp,
		weights.FeedForwardUpExperts, weights.FeedForwardDownExperts,
		weights.FeedForwardSharedUp, weights.FeedForwardSharedDown,
	} {
		feeds[node] = patternedValue(node.Shape, index+11, 0.06, 0)
	}
	outputs := []*tensor.Tensor{result.Output, result.Key, result.Value}
	want, err := reference.Execute(outputs, feeds)
	if err != nil {
		t.Fatal(err)
	}
	cuda, err := New(0)
	if err != nil {
		t.Fatal(err)
	}
	defer cuda.Close()
	got, err := cuda.Execute(context.Background(), outputs, feeds)
	if err != nil {
		t.Fatal(err)
	}
	compare(t, got[result.Output].Data, want[result.Output].Data, 8e-4)
	compare(t, got[result.Key].Data, want[result.Key].Data, 5e-5)
	compare(t, got[result.Value].Data, want[result.Value].Data, 5e-5)
}

func TestExecutorNemotronHAttentionBlockMatchesReference(t *testing.T) {
	cudatest.Require(t)
	builder := tensor.NewBuilder()
	spec := model.Spec{CommonSpec: model.CommonSpec{Architecture: "nemotron_h", BlockCount: 1, EmbeddingLength: 8,
		FeedForwardLength: 1,

		RMSNormEpsilon: 1e-5}, AttentionSpec: model.AttentionSpec{HeadCount: 2, HeadCountKV: 1,
		LayerHeadCounts: []uint32{2}, LayerKVHeadCounts: []uint32{1},
		KeyLength: 4, ValueLength: 4}, MoESpec: model.MoESpec{LayerFeedForward: []uint32{0}}, RecurrentSpec: model.RecurrentSpec{RecurrentLayers: []bool{false}},
	}
	input := builder.Input("input", dtype.F32, tensor.MustShape(8, 2))
	weights := model.LayerGraphWeights{
		AttentionNorm:       builder.Input("norm", dtype.F32, tensor.MustShape(8)),
		AttentionQ:          builder.Input("q", dtype.F32, tensor.MustShape(8, 8)),
		AttentionK:          builder.Input("k", dtype.F32, tensor.MustShape(8, 4)),
		AttentionV:          builder.Input("v", dtype.F32, tensor.MustShape(8, 4)),
		AttentionOutput:     builder.Input("output", dtype.F32, tensor.MustShape(8, 8)),
		AttentionOutputBias: builder.Input("output_bias", dtype.F32, tensor.MustShape(8)),
	}
	result, err := model.BuildNemotronHBlockCached(builder, input, spec, weights, []uint32{0, 1}, nil, nil, 0)
	if err != nil {
		t.Fatal(err)
	}
	feeds := map[*tensor.Tensor]reference.Value{
		input:                 patternedValue(input.Shape, 3, 0.15, 0),
		weights.AttentionNorm: patternedValue(weights.AttentionNorm.Shape, 5, 0.03, 1),
	}
	for index, node := range []*tensor.Tensor{
		weights.AttentionQ, weights.AttentionK, weights.AttentionV,
		weights.AttentionOutput, weights.AttentionOutputBias,
	} {
		feeds[node] = patternedValue(node.Shape, index+11, 0.06, 0)
	}
	outputs := []*tensor.Tensor{result.Output, result.Key, result.Value}
	want, err := reference.Execute(outputs, feeds)
	if err != nil {
		t.Fatal(err)
	}
	cuda, err := New(0)
	if err != nil {
		t.Fatal(err)
	}
	defer cuda.Close()
	got, err := cuda.Execute(context.Background(), outputs, feeds)
	if err != nil {
		t.Fatal(err)
	}
	compare(t, got[result.Output].Data, want[result.Output].Data, 6e-4)
	compare(t, got[result.Key].Data, want[result.Key].Data, 5e-5)
	compare(t, got[result.Value].Data, want[result.Value].Data, 5e-5)
}

func TestExecutorMatchesReference(t *testing.T) {
	cudatest.Require(t)
	builder := tensor.NewBuilder()
	shape := tensor.MustShape(8, 3)
	left := builder.Input("left", dtype.F32, shape)
	right := builder.Input("right", dtype.F32, shape)
	add := builder.Add(left, right)
	multiply := builder.Multiply(add, right)
	divide := builder.Divide(multiply, right)
	scale := builder.Scale(multiply, 0.25)
	layerNorm := builder.LayerNorm(scale, 1e-5)
	reluSquared := builder.ReLUSquared(scale)
	norm := builder.RMSNorm(scale, 1e-5)
	silu := builder.SiLU(norm)
	geluErf := builder.GELUErf(norm)
	sigmoid := builder.Sigmoid(norm)
	softplus := builder.Softplus(builder.GELU(silu))
	xielu := builder.XIELU(scale, 0.8, 0.2, 0.5, -0.1)
	l2Norm := builder.L2Norm(builder.Add(sigmoid, softplus), 1e-6)
	output := builder.Softmax(l2Norm)
	if err := builder.Err(); err != nil {
		t.Fatal(err)
	}
	leftData := make([]float32, 24)
	rightData := make([]float32, 24)
	for i := range leftData {
		leftData[i] = float32(i-11) / 3
		rightData[i] = float32((i%7)+1) / 5
	}
	leftValue, _ := reference.NewValue(shape, leftData)
	rightValue, _ := reference.NewValue(shape, rightData)
	feeds := map[*tensor.Tensor]reference.Value{left: leftValue, right: rightValue}
	want, err := reference.Execute([]*tensor.Tensor{output, layerNorm, reluSquared, geluErf, xielu, divide}, feeds)
	if err != nil {
		t.Fatal(err)
	}

	cuda, err := New(0)
	if err != nil {
		t.Fatal(err)
	}
	defer cuda.Close()
	got, err := cuda.Execute(
		context.Background(),
		[]*tensor.Tensor{output, layerNorm, reluSquared, geluErf, xielu, divide},
		feeds,
	)
	if err != nil {
		t.Fatal(err)
	}
	compare(t, got[output].Data, want[output].Data, 2e-5)
	compare(t, got[layerNorm].Data, want[layerNorm].Data, 2e-5)
	compare(t, got[reluSquared].Data, want[reluSquared].Data, 2e-5)
	compare(t, got[geluErf].Data, want[geluErf].Data, 2e-5)
	compare(t, got[xielu].Data, want[xielu].Data, 2e-5)
	compare(t, got[divide].Data, want[divide].Data, 2e-5)
}

func TestExecutorMPTVariantsMatchReference(t *testing.T) {
	cudatest.Require(t)
	builder := tensor.NewBuilder()
	feeds := make(map[*tensor.Tensor]reference.Value)
	seed := 5
	input := func(name string, shape tensor.Shape, scale, offset float32) *tensor.Tensor {
		item := builder.Input(name, dtype.F32, shape)
		feeds[item] = patternedValue(shape, seed, scale, offset)
		seed += 2
		return item
	}
	spec := model.Spec{CommonSpec: model.CommonSpec{Architecture: "mpt", EmbeddingLength: 4, FeedForwardLength: 6,

		LayerNormEpsilon: 1e-5}, AttentionSpec: model.AttentionSpec{HeadCount: 2, HeadCountKV: 2, KeyLength: 2, ValueLength: 2,
		RopeDisabled: true, MaxALiBiBias: 8,
		AttentionClamp: 2},
	}
	weights := model.LayerGraphWeights{
		AttentionNorm:              input("attn_norm", tensor.MustShape(4), 0.03, 0.9),
		AttentionQKV:               input("qkv", tensor.MustShape(4, 12), 0.03, -0.1),
		AttentionQKVBias:           input("qkv_bias", tensor.MustShape(12), 0.02, -0.03),
		AttentionQNorm:             input("q_norm", tensor.MustShape(4), 0.03, 0.9),
		AttentionKNorm:             input("k_norm", tensor.MustShape(4), 0.03, 0.9),
		AttentionQNormBias:         input("q_norm_bias", tensor.MustShape(4), 0.02, -0.02),
		AttentionKNormBias:         input("k_norm_bias", tensor.MustShape(4), 0.02, -0.02),
		AttentionOutput:            input("attn_out", tensor.MustShape(4, 4), 0.03, -0.1),
		AttentionOutputBias:        input("attn_out_bias", tensor.MustShape(4), 0.02, -0.03),
		FeedForwardNorm:            input("ffn_norm", tensor.MustShape(4), 0.03, 0.9),
		FeedForwardUp:              input("ffn_up", tensor.MustShape(4, 6), 0.03, -0.1),
		FeedForwardUpBias:          input("ffn_up_bias", tensor.MustShape(6), 0.02, -0.03),
		FeedForwardActivationScale: input("ffn_act_scales", tensor.MustShape(6), 0.02, 0.8),
		FeedForwardDown:            input("ffn_down", tensor.MustShape(6, 4), 0.03, -0.1),
		FeedForwardDownBias:        input("ffn_down_bias", tensor.MustShape(4), 0.02, -0.03),
	}
	current := input("current", tensor.MustShape(4, 3), 0.08, -0.1)
	output, err := model.BuildDenseBlock(builder, current, spec, weights, []uint32{0, 1, 2})
	if err != nil {
		t.Fatal(err)
	}
	want, err := reference.Execute([]*tensor.Tensor{output}, feeds)
	if err != nil {
		t.Fatal(err)
	}
	cuda, err := New(0)
	if err != nil {
		t.Fatal(err)
	}
	defer cuda.Close()
	got, err := cuda.Execute(context.Background(), []*tensor.Tensor{output}, feeds)
	if err != nil {
		t.Fatal(err)
	}
	compare(t, got[output].Data, want[output].Data, 2e-3)
}

func TestExecutorGroupedMulMatMatchesReference(t *testing.T) {
	cudatest.Require(t)
	builder := tensor.NewBuilder()
	left := builder.Input("left", dtype.F32, tensor.MustShape(4, 3, 2))
	right := builder.Input("right", dtype.F32, tensor.MustShape(4, 2, 3))
	output := builder.GroupedMulMat(left, right)
	feeds := map[*tensor.Tensor]reference.Value{
		left:  patternedValue(left.Shape, 11, 0.13, -0.2),
		right: patternedValue(right.Shape, 7, 0.09, 0.05),
	}
	want, err := reference.Execute([]*tensor.Tensor{output}, feeds)
	if err != nil {
		t.Fatal(err)
	}
	cuda, err := New(0)
	if err != nil {
		t.Fatal(err)
	}
	defer cuda.Close()
	got, err := cuda.Execute(context.Background(), []*tensor.Tensor{output}, feeds)
	if err != nil {
		t.Fatal(err)
	}
	compare(t, got[output].Data, want[output].Data, 1e-5)
}

func TestExecutorQuantizedGroupedMulMatMatchesReference(t *testing.T) {
	cudatest.Require(t)
	leftShape := tensor.MustShape(32, 3, 2)
	rightShape := tensor.MustShape(32, 2, 2)
	leftValue := patternedValue(leftShape, 11, 0.03, -0.1)
	rightValue := patternedValue(rightShape, 7, 0.05, 0.02)
	storage, err := quant.Quantize(dtype.Q8_0, leftValue.Data)
	if err != nil {
		t.Fatal(err)
	}
	dequantized, err := quant.Dequantize(dtype.Q8_0, storage, uint64(len(leftValue.Data)))
	if err != nil {
		t.Fatal(err)
	}
	referenceBuilder := tensor.NewBuilder()
	referenceLeft := referenceBuilder.Input("left", dtype.F32, leftShape)
	referenceRight := referenceBuilder.Input("right", dtype.F32, rightShape)
	referenceOutput := referenceBuilder.GroupedMulMat(referenceLeft, referenceRight)
	want, err := reference.Execute([]*tensor.Tensor{referenceOutput}, map[*tensor.Tensor]reference.Value{
		referenceLeft: {Shape: leftShape, Data: dequantized}, referenceRight: rightValue,
	})
	if err != nil {
		t.Fatal(err)
	}
	builder := tensor.NewBuilder()
	left := builder.Input("left", dtype.Q8_0, leftShape)
	right := builder.Input("right", dtype.F32, rightShape)
	output := builder.GroupedMulMat(left, right)
	worker, err := device.New(0)
	if err != nil {
		t.Fatal(err)
	}
	defer worker.Close()
	var pointer driver.DevicePtr
	if err = worker.Do(context.Background(), func(state *device.State) error {
		var allocateErr error
		pointer, allocateErr = state.Driver.MemAlloc(uint64(len(storage)))
		if allocateErr != nil {
			return allocateErr
		}
		return state.Driver.MemcpyHtoD(pointer, storage)
	}); err != nil {
		t.Fatal(err)
	}
	defer worker.Do(context.Background(), func(state *device.State) error { return state.Driver.MemFree(pointer) })
	cuda, err := NewWithWorker(worker)
	if err != nil {
		t.Fatal(err)
	}
	defer cuda.Close()
	got, err := cuda.ExecuteWithDeviceFeeds(context.Background(), []*tensor.Tensor{output},
		map[*tensor.Tensor]reference.Value{right: rightValue}, map[*tensor.Tensor]driver.DevicePtr{left: pointer})
	if err != nil {
		t.Fatal(err)
	}
	compare(t, got[output].Data, want[referenceOutput].Data, 1e-4)
}

func TestExecutorMoEMatchesReference(t *testing.T) {
	cudatest.Require(t)
	builder := tensor.NewBuilder()
	input := builder.Input("input", dtype.F32, tensor.MustShape(2, 2))
	router := builder.Input("router", dtype.F32, tensor.MustShape(2, 3))
	gate := builder.Input("gate", dtype.F32, tensor.MustShape(2, 2, 3))
	up := builder.Input("up", dtype.F32, tensor.MustShape(2, 2, 3))
	down := builder.Input("down", dtype.F32, tensor.MustShape(2, 2, 3))
	output := builder.MoE(input, router, gate, up, down, 2, true, 1.25)
	if err := builder.Err(); err != nil {
		t.Fatal(err)
	}
	value := func(shape tensor.Shape, data []float32) reference.Value {
		result, err := reference.NewValue(shape, data)
		if err != nil {
			t.Fatal(err)
		}
		return result
	}
	feeds := map[*tensor.Tensor]reference.Value{
		input: value(input.Shape, []float32{0.5, -1, 1.5, 0.25}),
		router: value(router.Shape, []float32{
			1, -0.5,
			-0.25, 0.75,
			0.5, 0.5,
		}),
		gate: value(gate.Shape, []float32{
			1, 0, 0, 1,
			0.5, -0.5, 1, 0.25,
			-1, 0.5, 0.25, 1,
		}),
		up: value(up.Shape, []float32{
			0.25, 1, -0.5, 0.75,
			1, 0.5, 0.5, -1,
			0.75, -0.25, 1, 0.5,
		}),
		down: value(down.Shape, []float32{
			1, -0.5, 0.25, 0.75,
			-0.25, 1, 0.5, -0.75,
			0.75, 0.25, -1, 0.5,
		}),
	}
	want, err := reference.Execute([]*tensor.Tensor{output}, feeds)
	if err != nil {
		t.Fatal(err)
	}
	cuda, err := New(0)
	if err != nil {
		t.Fatal(err)
	}
	defer cuda.Close()
	got, err := cuda.Execute(context.Background(), []*tensor.Tensor{output}, feeds)
	if err != nil {
		t.Fatal(err)
	}
	compare(t, got[output].Data, want[output].Data, 5e-5)
}

func TestExecutorMoEExpertScaleMatchesReference(t *testing.T) {
	cudatest.Require(t)
	builder := tensor.NewBuilder()
	input := builder.Input("input", dtype.F32, tensor.MustShape(2, 2))
	router := builder.Input("router", dtype.F32, tensor.MustShape(2, 2))
	gate := builder.Input("gate", dtype.F32, tensor.MustShape(2, 2, 2))
	up := builder.Input("up", dtype.F32, tensor.MustShape(2, 2, 2))
	down := builder.Input("down", dtype.F32, tensor.MustShape(2, 2, 2))
	expertScale := builder.Input("expert_scale", dtype.F32, tensor.MustShape(2))
	output := builder.MoEGELUWithRouterInput(
		input, input, router, gate, up, down, expertScale, 1, true, 1,
	)
	scaleValue, err := reference.NewValue(expertScale.Shape, []float32{0.5, 1.75})
	if err != nil {
		t.Fatal(err)
	}
	feeds := map[*tensor.Tensor]reference.Value{
		input:       patternedValue(input.Shape, 3, 0.2, 0),
		router:      patternedValue(router.Shape, 5, 0.15, 0),
		gate:        patternedValue(gate.Shape, 7, 0.1, 0),
		up:          patternedValue(up.Shape, 11, 0.1, 0),
		down:        patternedValue(down.Shape, 13, 0.1, 0),
		expertScale: scaleValue,
	}
	want, err := reference.Execute([]*tensor.Tensor{output}, feeds)
	if err != nil {
		t.Fatal(err)
	}
	cuda, err := New(0)
	if err != nil {
		t.Fatal(err)
	}
	defer cuda.Close()
	got, err := cuda.Execute(context.Background(), []*tensor.Tensor{output}, feeds)
	if err != nil {
		t.Fatal(err)
	}
	compare(t, got[output].Data, want[output].Data, 7e-5)
}

func TestExecutorGroupedMoEMatchesReference(t *testing.T) {
	cudatest.Require(t)
	builder := tensor.NewBuilder()
	input := builder.Input("input", dtype.F32, tensor.MustShape(4, 2))
	routerInput := builder.Input("router-input", dtype.F32, tensor.MustShape(4, 2))
	router := builder.Input("router", dtype.F32, tensor.MustShape(4, 4))
	gate := builder.Input("gate", dtype.F32, tensor.MustShape(4, 3, 2))
	up := builder.Input("up", dtype.F32, tensor.MustShape(4, 3, 2))
	down := builder.Input("down", dtype.F32, tensor.MustShape(3, 4, 2))
	output := builder.MoEGroupedWithRouterInput(
		input, routerInput, router, gate, up, down, 2, true, 1.25, 2,
	)
	feeds := map[*tensor.Tensor]reference.Value{
		input:       patternedValue(input.Shape, 3, 0.2, 0),
		routerInput: patternedValue(routerInput.Shape, 5, 0.15, 0),
		router:      patternedValue(router.Shape, 7, 0.1, 0),
		gate:        patternedValue(gate.Shape, 11, 0.08, 0),
		up:          patternedValue(up.Shape, 13, 0.08, 0),
		down:        patternedValue(down.Shape, 17, 0.08, 0),
	}
	want, err := reference.Execute([]*tensor.Tensor{output}, feeds)
	if err != nil {
		t.Fatal(err)
	}
	cuda, err := New(0)
	if err != nil {
		t.Fatal(err)
	}
	defer cuda.Close()
	got, err := cuda.Execute(context.Background(), []*tensor.Tensor{output}, feeds)
	if err != nil {
		t.Fatal(err)
	}
	compare(t, got[output].Data, want[output].Data, 7e-5)
}

func TestExecutorGroupedMoESupportsTopKPolicyBoundary(t *testing.T) {
	cudatest.Require(t)
	const expertCount = uint64(tensor.MaxMoETopK)
	builder := tensor.NewBuilder()
	input := builder.Input("input", dtype.F32, tensor.MustShape(1, 1))
	router := builder.Input("router", dtype.F32, tensor.MustShape(1, expertCount))
	gate := builder.Input("gate", dtype.F32, tensor.MustShape(1, 1, expertCount))
	up := builder.Input("up", dtype.F32, tensor.MustShape(1, 1, expertCount))
	down := builder.Input("down", dtype.F32, tensor.MustShape(1, 1, expertCount))
	output := builder.MoE(input, router, gate, up, down, uint32(expertCount), true, 1)
	if err := builder.Err(); err != nil {
		t.Fatal(err)
	}
	feeds := map[*tensor.Tensor]reference.Value{
		input:  patternedValue(input.Shape, 3, 0.1, 0.2),
		router: patternedValue(router.Shape, 5, 0.1, 0),
		gate:   patternedValue(gate.Shape, 7, 0.1, 0),
		up:     patternedValue(up.Shape, 11, 0.1, 0),
		down:   patternedValue(down.Shape, 13, 0.1, 0),
	}
	want, err := reference.Execute([]*tensor.Tensor{output}, feeds)
	if err != nil {
		t.Fatal(err)
	}
	cuda, err := New(0)
	if err != nil {
		t.Fatal(err)
	}
	defer cuda.Close()
	got, err := cuda.Execute(context.Background(), []*tensor.Tensor{output}, feeds)
	if err != nil {
		t.Fatal(err)
	}
	compare(t, got[output].Data, want[output].Data, 1e-5)
}

func TestExecutorGroveMoEBlockMatchesReference(t *testing.T) {
	cudatest.Require(t)
	builder := tensor.NewBuilder()
	spec := model.Spec{CommonSpec: model.CommonSpec{Architecture: "grovemoe", BlockCount: 1, EmbeddingLength: 8,
		FeedForwardLength: 24,

		RMSNormEpsilon: 1e-6}, AttentionSpec: model.AttentionSpec{HeadCount: 2, HeadCountKV: 1, KeyLength: 4, ValueLength: 4,
		RopeFrequencyBase: 10000}, MoESpec: model.MoESpec{ExpertCount: 4, ExpertUsedCount: 2,
		ExpertFeedForward: 6, ExpertChunkFeedForward: 3, ExpertWeightsScale: 1.25,
		ExpertGroupScale: 0.5, ExpertsPerGroup: 2},
	}
	input := builder.Input("input", dtype.F32, tensor.MustShape(8, 3))
	weights := model.LayerGraphWeights{
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
	result, err := model.BuildDenseBlockCachedForLayer(
		builder, input, spec, weights, []uint32{0, 1, 2}, nil, nil, 0,
	)
	if err != nil {
		t.Fatal(err)
	}
	feeds := map[*tensor.Tensor]reference.Value{input: patternedValue(input.Shape, 3, 0.2, 0)}
	for index, node := range []*tensor.Tensor{
		weights.AttentionQ, weights.AttentionK, weights.AttentionV, weights.AttentionOutput,
		weights.FeedForwardRouter, weights.FeedForwardGateExperts,
		weights.FeedForwardUpExperts, weights.FeedForwardDownExperts,
		weights.FeedForwardGateChunkExperts, weights.FeedForwardUpChunkExperts,
		weights.FeedForwardDownChunkExperts,
	} {
		feeds[node] = patternedValue(node.Shape, index+7, 0.05, 0)
	}
	for index, node := range []*tensor.Tensor{
		weights.AttentionNorm, weights.AttentionQNorm, weights.AttentionKNorm, weights.FeedForwardNorm,
	} {
		feeds[node] = patternedValue(node.Shape, index+31, 0.03, 1)
	}
	outputs := []*tensor.Tensor{result.Output, result.Key, result.Value}
	want, err := reference.Execute(outputs, feeds)
	if err != nil {
		t.Fatal(err)
	}
	cuda, err := New(0)
	if err != nil {
		t.Fatal(err)
	}
	defer cuda.Close()
	got, err := cuda.Execute(context.Background(), outputs, feeds)
	if err != nil {
		t.Fatal(err)
	}
	compare(t, got[result.Output].Data, want[result.Output].Data, 8e-4)
	compare(t, got[result.Key].Data, want[result.Key].Data, 5e-5)
	compare(t, got[result.Value].Data, want[result.Value].Data, 5e-5)
}

func TestExecutorLlama4MoEBlockMatchesReference(t *testing.T) {
	cudatest.Require(t)
	builder := tensor.NewBuilder()
	spec := model.Spec{CommonSpec: model.CommonSpec{Architecture: "llama4", BlockCount: 4, EmbeddingLength: 4, FeedForwardLength: 6,

		RMSNormEpsilon: 1e-5}, AttentionSpec: model.AttentionSpec{HeadCount: 2, HeadCountKV: 1, KeyLength: 2, ValueLength: 2,
		RopeDimensionCount: 2, RopeFrequencyBase: 10000, RopeFrequencySWA: 10000,
		SlidingWindow: 4, SlidingPattern: 4, NoRopeLayerStep: 4}, MoESpec: model.MoESpec{ExpertCount: 4, ExpertUsedCount: 2, ExpertFeedForward: 4, SharedExpertFF: 4,
		ExpertWeightsScale: 1, ExpertGatingFunc: 2, MoELayerStep: 4},
	}
	input := builder.Input("input", dtype.F32, tensor.MustShape(4, 2))
	weights := model.LayerGraphWeights{
		AttentionNorm:             builder.Input("attn_norm", dtype.F32, tensor.MustShape(4)),
		AttentionQ:                builder.Input("attn_q", dtype.F32, tensor.MustShape(4, 4)),
		AttentionK:                builder.Input("attn_k", dtype.F32, tensor.MustShape(4, 2)),
		AttentionV:                builder.Input("attn_v", dtype.F32, tensor.MustShape(4, 2)),
		AttentionOutput:           builder.Input("attn_output", dtype.F32, tensor.MustShape(4, 4)),
		AttentionTemperatureScale: builder.Input("attn_temp", dtype.F32, tensor.MustShape(1, 1, 2)),
		FeedForwardNorm:           builder.Input("ffn_norm", dtype.F32, tensor.MustShape(4)),
		FeedForwardRouter:         builder.Input("router", dtype.F32, tensor.MustShape(4, 4)),
		FeedForwardGateExperts:    builder.Input("gate_exps", dtype.F32, tensor.MustShape(4, 4, 4)),
		FeedForwardUpExperts:      builder.Input("up_exps", dtype.F32, tensor.MustShape(4, 4, 4)),
		FeedForwardDownExperts:    builder.Input("down_exps", dtype.F32, tensor.MustShape(4, 4, 4)),
		FeedForwardSharedGate:     builder.Input("shared_gate", dtype.F32, tensor.MustShape(4, 4)),
		FeedForwardSharedUp:       builder.Input("shared_up", dtype.F32, tensor.MustShape(4, 4)),
		FeedForwardSharedDown:     builder.Input("shared_down", dtype.F32, tensor.MustShape(4, 4)),
	}
	result, err := model.BuildDenseBlockCachedForLayer(
		builder, input, spec, weights, []uint32{8192, 8193}, nil, nil, 3,
	)
	if err != nil {
		t.Fatal(err)
	}
	feeds := map[*tensor.Tensor]reference.Value{input: patternedValue(input.Shape, 3, 0.2, 0)}
	for index, node := range []*tensor.Tensor{
		weights.AttentionQ, weights.AttentionK, weights.AttentionV, weights.AttentionOutput,
		weights.FeedForwardRouter, weights.FeedForwardGateExperts, weights.FeedForwardUpExperts,
		weights.FeedForwardDownExperts, weights.FeedForwardSharedGate, weights.FeedForwardSharedUp,
		weights.FeedForwardSharedDown,
	} {
		feeds[node] = patternedValue(node.Shape, index+11, 0.08, 0)
	}
	feeds[weights.AttentionNorm] = patternedValue(weights.AttentionNorm.Shape, 29, 0.03, 1)
	feeds[weights.FeedForwardNorm] = patternedValue(weights.FeedForwardNorm.Shape, 31, 0.03, 1)
	feeds[weights.AttentionTemperatureScale] = reference.Value{
		Shape: weights.AttentionTemperatureScale.Shape, Data: []float32{1.0693147, 1.0693147},
	}
	outputs := []*tensor.Tensor{result.Output, result.Key, result.Value}
	want, err := reference.Execute(outputs, feeds)
	if err != nil {
		t.Fatal(err)
	}
	cuda, err := New(0)
	if err != nil {
		t.Fatal(err)
	}
	defer cuda.Close()
	got, err := cuda.Execute(context.Background(), outputs, feeds)
	if err != nil {
		t.Fatal(err)
	}
	compare(t, got[result.Output].Data, want[result.Output].Data, 8e-4)
	compare(t, got[result.Key].Data, want[result.Key].Data, 1e-6)
	compare(t, got[result.Value].Data, want[result.Value].Data, 1e-6)
}

func TestExecutorMistral3TemperatureMoEBlockMatchesReference(t *testing.T) {
	cudatest.Require(t)
	builder := tensor.NewBuilder()
	spec := model.Spec{CommonSpec: model.CommonSpec{Architecture: "mistral3", EmbeddingLength: 4, FeedForwardLength: 4,

		RMSNormEpsilon: 1e-5}, AttentionSpec: model.AttentionSpec{HeadCount: 2, HeadCountKV: 1, KeyLength: 2, ValueLength: 2,
		RopeDimensionCount: 2, RopeFrequencyBase: 10000,
		AttentionTempScale: 0.1, AttentionTempFloor: 8}, MoESpec: model.MoESpec{ExpertCount: 4, ExpertUsedCount: 2, ExpertFeedForward: 4,
		ExpertWeightsScale: 1, ExpertWeightsNorm: true},
	}
	input := builder.Input("input", dtype.F32, tensor.MustShape(4, 2))
	weights := model.LayerGraphWeights{
		AttentionNorm:             builder.Input("attn_norm", dtype.F32, tensor.MustShape(4)),
		AttentionQ:                builder.Input("attn_q", dtype.F32, tensor.MustShape(4, 4)),
		AttentionK:                builder.Input("attn_k", dtype.F32, tensor.MustShape(4, 2)),
		AttentionV:                builder.Input("attn_v", dtype.F32, tensor.MustShape(4, 2)),
		AttentionOutput:           builder.Input("attn_output", dtype.F32, tensor.MustShape(4, 4)),
		AttentionTemperatureScale: builder.Input("attn_temp", dtype.F32, tensor.MustShape(1, 1, 2)),
		FeedForwardNorm:           builder.Input("ffn_norm", dtype.F32, tensor.MustShape(4)),
		FeedForwardRouter:         builder.Input("router", dtype.F32, tensor.MustShape(4, 4)),
		FeedForwardGateExperts:    builder.Input("gate_exps", dtype.F32, tensor.MustShape(4, 4, 4)),
		FeedForwardUpExperts:      builder.Input("up_exps", dtype.F32, tensor.MustShape(4, 4, 4)),
		FeedForwardDownExperts:    builder.Input("down_exps", dtype.F32, tensor.MustShape(4, 4, 4)),
	}
	result, err := model.BuildDenseBlockCachedForLayer(
		builder, input, spec, weights, []uint32{8, 9}, nil, nil, 0,
	)
	if err != nil {
		t.Fatal(err)
	}
	feeds := map[*tensor.Tensor]reference.Value{input: patternedValue(input.Shape, 3, 0.2, 0)}
	for index, node := range []*tensor.Tensor{
		weights.AttentionQ, weights.AttentionK, weights.AttentionV, weights.AttentionOutput,
		weights.FeedForwardRouter, weights.FeedForwardGateExperts,
		weights.FeedForwardUpExperts, weights.FeedForwardDownExperts,
	} {
		feeds[node] = patternedValue(node.Shape, index+11, 0.08, 0)
	}
	feeds[weights.AttentionNorm] = patternedValue(weights.AttentionNorm.Shape, 29, 0.03, 1)
	feeds[weights.FeedForwardNorm] = patternedValue(weights.FeedForwardNorm.Shape, 31, 0.03, 1)
	feeds[weights.AttentionTemperatureScale] = reference.Value{
		Shape: weights.AttentionTemperatureScale.Shape,
		Data:  []float32{1 + 0.1*float32(math.Log(2)), 1 + 0.1*float32(math.Log(2))},
	}
	outputs := []*tensor.Tensor{result.Output, result.Key, result.Value}
	want, err := reference.Execute(outputs, feeds)
	if err != nil {
		t.Fatal(err)
	}
	cuda, err := New(0)
	if err != nil {
		t.Fatal(err)
	}
	defer cuda.Close()
	got, err := cuda.Execute(context.Background(), outputs, feeds)
	if err != nil {
		t.Fatal(err)
	}
	compare(t, got[result.Output].Data, want[result.Output].Data, 8e-4)
	compare(t, got[result.Key].Data, want[result.Key].Data, 5e-6)
	compare(t, got[result.Value].Data, want[result.Value].Data, 5e-6)
}

func TestExecutorGPTOSSBlockMatchesReference(t *testing.T) {
	cudatest.Require(t)
	builder := tensor.NewBuilder()
	spec := model.Spec{CommonSpec: model.CommonSpec{Architecture: "gpt-oss", BlockCount: 2, EmbeddingLength: 4, FeedForwardLength: 4,

		RMSNormEpsilon: 1e-5}, AttentionSpec: model.AttentionSpec{HeadCount: 2, HeadCountKV: 1, KeyLength: 2, ValueLength: 2,
		RopeDimensionCount: 2, RopeFrequencyBase: 10000, RopeFrequencySWA: 2000,
		SlidingWindow: 4, SlidingPattern: 2}, MoESpec: model.MoESpec{ExpertCount: 4, ExpertUsedCount: 2, ExpertFeedForward: 4,
		ExpertWeightsScale: 1, ExpertGatingFunc: 3},
	}
	input := builder.Input("input", dtype.F32, tensor.MustShape(4, 2))
	weights := model.LayerGraphWeights{
		AttentionNorm:          builder.Input("attn_norm", dtype.F32, tensor.MustShape(4)),
		AttentionPostNorm:      builder.Input("post_attn_norm", dtype.F32, tensor.MustShape(4)),
		AttentionQ:             builder.Input("attn_q", dtype.F32, tensor.MustShape(4, 4)),
		AttentionK:             builder.Input("attn_k", dtype.F32, tensor.MustShape(4, 2)),
		AttentionV:             builder.Input("attn_v", dtype.F32, tensor.MustShape(4, 2)),
		AttentionOutput:        builder.Input("attn_output", dtype.F32, tensor.MustShape(4, 4)),
		AttentionOutputBias:    builder.Input("attn_output_bias", dtype.F32, tensor.MustShape(4)),
		AttentionSinks:         builder.Input("attn_sinks", dtype.F32, tensor.MustShape(2)),
		FeedForwardRouter:      builder.Input("router", dtype.F32, tensor.MustShape(4, 4)),
		FeedForwardRouterBias:  builder.Input("router_bias", dtype.F32, tensor.MustShape(4)),
		FeedForwardGateExperts: builder.Input("gate_exps", dtype.F32, tensor.MustShape(4, 4, 4)),
		FeedForwardGateBias:    builder.Input("gate_bias", dtype.F32, tensor.MustShape(4, 4)),
		FeedForwardUpExperts:   builder.Input("up_exps", dtype.F32, tensor.MustShape(4, 4, 4)),
		FeedForwardUpBias:      builder.Input("up_bias", dtype.F32, tensor.MustShape(4, 4)),
		FeedForwardDownExperts: builder.Input("down_exps", dtype.F32, tensor.MustShape(4, 4, 4)),
		FeedForwardDownBias:    builder.Input("down_bias", dtype.F32, tensor.MustShape(4, 4)),
	}
	result, err := model.BuildDenseBlockCachedForLayer(builder, input, spec, weights, []uint32{0, 1}, nil, nil, 0)
	if err != nil {
		t.Fatal(err)
	}
	feeds := map[*tensor.Tensor]reference.Value{input: patternedValue(input.Shape, 3, 0.2, 0)}
	for index, node := range []*tensor.Tensor{
		weights.AttentionQ, weights.AttentionK, weights.AttentionV, weights.AttentionOutput,
		weights.FeedForwardRouter, weights.FeedForwardGateExperts, weights.FeedForwardUpExperts,
		weights.FeedForwardDownExperts,
	} {
		feeds[node] = patternedValue(node.Shape, index+11, 0.08, 0)
	}
	for index, node := range []*tensor.Tensor{
		weights.AttentionNorm, weights.AttentionPostNorm, weights.AttentionOutputBias,
		weights.AttentionSinks, weights.FeedForwardRouterBias, weights.FeedForwardGateBias,
		weights.FeedForwardUpBias, weights.FeedForwardDownBias,
	} {
		feeds[node] = patternedValue(node.Shape, index+31, 0.03, 0.7)
	}
	outputs := []*tensor.Tensor{result.Output, result.Key, result.Value}
	want, err := reference.Execute(outputs, feeds)
	if err != nil {
		t.Fatal(err)
	}
	cuda, err := New(0)
	if err != nil {
		t.Fatal(err)
	}
	defer cuda.Close()
	got, err := cuda.Execute(context.Background(), outputs, feeds)
	if err != nil {
		t.Fatal(err)
	}
	compare(t, got[result.Output].Data, want[result.Output].Data, 8e-4)
	compare(t, got[result.Key].Data, want[result.Key].Data, 1e-6)
	compare(t, got[result.Value].Data, want[result.Value].Data, 1e-6)
}

func TestExecutorGLM4MoEBlockMatchesReference(t *testing.T) {
	cudatest.Require(t)
	builder := tensor.NewBuilder()
	spec := model.Spec{CommonSpec: model.CommonSpec{Architecture: "glm4moe", BlockCount: 2,
		EmbeddingLength: 8, FeedForwardLength: 12,

		RMSNormEpsilon: 1e-6}, AttentionSpec: model.AttentionSpec{HeadCount: 2, HeadCountKV: 1, KeyLength: 4, ValueLength: 4,
		RopeDimensionCount: 4, RopeSections: [4]int32{1, 1, 0, 0},
		RopeFrequencyBase: 10000}, MoESpec: model.MoESpec{LeadingDenseBlocks: 1,
		ExpertCount:     4,
		ExpertUsedCount: 2, ExpertFeedForward: 6, SharedExpertFF: 12,
		ExpertWeightsScale: 1.25, ExpertWeightsNorm: true, ExpertGatingFunc: 2},
	}
	input := builder.Input("input", dtype.F32, tensor.MustShape(8, 3))
	weights := model.LayerGraphWeights{
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
	result, err := model.BuildDenseBlockCachedForLayer(
		builder, input, spec, weights, []uint32{0, 1, 2}, nil, nil, 1,
	)
	if err != nil {
		t.Fatal(err)
	}
	feeds := map[*tensor.Tensor]reference.Value{input: patternedValue(input.Shape, 3, 0.2, 0)}
	for index, node := range []*tensor.Tensor{
		weights.AttentionQ, weights.AttentionK, weights.AttentionV, weights.AttentionOutput,
		weights.FeedForwardRouter, weights.FeedForwardGateExperts,
		weights.FeedForwardUpExperts, weights.FeedForwardDownExperts,
		weights.FeedForwardExpertBias, weights.FeedForwardSharedGate,
		weights.FeedForwardSharedUp, weights.FeedForwardSharedDown,
	} {
		feeds[node] = patternedValue(node.Shape, index+7, 0.05, 0)
	}
	for index, node := range []*tensor.Tensor{
		weights.AttentionNorm, weights.AttentionQNorm, weights.AttentionKNorm, weights.FeedForwardNorm,
	} {
		feeds[node] = patternedValue(node.Shape, index+37, 0.03, 1)
	}
	outputs := []*tensor.Tensor{result.Output, result.Key, result.Value}
	want, err := reference.Execute(outputs, feeds)
	if err != nil {
		t.Fatal(err)
	}
	cuda, err := New(0)
	if err != nil {
		t.Fatal(err)
	}
	defer cuda.Close()
	got, err := cuda.Execute(context.Background(), outputs, feeds)
	if err != nil {
		t.Fatal(err)
	}
	compare(t, got[result.Output].Data, want[result.Output].Data, 8e-4)
	compare(t, got[result.Key].Data, want[result.Key].Data, 5e-5)
	compare(t, got[result.Value].Data, want[result.Value].Data, 5e-5)
}

func TestExecutorNativeQuantizedMoEMatchesReference(t *testing.T) {
	cudatest.Require(t)
	for _, dataType := range []dtype.Type{
		dtype.Q8_0,
		dtype.Q4_0,
		dtype.Q4K,
		dtype.IQ4XS,
		dtype.TQ2_0,
		dtype.MXFP4,
		dtype.Q1_0,
	} {
		t.Run(dataType.String(), func(t *testing.T) {
			testExecutorNativeQuantizedMoE(t, dataType)
		})
	}
}

func testExecutorNativeQuantizedMoE(t *testing.T, dataType dtype.Type) {
	t.Helper()
	traits, ok := dataType.Traits()
	if !ok {
		t.Fatalf("missing traits for %s", dataType)
	}
	width := traits.BlockSize
	inputShape := tensor.MustShape(width, 1)
	routerShape := tensor.MustShape(width, 2)
	wideRouterInputShape := tensor.MustShape(2*width, 1)
	wideRouterShape := tensor.MustShape(2*width, 2)
	gateShape := tensor.MustShape(width, width, 2)
	downShape := tensor.MustShape(width, width, 2)
	inputValue := patternedValue(inputShape, 3, 0.02, 0)
	routerValue := patternedValue(routerShape, 5, 0.01, 0)
	wideRouterInputValue := patternedValue(wideRouterInputShape, 6, 0.01, 0)
	wideRouterValue := patternedValue(wideRouterShape, 8, 0.01, 0)
	selectionBiasValue := reference.Value{Shape: tensor.MustShape(2), Data: []float32{0.2, -0.1}}
	gateValue := patternedValue(gateShape, 7, 0.01, 0)
	upValue := patternedValue(gateShape, 11, 0.01, 0)
	downValue := patternedValue(downShape, 13, 0.01, 0)
	gateUpShape := tensor.MustShape(width, 2*width, 2)
	gateUpData := make([]float32, 4*width*width)
	expertSize := int(width * width)
	for expert := range 2 {
		destination := expert * 2 * expertSize
		copy(gateUpData[destination:destination+expertSize], gateValue.Data[expert*expertSize:(expert+1)*expertSize])
		copy(gateUpData[destination+expertSize:destination+2*expertSize], upValue.Data[expert*expertSize:(expert+1)*expertSize])
	}
	gateUpValue, err := reference.NewValue(gateUpShape, gateUpData)
	if err != nil {
		t.Fatal(err)
	}

	quantize := func(value reference.Value) ([]byte, reference.Value) {
		t.Helper()
		storage, err := quant.Quantize(dataType, value.Data)
		if err != nil {
			t.Fatal(err)
		}
		dequantized, err := quant.Dequantize(dataType, storage, uint64(len(value.Data)))
		if err != nil {
			t.Fatal(err)
		}
		result, err := reference.NewValue(value.Shape, dequantized)
		if err != nil {
			t.Fatal(err)
		}
		return storage, result
	}
	gateStorage, gateReference := quantize(gateValue)
	upStorage, upReference := quantize(upValue)
	downStorage, downReference := quantize(downValue)
	gateUpStorage, gateUpReference := quantize(gateUpValue)

	referenceBuilder := tensor.NewBuilder()
	referenceInput := referenceBuilder.Input("input", dtype.F32, inputShape)
	referenceRouter := referenceBuilder.Input("router", dtype.F32, routerShape)
	referenceWideRouterInput := referenceBuilder.Input("wide_router_input", dtype.F32, wideRouterInputShape)
	referenceWideRouter := referenceBuilder.Input("wide_router", dtype.F32, wideRouterShape)
	referenceSelectionBias := referenceBuilder.Input("selection_bias", dtype.F32, tensor.MustShape(2))
	referenceGate := referenceBuilder.Input("gate", dtype.F32, gateShape)
	referenceUp := referenceBuilder.Input("up", dtype.F32, gateShape)
	referenceDown := referenceBuilder.Input("down", dtype.F32, downShape)
	referenceGateUp := referenceBuilder.Input("gate_up", dtype.F32, gateUpShape)
	referenceOutput := referenceBuilder.MoE(
		referenceInput, referenceRouter, referenceGate, referenceUp, referenceDown, 2, true, 1.25,
	)
	referenceFusedOutput := referenceBuilder.MoESigmoidFusedGateUp(
		referenceInput, referenceRouter, referenceGateUp, referenceDown, nil, 2, true, 1.25,
	)
	referenceSquaredOutput := referenceBuilder.MoEReLUSquaredWithRouterInput(
		referenceInput, referenceWideRouterInput, referenceWideRouter, referenceUp, referenceDown,
		referenceSelectionBias, 2, true, 1.25, tensor.MoERoutingSigmoid,
	)
	want, err := reference.Execute(
		[]*tensor.Tensor{referenceOutput, referenceFusedOutput, referenceSquaredOutput},
		map[*tensor.Tensor]reference.Value{
			referenceInput:           inputValue,
			referenceRouter:          routerValue,
			referenceWideRouterInput: wideRouterInputValue,
			referenceWideRouter:      wideRouterValue,
			referenceSelectionBias:   selectionBiasValue,
			referenceGate:            gateReference,
			referenceUp:              upReference,
			referenceDown:            downReference,
			referenceGateUp:          gateUpReference,
		},
	)
	if err != nil {
		t.Fatal(err)
	}

	builder := tensor.NewBuilder()
	input := builder.Input("input", dtype.F32, inputShape)
	router := builder.Input("router", dtype.F32, routerShape)
	wideRouterInput := builder.Input("wide_router_input", dtype.F32, wideRouterInputShape)
	wideRouter := builder.Input("wide_router", dtype.F32, wideRouterShape)
	selectionBias := builder.Input("selection_bias", dtype.F32, tensor.MustShape(2))
	gate := builder.Input("gate", dataType, gateShape)
	up := builder.Input("up", dataType, gateShape)
	down := builder.Input("down", dataType, downShape)
	gateUp := builder.Input("gate_up", dataType, gateUpShape)
	output := builder.MoE(input, router, gate, up, down, 2, true, 1.25)
	fusedOutput := builder.MoESigmoidFusedGateUp(input, router, gateUp, down, nil, 2, true, 1.25)
	squaredOutput := builder.MoEReLUSquaredWithRouterInput(
		input, wideRouterInput, wideRouter, up, down, selectionBias, 2, true, 1.25, tensor.MoERoutingSigmoid,
	)
	if err := builder.Err(); err != nil {
		t.Fatal(err)
	}

	worker, err := device.New(0)
	if err != nil {
		t.Fatal(err)
	}
	defer worker.Close()
	deviceFeeds := make(map[*tensor.Tensor]driver.DevicePtr, 4)
	storages := []struct {
		node *tensor.Tensor
		data []byte
	}{{gate, gateStorage}, {up, upStorage}, {down, downStorage}, {gateUp, gateUpStorage}}
	var allocations []driver.DevicePtr
	err = worker.Do(context.Background(), func(state *device.State) error {
		for _, item := range storages {
			pointer, allocateErr := state.Driver.MemAlloc(uint64(len(item.data)))
			if allocateErr != nil {
				return allocateErr
			}
			allocations = append(allocations, pointer)
			deviceFeeds[item.node] = pointer
			if copyErr := state.Driver.MemcpyHtoD(pointer, item.data); copyErr != nil {
				return copyErr
			}
		}
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
	defer worker.Do(context.Background(), func(state *device.State) error {
		for _, pointer := range allocations {
			if err := state.Driver.MemFree(pointer); err != nil {
				return err
			}
		}
		return nil
	})
	cuda, err := NewWithWorker(worker)
	if err != nil {
		t.Fatal(err)
	}
	defer cuda.Close()
	got, err := cuda.ExecuteWithDeviceFeeds(
		context.Background(),
		[]*tensor.Tensor{output, fusedOutput, squaredOutput},
		map[*tensor.Tensor]reference.Value{
			input: inputValue, router: routerValue, wideRouterInput: wideRouterInputValue,
			wideRouter: wideRouterValue, selectionBias: selectionBiasValue,
		},
		deviceFeeds,
	)
	if err != nil {
		t.Fatal(err)
	}
	compare(t, got[output].Data, want[referenceOutput].Data, 5e-4)
	compare(t, got[fusedOutput].Data, want[referenceFusedOutput].Data, 5e-4)
	compare(t, got[squaredOutput].Data, want[referenceSquaredOutput].Data, 5e-4)
}

func TestExecutorSigmoidMoEWithSelectionBiasMatchesReference(t *testing.T) {
	cudatest.Require(t)
	builder := tensor.NewBuilder()
	input := builder.Input("input", dtype.F32, tensor.MustShape(3, 2))
	router := builder.Input("router", dtype.F32, tensor.MustShape(3, 4))
	gate := builder.Input("gate", dtype.F32, tensor.MustShape(3, 5, 4))
	up := builder.Input("up", dtype.F32, tensor.MustShape(3, 5, 4))
	down := builder.Input("down", dtype.F32, tensor.MustShape(5, 3, 4))
	bias := builder.Input("bias", dtype.F32, tensor.MustShape(4))
	output := builder.MoESigmoid(input, router, gate, up, down, bias, 2, true, 1.25)
	feeds := map[*tensor.Tensor]reference.Value{
		input:  patternedValue(input.Shape, 3, 0.4, 0),
		router: patternedValue(router.Shape, 5, 0.3, 0),
		gate:   patternedValue(gate.Shape, 7, 0.2, 0),
		up:     patternedValue(up.Shape, 11, 0.2, 0),
		down:   patternedValue(down.Shape, 13, 0.2, 0),
		bias:   patternedValue(bias.Shape, 17, 0.4, 0),
	}
	want, err := reference.Execute([]*tensor.Tensor{output}, feeds)
	if err != nil {
		t.Fatal(err)
	}
	cuda, err := New(0)
	if err != nil {
		t.Fatal(err)
	}
	defer cuda.Close()
	got, err := cuda.Execute(context.Background(), []*tensor.Tensor{output}, feeds)
	if err != nil {
		t.Fatal(err)
	}
	compare(t, got[output].Data, want[output].Data, 7e-5)
}

func TestExecutorRetainedOutputLifetime(t *testing.T) {
	cudatest.Require(t)
	builder := tensor.NewBuilder()
	shape := tensor.MustShape(4)
	left := builder.Input("left", dtype.F32, shape)
	right := builder.Input("right", dtype.F32, shape)
	output := builder.Add(left, right)
	leftValue, _ := reference.NewValue(shape, []float32{1, 2, 3, 4})
	rightValue, _ := reference.NewValue(shape, []float32{10, 20, 30, 40})
	cuda, err := New(0)
	if err != nil {
		t.Fatal(err)
	}
	defer cuda.Close()
	feeds := map[*tensor.Tensor]reference.Value{left: leftValue, right: rightValue}
	if _, err := cuda.Execute(context.Background(), []*tensor.Tensor{output}, feeds); err != nil {
		t.Fatal(err)
	}
	before, err := cuda.worker.MemoryStats(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	retained, err := cuda.ExecuteRetainedWithDeviceFeeds(
		context.Background(),
		[]*tensor.Tensor{output},
		feeds,
		nil,
	)
	if err != nil {
		t.Fatal(err)
	}
	during, err := cuda.worker.MemoryStats(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if during.CurrentBytes-before.CurrentBytes != minimumDeviceBufferBytes {
		t.Fatalf(
			"retained pool bytes = %d, want %d",
			during.CurrentBytes-before.CurrentBytes,
			minimumDeviceBufferBytes,
		)
	}
	got, err := retained.CopyToHost(context.Background(), output)
	if err != nil {
		t.Fatal(err)
	}
	compare(t, got.Data, []float32{11, 22, 33, 44}, 0)
	canceled, cancel := context.WithCancel(context.Background())
	cancel()
	if err := retained.Release(canceled); !errors.Is(err, context.Canceled) {
		t.Fatalf("canceled release error = %v", err)
	}
	if _, ok := retained.Value(output); !ok {
		t.Fatal("canceled release discarded retained output")
	}
	if err := retained.Release(context.Background()); err != nil {
		t.Fatal(err)
	}
	if err := retained.Release(context.Background()); err != nil {
		t.Fatal(err)
	}
	after, err := cuda.worker.MemoryStats(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if after.CurrentBytes != during.CurrentBytes {
		t.Fatalf("pooled bytes after release = %d, want %d", after.CurrentBytes, during.CurrentBytes)
	}
	if _, ok := retained.Value(output); ok {
		t.Fatal("released output remains accessible")
	}
	reused, err := cuda.ExecuteRetainedWithDeviceFeeds(
		context.Background(),
		[]*tensor.Tensor{output},
		feeds,
		nil,
	)
	if err != nil {
		t.Fatal(err)
	}
	reusedStats, err := cuda.worker.MemoryStats(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if reusedStats.Allocations != after.Allocations || reusedStats.CurrentBytes != after.CurrentBytes {
		t.Fatalf("retained pool did not reuse allocation: before=%+v after=%+v", after, reusedStats)
	}
	if err := reused.Release(context.Background()); err != nil {
		t.Fatal(err)
	}
}

func TestExecutorRetainedFlatSlicesShareProducerStorage(t *testing.T) {
	cudatest.Require(t)
	builder := tensor.NewBuilder()
	shape := tensor.MustShape(8)
	input := builder.Input("input", dtype.F32, shape)
	producer := builder.Scale(input, 2)
	first := builder.FlatSlice(producer, 1, 3)
	second := builder.FlatSlice(producer, 5, 2)
	cuda, err := New(0)
	if err != nil {
		t.Fatal(err)
	}
	defer cuda.Close()
	retained, err := cuda.ExecuteRetainedWithDeviceFeeds(
		context.Background(),
		[]*tensor.Tensor{first, second},
		map[*tensor.Tensor]reference.Value{
			input: {Shape: shape, Data: []float32{1, 2, 3, 4, 5, 6, 7, 8}},
		},
		nil,
	)
	if err != nil {
		t.Fatal(err)
	}
	defer retained.Release(context.Background())
	firstValue, firstOK := retained.Value(first)
	secondValue, secondOK := retained.Value(second)
	if !firstOK || !secondOK || secondValue.Pointer-firstValue.Pointer != 4*4 {
		t.Fatalf("retained slice pointers = %+v/%+v", firstValue, secondValue)
	}
	firstHost, err := retained.CopyToHost(context.Background(), first)
	if err != nil {
		t.Fatal(err)
	}
	secondHost, err := retained.CopyToHost(context.Background(), second)
	if err != nil {
		t.Fatal(err)
	}
	compare(t, firstHost.Data, []float32{4, 6, 8}, 0)
	compare(t, secondHost.Data, []float32{12, 14}, 0)
}

func TestExecutorStableTargetAppendsWithoutPrefixCopy(t *testing.T) {
	cudatest.Require(t)
	cuda, err := New(0)
	if err != nil {
		t.Fatal(err)
	}
	defer cuda.Close()
	buffer, err := cuda.AllocateDeviceBuffer(context.Background(), 8*4)
	if err != nil {
		t.Fatal(err)
	}
	defer buffer.Release(context.Background())

	initialBuilder := tensor.NewBuilder()
	initialShape := tensor.MustShape(3)
	initialInput := initialBuilder.Input("initial", dtype.F32, initialShape)
	initialOutput := initialBuilder.Scale(initialInput, 1)
	initialTarget, err := buffer.Value(initialShape)
	if err != nil {
		t.Fatal(err)
	}
	initialGraph, err := Compile(initialOutput)
	if err != nil {
		t.Fatal(err)
	}
	initial, err := cuda.ExecuteRetainedCompiledWithTargets(
		context.Background(),
		initialGraph,
		map[*tensor.Tensor]reference.Value{
			initialInput: {Shape: initialShape, Data: []float32{1, 2, 3}},
		},
		nil,
		map[*tensor.Tensor]DeviceValue{initialOutput: initialTarget},
	)
	if err != nil {
		t.Fatal(err)
	}
	if err := initial.Release(context.Background()); err != nil {
		t.Fatal(err)
	}

	appendBuilder := tensor.NewBuilder()
	past := appendBuilder.Input("past", dtype.F32, initialShape)
	newShape := tensor.MustShape(2)
	added := appendBuilder.Input("added", dtype.F32, newShape)
	joined := appendBuilder.Concat(past, added, 0)
	joinedTarget, err := buffer.Value(joined.Shape)
	if err != nil {
		t.Fatal(err)
	}
	appendGraph, err := Compile(joined)
	if err != nil {
		t.Fatal(err)
	}
	retained, err := cuda.ExecuteRetainedCompiledWithTargets(
		context.Background(),
		appendGraph,
		map[*tensor.Tensor]reference.Value{
			added: {Shape: newShape, Data: []float32{4, 5}},
		},
		map[*tensor.Tensor]driver.DevicePtr{past: initialTarget.Pointer},
		map[*tensor.Tensor]DeviceValue{joined: joinedTarget},
	)
	if err != nil {
		t.Fatal(err)
	}
	defer retained.Release(context.Background())
	got, err := retained.CopyToHost(context.Background(), joined)
	if err != nil {
		t.Fatal(err)
	}
	compare(t, got.Data, []float32{1, 2, 3, 4, 5}, 0)
}

func TestExecutorCopyDeviceValuesConcatenatesSegments(t *testing.T) {
	cudatest.Require(t)
	builder := tensor.NewBuilder()
	shape := tensor.MustShape(4)
	input := builder.Input("input", dtype.F32, shape)
	output := builder.Scale(input, 1)
	inputValue, _ := reference.NewValue(shape, []float32{1, 2, 3, 4})
	cuda, err := New(0)
	if err != nil {
		t.Fatal(err)
	}
	defer cuda.Close()
	retained, err := cuda.ExecuteRetainedWithDeviceFeeds(
		context.Background(),
		[]*tensor.Tensor{output},
		map[*tensor.Tensor]reference.Value{input: inputValue},
		nil,
	)
	if err != nil {
		t.Fatal(err)
	}
	defer retained.Release(context.Background())
	source, ok := retained.Value(output)
	if !ok {
		t.Fatal("retained output is unavailable")
	}
	copiedOwner, copied, err := cuda.CopyDeviceValues(
		context.Background(),
		[]DeviceCopy{{
			Shape: shape,
			Segments: []DeviceCopySegment{
				{Source: source.Pointer + 8, Bytes: 8},
				{Source: source.Pointer, Bytes: 8},
			},
		}},
	)
	if err != nil {
		t.Fatal(err)
	}
	defer copiedOwner.Release(context.Background())
	data := make([]float32, 4)
	err = cuda.worker.Do(context.Background(), func(state *device.State) error {
		return state.Driver.MemcpyDtoH(driver.Bytes(data), copied[0].Pointer)
	})
	if err != nil {
		t.Fatal(err)
	}
	compare(t, data, []float32{3, 4, 1, 2}, 0)
}

func TestExecutorMulMatMatchesReference(t *testing.T) {
	cudatest.Require(t)
	builder := tensor.NewBuilder()
	left := builder.Input("left", dtype.F32, tensor.MustShape(5, 3))
	right := builder.Input("right", dtype.F32, tensor.MustShape(5, 4))
	output := builder.MulMat(left, right)
	if err := builder.Err(); err != nil {
		t.Fatal(err)
	}
	leftValue, _ := reference.NewValue(left.Shape, []float32{
		1, 2, 3, 4, 5,
		2, 3, 4, 5, 6,
		3, 4, 5, 6, 7,
	})
	rightValue, _ := reference.NewValue(right.Shape, []float32{
		1, 0, 0, 0, 0,
		0, 1, 0, 0, 0,
		0, 0, 1, 0, 0,
		1, 1, 1, 1, 1,
	})
	feeds := map[*tensor.Tensor]reference.Value{left: leftValue, right: rightValue}
	want, err := reference.Execute([]*tensor.Tensor{output}, feeds)
	if err != nil {
		t.Fatal(err)
	}
	cuda, err := New(0)
	if err != nil {
		t.Fatal(err)
	}
	defer cuda.Close()
	got, err := cuda.Execute(context.Background(), []*tensor.Tensor{output}, feeds)
	if err != nil {
		t.Fatal(err)
	}
	compare(t, got[output].Data, want[output].Data, 1e-5)
}

func TestExecutorQ8DecodeMulMatMatchesDequantizedReference(t *testing.T) {
	cudatest.Require(t)
	leftShape := tensor.MustShape(64, 19)
	rightShape := tensor.MustShape(64, 1)
	leftValue := patternedValue(leftShape, 11, 0.03, -0.1)
	rightValue := patternedValue(rightShape, 7, 0.05, 0.02)
	storage, err := quant.Quantize(dtype.Q8_0, leftValue.Data)
	if err != nil {
		t.Fatal(err)
	}
	dequantized, err := quant.Dequantize(dtype.Q8_0, storage, uint64(len(leftValue.Data)))
	if err != nil {
		t.Fatal(err)
	}
	referenceBuilder := tensor.NewBuilder()
	referenceLeft := referenceBuilder.Input("left", dtype.F32, leftShape)
	referenceRight := referenceBuilder.Input("right", dtype.F32, rightShape)
	referenceOutput := referenceBuilder.MulMat(referenceLeft, referenceRight)
	want, err := reference.Execute([]*tensor.Tensor{referenceOutput}, map[*tensor.Tensor]reference.Value{
		referenceLeft: {Shape: leftShape, Data: dequantized}, referenceRight: rightValue,
	})
	if err != nil {
		t.Fatal(err)
	}
	builder := tensor.NewBuilder()
	left := builder.Input("left", dtype.Q8_0, leftShape)
	right := builder.Input("right", dtype.F32, rightShape)
	output := builder.MulMat(left, right)
	worker, err := device.New(0)
	if err != nil {
		t.Fatal(err)
	}
	defer worker.Close()
	var pointer driver.DevicePtr
	if err = worker.Do(context.Background(), func(state *device.State) error {
		var allocateErr error
		pointer, allocateErr = state.Driver.MemAlloc(uint64(len(storage)))
		if allocateErr != nil {
			return allocateErr
		}
		return state.Driver.MemcpyHtoD(pointer, storage)
	}); err != nil {
		t.Fatal(err)
	}
	defer worker.Do(context.Background(), func(state *device.State) error { return state.Driver.MemFree(pointer) })
	cuda, err := NewWithWorker(worker)
	if err != nil {
		t.Fatal(err)
	}
	defer cuda.Close()
	got, err := cuda.ExecuteWithDeviceFeeds(context.Background(), []*tensor.Tensor{output},
		map[*tensor.Tensor]reference.Value{right: rightValue}, map[*tensor.Tensor]driver.DevicePtr{left: pointer})
	if err != nil {
		t.Fatal(err)
	}
	compare(t, got[output].Data, want[referenceOutput].Data, 1e-2)
	referenceSelection := referenceBuilder.TopK(referenceOutput, 1)
	wantSelection, err := reference.Execute(
		[]*tensor.Tensor{referenceSelection},
		map[*tensor.Tensor]reference.Value{
			referenceLeft: {Shape: leftShape, Data: dequantized}, referenceRight: rightValue,
		},
	)
	if err != nil {
		t.Fatal(err)
	}
	selection := builder.TopK(output, 1)
	gotSelection, err := cuda.ExecuteWithDeviceFeeds(
		context.Background(), []*tensor.Tensor{selection},
		map[*tensor.Tensor]reference.Value{right: rightValue},
		map[*tensor.Tensor]driver.DevicePtr{left: pointer},
	)
	if err != nil {
		t.Fatal(err)
	}
	compare(t, gotSelection[selection].Data, wantSelection[referenceSelection].Data, 0)
}

func TestExecutorActivatedGatesMatchReference(t *testing.T) {
	cudatest.Require(t)
	for _, build := range []struct {
		name string
		fn   func(*tensor.Builder, *tensor.Tensor, *tensor.Tensor) *tensor.Tensor
	}{
		{"silu", func(builder *tensor.Builder, gate, up *tensor.Tensor) *tensor.Tensor {
			return builder.SwiGLU(gate, up)
		}},
		{"sigmoid", func(builder *tensor.Builder, gate, up *tensor.Tensor) *tensor.Tensor {
			return builder.Multiply(builder.Sigmoid(gate), up)
		}},
	} {
		t.Run(build.name, func(t *testing.T) {
			builder := tensor.NewBuilder()
			gate := builder.Input("gate", dtype.F32, tensor.MustShape(8, 2))
			up := builder.Input("up", dtype.F32, gate.Shape)
			activated := build.fn(builder, gate, up)
			projection := builder.Input("projection", dtype.F32, tensor.MustShape(8, 7))
			output := builder.MulMat(projection, activated)
			feeds := map[*tensor.Tensor]reference.Value{
				gate: patternedValue(gate.Shape, 7, 0.05, -0.1),
				up:   patternedValue(up.Shape, 11, 0.03, 0.02),
				projection: patternedValue(
					projection.Shape, 13, 0.02, -0.03,
				),
			}
			outputs := []*tensor.Tensor{activated, output}
			want, err := reference.Execute(outputs, feeds)
			if err != nil {
				t.Fatal(err)
			}
			cuda, err := New(0)
			if err != nil {
				t.Fatal(err)
			}
			defer cuda.Close()
			got, err := cuda.Execute(context.Background(), outputs, feeds)
			if err != nil {
				t.Fatal(err)
			}
			for _, item := range outputs {
				compare(t, got[item].Data, want[item].Data, 1e-5)
			}
		})
	}
}

func TestExecutorSSMConvMatchesReference(t *testing.T) {
	cudatest.Require(t)
	builder := tensor.NewBuilder()
	input := builder.Input("input", dtype.F32, tensor.MustShape(6, 3, 2))
	weights := builder.Input("weights", dtype.F32, tensor.MustShape(4, 3))
	output := builder.SSMConv(input, weights)
	if err := builder.Err(); err != nil {
		t.Fatal(err)
	}
	inputValue := patternedValue(input.Shape, 7, 0.2, -0.3)
	weightValue := patternedValue(weights.Shape, 11, 0.1, 0.05)
	feeds := map[*tensor.Tensor]reference.Value{
		input:   inputValue,
		weights: weightValue,
	}
	want, err := reference.Execute([]*tensor.Tensor{output}, feeds)
	if err != nil {
		t.Fatal(err)
	}
	cuda, err := New(0)
	if err != nil {
		t.Fatal(err)
	}
	defer cuda.Close()
	got, err := cuda.Execute(context.Background(), []*tensor.Tensor{output}, feeds)
	if err != nil {
		t.Fatal(err)
	}
	compare(t, got[output].Data, want[output].Data, 1e-5)
}
