//go:build windows

package executor

import (
	"context"
	"testing"

	"overgo/internal/model"
	"overgo/internal/tensor"
	"overgo/internal/tensor/dtype"
	"overgo/internal/tensor/reference"
)

// TestExecutorConditionedDiffusionBlockMatchesReference: the adaptive-
// layernorm diffusion block (axis-partitioned rotary self-attention,
// precomputed cross-attention context, tanh-GELU feed-forward, modulated
// head) as ONE graph on both backends. Small synthetic geometry: dim 12,
// 2 heads of width 6 (axis channels 2/2/2), a 2x1x2 token grid, 3 context
// tokens.
func TestExecutorConditionedDiffusionBlockMatchesReference(t *testing.T) {
	builder := tensor.NewBuilder()
	const (
		dim       = uint64(12)
		heads     = uint64(2)
		ffn       = uint64(20)
		tokens    = uint64(4)
		ctxTokens = uint64(3)
		headWide  = dim / heads
	)
	input := builder.Input("input", dtype.F32, tensor.MustShape(dim, tokens))
	conditioning := builder.Input("conditioning", dtype.F32, tensor.MustShape(6*dim))
	headConditioning := builder.Input("head_conditioning", dtype.F32, tensor.MustShape(dim))
	contextInput := builder.Input("context", dtype.F32, tensor.MustShape(dim, ctxTokens))
	weightInput := func(name string, dimensions ...uint64) *tensor.Tensor {
		return builder.Input(name, dtype.F32, tensor.MustShape(dimensions...))
	}
	selfWeights := model.ConditionedDiffusionAttentionWeights{
		Query: weightInput("self_q", dim, dim), QueryBias: weightInput("self_qb", dim),
		Key: weightInput("self_k", dim, dim), KeyBias: weightInput("self_kb", dim),
		Value: weightInput("self_v", dim, dim), ValueBias: weightInput("self_vb", dim),
		Output: weightInput("self_o", dim, dim), OutputBias: weightInput("self_ob", dim),
		QueryNorm: weightInput("self_nq", dim), KeyNorm: weightInput("self_nk", dim),
	}
	crossWeights := model.ConditionedDiffusionAttentionWeights{
		Query: weightInput("cross_q", dim, dim), QueryBias: weightInput("cross_qb", dim),
		Key: weightInput("cross_k", dim, dim), KeyBias: weightInput("cross_kb", dim),
		Value: weightInput("cross_v", dim, dim), ValueBias: weightInput("cross_vb", dim),
		Output: weightInput("cross_o", dim, dim), OutputBias: weightInput("cross_ob", dim),
		QueryNorm: weightInput("cross_nq", dim), KeyNorm: weightInput("cross_nk", dim),
	}
	options := model.ConditionedDiffusionBlockOptions{
		Dim: dim, Heads: heads, FFNDim: ffn,
		Epsilon:      1e-6,
		RotaryBase:   10000,
		AxisChannels: model.ThreeAxisRotaryChannels(headWide),
		AxisPositions: [3][]uint32{
			{0, 0, 1, 1}, // frames
			{0, 0, 0, 0}, // height
			{0, 1, 0, 1}, // width
		},
	}
	program, err := model.CompileConditionedDiffusionProgram(options)
	if err != nil {
		t.Fatal(err)
	}
	crossKey, crossValue, err := program.BuildCrossContext(builder, contextInput, crossWeights)
	if err != nil {
		t.Fatal(err)
	}
	blockWeights := model.ConditionedDiffusionBlockWeights{
		Modulation:      weightInput("modulation", 6*dim),
		SelfAttention:   selfWeights,
		CrossAttention:  crossWeights,
		CrossNormWeight: weightInput("norm3_w", dim),
		CrossNormBias:   weightInput("norm3_b", dim),
		FFNExpand:       weightInput("ffn0_w", dim, ffn),
		FFNExpandBias:   weightInput("ffn0_b", ffn),
		FFNContract:     weightInput("ffn2_w", ffn, dim),
		FFNContractBias: weightInput("ffn2_b", dim),
	}
	result, err := program.BuildBlock(
		builder, input, conditioning, crossKey, crossValue, blockWeights,
	)
	if err != nil {
		t.Fatal(err)
	}
	head, err := program.BuildHead(
		builder, result.Output, headConditioning,
		weightInput("head_mod", 2*dim), weightInput("head_w", dim, 8), weightInput("head_b", 8),
	)
	if err != nil {
		t.Fatal(err)
	}
	if err := builder.Err(); err != nil {
		t.Fatal(err)
	}
	feeds := map[*tensor.Tensor]reference.Value{
		input:            patternedValue(input.Shape, 1, 0.21, 0),
		conditioning:     patternedValue(conditioning.Shape, 2, 0.07, 0),
		headConditioning: patternedValue(headConditioning.Shape, 3, 0.05, 0),
		contextInput:     patternedValue(contextInput.Shape, 4, 0.17, 0),
	}
	seed := 5
	for _, bundle := range []model.ConditionedDiffusionAttentionWeights{selfWeights, crossWeights} {
		for _, node := range []*tensor.Tensor{
			bundle.Query, bundle.Key, bundle.Value, bundle.Output,
		} {
			feeds[node] = patternedValue(node.Shape, seed, 0.05, 0)
			seed++
		}
		for _, node := range []*tensor.Tensor{
			bundle.QueryBias, bundle.KeyBias, bundle.ValueBias, bundle.OutputBias,
		} {
			feeds[node] = patternedValue(node.Shape, seed, 0.02, 0)
			seed++
		}
		for _, node := range []*tensor.Tensor{bundle.QueryNorm, bundle.KeyNorm} {
			feeds[node] = patternedValue(node.Shape, seed, 0.04, 1)
			seed++
		}
	}
	for _, node := range []*tensor.Tensor{
		blockWeights.Modulation, blockWeights.FFNExpand, blockWeights.FFNContract,
	} {
		feeds[node] = patternedValue(node.Shape, seed, 0.05, 0)
		seed++
	}
	for _, node := range []*tensor.Tensor{
		blockWeights.FFNExpandBias, blockWeights.FFNContractBias, blockWeights.CrossNormBias,
	} {
		feeds[node] = patternedValue(node.Shape, seed, 0.02, 0)
		seed++
	}
	feeds[blockWeights.CrossNormWeight] = patternedValue(blockWeights.CrossNormWeight.Shape, seed, 0.03, 1)
	seed++
	headWeights := []*tensor.Tensor{}
	for _, node := range builder.Nodes() {
		if node.Op == tensor.OpInput {
			if _, fed := feeds[node]; !fed {
				headWeights = append(headWeights, node)
			}
		}
	}
	for _, node := range headWeights {
		feeds[node] = patternedValue(node.Shape, seed, 0.05, 0)
		seed++
	}
	outputs := []*tensor.Tensor{
		result.SelfQueryProjected, result.SelfQueryNormed, result.SelfQueryRotated,
		result.SelfKeyNormed, result.SelfKeyRotated, result.SelfValueProjected,
		result.SelfAttention, result.SelfProjected, result.SelfResidual,
		result.CrossProjected, result.CrossResidual, result.FeedForward,
		result.Output, head,
	}
	names := []string{
		"self_q_projected", "self_q_norm", "self_q_rope",
		"self_k_norm", "self_k_rope", "self_v_projected",
		"self_sdpa", "self_attn", "self_residual",
		"cross_attn", "cross_residual", "ffn",
		"output", "head",
	}
	want, err := reference.Execute(outputs, feeds)
	if err != nil {
		t.Fatal(err)
	}
	cuda := newFixtureExecutor(t)
	got, err := cuda.Execute(context.WithoutCancel(t.Context()), outputs, feeds)
	if err != nil {
		t.Fatal(err)
	}
	failed := false
	for index, output := range outputs {
		maxDiff, at := maxAbsDifference(t, got[output].Data, want[output].Data)
		t.Logf("%s max_abs_diff=%.6g at=%d", names[index], maxDiff, at)
		failed = failed || maxDiff > accuracyModel
	}
	if failed {
		t.Fatalf("conditioned diffusion block CUDA/reference divergence exceeds %g", accuracyModel)
	}
}
