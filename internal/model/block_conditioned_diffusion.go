// Conditioned diffusion transformer stages: adaptive-layernorm blocks with
// full-width QK RMS norms, axis-partitioned rotary self-attention over a
// (frames, height, width) token grid, fixed-context cross-attention, and a
// tanh-GELU feed-forward. Generic and config-driven — nothing here names a
// model family. Composes only cataloged tensor ops, so one graph definition
// executes on both the reference and CUDA backends.
package model

import (
	"errors"
	"fmt"
	"math"

	"overgo/internal/tensor"
)

// ConditionedDiffusionAttentionWeights: one biased attention projection
// bundle; QueryNorm/KeyNorm are full model-width RMS norms applied before
// the head split.
type ConditionedDiffusionAttentionWeights struct {
	Query, QueryBias   *tensor.Tensor
	Key, KeyBias       *tensor.Tensor
	Value, ValueBias   *tensor.Tensor
	Output, OutputBias *tensor.Tensor
	QueryNorm, KeyNorm *tensor.Tensor
}

// requireQueryOutput: the projections a block consumes directly; K/V (and
// the key norm) belong to the cross-context stage when attention keys come
// precomputed.
func (w ConditionedDiffusionAttentionWeights) requireQueryOutput(scope string, required *graphWeights) {
	required.add(scope+" query", w.Query)
	required.add(scope+" query bias", w.QueryBias)
	required.add(scope+" output", w.Output)
	required.add(scope+" output bias", w.OutputBias)
	required.add(scope+" query norm", w.QueryNorm)
}

func (w ConditionedDiffusionAttentionWeights) require(scope string, required *graphWeights) {
	w.requireQueryOutput(scope, required)
	required.add(scope+" key", w.Key)
	required.add(scope+" key bias", w.KeyBias)
	required.add(scope+" value", w.Value)
	required.add(scope+" value bias", w.ValueBias)
	required.add(scope+" key norm", w.KeyNorm)
}

// ConditionedDiffusionBlockWeights: one block. CrossNormWeight/Bias may both
// be nil, which selects the identity pre-cross-attention norm variant.
type ConditionedDiffusionBlockWeights struct {
	Modulation                   *tensor.Tensor // [6*dim] learned base modulation
	SelfAttention                ConditionedDiffusionAttentionWeights
	CrossAttention               ConditionedDiffusionAttentionWeights
	CrossNormWeight              *tensor.Tensor
	CrossNormBias                *tensor.Tensor
	FFNExpand, FFNExpandBias     *tensor.Tensor
	FFNContract, FFNContractBias *tensor.Tensor
}

// ConditionedDiffusionBlockOptions: block geometry and rotary layout.
// AxisPositions carry one grid coordinate per token per axis; AxisChannels
// partition the head width into contiguous per-axis rotary spans.
type ConditionedDiffusionBlockOptions struct {
	Dim, Heads, FFNDim uint64
	Epsilon            float32
	RotaryBase         float32
	AxisChannels       [3]uint64
	AxisPositions      [3][]uint32
}

// ThreeAxisRotaryChannels: canonical temporal/height/width split of an even
// head width: both spatial axes take 2*(width/6) channels and the temporal
// axis the remainder.
func ThreeAxisRotaryChannels(headWidth uint64) [3]uint64 {
	spatial := 2 * (headWidth / 6)
	return [3]uint64{headWidth - 2*spatial, spatial, spatial}
}

// ConditionedDiffusionBlockResult: block output plus every intra-block seam
// in golden-trace order for parity probes.
type ConditionedDiffusionBlockResult struct {
	Output *tensor.Tensor

	SelfQueryProjected, SelfKeyProjected, SelfValueProjected *tensor.Tensor
	SelfQueryNormed, SelfKeyNormed                           *tensor.Tensor
	SelfQueryRotated, SelfKeyRotated                         *tensor.Tensor
	SelfAttention                                            *tensor.Tensor // pre-projection SDPA output
	SelfProjected                                            *tensor.Tensor // post output projection
	SelfResidual                                             *tensor.Tensor
	CrossProjected                                           *tensor.Tensor
	CrossResidual                                            *tensor.Tensor
	FeedForward                                              *tensor.Tensor
}

// BuildAxisPartitionedRoPE: adjacent-pair rotary over contiguous per-axis
// channel spans with axis-local frequency exponents: pair j of an axis span
// of width w rotates by position*base^(-2j/w). Slice, rotate, reassemble —
// every stage is a cataloged op.
func BuildAxisPartitionedRoPE(
	builder *tensor.Builder,
	input *tensor.Tensor,
	channels [3]uint64,
	positions [3][]uint32,
	frequencyBase float32,
) *tensor.Tensor {
	if builder == nil {
		return nil
	}
	if input == nil || input.Shape.Rank != 3 {
		return nil
	}
	headWidth := input.Shape.Dims[0]
	if channels[0]+channels[1]+channels[2] != headWidth {
		return nil
	}
	heads, tokens := input.Shape.Dims[1], input.Shape.Dims[2]
	var joined *tensor.Tensor
	offset := uint64(0)
	for axis := range channels {
		span := channels[axis]
		part := builder.Reshape(
			builder.GroupSlice(input, offset, span, 1, span),
			span, heads, tokens,
		)
		rotated := builder.RoPENormal(part, positions[axis], uint32(span), frequencyBase)
		if joined == nil {
			joined = rotated
		} else {
			joined = builder.Concat(joined, rotated, 0)
		}
		offset += span
	}
	return joined
}

// buildAdaptiveShiftScale: x*(1+scale) + shift with [dim] rows broadcast
// over tokens.
func buildAdaptiveShiftScale(builder *tensor.Builder, x, shift, scale *tensor.Tensor) *tensor.Tensor {
	return builder.Add(builder.Add(x, builder.Multiply(x, scale)), shift)
}

func buildBiasedProjection(builder *tensor.Builder, weight, bias, input *tensor.Tensor) *tensor.Tensor {
	return builder.Add(builder.MulMat(weight, input), bias)
}

// BuildConditionedDiffusionCrossContext: fixed-context cross-attention K/V,
// projected once per context: K = RMSNorm_w(W_k ctx + b_k), V = W_v ctx +
// b_v, both reshaped to [headWidth, heads, contextTokens].
func BuildConditionedDiffusionCrossContext(
	builder *tensor.Builder,
	context *tensor.Tensor,
	weights ConditionedDiffusionAttentionWeights,
	heads uint64,
	epsilon float32,
) (key, value *tensor.Tensor, err error) {
	if builder == nil || context == nil || context.Shape.Rank != 2 {
		return nil, nil, errors.New("conditioned diffusion cross context input is nil or misshaped")
	}
	required := graphWeights{
		requireGraphWeight("cross key", weights.Key),
		requireGraphWeight("cross key bias", weights.KeyBias),
		requireGraphWeight("cross value", weights.Value),
		requireGraphWeight("cross value bias", weights.ValueBias),
		requireGraphWeight("cross key norm", weights.KeyNorm),
	}
	if err := required.validate("conditioned diffusion cross context"); err != nil {
		return nil, nil, err
	}
	dim := context.Shape.Dims[0]
	tokens := context.Shape.Dims[1]
	if heads == 0 || dim%heads != 0 {
		return nil, nil, fmt.Errorf("conditioned diffusion cross context: dim %d incompatible with heads %d", dim, heads)
	}
	headWidth := dim / heads
	k := builder.WeightedRMSNorm(buildBiasedProjection(builder, weights.Key, weights.KeyBias, context), weights.KeyNorm, epsilon)
	v := buildBiasedProjection(builder, weights.Value, weights.ValueBias, context)
	key = builder.Reshape(k, headWidth, heads, tokens)
	value = builder.Reshape(v, headWidth, heads, tokens)
	if err := builder.Err(); err != nil {
		return nil, nil, err
	}
	return key, value, nil
}

// BuildConditionedDiffusionBlock: one adaptive-layernorm diffusion
// transformer block. conditioning is the per-execution [6*dim] timestep
// embedding added to the block's learned modulation; chunk order is
// (self shift, self scale, self gate, ffn shift, ffn scale, ffn gate).
// crossKey/crossValue come from BuildConditionedDiffusionCrossContext.
func BuildConditionedDiffusionBlock(
	builder *tensor.Builder,
	input, conditioning *tensor.Tensor,
	crossKey, crossValue *tensor.Tensor,
	options ConditionedDiffusionBlockOptions,
	weights ConditionedDiffusionBlockWeights,
) (ConditionedDiffusionBlockResult, error) {
	var result ConditionedDiffusionBlockResult
	if builder == nil || input == nil || conditioning == nil || crossKey == nil || crossValue == nil {
		return result, errors.New("conditioned diffusion block input is nil")
	}
	required := graphWeights{
		requireGraphWeight("modulation", weights.Modulation),
		requireGraphWeight("feed-forward expand", weights.FFNExpand),
		requireGraphWeight("feed-forward expand bias", weights.FFNExpandBias),
		requireGraphWeight("feed-forward contract", weights.FFNContract),
		requireGraphWeight("feed-forward contract bias", weights.FFNContractBias),
	}
	weights.SelfAttention.require("self-attention", &required)
	weights.CrossAttention.requireQueryOutput("cross-attention", &required)
	if err := required.validate("conditioned diffusion block"); err != nil {
		return result, err
	}
	if (weights.CrossNormWeight == nil) != (weights.CrossNormBias == nil) {
		return result, errors.New("conditioned diffusion block cross norm affine pair is incomplete")
	}
	dim, heads := options.Dim, options.Heads
	if dim == 0 || heads == 0 || dim%heads != 0 || options.FFNDim == 0 || options.Epsilon <= 0 {
		return result, fmt.Errorf("conditioned diffusion block geometry dim=%d heads=%d ffn=%d eps=%g is invalid", dim, heads, options.FFNDim, options.Epsilon)
	}
	if input.Shape.Rank != 2 || input.Shape.Dims[0] != dim {
		return result, errors.New("conditioned diffusion block input shape is incompatible")
	}
	headWidth := dim / heads
	if options.AxisChannels[0]+options.AxisChannels[1]+options.AxisChannels[2] != headWidth {
		return result, fmt.Errorf("conditioned diffusion block axis channels %v do not cover head width %d", options.AxisChannels, headWidth)
	}
	tokens := input.Shape.Dims[1]
	for axis := range options.AxisPositions {
		if uint64(len(options.AxisPositions[axis])) != tokens {
			return result, fmt.Errorf("conditioned diffusion block axis %d has %d positions, need %d", axis, len(options.AxisPositions[axis]), tokens)
		}
	}
	if conditioning.Shape.Rank != 1 || conditioning.Shape.Dims[0] != 6*dim {
		return result, errors.New("conditioned diffusion block conditioning must be a [6*dim] vector")
	}

	modulation := builder.Add(weights.Modulation, conditioning)
	chunk := func(index uint64) *tensor.Tensor {
		return builder.FlatSlice(modulation, index*dim, dim)
	}
	attentionScale := float32(1 / math.Sqrt(float64(headWidth)))

	selfIn := buildAdaptiveShiftScale(builder, builder.LayerNorm(input, options.Epsilon), chunk(0), chunk(1))
	result.SelfQueryProjected = buildBiasedProjection(builder, weights.SelfAttention.Query, weights.SelfAttention.QueryBias, selfIn)
	result.SelfKeyProjected = buildBiasedProjection(builder, weights.SelfAttention.Key, weights.SelfAttention.KeyBias, selfIn)
	result.SelfValueProjected = buildBiasedProjection(builder, weights.SelfAttention.Value, weights.SelfAttention.ValueBias, selfIn)
	result.SelfQueryNormed = builder.WeightedRMSNorm(result.SelfQueryProjected, weights.SelfAttention.QueryNorm, options.Epsilon)
	result.SelfKeyNormed = builder.WeightedRMSNorm(result.SelfKeyProjected, weights.SelfAttention.KeyNorm, options.Epsilon)
	query := builder.Reshape(result.SelfQueryNormed, headWidth, heads, tokens)
	key := builder.Reshape(result.SelfKeyNormed, headWidth, heads, tokens)
	value := builder.Reshape(result.SelfValueProjected, headWidth, heads, tokens)
	result.SelfQueryRotated = BuildAxisPartitionedRoPE(builder, query, options.AxisChannels, options.AxisPositions, options.RotaryBase)
	result.SelfKeyRotated = BuildAxisPartitionedRoPE(builder, key, options.AxisChannels, options.AxisPositions, options.RotaryBase)
	attention := builder.Attention(result.SelfQueryRotated, result.SelfKeyRotated, value, attentionScale, false)
	result.SelfAttention = builder.Reshape(attention, dim, tokens)
	result.SelfProjected = buildBiasedProjection(builder, weights.SelfAttention.Output, weights.SelfAttention.OutputBias, result.SelfAttention)
	result.SelfResidual = builder.Add(input, builder.Multiply(result.SelfProjected, chunk(2)))

	crossIn := result.SelfResidual
	if weights.CrossNormWeight != nil {
		crossIn = builder.Add(
			builder.Multiply(builder.LayerNorm(result.SelfResidual, options.Epsilon), weights.CrossNormWeight),
			weights.CrossNormBias,
		)
	}
	crossQuery := builder.WeightedRMSNorm(
		buildBiasedProjection(builder, weights.CrossAttention.Query, weights.CrossAttention.QueryBias, crossIn),
		weights.CrossAttention.QueryNorm, options.Epsilon,
	)
	crossAttention := builder.Attention(
		builder.Reshape(crossQuery, headWidth, heads, tokens),
		crossKey, crossValue, attentionScale, false,
	)
	result.CrossProjected = buildBiasedProjection(
		builder, weights.CrossAttention.Output, weights.CrossAttention.OutputBias,
		builder.Reshape(crossAttention, dim, tokens),
	)
	result.CrossResidual = builder.Add(result.SelfResidual, result.CrossProjected)

	ffnIn := buildAdaptiveShiftScale(builder, builder.LayerNorm(result.CrossResidual, options.Epsilon), chunk(3), chunk(4))
	hidden := builder.GELUTanhExact(buildBiasedProjection(builder, weights.FFNExpand, weights.FFNExpandBias, ffnIn))
	result.FeedForward = buildBiasedProjection(builder, weights.FFNContract, weights.FFNContractBias, hidden)
	result.Output = builder.Add(result.CrossResidual, builder.Multiply(result.FeedForward, chunk(5)))
	if err := builder.Err(); err != nil {
		return ConditionedDiffusionBlockResult{}, err
	}
	return result, nil
}

// BuildConditionedDiffusionHead: final modulated projection. conditioning is
// the [dim] timestep embedding; modulation is the learned [2*dim]
// (shift, scale) pair; output projects each token to its patch values.
func BuildConditionedDiffusionHead(
	builder *tensor.Builder,
	input, conditioning *tensor.Tensor,
	modulation, weight, bias *tensor.Tensor,
	epsilon float32,
) (*tensor.Tensor, error) {
	if builder == nil || input == nil || conditioning == nil {
		return nil, errors.New("conditioned diffusion head input is nil")
	}
	required := graphWeights{
		requireGraphWeight("head modulation", modulation),
		requireGraphWeight("head weight", weight),
		requireGraphWeight("head bias", bias),
	}
	if err := required.validate("conditioned diffusion head"); err != nil {
		return nil, err
	}
	if input.Shape.Rank != 2 || epsilon <= 0 {
		return nil, errors.New("conditioned diffusion head input shape or epsilon is invalid")
	}
	dim := input.Shape.Dims[0]
	if conditioning.Shape.Rank != 1 || conditioning.Shape.Dims[0] != dim ||
		modulation.Shape.Rank != 1 || modulation.Shape.Dims[0] != 2*dim {
		return nil, errors.New("conditioned diffusion head conditioning shape is incompatible")
	}
	shift := builder.Add(builder.FlatSlice(modulation, 0, dim), conditioning)
	scale := builder.Add(builder.FlatSlice(modulation, dim, dim), conditioning)
	modulated := buildAdaptiveShiftScale(builder, builder.LayerNorm(input, epsilon), shift, scale)
	output := buildBiasedProjection(builder, weight, bias, modulated)
	if err := builder.Err(); err != nil {
		return nil, err
	}
	return output, nil
}
