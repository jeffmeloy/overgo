// Conditioned diffusion: adaptive norm, axis rotary, cross attention, FFN.
package model

import (
	"errors"
	"fmt"

	"overgo/internal/hostmath"
	"overgo/internal/tensor"
)

const (
	conditionedSelfShift uint64 = iota
	conditionedSelfScale
	conditionedSelfGate
	conditionedFFNShift
	conditionedFFNScale
	conditionedFFNGate
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

// ConditionedDiffusionBlockOptions defines geometry and rotary partitions.
type ConditionedDiffusionBlockOptions struct {
	Dim, Heads, FFNDim uint64
	Epsilon            float32
	RotaryBase         float32
	AxisChannels       [3]uint64
	AxisPositions      [3][]uint32
	// RoundAttentionStorage inserts BF16 storage rounds.
	RoundAttentionStorage bool
}

// ConditionedDiffusionProgram owns validated geometry.
type ConditionedDiffusionProgram struct {
	options ConditionedDiffusionBlockOptions
}

// CompileConditionedDiffusionProgram validates immutable graph geometry.
func CompileConditionedDiffusionProgram(options ConditionedDiffusionBlockOptions) (ConditionedDiffusionProgram, error) {
	headWidth, validHeads := tensor.EqualPartition(options.Dim, options.Heads)
	if !validHeads || options.FFNDim == tensor.FirstOffset || !positiveFinite(options.Epsilon) {
		return ConditionedDiffusionProgram{}, fmt.Errorf(
			"conditioned diffusion geometry dim=%d heads=%d ffn=%d eps=%g is invalid",
			options.Dim, options.Heads, options.FFNDim, options.Epsilon,
		)
	}
	if !tensor.PartitionsCover(headWidth, options.AxisChannels[:]...) {
		return ConditionedDiffusionProgram{}, fmt.Errorf(
			"conditioned diffusion axis channels %v do not cover head width %d",
			options.AxisChannels, headWidth,
		)
	}
	return ConditionedDiffusionProgram{options: options}, nil
}

func (p ConditionedDiffusionProgram) BuildCrossContext(
	builder *tensor.Builder,
	context *tensor.Tensor,
	weights ConditionedDiffusionAttentionWeights,
) (*tensor.Tensor, *tensor.Tensor, error) {
	return buildConditionedDiffusionCrossContext(
		builder, context, weights, p.options.Heads, p.options.Epsilon,
	)
}

func (p ConditionedDiffusionProgram) BuildBlock(
	builder *tensor.Builder,
	input, conditioning, crossKey, crossValue *tensor.Tensor,
	weights ConditionedDiffusionBlockWeights,
) (ConditionedDiffusionBlockResult, error) {
	return buildConditionedDiffusionBlock(
		builder, input, conditioning, crossKey, crossValue, nil, nil, p.options, weights,
	)
}

// BuildBlockWithSelfHistory attends over retained rotated keys and values
// before the current bidirectional chunk.
func (p ConditionedDiffusionProgram) BuildBlockWithSelfHistory(
	builder *tensor.Builder,
	input, conditioning, crossKey, crossValue, historyKey, historyValue *tensor.Tensor,
	weights ConditionedDiffusionBlockWeights,
) (ConditionedDiffusionBlockResult, error) {
	return buildConditionedDiffusionBlock(
		builder, input, conditioning, crossKey, crossValue, historyKey, historyValue, p.options, weights,
	)
}

func (p ConditionedDiffusionProgram) BuildHead(
	builder *tensor.Builder,
	input, conditioning, modulation, weight, bias *tensor.Tensor,
) (*tensor.Tensor, error) {
	return buildConditionedDiffusionHead(
		builder, input, conditioning, modulation, weight, bias, p.options.Epsilon,
	)
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
	SelfValue                                                *tensor.Tensor // [headWidth, heads, current tokens]
	SelfKeyCache, SelfValueCache                             *tensor.Tensor // retained history plus current tokens
	SelfAttention                                            *tensor.Tensor // pre-projection SDPA output
	SelfProjected                                            *tensor.Tensor // post output projection
	SelfResidual                                             *tensor.Tensor
	CrossProjected                                           *tensor.Tensor
	CrossResidual                                            *tensor.Tensor
	FeedForward                                              *tensor.Tensor
}

// buildAxisPartitionedRoPE: adjacent-pair rotary over contiguous per-axis
// channel spans with axis-local frequency exponents: pair j of an axis span
// of width w rotates by position*base^(-2j/w). Slice, rotate, reassemble —
// every stage is a cataloged op.
func buildAxisPartitionedRoPE(
	builder *tensor.Builder,
	input *tensor.Tensor,
	channels [3]uint64,
	positions [3][]uint32,
	frequencyBase float32,
) *tensor.Tensor {
	if builder == nil {
		return nil
	}
	if input == nil {
		return nil
	}
	headWidth, heads, tokens, validInput := tensor.Extents3(input.Shape)
	if !validInput || !tensor.PartitionsCover(headWidth, channels[:]...) {
		return nil
	}
	var joined *tensor.Tensor
	offset := uint64(tensor.FirstOffset)
	for axis := range channels {
		span := channels[axis]
		part := builder.Reshape(
			builder.GroupSlice(input, offset, span, tensor.SingletonExtent, span),
			span, heads, tokens,
		)
		rotated := builder.RoPEWithOptions(part, tensor.RoPEOptions{Layout: tensor.RoPELayoutNormal, Positions: positions[axis], RotaryDimensions: uint32(span), FrequencyBase: frequencyBase, FrequencyScale: tensor.UnitFrequencyScale})
		if joined == nil {
			joined = rotated
		} else {
			joined = builder.Concat(joined, rotated, tensor.FirstOffset)
		}
		offset += span
	}
	return joined
}

func buildBiasedProjection(builder *tensor.Builder, weight, bias, input *tensor.Tensor) *tensor.Tensor {
	return builder.Add(builder.MulMat(weight, input), bias)
}

// buildConditionedDiffusionCrossContext: fixed-context cross-attention K/V,
// projected once per context: K = RMSNorm_w(W_k ctx + b_k), V = W_v ctx +
// b_v, both reshaped to [headWidth, heads, contextTokens].
func buildConditionedDiffusionCrossContext(
	builder *tensor.Builder,
	context *tensor.Tensor,
	weights ConditionedDiffusionAttentionWeights,
	heads uint64,
	epsilon float32,
) (key, value *tensor.Tensor, err error) {
	if builder == nil || context == nil {
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
	dim, tokens, validContext := tensor.MatrixExtents(context.Shape)
	headWidth, validHeads := tensor.EqualPartition(dim, heads)
	if !validContext || !validHeads {
		return nil, nil, fmt.Errorf("conditioned diffusion cross context: dim %d incompatible with heads %d", dim, heads)
	}
	k := builder.WeightedRMSNorm(buildBiasedProjection(builder, weights.Key, weights.KeyBias, context), weights.KeyNorm, epsilon)
	v := buildBiasedProjection(builder, weights.Value, weights.ValueBias, context)
	key = builder.Reshape(k, headWidth, heads, tokens)
	value = builder.Reshape(v, headWidth, heads, tokens)
	if err := builder.Err(); err != nil {
		return nil, nil, err
	}
	return key, value, nil
}

// buildConditionedDiffusionBlock: one adaptive-layernorm diffusion
// transformer block. conditioning is the per-execution [6*dim] timestep
// embedding added to the block's learned modulation; chunk order is
// (self shift, self scale, self gate, ffn shift, ffn scale, ffn gate).
// crossKey/crossValue: preprojected fixed context.
func buildConditionedDiffusionBlock(
	builder *tensor.Builder,
	input, conditioning *tensor.Tensor,
	crossKey, crossValue *tensor.Tensor,
	historyKey, historyValue *tensor.Tensor,
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
	tokens, validInput := tensor.MatrixRows(input.Shape, dim)
	if !validInput {
		return result, errors.New("conditioned diffusion block input shape is incompatible")
	}
	headWidth, validHeads := tensor.EqualPartition(dim, heads)
	if !validHeads || !tensor.PartitionsCover(headWidth, options.AxisChannels[:]...) {
		return result, fmt.Errorf("conditioned diffusion block axis channels %v do not cover head width %d", options.AxisChannels, headWidth)
	}
	for axis := range options.AxisPositions {
		if uint64(len(options.AxisPositions[axis])) != tokens {
			return result, fmt.Errorf("conditioned diffusion block axis %d has %d positions, need %d", axis, len(options.AxisPositions[axis]), tokens)
		}
	}
	if !conditioning.Shape.Equal(weights.Modulation.Shape) {
		return result, errors.New("conditioned diffusion block conditioning must be a [6*dim] vector")
	}

	modulation := builder.Add(weights.Modulation, conditioning)
	chunk := func(index uint64) *tensor.Tensor {
		return builder.FlatSlice(modulation, index*dim, dim)
	}
	attentionScale := hostmath.InvSqrt32(headWidth)

	selfIn := builder.AdaptiveShiftScale(
		builder.LayerNorm(input, options.Epsilon),
		chunk(conditionedSelfShift), chunk(conditionedSelfScale),
	)
	result.SelfQueryProjected = buildBiasedProjection(builder, weights.SelfAttention.Query, weights.SelfAttention.QueryBias, selfIn)
	result.SelfKeyProjected = buildBiasedProjection(builder, weights.SelfAttention.Key, weights.SelfAttention.KeyBias, selfIn)
	result.SelfValueProjected = buildBiasedProjection(builder, weights.SelfAttention.Value, weights.SelfAttention.ValueBias, selfIn)
	result.SelfQueryNormed = builder.WeightedRMSNorm(result.SelfQueryProjected, weights.SelfAttention.QueryNorm, options.Epsilon)
	result.SelfKeyNormed = builder.WeightedRMSNorm(result.SelfKeyProjected, weights.SelfAttention.KeyNorm, options.Epsilon)
	query := builder.Reshape(result.SelfQueryNormed, headWidth, heads, tokens)
	key := builder.Reshape(result.SelfKeyNormed, headWidth, heads, tokens)
	value := builder.Reshape(result.SelfValueProjected, headWidth, heads, tokens)
	result.SelfQueryRotated = buildAxisPartitionedRoPE(builder, query, options.AxisChannels, options.AxisPositions, options.RotaryBase)
	result.SelfKeyRotated = buildAxisPartitionedRoPE(builder, key, options.AxisChannels, options.AxisPositions, options.RotaryBase)
	result.SelfValue = value
	attentionKey, attentionValue := result.SelfKeyRotated, value
	if historyKey != nil || historyValue != nil {
		if historyKey == nil || historyValue == nil {
			return result, errors.New("conditioned diffusion self history shape is incompatible")
		}
		_, historyAxis, validHistory := tensor.TrailingExtent(historyKey.Shape, headWidth, heads)
		if !validHistory || !historyKey.Shape.Equal(historyValue.Shape) {
			return result, errors.New("conditioned diffusion self history shape is incompatible")
		}
		attentionKey = builder.Concat(historyKey, attentionKey, historyAxis)
		attentionValue = builder.Concat(historyValue, attentionValue, historyAxis)
	}
	result.SelfKeyCache, result.SelfValueCache = attentionKey, attentionValue
	roundStorage := func(x *tensor.Tensor) *tensor.Tensor {
		if options.RoundAttentionStorage {
			return builder.BF16Round(x)
		}
		return x
	}
	attention := builder.AttentionWithOptions(
		roundStorage(result.SelfQueryRotated), roundStorage(attentionKey),
		roundStorage(attentionValue), tensor.AttentionOptions{Scale: attentionScale, Causal: false})

	result.SelfAttention = builder.Reshape(attention, dim, tokens)
	result.SelfProjected = buildBiasedProjection(builder, weights.SelfAttention.Output, weights.SelfAttention.OutputBias, result.SelfAttention)
	result.SelfResidual = builder.Add(input, builder.Multiply(result.SelfProjected, chunk(conditionedSelfGate)))

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
	crossAttention := builder.AttentionWithOptions(
		roundStorage(builder.Reshape(crossQuery, headWidth, heads, tokens)),
		roundStorage(crossKey), roundStorage(crossValue), tensor.AttentionOptions{Scale: attentionScale, Causal: false})

	result.CrossProjected = buildBiasedProjection(
		builder, weights.CrossAttention.Output, weights.CrossAttention.OutputBias,
		builder.Reshape(crossAttention, dim, tokens),
	)
	result.CrossResidual = builder.Add(result.SelfResidual, result.CrossProjected)

	ffnIn := builder.AdaptiveShiftScale(
		builder.LayerNorm(result.CrossResidual, options.Epsilon),
		chunk(conditionedFFNShift), chunk(conditionedFFNScale),
	)
	hidden := builder.GELUTanhExact(buildBiasedProjection(builder, weights.FFNExpand, weights.FFNExpandBias, ffnIn))
	result.FeedForward = buildBiasedProjection(builder, weights.FFNContract, weights.FFNContractBias, hidden)
	result.Output = builder.Add(result.CrossResidual, builder.Multiply(result.FeedForward, chunk(conditionedFFNGate)))
	if err := builder.Err(); err != nil {
		return ConditionedDiffusionBlockResult{}, err
	}
	return result, nil
}

// buildConditionedDiffusionHead: final modulated projection. conditioning is
// the [dim] timestep embedding; modulation is the learned [2*dim]
// (shift, scale) pair; output projects each token to its patch values.
func buildConditionedDiffusionHead(
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
	dim, _, validInput := tensor.MatrixExtents(input.Shape)
	if !validInput || !positiveFinite(epsilon) {
		return nil, errors.New("conditioned diffusion head input shape or epsilon is invalid")
	}
	if !tensor.IsVector(conditioning.Shape, dim) || !tensor.IsVector(modulation.Shape, tensor.PairedExtent*dim) {
		return nil, errors.New("conditioned diffusion head conditioning shape is incompatible")
	}
	shift := builder.Add(builder.FlatSlice(modulation, tensor.FirstOffset, dim), conditioning)
	scale := builder.Add(builder.FlatSlice(modulation, dim, dim), conditioning)
	modulated := builder.AdaptiveShiftScale(builder.LayerNorm(input, epsilon), shift, scale)
	output := buildBiasedProjection(builder, weight, bias, modulated)
	if err := builder.Err(); err != nil {
		return nil, err
	}
	return output, nil
}
