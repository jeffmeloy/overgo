package model

import (
	"errors"

	"overgo/internal/hostmath"
	"overgo/internal/tensor"
)

type qkNormKind uint8

const (
	qkNormNone qkNormKind = iota
	qkNormWeighted
	qkNormOptionalWeighted
	qkNormConfigured
	qkNormConfiguredNoBias
	qkNormAffine
	qkNormLayer
	qkNormRMS
)

// qkPreprocessPlan: compiled Q/K normalization stages.
type qkPreprocessPlan struct {
	Projection qkNormKind
	Heads      qkNormKind
	PostRotary qkNormKind
}

type qkStage uint8

const (
	qkProjection qkStage = iota
	qkHeads
	qkPostRotary
)

type queryScalePolicy uint8

const (
	queryScalePolicyScores queryScalePolicy = iota
	queryScalePolicyTemperatureWithoutRoPE
	queryScalePolicyConfiguredTemperature
	queryScalePolicyPreDot
	queryScalePolicyEmbeddingHead
)

// denseStagePolicy: architecture-owned dense stage selection.
type denseStagePolicy struct {
	QK                           qkPreprocessPlan
	QKHeadsMinBlocks             uint32
	NonRecurrentQKNoBias         bool
	PostRotaryRMSExcludedExperts uint32
	AttentionGate                attentionGateKind
	AttentionHeadGate            bool
	AttentionFlatGate            bool
	AttentionFlatGateElse        bool
	AttentionSubNorm             bool
	AttentionValueScale          bool
	Residual                     residualStageKind
	ResidualParallelOnly         bool
	QueryScale                   queryScalePolicy
	EmbeddingHeadScaleBlocks     uint32
}

func (s Spec) qkPreprocessPlan(profile ArchitectureProfile, layer uint32) qkPreprocessPlan {
	policy := profile.DenseStages
	plan := policy.QK
	if policy.QKHeadsMinBlocks != tensor.FirstOffset && s.BlockCount < policy.QKHeadsMinBlocks {
		plan.Heads = qkNormNone
	}
	if policy.PostRotaryRMSExcludedExperts != tensor.FirstOffset && s.UsesRoPE(layer) &&
		s.ExpertCount != policy.PostRotaryRMSExcludedExperts {
		plan.PostRotary = qkNormRMS
	}
	if policy.NonRecurrentQKNoBias && !s.IsRecurrentLayer(layer) {
		plan.Heads = qkNormConfiguredNoBias
	}
	return plan
}

func (p qkPreprocessPlan) Apply(
	builder *tensor.Builder,
	query, key *tensor.Tensor,
	spec Spec,
	weights LayerGraphWeights,
	normalization normalizationPlan,
	stage qkStage,
) (*tensor.Tensor, *tensor.Tensor, error) {
	kinds := [...]qkNormKind{p.Projection, p.Heads, p.PostRotary}
	return applyQKNorm(builder, query, key, spec, weights, normalization, kinds[stage])
}

func applyQKNorm(
	builder *tensor.Builder,
	query, key *tensor.Tensor,
	spec Spec,
	weights LayerGraphWeights,
	normalization normalizationPlan,
	kind qkNormKind,
) (*tensor.Tensor, *tensor.Tensor, error) {
	qWeight, kWeight := weights.AttentionQNorm, weights.AttentionKNorm
	switch kind {
	case qkNormNone:
		return query, key, nil
	case qkNormOptionalWeighted, qkNormConfigured, qkNormLayer:
		if (qWeight == nil) != (kWeight == nil) {
			return nil, nil, errors.New("dense block Q/K norm weights are incomplete")
		}
		if qWeight == nil {
			return query, key, nil
		}
	}
	if kind != qkNormRMS && (qWeight == nil || kWeight == nil) {
		return nil, nil, errors.New("dense block architecture requires Q/K norm weights")
	}
	switch kind {
	case qkNormWeighted, qkNormOptionalWeighted:
		query = builder.WeightedRMSNorm(query, qWeight, spec.RMSNormEpsilon)
		key = builder.WeightedRMSNorm(key, kWeight, spec.RMSNormEpsilon)
	case qkNormConfigured:
		query = normalization.Apply(builder, query, qWeight, weights.AttentionQNormBias)
		key = normalization.Apply(builder, key, kWeight, weights.AttentionKNormBias)
	case qkNormConfiguredNoBias:
		query = normalization.Apply(builder, query, qWeight, nil)
		key = normalization.Apply(builder, key, kWeight, nil)
	case qkNormAffine:
		query = builder.Multiply(builder.LayerNorm(query, spec.QKNormEpsilon), qWeight)
		key = builder.Multiply(builder.LayerNorm(key, spec.QKNormEpsilon), kWeight)
		if weights.AttentionQNormBias != nil {
			query = builder.Add(query, weights.AttentionQNormBias)
		}
		if weights.AttentionKNormBias != nil {
			key = builder.Add(key, weights.AttentionKNormBias)
		}
	case qkNormLayer:
		query = builder.Multiply(builder.LayerNorm(query, spec.LayerNormEpsilon), qWeight)
		key = builder.Multiply(builder.LayerNorm(key, spec.LayerNormEpsilon), kWeight)
	case qkNormRMS:
		query = builder.RMSNorm(query, spec.RMSNormEpsilon)
		key = builder.RMSNorm(key, spec.RMSNormEpsilon)
	}
	return query, key, nil
}

type queryScaleKind uint8

const (
	queryScaleScores queryScaleKind = iota
	queryScalePreDot
	queryScaleTemperature
)

// queryScalePlan: compiled pre-attention scaling order.
type queryScalePlan struct {
	kind          queryScaleKind
	embeddingHead bool
}

type attentionGateKind uint8

const (
	attentionGateNone attentionGateKind = iota
	attentionGateSoftplus
	attentionGateSigmoid
)

type attentionGateStage uint8

const (
	attentionGateHeads attentionGateStage = iota
	attentionGateFlat
)

// attentionOutputPlan: compiled attention-output stages.
type attentionOutputPlan struct {
	gate          attentionGateKind
	headGate      bool
	flatGate      bool
	flatGateElse  bool
	subNorm       bool
	valueScale    float32
	scaleValue    bool
	sandwichNorm  bool
	postNorm      bool
	residualScale float32
}

func (s Spec) attentionOutputPlan(profile ArchitectureProfile, norm normalizationPlan) attentionOutputPlan {
	policy := profile.DenseStages
	plan := attentionOutputPlan{
		valueScale: s.AttentionValueScale, scaleValue: policy.AttentionValueScale,
		sandwichNorm: s.SandwichNorm,
		postNorm:     norm.PostAttention, residualScale: s.ResidualScale,
		gate: policy.AttentionGate, headGate: policy.AttentionHeadGate,
		flatGate: policy.AttentionFlatGate, flatGateElse: policy.AttentionFlatGateElse,
		subNorm: policy.AttentionSubNorm,
	}
	return plan
}

func (p attentionOutputPlan) PrepareGate(
	builder *tensor.Builder,
	normalized *tensor.Tensor,
	weights LayerGraphWeights,
) *tensor.Tensor {
	if p.gate == attentionGateNone || weights.AttentionOutputGate == nil {
		return nil
	}
	gate := builder.MulMat(weights.AttentionOutputGate, normalized)
	if p.gate == attentionGateSoftplus {
		return builder.Softplus(gate)
	}
	return builder.Sigmoid(gate)
}

func (p attentionOutputPlan) ApplyGate(
	builder *tensor.Builder,
	attention, gate *tensor.Tensor,
	weights LayerGraphWeights,
	headCount uint32,
	tokens uint64,
	stage attentionGateStage,
) *tensor.Tensor {
	if gate == nil {
		return attention
	}
	_, gateOutputs, validGate := tensor.MatrixExtents(weights.AttentionOutputGate.Shape)
	perHeadGate := validGate && gateOutputs == uint64(headCount)
	if stage == attentionGateHeads {
		if !p.headGate || !perHeadGate {
			return attention
		}
		gate = builder.Reshape(gate, tensor.SingletonExtent, uint64(headCount), tokens)
	} else if !p.flatGate && !(p.flatGateElse && !perHeadGate) {
		return attention
	}
	return builder.Multiply(attention, gate)
}

func (p attentionOutputPlan) ApplyProjection(
	builder *tensor.Builder,
	attention *tensor.Tensor,
	spec Spec,
	weights LayerGraphWeights,
) (*tensor.Tensor, error) {
	if p.subNorm {
		attention = builder.WeightedRMSNorm(attention, weights.AttentionSubNorm, spec.RMSNormEpsilon)
	}
	attention = builder.MulMat(weights.AttentionOutput, attention)
	if p.scaleValue {
		attention = builder.Scale(attention, p.valueScale)
	}
	if weights.AttentionOutputScale != nil {
		attention = builder.Multiply(attention, weights.AttentionOutputScale)
	}
	if weights.AttentionOutputBias != nil {
		attention = builder.Add(attention, weights.AttentionOutputBias)
	}
	if p.sandwichNorm {
		attention = builder.WeightedRMSNorm(attention, weights.AttentionNorm, spec.RMSNormEpsilon)
	}
	if p.postNorm {
		if weights.AttentionPostNorm == nil || weights.FeedForwardPostNorm == nil {
			return nil, errors.New("dense post-normalized block requires post norm weights")
		}
		attention = builder.WeightedRMSNorm(attention, weights.AttentionPostNorm, spec.RMSNormEpsilon)
	}
	if positiveFinite(p.residualScale) {
		attention = builder.Scale(attention, p.residualScale)
	}
	return attention, nil
}

type residualStageKind uint8

const (
	residualSequential residualStageKind = iota
	residualShared
	residualFeedForwardNormalized
	residualOriginalNorm
	residualStable
	residualGPTOSS
)

// residualStagePlan: compiled residual and FFN normalization flow.
type residualStagePlan struct {
	kind          residualStageKind
	postOnly      bool
	sandwichNorm  bool
	residualScale float32
}

func (s Spec) residualStagePlan(profile ArchitectureProfile, norm normalizationPlan) residualStagePlan {
	plan := residualStagePlan{
		postOnly: !norm.PreAttention, sandwichNorm: s.SandwichNorm,
		residualScale: s.ResidualScale,
	}
	policy := profile.DenseStages
	if policy.ResidualParallelOnly && !s.ParallelResidual {
		plan.kind = residualSequential
	} else if policy.Residual != residualSequential {
		plan.kind = policy.Residual
	} else if profile.Residual == residualParallel {
		plan.kind = residualShared
	}
	return plan
}

func (p residualStagePlan) AfterAttention(
	builder *tensor.Builder,
	residual, normalized *tensor.Tensor,
	spec Spec,
	weights LayerGraphWeights,
) *tensor.Tensor {
	if p.kind == residualGPTOSS {
		return builder.WeightedRMSNorm(residual, weights.AttentionPostNorm, spec.RMSNormEpsilon)
	}
	return normalized
}

func (p residualStagePlan) FeedForwardInput(
	builder *tensor.Builder,
	input, residual, normalized, feedForwardNormalized *tensor.Tensor,
	spec Spec,
	weights LayerGraphWeights,
	normalization normalizationPlan,
) *tensor.Tensor {
	switch p.kind {
	case residualFeedForwardNormalized:
		return feedForwardNormalized
	case residualOriginalNorm:
		return normalization.Apply(builder, input, weights.FeedForwardNorm, weights.FeedForwardNormBias)
	case residualShared, residualGPTOSS:
		return normalized
	case residualStable:
		if weights.FeedForwardNorm == nil {
			return normalized
		}
	}
	if p.postOnly || p.sandwichNorm {
		return residual
	}
	if p.kind == residualStable && weights.FeedForwardNormBias == nil {
		return builder.Multiply(builder.LayerNorm(residual, spec.LayerNormEpsilon), weights.FeedForwardNorm)
	}
	return normalization.Apply(builder, residual, weights.FeedForwardNorm, weights.FeedForwardNormBias)
}

func (p residualStagePlan) ApplyFeedForwardOutput(
	builder *tensor.Builder,
	feedForward *tensor.Tensor,
	spec Spec,
	weights LayerGraphWeights,
	norm normalizationPlan,
) *tensor.Tensor {
	if p.sandwichNorm {
		feedForward = builder.WeightedRMSNorm(feedForward, weights.FeedForwardNorm, spec.RMSNormEpsilon)
	}
	if norm.PostFeedForward {
		feedForward = builder.WeightedRMSNorm(feedForward, weights.FeedForwardPostNorm, spec.RMSNormEpsilon)
	}
	if positiveFinite(p.residualScale) {
		feedForward = builder.Scale(feedForward, p.residualScale)
	}
	return feedForward
}

func (s Spec) queryScalePlan(profile ArchitectureProfile, layer uint32) queryScalePlan {
	switch profile.DenseStages.QueryScale {
	case queryScalePolicyTemperatureWithoutRoPE:
		if s.UsesRoPE(layer) {
			return queryScalePlan{kind: queryScaleScores}
		}
		return queryScalePlan{kind: queryScaleTemperature}
	case queryScalePolicyConfiguredTemperature:
		if !positiveFinite(s.AttentionTempScale) {
			return queryScalePlan{kind: queryScaleScores}
		}
		return queryScalePlan{kind: queryScaleTemperature}
	case queryScalePolicyPreDot:
		return queryScalePlan{kind: queryScalePreDot}
	case queryScalePolicyEmbeddingHead:
		return queryScalePlan{
			kind:          queryScalePreDot,
			embeddingHead: profile.DenseStages.EmbeddingHeadScaleBlocks == s.BlockCount,
		}
	default:
		return queryScalePlan{kind: queryScaleScores}
	}
}

func (p queryScalePlan) Apply(
	builder *tensor.Builder,
	query *tensor.Tensor,
	spec Spec,
	weights LayerGraphWeights,
	scale float32,
) (*tensor.Tensor, float32) {
	switch p.kind {
	case queryScaleTemperature:
		return builder.Multiply(query, weights.AttentionTemperatureScale), scale
	case queryScalePreDot:
		if p.embeddingHead {
			scale = hostmath.InvSqrt32(uint64(spec.EmbeddingLength) / uint64(spec.HeadCount))
		}
		return builder.Scale(query, scale), tensor.UnitScale
	default:
		return query, scale
	}
}
