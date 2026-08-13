package model

import (
	"errors"
	"math"

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

// QKPreprocessPlan: compiled Q/K normalization stages.
type QKPreprocessPlan struct {
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

// DenseStagePolicy: architecture-owned dense stage selection.
type DenseStagePolicy struct {
	QK                       QKPreprocessPlan
	QKHeadsMinBlocks         uint32
	NonRecurrentQKNoBias     bool
	PostRotaryRMSNon128      bool
	AttentionGate            attentionGateKind
	AttentionHeadGate        bool
	AttentionFlatGate        bool
	AttentionFlatGateElse    bool
	AttentionSubNorm         bool
	AttentionValueScale      bool
	Residual                 residualStageKind
	ResidualParallelOnly     bool
	QueryScale               queryScalePolicy
	EmbeddingHeadScaleBlocks uint32
}

func (s Spec) qkPreprocessPlan(layer uint32) QKPreprocessPlan {
	policy := s.Profile().DenseStages
	plan := policy.QK
	if policy.QKHeadsMinBlocks > 0 && s.BlockCount < policy.QKHeadsMinBlocks {
		plan.Heads = qkNormNone
	}
	if policy.PostRotaryRMSNon128 && s.UsesRoPE(layer) && s.ExpertCount != 128 {
		plan.PostRotary = qkNormRMS
	}
	if policy.NonRecurrentQKNoBias && !s.IsRecurrentLayer(layer) {
		plan.Heads = qkNormConfiguredNoBias
	}
	return plan
}

func (p QKPreprocessPlan) Apply(
	builder *tensor.Builder,
	query, key *tensor.Tensor,
	spec Spec,
	weights LayerGraphWeights,
	stage qkStage,
) (*tensor.Tensor, *tensor.Tensor, error) {
	kinds := [...]qkNormKind{p.Projection, p.Heads, p.PostRotary}
	return applyQKNorm(builder, query, key, spec, weights, kinds[stage])
}

func applyQKNorm(
	builder *tensor.Builder,
	query, key *tensor.Tensor,
	spec Spec,
	weights LayerGraphWeights,
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
		query = ApplyNormalization(builder, query, qWeight, weights.AttentionQNormBias, spec)
		key = ApplyNormalization(builder, key, kWeight, weights.AttentionKNormBias, spec)
	case qkNormConfiguredNoBias:
		query = ApplyNormalization(builder, query, qWeight, nil, spec)
		key = ApplyNormalization(builder, key, kWeight, nil, spec)
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

// QueryScalePlan: compiled pre-attention scaling order.
type QueryScalePlan struct {
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

// AttentionOutputPlan: compiled attention-output stages.
type AttentionOutputPlan struct {
	gate          attentionGateKind
	headGate      bool
	flatGate      bool
	flatGateElse  bool
	subNorm       bool
	valueScale    float32
	sandwichNorm  bool
	postNorm      bool
	residualScale float32
}

func (s Spec) attentionOutputPlan(norm NormalizationPlan) AttentionOutputPlan {
	policy := s.Profile().DenseStages
	plan := AttentionOutputPlan{
		valueScale: s.AttentionValueScale, sandwichNorm: s.SandwichNorm,
		postNorm: norm.PostAttention, residualScale: s.ResidualScale,
		gate: policy.AttentionGate, headGate: policy.AttentionHeadGate,
		flatGate: policy.AttentionFlatGate, flatGateElse: policy.AttentionFlatGateElse,
		subNorm: policy.AttentionSubNorm,
	}
	if !policy.AttentionValueScale {
		plan.valueScale = 0
	}
	return plan
}

func (p AttentionOutputPlan) PrepareGate(
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

func (p AttentionOutputPlan) ApplyGate(
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
	headWidth := weights.AttentionOutputGate.Shape.Dims[1] == uint64(headCount)
	if stage == attentionGateHeads {
		if !p.headGate || !headWidth {
			return attention
		}
		gate = builder.Reshape(gate, 1, uint64(headCount), tokens)
	} else if !p.flatGate && !(p.flatGateElse && !headWidth) {
		return attention
	}
	return builder.Multiply(attention, gate)
}

func (p AttentionOutputPlan) ApplyProjection(
	builder *tensor.Builder,
	attention *tensor.Tensor,
	spec Spec,
	weights LayerGraphWeights,
) (*tensor.Tensor, error) {
	if p.subNorm {
		attention = builder.WeightedRMSNorm(attention, weights.AttentionSubNorm, spec.RMSNormEpsilon)
	}
	attention = builder.MulMat(weights.AttentionOutput, attention)
	if p.valueScale != 0 {
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
	if p.residualScale > 0 {
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

// ResidualStagePlan: compiled residual and FFN normalization flow.
type ResidualStagePlan struct {
	kind          residualStageKind
	postOnly      bool
	sandwichNorm  bool
	residualScale float32
}

func (s Spec) residualStagePlan(profile ArchitectureProfile, norm NormalizationPlan) ResidualStagePlan {
	plan := ResidualStagePlan{
		postOnly: !norm.PreAttention, sandwichNorm: s.SandwichNorm,
		residualScale: s.ResidualScale,
	}
	policy := profile.DenseStages
	if policy.ResidualParallelOnly && !s.ParallelResidual {
		plan.kind = residualSequential
	} else if policy.Residual != residualSequential {
		plan.kind = policy.Residual
	} else if profile.Residual == ResidualParallel {
		plan.kind = residualShared
	}
	return plan
}

func (p ResidualStagePlan) AfterAttention(
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

func (p ResidualStagePlan) FeedForwardInput(
	builder *tensor.Builder,
	input, residual, normalized, feedForwardNormalized *tensor.Tensor,
	spec Spec,
	weights LayerGraphWeights,
) *tensor.Tensor {
	switch p.kind {
	case residualFeedForwardNormalized:
		return feedForwardNormalized
	case residualOriginalNorm:
		return ApplyNormalization(builder, input, weights.FeedForwardNorm, weights.FeedForwardNormBias, spec)
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
	return ApplyNormalization(builder, residual, weights.FeedForwardNorm, weights.FeedForwardNormBias, spec)
}

func (p ResidualStagePlan) ApplyFeedForwardOutput(
	builder *tensor.Builder,
	feedForward *tensor.Tensor,
	spec Spec,
	weights LayerGraphWeights,
	norm NormalizationPlan,
) *tensor.Tensor {
	if p.sandwichNorm {
		feedForward = builder.WeightedRMSNorm(feedForward, weights.FeedForwardNorm, spec.RMSNormEpsilon)
	}
	if norm.PostFeedForward {
		feedForward = builder.WeightedRMSNorm(feedForward, weights.FeedForwardPostNorm, spec.RMSNormEpsilon)
	}
	if p.residualScale > 0 {
		feedForward = builder.Scale(feedForward, p.residualScale)
	}
	return feedForward
}

func (s Spec) queryScalePlan(profile ArchitectureProfile, layer uint32) QueryScalePlan {
	switch profile.DenseStages.QueryScale {
	case queryScalePolicyTemperatureWithoutRoPE:
		if s.UsesRoPE(layer) {
			return QueryScalePlan{kind: queryScaleScores}
		}
		return QueryScalePlan{kind: queryScaleTemperature}
	case queryScalePolicyConfiguredTemperature:
		if s.AttentionTempScale == 0 {
			return QueryScalePlan{kind: queryScaleScores}
		}
		return QueryScalePlan{kind: queryScaleTemperature}
	case queryScalePolicyPreDot:
		return QueryScalePlan{kind: queryScalePreDot}
	case queryScalePolicyEmbeddingHead:
		return QueryScalePlan{
			kind:          queryScalePreDot,
			embeddingHead: profile.DenseStages.EmbeddingHeadScaleBlocks == s.BlockCount,
		}
	default:
		return QueryScalePlan{kind: queryScaleScores}
	}
}

func (p QueryScalePlan) Apply(
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
			scale = float32(1 / math.Sqrt(float64(spec.EmbeddingLength)/float64(spec.HeadCount)))
		}
		return builder.Scale(query, scale), 1
	default:
		return query, scale
	}
}
