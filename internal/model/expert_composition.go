package model

import (
	"errors"
	"math"

	"llamacpp2go/internal/tensor"
)

type expertCompositionKind uint8

const (
	expertRoutedOnly expertCompositionKind = iota
	expertSharedAdd
	expertSharedAverage
	expertSharedLimited
	expertSharedGated
	expertGrouped
	expertGrok
	expertArctic
)

type expertCompositionCondition uint8

const (
	expertCompositionAlways expertCompositionCondition = iota
	expertCompositionWithShared
	expertCompositionWithExpertsAndShared
	expertCompositionUnlessSigmoidWithoutShared
)

type expertNormalizationPolicy uint8

const (
	expertNormalizeAlways expertNormalizationPolicy = iota
	expertNormalizeMetadata
	expertNormalizeNever
)

type expertRoutingPolicy uint8

const (
	expertRouteMetadata expertRoutingPolicy = iota
	expertRouteSigmoid
	expertRouteSelectedSoftmax
)

// ExpertPolicy: routed/shared expert planning policy.
type ExpertPolicy struct {
	Composition         expertCompositionKind
	Condition           expertCompositionCondition
	Normalization       expertNormalizationPolicy
	Routing             expertRoutingPolicy
	Activation          tensor.MoEActivation
	SelectionBias       bool
	RouterInputOriginal bool
	ResidualScale       bool
	ClampSwiGLU         bool
	FusedGateUp         bool
	OptionalGate        bool
}

func (p ExpertPolicy) compositionKind(spec Spec) expertCompositionKind {
	switch p.Condition {
	case expertCompositionWithShared:
		if spec.SharedExpertFF == 0 {
			return expertRoutedOnly
		}
	case expertCompositionWithExpertsAndShared:
		if spec.ExpertCount == 0 || spec.SharedExpertFF == 0 {
			return expertRoutedOnly
		}
	case expertCompositionUnlessSigmoidWithoutShared:
		if spec.ExpertGatingFunc == 2 && spec.SharedExpertFF == 0 {
			return expertRoutedOnly
		}
	}
	return p.Composition
}

// ExpertCompositionPlan: compiled routed/shared expert composition.
type ExpertCompositionPlan struct {
	kind                expertCompositionKind
	routerInputOriginal bool
	residualScale       float32
}

func (p ExpertCompositionPlan) Validate(spec Spec) error {
	if p.kind == expertGrouped && (spec.ExpertsPerGroup == 0 || spec.ExpertCount == 0 ||
		spec.ExpertCount%spec.ExpertsPerGroup != 0) {
		return errors.New("GroveMoE expert grouping is invalid")
	}
	return nil
}

func (s Spec) expertCompositionPlan() ExpertCompositionPlan {
	policy := s.Profile().Experts
	plan := ExpertCompositionPlan{
		kind: policy.compositionKind(s), routerInputOriginal: policy.RouterInputOriginal,
	}
	if policy.ResidualScale && s.ExpertCount > 0 {
		plan.residualScale = s.ResidualScale
	}
	return plan
}

func (p ExpertCompositionPlan) buildArctic(
	builder *tensor.Builder,
	input, residual *tensor.Tensor,
	spec Spec,
	weights LayerGraphWeights,
	experts MoEGraphPlan,
) (*tensor.Tensor, error) {
	denseInput := builder.WeightedRMSNorm(residual, weights.FeedForwardNorm, spec.RMSNormEpsilon)
	denseGate := builder.MulMat(weights.FeedForwardGate, denseInput)
	denseUp := builder.MulMat(weights.FeedForwardUp, denseInput)
	dense := builder.MulMat(weights.FeedForwardDown, builder.SwiGLU(denseGate, denseUp))
	expertInput := builder.WeightedRMSNorm(input, weights.FeedForwardExpertNorm, spec.RMSNormEpsilon)
	routed := experts.BuildLayer(builder, expertInput, nil, weights)
	if err := builder.Err(); err != nil {
		return nil, err
	}
	return builder.Add(dense, routed), nil
}

func (p ExpertCompositionPlan) Build(
	builder *tensor.Builder,
	input, residual, normalized *tensor.Tensor,
	spec Spec,
	weights LayerGraphWeights,
	layerPlan LayerPlan,
) (*tensor.Tensor, error) {
	if p.kind == expertArctic {
		return p.buildArctic(builder, input, residual, spec, weights, layerPlan.Experts)
	}
	var routerInput *tensor.Tensor
	if p.routerInputOriginal {
		routerInput = input
	}
	routed := layerPlan.Experts.BuildLayer(builder, normalized, routerInput, weights)
	switch p.kind {
	case expertRoutedOnly:
	case expertSharedAdd:
		routed = builder.Add(routed, buildSharedSwiGLU(builder, normalized, weights))
	case expertSharedAverage:
		routed = builder.Scale(
			builder.Add(routed, buildSharedSwiGLU(builder, normalized, weights)), 0.5,
		)
	case expertSharedLimited:
		gate := builder.MulMat(weights.FeedForwardSharedGate, normalized)
		up := builder.MulMat(weights.FeedForwardSharedUp, normalized)
		shared := builder.MulMat(
			weights.FeedForwardSharedDown,
			limitedSwiGLU(builder, gate, up, spec.LayerSharedSwiGLUClampLimit(layerPlan.Layer)),
		)
		routed = builder.Add(routed, shared)
	case expertSharedGated:
		gateWeight := builder.Reshape(
			weights.FeedForwardSharedRouter, uint64(spec.EmbeddingLength), 1,
		)
		gate := builder.Sigmoid(builder.MulMat(gateWeight, normalized))
		shared := buildSharedSwiGLU(builder, normalized, weights)
		routed = builder.Add(routed, builder.Multiply(shared, gate))
	case expertGrouped:
		if spec.ExpertsPerGroup == 0 {
			return nil, errors.New("GroveMoE expert grouping is invalid")
		}
		chunkTopK := spec.ExpertUsedCount
		chunkExperts := spec.ExpertCount / spec.ExpertsPerGroup
		if chunkTopK > chunkExperts {
			chunkTopK = chunkExperts
		}
		chunkPlan := layerPlan.Experts
		chunkPlan.TopK = chunkTopK
		chunkPlan.ExpertIndexDivisor = spec.ExpertsPerGroup
		chunk := chunkPlan.Build(
			builder, routed, weights.FeedForwardRouter, MoEGraphInputs{
				RouterInput: normalized, Gate: weights.FeedForwardGateChunkExperts,
				Up: weights.FeedForwardUpChunkExperts, Down: weights.FeedForwardDownChunkExperts,
			},
		)
		routed = builder.Add(routed, builder.Scale(chunk, spec.ExpertGroupScale))
	case expertGrok:
		if weights.FeedForwardUp != nil || weights.FeedForwardGate != nil ||
			weights.FeedForwardDown != nil {
			if weights.FeedForwardUp == nil || weights.FeedForwardGate == nil ||
				weights.FeedForwardDown == nil {
				return nil, errors.New("Grok dense FFN weights are incomplete")
			}
			gate := builder.MulMat(weights.FeedForwardGate, normalized)
			up := builder.MulMat(weights.FeedForwardUp, normalized)
			dense := builder.MulMat(weights.FeedForwardDown, builder.GEGLU(gate, up))
			routed = builder.Scale(builder.Add(dense, routed), float32(math.Sqrt(0.5)))
		}
		if weights.FeedForwardPostNorm == nil {
			return nil, errors.New("Grok feed-forward post norm is nil")
		}
		routed = builder.WeightedRMSNorm(
			routed, weights.FeedForwardPostNorm, spec.RMSNormEpsilon,
		)
	}
	if p.residualScale > 0 {
		routed = builder.Scale(routed, p.residualScale)
	}
	if err := builder.Err(); err != nil {
		return nil, err
	}
	return routed, nil
}
