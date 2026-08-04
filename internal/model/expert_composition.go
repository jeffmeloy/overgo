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
	plan := ExpertCompositionPlan{}
	shared := s.SharedExpertFF > 0
	switch s.Architecture {
	case "arctic":
		plan.kind = expertArctic
	case "grok":
		plan.kind = expertGrok
	case "grovemoe":
		plan.kind = expertGrouped
	case "cohere2moe":
		if shared {
			plan.kind = expertSharedAverage
		}
	case "step35":
		if shared {
			plan.kind = expertSharedLimited
		}
	case "qwen2moe":
		plan.kind = expertSharedGated
	case "hy_v3", "deepseek2-ocr", "glm4moe", "llama4", "hunyuan-moe",
		"bailingmoe", "deepseek":
		plan.kind = expertSharedAdd
	case "exaone-moe", "bailingmoe2", "lfm2moe", "dots1":
		if s.ExpertGatingFunc != 2 || shared {
			plan.kind = expertSharedAdd
		}
	case "ernie4_5-moe", "laguna", "afmoe", "granitemoe", "granitehybrid":
		if shared {
			plan.kind = expertSharedAdd
		}
	}
	if s.Architecture == "granite" && s.ExpertCount > 0 && shared {
		plan.kind = expertSharedAdd
	}
	plan.routerInputOriginal = s.Architecture == "smallthinker"
	if s.Architecture == "granitemoe" || s.Architecture == "granitehybrid" ||
		s.Architecture == "granite" && s.ExpertCount > 0 {
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
