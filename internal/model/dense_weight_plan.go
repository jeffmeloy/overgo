package model

import (
	"errors"
	"fmt"

	"llamacpp2go/internal/tensor"
)

type denseBiasCatalogPolicy uint8

const (
	denseBiasCatalogStandard denseBiasCatalogPolicy = iota
	denseBiasCatalogGPTJ
	denseBiasCatalogJais2
	denseBiasCatalogJais
)

// DenseWeightPolicy: architecture-owned dense tensor requirements.
type DenseWeightPolicy struct {
	AllowUngatedExperts        bool
	RequireExpertBias          bool
	RequirePostNorm            bool
	RequireSubNorm             bool
	RequireAttentionOutputBias bool
	ValidateOptionalQKNorm     bool
	ValidateFalconNorm         bool
	RequireOpenAIBiases        bool
	RequireAttentionGate       bool
	RequireAttentionSinks      bool
	SkipFeedForwardNorm        bool
	RMSNormBias                bool
	AllowActivationScale       bool
	BiasCatalog                denseBiasCatalogPolicy
}

// DenseWeightPlan: compiled dense graph tensor contract.
type DenseWeightPlan struct {
	supportsExperts            bool
	allowUngatedExperts        bool
	requireExpertBias          bool
	requireShared              bool
	requireSharedRouter        bool
	requireChunkExperts        bool
	requireOpenAIBiases        bool
	requireArcticDense         bool
	requireQKNorm              bool
	requirePostNorm            bool
	requireAttentionGate       bool
	requireAttentionOutputBias bool
	requireTemperature         bool
	requireSubNorm             bool
	requireBaseNorm            bool
	requireFeedForwardNorm     bool
	requireNormBias            bool
	validateOptionalQKNorm     bool
	validateFalconNorm         bool
	allowActivationScale       bool
}

func (s Spec) denseWeightPlan(profile ArchitectureProfile, layer uint32) DenseWeightPlan {
	postOnly := !s.NormPlan().PreAttention
	composition := s.expertCompositionPlan()
	policy := profile.DenseWeights
	qk := s.qkPreprocessPlan(layer)
	requireQKNorm := func(kind qkNormKind) bool {
		return kind == qkNormWeighted || kind == qkNormConfiguredNoBias ||
			kind == qkNormAffine || kind == qkNormRMS
	}
	queryScale := s.queryScalePlan(profile, layer)
	plan := DenseWeightPlan{
		supportsExperts: profile.Has(ArchitectureMoE) || s.ExpertCount > 0,
		requirePostNorm: postOnly || policy.RequirePostNorm,
		requireSubNorm:  policy.RequireSubNorm,
		requireAttentionOutputBias: profile.FeedForward == FeedForwardSequentialGELU ||
			policy.RequireAttentionOutputBias,
		requireTemperature:     queryScale.kind == queryScaleTemperature,
		validateOptionalQKNorm: policy.ValidateOptionalQKNorm,
		validateFalconNorm:     policy.ValidateFalconNorm,
		allowActivationScale:   policy.AllowActivationScale,
	}
	plan.allowUngatedExperts = policy.AllowUngatedExperts
	plan.requireExpertBias = policy.RequireExpertBias
	plan.requireShared = plan.supportsExperts &&
		(composition.kind == expertSharedAdd || composition.kind == expertSharedAverage ||
			composition.kind == expertSharedLimited || composition.kind == expertSharedGated)
	plan.requireSharedRouter = composition.kind == expertSharedGated
	plan.requireChunkExperts = composition.kind == expertGrouped
	plan.requireOpenAIBiases = policy.RequireOpenAIBiases
	plan.requireArcticDense = composition.kind == expertArctic
	plan.requireAttentionGate = policy.RequireAttentionGate
	plan.requireQKNorm = plan.requirePostNorm || requireQKNorm(qk.Projection) ||
		requireQKNorm(qk.Heads) || requireQKNorm(qk.PostRotary) || plan.requireAttentionGate
	plan.requireBaseNorm = !postOnly && !s.UsesUnweightedLayerNorm()
	plan.requireFeedForwardNorm = plan.requireBaseNorm && profile.Residual != ResidualParallel &&
		!policy.SkipFeedForwardNorm
	plan.requireNormBias = plan.requireBaseNorm && s.RequiresLayerNormBias()
	return plan
}

func (p DenseWeightPlan) Validate(
	spec Spec,
	profile ArchitectureProfile,
	weights LayerGraphWeights,
	usesExperts bool,
) error {
	required := map[string]*tensor.Tensor{"attention output": weights.AttentionOutput}
	if usesExperts {
		if !p.supportsExperts {
			return errors.New("dense block expert weights require a supported MoE architecture")
		}
		required["feed-forward router"] = weights.FeedForwardRouter
		required["feed-forward expert down"] = weights.FeedForwardDownExperts
		if weights.FeedForwardGateUpExperts != nil {
			required["feed-forward fused expert gate/up"] = weights.FeedForwardGateUpExperts
		} else {
			required["feed-forward expert up"] = weights.FeedForwardUpExperts
			if !p.allowUngatedExperts || weights.FeedForwardGateExperts != nil {
				required["feed-forward expert gate"] = weights.FeedForwardGateExperts
			}
		}
		if p.requireExpertBias {
			required["feed-forward expert correction bias"] = weights.FeedForwardExpertBias
		}
		if p.requireShared {
			required["feed-forward shared gate"] = weights.FeedForwardSharedGate
			required["feed-forward shared up"] = weights.FeedForwardSharedUp
			required["feed-forward shared down"] = weights.FeedForwardSharedDown
		}
		if p.requireSharedRouter {
			required["feed-forward shared router"] = weights.FeedForwardSharedRouter
		}
		if p.requireChunkExperts {
			required["feed-forward chunk expert gate"] = weights.FeedForwardGateChunkExperts
			required["feed-forward chunk expert up"] = weights.FeedForwardUpChunkExperts
			required["feed-forward chunk expert down"] = weights.FeedForwardDownChunkExperts
		}
		if p.requireOpenAIBiases {
			required["feed-forward router bias"] = weights.FeedForwardRouterBias
			required["feed-forward expert gate bias"] = weights.FeedForwardGateBias
			required["feed-forward expert up bias"] = weights.FeedForwardUpBias
			required["feed-forward expert down bias"] = weights.FeedForwardDownBias
		}
		if p.requireArcticDense {
			required["feed-forward expert norm"] = weights.FeedForwardExpertNorm
			required["feed-forward gate"] = weights.FeedForwardGate
			required["feed-forward up"] = weights.FeedForwardUp
			required["feed-forward down"] = weights.FeedForwardDown
		}
	} else {
		required["feed-forward up"] = weights.FeedForwardUp
		required["feed-forward down"] = weights.FeedForwardDown
		if profile.FeedForward == FeedForwardSwiGLU {
			required["feed-forward gate"] = weights.FeedForwardGate
		}
	}
	if weights.AttentionQKV != nil {
		required["attention QKV"] = weights.AttentionQKV
		if weights.AttentionQBias != nil || weights.AttentionKBias != nil || weights.AttentionVBias != nil {
			return errors.New("dense fused QKV cannot use separate projection biases")
		}
	} else {
		required["attention Q"] = weights.AttentionQ
		required["attention K"] = weights.AttentionK
		required["attention V"] = weights.AttentionV
		if weights.AttentionQKVBias != nil {
			return errors.New("dense fused QKV bias has no fused projection")
		}
	}
	if p.validateFalconNorm && weights.AttentionNorm2 == nil && weights.AttentionNorm2Bias != nil {
		return errors.New("Falcon secondary attention norm bias has no weight")
	}
	if p.requireSubNorm {
		required["attention sub norm"] = weights.AttentionSubNorm
		required["feed-forward sub norm"] = weights.FeedForwardSubNorm
	}
	if profile.FeedForward == FeedForwardSequentialGELU {
		required["feed-forward up bias"] = weights.FeedForwardUpBias
		required["feed-forward down bias"] = weights.FeedForwardDownBias
	}
	if p.requireQKNorm {
		required["attention Q norm"] = weights.AttentionQNorm
		required["attention K norm"] = weights.AttentionKNorm
	}
	if p.requirePostNorm {
		required["attention post norm"] = weights.AttentionPostNorm
		required["feed-forward post norm"] = weights.FeedForwardPostNorm
	}
	if p.requireBaseNorm {
		required["attention norm"] = weights.AttentionNorm
		if p.requireFeedForwardNorm {
			required["feed-forward norm"] = weights.FeedForwardNorm
		}
		if p.requireNormBias {
			required["attention norm bias"] = weights.AttentionNormBias
			if p.requireFeedForwardNorm {
				required["feed-forward norm bias"] = weights.FeedForwardNormBias
			}
		}
	}
	if p.requireAttentionOutputBias {
		required["attention output bias"] = weights.AttentionOutputBias
	}
	if profile.DenseWeights.RequireAttentionSinks {
		required["attention sinks"] = weights.AttentionSinks
		required["attention post norm"] = weights.AttentionPostNorm
	}
	if p.requireAttentionGate {
		required["attention output gate"] = weights.AttentionOutputGate
	}
	if p.requireTemperature {
		required["attention temperature scale"] = weights.AttentionTemperatureScale
	}
	if p.validateOptionalQKNorm &&
		(weights.AttentionQNorm == nil) != (weights.AttentionKNorm == nil) {
		return fmt.Errorf("%s Q/K norm weights must both be present or absent", spec.Architecture)
	}
	for name, item := range required {
		if item == nil {
			return fmt.Errorf("dense block %s weight is nil", name)
		}
	}
	return nil
}
