package model

import (
	"errors"
	"fmt"
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
	required := graphWeights{requireGraphWeight("attention output", weights.AttentionOutput)}
	if usesExperts {
		if !p.supportsExperts {
			return errors.New("dense block expert weights require a supported MoE architecture")
		}
		required.add("feed-forward router", weights.FeedForwardRouter)
		required.add("feed-forward expert down", weights.FeedForwardDownExperts)
		if weights.FeedForwardGateUpExperts != nil {
			required.add("feed-forward fused expert gate/up", weights.FeedForwardGateUpExperts)
		} else {
			required.add("feed-forward expert up", weights.FeedForwardUpExperts)
			if !p.allowUngatedExperts || weights.FeedForwardGateExperts != nil {
				required.add("feed-forward expert gate", weights.FeedForwardGateExperts)
			}
		}
		if p.requireExpertBias {
			required.add("feed-forward expert correction bias", weights.FeedForwardExpertBias)
		}
		if p.requireShared {
			required.add("feed-forward shared gate", weights.FeedForwardSharedGate)
			required.add("feed-forward shared up", weights.FeedForwardSharedUp)
			required.add("feed-forward shared down", weights.FeedForwardSharedDown)
		}
		if p.requireSharedRouter {
			required.add("feed-forward shared router", weights.FeedForwardSharedRouter)
		}
		if p.requireChunkExperts {
			required.add("feed-forward chunk expert gate", weights.FeedForwardGateChunkExperts)
			required.add("feed-forward chunk expert up", weights.FeedForwardUpChunkExperts)
			required.add("feed-forward chunk expert down", weights.FeedForwardDownChunkExperts)
		}
		if p.requireOpenAIBiases {
			required.add("feed-forward router bias", weights.FeedForwardRouterBias)
			required.add("feed-forward expert gate bias", weights.FeedForwardGateBias)
			required.add("feed-forward expert up bias", weights.FeedForwardUpBias)
			required.add("feed-forward expert down bias", weights.FeedForwardDownBias)
		}
		if p.requireArcticDense {
			required.add("feed-forward expert norm", weights.FeedForwardExpertNorm)
			required.add("feed-forward gate", weights.FeedForwardGate)
			required.add("feed-forward up", weights.FeedForwardUp)
			required.add("feed-forward down", weights.FeedForwardDown)
		}
	} else {
		required.add("feed-forward up", weights.FeedForwardUp)
		required.add("feed-forward down", weights.FeedForwardDown)
		if profile.FeedForward == FeedForwardSwiGLU {
			required.add("feed-forward gate", weights.FeedForwardGate)
		}
	}
	if weights.AttentionQKV != nil {
		required.add("attention QKV", weights.AttentionQKV)
		if weights.AttentionQBias != nil || weights.AttentionKBias != nil || weights.AttentionVBias != nil {
			return errors.New("dense fused QKV cannot use separate projection biases")
		}
	} else {
		required.add("attention Q", weights.AttentionQ)
		required.add("attention K", weights.AttentionK)
		required.add("attention V", weights.AttentionV)
		if weights.AttentionQKVBias != nil {
			return errors.New("dense fused QKV bias has no fused projection")
		}
	}
	if p.validateFalconNorm && weights.AttentionNorm2 == nil && weights.AttentionNorm2Bias != nil {
		return errors.New("Falcon secondary attention norm bias has no weight")
	}
	if p.requireSubNorm {
		required.add("attention sub norm", weights.AttentionSubNorm)
		required.add("feed-forward sub norm", weights.FeedForwardSubNorm)
	}
	if profile.FeedForward == FeedForwardSequentialGELU {
		required.add("feed-forward up bias", weights.FeedForwardUpBias)
		required.add("feed-forward down bias", weights.FeedForwardDownBias)
	}
	if p.requireQKNorm {
		required.add("attention Q norm", weights.AttentionQNorm)
		required.add("attention K norm", weights.AttentionKNorm)
	}
	if p.requirePostNorm {
		required.add("attention post norm", weights.AttentionPostNorm)
		required.add("feed-forward post norm", weights.FeedForwardPostNorm)
	}
	if p.requireBaseNorm {
		required.add("attention norm", weights.AttentionNorm)
		if p.requireFeedForwardNorm {
			required.add("feed-forward norm", weights.FeedForwardNorm)
		}
		if p.requireNormBias {
			required.add("attention norm bias", weights.AttentionNormBias)
			if p.requireFeedForwardNorm {
				required.add("feed-forward norm bias", weights.FeedForwardNormBias)
			}
		}
	}
	if p.requireAttentionOutputBias {
		required.add("attention output bias", weights.AttentionOutputBias)
	}
	if profile.DenseWeights.RequireAttentionSinks {
		required.add("attention sinks", weights.AttentionSinks)
		required.add("attention post norm", weights.AttentionPostNorm)
	}
	if p.requireAttentionGate {
		required.add("attention output gate", weights.AttentionOutputGate)
	}
	if p.requireTemperature {
		required.add("attention temperature scale", weights.AttentionTemperatureScale)
	}
	if p.validateOptionalQKNorm &&
		(weights.AttentionQNorm == nil) != (weights.AttentionKNorm == nil) {
		return fmt.Errorf("%s Q/K norm weights must both be present or absent", spec.Architecture)
	}
	return required.validate("dense block")
}
