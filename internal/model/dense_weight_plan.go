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
	AllowUngatedExperts           bool
	RequireExpertBias             bool
	RequirePostNorm               bool
	RequireSubNorm                bool
	RequireAttentionOutputBias    bool
	ValidateOptionalQKNorm        bool
	UseSecondaryAttentionNorm     bool
	RequireExpertProjectionBiases bool
	RequireAttentionGate          bool
	RequireAttentionSinks         bool
	SkipFeedForwardNorm           bool
	RMSNormBias                   bool
	AllowActivationScale          bool
	BiasCatalog                   denseBiasCatalogPolicy
}

// DenseWeightPlan: compiled dense graph tensor contract.
type DenseWeightPlan struct {
	supportsExperts               bool
	allowUngatedExperts           bool
	requireExpertBias             bool
	requireShared                 bool
	requireSharedRouter           bool
	requireChunkExperts           bool
	requireExpertProjectionBiases bool
	requireSeparateDenseBranch    bool
	requireQKNorm                 bool
	requirePostNorm               bool
	requireAttentionGate          bool
	requireAttentionSinks         bool
	requireAttentionOutputBias    bool
	requireTemperature            bool
	requireSubNorm                bool
	requireBaseNorm               bool
	requireFeedForwardNorm        bool
	requireNormBias               bool
	validateOptionalQKNorm        bool
	useSecondaryAttentionNorm     bool
	allowActivationScale          bool
}

func (s Spec) denseWeightPlan(profile ArchitectureProfile, layer uint32) DenseWeightPlan {
	postOnly := !profile.Runtime.normalizationPlan(s, profile).PreAttention
	composition := s.expertCompositionPlan(profile)
	policy := profile.DenseWeights
	qk := s.qkPreprocessPlan(profile, layer)
	requireQKNorm := func(kind qkNormKind) bool {
		return kind == qkNormWeighted || kind == qkNormConfiguredNoBias ||
			kind == qkNormAffine || kind == qkNormRMS
	}
	queryScale := s.queryScalePlan(profile, layer)
	plan := DenseWeightPlan{
		supportsExperts: profile.Experts.Catalog != expertCatalogNone,
		requirePostNorm: postOnly || policy.RequirePostNorm,
		requireSubNorm:  policy.RequireSubNorm,
		requireAttentionOutputBias: profile.FeedForward == FeedForwardSequentialGELU ||
			policy.RequireAttentionOutputBias,
		requireTemperature:        queryScale.kind == queryScaleTemperature,
		validateOptionalQKNorm:    policy.ValidateOptionalQKNorm,
		useSecondaryAttentionNorm: policy.UseSecondaryAttentionNorm,
		allowActivationScale:      policy.AllowActivationScale,
	}
	plan.allowUngatedExperts = policy.AllowUngatedExperts
	plan.requireExpertBias = policy.RequireExpertBias
	plan.requireShared = plan.supportsExperts &&
		(composition.kind == expertSharedAdd || composition.kind == expertSharedAverage ||
			composition.kind == expertSharedLimited || composition.kind == expertSharedGated)
	plan.requireSharedRouter = composition.kind == expertSharedGated
	plan.requireChunkExperts = composition.kind == expertGrouped
	plan.requireExpertProjectionBiases = policy.RequireExpertProjectionBiases
	plan.requireSeparateDenseBranch = composition.kind == expertDenseRoutedSeparateNorm
	plan.requireAttentionGate = policy.RequireAttentionGate
	plan.requireAttentionSinks = policy.RequireAttentionSinks
	plan.requireQKNorm = plan.requirePostNorm || requireQKNorm(qk.Projection) ||
		requireQKNorm(qk.Heads) || requireQKNorm(qk.PostRotary) || plan.requireAttentionGate
	plan.requireBaseNorm = !postOnly && !s.UsesUnweightedLayerNorm() && !s.UsesUnweightedRMSNorm()
	plan.requireFeedForwardNorm = plan.requireBaseNorm && profile.Residual != ResidualParallel &&
		!policy.SkipFeedForwardNorm
	plan.requireNormBias = plan.requireBaseNorm && s.RequiresLayerNormBias()
	return plan
}

func (p DenseWeightPlan) Validate(
	spec Spec,
	weights LayerGraphWeights,
	usesExperts bool,
	feedForward FeedForwardPolicy,
) error {
	required := graphWeights{weights.AttentionOutput}
	if usesExperts {
		if !p.supportsExperts {
			return errors.New("dense block expert weights require a supported MoE architecture")
		}
		required = append(required, weights.FeedForwardRouter)
		required = append(required, weights.FeedForwardDownExperts)
		if weights.FeedForwardGateUpExperts != nil {
			required = append(required, weights.FeedForwardGateUpExperts)
		} else {
			required = append(required, weights.FeedForwardUpExperts)
			if !p.allowUngatedExperts || weights.FeedForwardGateExperts != nil {
				required = append(required, weights.FeedForwardGateExperts)
			}
		}
		if p.requireExpertBias {
			required = append(required, weights.FeedForwardExpertBias)
		}
		if p.requireShared {
			required = append(required, weights.FeedForwardSharedGate)
			required = append(required, weights.FeedForwardSharedUp)
			required = append(required, weights.FeedForwardSharedDown)
		}
		if p.requireSharedRouter {
			required = append(required, weights.FeedForwardSharedRouter)
		}
		if p.requireChunkExperts {
			required = append(required, weights.FeedForwardGateChunkExperts)
			required = append(required, weights.FeedForwardUpChunkExperts)
			required = append(required, weights.FeedForwardDownChunkExperts)
		}
		if p.requireExpertProjectionBiases {
			required = append(required, weights.FeedForwardRouterBias)
			required = append(required, weights.FeedForwardGateBias)
			required = append(required, weights.FeedForwardUpBias)
			required = append(required, weights.FeedForwardDownBias)
		}
		if p.requireSeparateDenseBranch {
			required = append(required, weights.FeedForwardExpertNorm)
			required = append(required, weights.FeedForwardGate)
			required = append(required, weights.FeedForwardUp)
			required = append(required, weights.FeedForwardDown)
		}
	} else {
		required = append(required, weights.FeedForwardUp)
		required = append(required, weights.FeedForwardDown)
		if feedForward == FeedForwardSwiGLU || feedForward == FeedForwardGEGLU {
			required = append(required, weights.FeedForwardGate)
		}
	}
	if weights.AttentionQKV != nil {
		required = append(required, weights.AttentionQKV)
		if weights.AttentionQBias != nil || weights.AttentionKBias != nil || weights.AttentionVBias != nil {
			return errors.New("dense fused QKV cannot use separate projection biases")
		}
	} else {
		required = append(required, weights.AttentionQ)
		required = append(required, weights.AttentionK)
		required = append(required, weights.AttentionV)
		if weights.AttentionQKVBias != nil {
			return errors.New("dense fused QKV bias has no fused projection")
		}
	}
	if p.useSecondaryAttentionNorm && weights.AttentionNorm2 == nil && weights.AttentionNorm2Bias != nil {
		return errors.New("secondary attention norm bias has no weight")
	}
	if p.requireSubNorm {
		required = append(required, weights.AttentionSubNorm)
		required = append(required, weights.FeedForwardSubNorm)
	}
	if feedForward == FeedForwardSequentialGELU {
		required = append(required, weights.FeedForwardUpBias)
		required = append(required, weights.FeedForwardDownBias)
	}
	if p.requireQKNorm {
		required = append(required, weights.AttentionQNorm)
		required = append(required, weights.AttentionKNorm)
	}
	if p.requirePostNorm {
		required = append(required, weights.AttentionPostNorm)
		required = append(required, weights.FeedForwardPostNorm)
	}
	if p.requireBaseNorm {
		required = append(required, weights.AttentionNorm)
		if p.requireFeedForwardNorm {
			required = append(required, weights.FeedForwardNorm)
		}
		if p.requireNormBias {
			required = append(required, weights.AttentionNormBias)
			if p.requireFeedForwardNorm {
				required = append(required, weights.FeedForwardNormBias)
			}
		}
	}
	if p.requireAttentionOutputBias {
		required = append(required, weights.AttentionOutputBias)
	}
	if p.requireAttentionSinks {
		required = append(required, weights.AttentionSinks)
		required = append(required, weights.AttentionPostNorm)
	}
	if p.requireAttentionGate {
		required = append(required, weights.AttentionOutputGate)
	}
	if p.requireTemperature {
		required = append(required, weights.AttentionTemperatureScale)
	}
	if p.validateOptionalQKNorm &&
		(weights.AttentionQNorm == nil) != (weights.AttentionKNorm == nil) {
		return fmt.Errorf("%s Q/K norm weights must both be present or absent", spec.Architecture)
	}
	return required.validate("dense block")
}
