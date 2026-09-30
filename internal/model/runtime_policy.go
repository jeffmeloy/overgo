package model

import (
	"overgo/internal/hostmath"
	"overgo/internal/tensor"
)

// EmbeddingScalePolicy: default input scaling.
type EmbeddingScalePolicy uint8

const (
	embeddingScaleIdentity EmbeddingScalePolicy = iota
	embeddingScaleSqrtWidth
)

// LogitScalePolicy: metadata scale direction.
type LogitScalePolicy uint8

const (
	logitScaleReciprocal LogitScalePolicy = iota
	logitScaleDirect
)

// NormalizationPlacementPolicy: block normalization placement.
type NormalizationPlacementPolicy uint8

const (
	normalizationPlacementCapabilities NormalizationPlacementPolicy = iota
	normalizationPlacementPostOnly
)

// NormalizationBiasPolicy: affine bias selection.
type NormalizationBiasPolicy uint8

const (
	normalizationBiasAutomatic NormalizationBiasPolicy = iota
	normalizationBiasNever
	normalizationBiasAlways
)

// runtimePolicy: scalar and normalization runtime decisions.
type runtimePolicy struct {
	EmbeddingScale          EmbeddingScalePolicy
	LogitScale              LogitScalePolicy
	NormalizationPlacement  NormalizationPlacementPolicy
	NormalizationBias       NormalizationBiasPolicy
	Recurrent               recurrentRuntimePolicy
	ActivationResidualScale float32
}

// recurrentRuntimePolicy: recurrent scalar contract.
type recurrentRuntimePolicy struct {
	HeadNormEpsilon       float32
	KeyNormEpsilon        float32
	PeriodicResidualScale float32
	TokenShiftCount       uint32
	InnerWidthMultiplier  uint32
}

func (p recurrentRuntimePolicy) validInnerWidth(spec Spec) bool {
	return p.InnerWidthMultiplier == tensor.FirstOffset ||
		uint64(spec.SSMInnerSize) == uint64(p.InnerWidthMultiplier)*uint64(spec.EmbeddingLength)
}

func (p runtimePolicy) inputEmbeddingScale(spec Spec) float32 {
	if positiveFinite(spec.EmbeddingScale) {
		return spec.EmbeddingScale
	}
	if p.EmbeddingScale == embeddingScaleSqrtWidth {
		return hostmath.Sqrt32(uint64(spec.EmbeddingLength))
	}
	return tensor.UnitScale
}

func (p runtimePolicy) outputLogitMultiplier(spec Spec) float32 {
	if !positiveFinite(spec.LogitScale) {
		return tensor.UnitScale
	}
	if p.LogitScale == logitScaleDirect {
		return spec.LogitScale
	}
	return tensor.UnitScale / spec.LogitScale
}

func (s Spec) resolvedAttentionScale(keyLength uint64) float32 {
	if positiveFinite(s.AttentionScale) {
		return s.AttentionScale
	}
	return hostmath.InvSqrt32(keyLength)
}

func (p runtimePolicy) normalizationPlan(spec Spec, profile ArchitectureProfile) normalizationPlan {
	operation := profile.Normalization
	pre := !profile.Has(architecturePostOnlyNorm)
	post := profile.Has(architecturePostNorm)
	if p.NormalizationPlacement == normalizationPlacementPostOnly {
		pre, post = false, true
	}
	bias := operation == normalizationLayer
	epsilon := spec.RMSNormEpsilon
	if operation == normalizationLayer || operation == NormalizationUnweightedLayer || operation == normalizationWeightOnlyLayer {
		epsilon = spec.LayerNormEpsilon
	}
	switch p.NormalizationBias {
	case normalizationBiasNever:
		bias = false
	case normalizationBiasAlways:
		bias = true
	}
	return normalizationPlan{
		Operation: operation, PreAttention: pre, PreFeedForward: pre,
		PostAttention: post, PostFeedForward: post, Bias: bias,
		RMSBias:        profile.DenseWeights.RMSNormBias,
		Epsilon:        epsilon,
		PostNormLayout: profile.PostNormLayout, FeedForwardLayout: profile.FFNNormLayout,
	}
}
