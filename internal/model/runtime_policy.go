package model

import (
	"overgo/internal/hostmath"
	"overgo/internal/tensor"
)

// EmbeddingScalePolicy: default input scaling.
type EmbeddingScalePolicy uint8

const (
	EmbeddingScaleIdentity EmbeddingScalePolicy = iota
	EmbeddingScaleSqrtWidth
)

// LogitScalePolicy: metadata scale direction.
type LogitScalePolicy uint8

const (
	LogitScaleReciprocal LogitScalePolicy = iota
	LogitScaleDirect
)

// NormalizationPlacementPolicy: block normalization placement.
type NormalizationPlacementPolicy uint8

const (
	NormalizationPlacementCapabilities NormalizationPlacementPolicy = iota
	NormalizationPlacementPostOnly
)

// NormalizationBiasPolicy: affine bias selection.
type NormalizationBiasPolicy uint8

const (
	NormalizationBiasAutomatic NormalizationBiasPolicy = iota
	NormalizationBiasNever
	NormalizationBiasAlways
)

// RuntimePolicy: scalar and normalization runtime decisions.
type RuntimePolicy struct {
	EmbeddingScale          EmbeddingScalePolicy
	LogitScale              LogitScalePolicy
	NormalizationPlacement  NormalizationPlacementPolicy
	NormalizationBias       NormalizationBiasPolicy
	Recurrent               RecurrentRuntimePolicy
	ActivationResidualScale float32
}

// RecurrentRuntimePolicy: recurrent scalar contract.
type RecurrentRuntimePolicy struct {
	HeadNormEpsilon       float32
	KeyNormEpsilon        float32
	PeriodicResidualScale float32
	TokenShiftCount       uint32
	InnerWidthMultiplier  uint32
}

func (p RecurrentRuntimePolicy) validInnerWidth(spec Spec) bool {
	return p.InnerWidthMultiplier == tensor.FirstOffset ||
		uint64(spec.SSMInnerSize) == uint64(p.InnerWidthMultiplier)*uint64(spec.EmbeddingLength)
}

func (p RuntimePolicy) inputEmbeddingScale(spec Spec) float32 {
	if positiveFinite(spec.EmbeddingScale) {
		return spec.EmbeddingScale
	}
	if p.EmbeddingScale == EmbeddingScaleSqrtWidth {
		return hostmath.Sqrt32(uint64(spec.EmbeddingLength))
	}
	return tensor.UnitScale
}

func (p RuntimePolicy) outputLogitMultiplier(spec Spec) float32 {
	if !positiveFinite(spec.LogitScale) {
		return tensor.UnitScale
	}
	if p.LogitScale == LogitScaleDirect {
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

func (p RuntimePolicy) normalizationPlan(spec Spec, profile ArchitectureProfile) NormalizationPlan {
	operation := profile.Normalization
	pre := !profile.Has(ArchitecturePostOnlyNorm)
	post := profile.Has(ArchitecturePostNorm)
	if p.NormalizationPlacement == NormalizationPlacementPostOnly {
		pre, post = false, true
	}
	bias := operation == NormalizationLayer
	epsilon := spec.RMSNormEpsilon
	if operation == NormalizationLayer || operation == NormalizationUnweightedLayer || operation == NormalizationWeightOnlyLayer {
		epsilon = spec.LayerNormEpsilon
	}
	switch p.NormalizationBias {
	case NormalizationBiasNever:
		bias = false
	case NormalizationBiasAlways:
		bias = true
	}
	return NormalizationPlan{
		Operation: operation, PreAttention: pre, PreFeedForward: pre,
		PostAttention: post, PostFeedForward: post, Bias: bias,
		RMSBias:        profile.DenseWeights.RMSNormBias,
		Epsilon:        epsilon,
		PostNormLayout: profile.PostNormLayout, FeedForwardLayout: profile.FFNNormLayout,
	}
}
