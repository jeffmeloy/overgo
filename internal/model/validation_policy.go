package model

import "slices"

// BaseRotaryValidationPolicy: base rotary invariant.
type BaseRotaryValidationPolicy uint8

const (
	BaseRotaryValidationDefault BaseRotaryValidationPolicy = iota
	BaseRotaryValidationPaired
)

// ExpertMetadataPolicy: expert metadata admission.
type ExpertMetadataPolicy uint8

const (
	ExpertMetadataNone ExpertMetadataPolicy = iota
	ExpertMetadataRequired
	ExpertMetadataWhenDeclared
)

// EncoderValidationPolicy: encoder metadata contract.
type EncoderValidationPolicy uint8

const (
	EncoderValidationNone EncoderValidationPolicy = iota
	EncoderValidationTokenTypesMatchingHeads
	EncoderValidationTokenTypesALiBi
	EncoderValidationRotaryOptionalExperts
	EncoderValidationFullRotary
	EncoderValidationRotary
	EncoderValidationRotaryPeriodicExperts
)

// AttentionValidationPolicy: attention metadata contract.
type AttentionValidationPolicy uint8

const (
	AttentionValidationNone AttentionValidationPolicy = iota
	AttentionValidationQKNormEpsilon
	AttentionValidationMatchingAttentionKVHeads
	AttentionValidationPerLayerAttentionAndFeedForward
	AttentionValidationPerLayerSlidingAttention
	AttentionValidationSparseLayerAttention
	AttentionValidationScaledSlidingAttention
	AttentionValidationSharedKVAlternatingState
	AttentionValidationPerLayerDualRotaryAttention
	AttentionValidationTargetHiddenDualRotaryAttention
	AttentionValidationRequiredSlidingFrequency
	AttentionValidationOptionalSlidingFrequency
	AttentionValidationRequiredSlidingRotary
	AttentionValidationRequiredSlidingRotaryExperts
	AttentionValidationPeriodicExperts
	AttentionValidationPartialRotaryRequired
	AttentionValidationPartialRotaryFixed
	AttentionValidationScaledPartialRotary
	AttentionValidationFullScaledRotary
	AttentionValidationFullHeadSlidingRotary
	AttentionValidationSlidingRotaryEmbeddingProjection
	AttentionValidationFullRotary
	AttentionValidationFullScaledRotaryXIELU
	AttentionValidationOptionalRotaryBase
	AttentionValidationHalvedFeedForward
	AttentionValidationMultiAxisAttention
	AttentionValidationVisualExpertAttention
	AttentionValidationLayerwiseQKNorm
	AttentionValidationOptionalRopeSections
	AttentionValidationOptionalRopeSectionsExperts
	AttentionValidationSharedKVAttention
	AttentionValidationOptionalRotaryBaseGQA
	AttentionValidationOptionalExperts
)

// MLAValidationPolicy: MLA/DSA metadata relation.
type MLAValidationPolicy uint8

const (
	MLAValidationNone MLAValidationPolicy = iota
	MLAValidationSparseLatentIndexer
	MLAValidationCompressedHyper
	MLAValidationHybridLinearAttention
	MLAValidationOptionalExpertsLatent
	MLAValidationScaledLatent
)

// RecurrentValidationPolicy: recurrent metadata relation.
type RecurrentValidationPolicy uint8

const (
	RecurrentValidationNone RecurrentValidationPolicy = iota
	RecurrentValidationAudioDecoder
	RecurrentValidationTargetLayerBlock
	RecurrentValidationSingleBlockTarget
	RecurrentValidationUngroupedStateSpace
	RecurrentValidationGroupedStateSpace
	RecurrentValidationGroupedStateSpaceAttention
	RecurrentValidationTimeMixV6
	RecurrentValidationTimeMixV6SharedKV
	RecurrentValidationTimeMixV7Gated
	RecurrentValidationTimeMixV7
	RecurrentValidationStateSpaceAttentionExperts
	RecurrentValidationGroupedStateSpaceOptionalExperts
	RecurrentValidationUngroupedScheduledStateSpace
	RecurrentValidationScheduledStateSpaceDense
	RecurrentValidationScheduledStateSpaceExperts
)

// HybridValidationPolicy: hybrid/draft metadata contract.
type HybridValidationPolicy uint8

const (
	HybridValidationNone HybridValidationPolicy = iota
	HybridValidationAlternatingGatedDelta
	HybridValidationAlternatingGatedDeltaHybrid
	HybridValidationAlternatingGatedDeltaExperts
	HybridValidationSharedExperts
	HybridValidationSparseSharedExperts
	HybridValidationSigmoidExperts
	HybridValidationCompressedHyperDraft
	HybridValidationNonCausalExperts
	HybridValidationNormalizedSharedExperts
	HybridValidationRoutedExperts
	HybridValidationProductSharedExperts
	HybridValidationProductExperts
	HybridValidationScaledDense
	HybridValidationScaledExperts
	HybridValidationScaledStateSpace
	HybridValidationModelFeedForwardExperts
	HybridValidationSharedExpertNorm
	HybridValidationRequiredExpertFeedForward
	HybridValidationSharedExpertProduct
	HybridValidationMultiHeadDraft
	HybridValidationFullRotaryVision
	HybridValidationDualExpertProduct
	HybridValidationWeightedExpertProduct
	HybridValidationScaledSigmoidExperts
	HybridValidationScaledSharedExperts
	HybridValidationFullHeadExperts
	HybridValidationOptionalExperts
	HybridValidationExtendedRotary
	HybridValidationChunkedExperts
	HybridValidationSelectedSoftmaxExperts
	HybridValidationBasicScaledExperts
	HybridValidationPerLayerYaRNExperts
	HybridValidationSlidingSigmoidExperts
	HybridValidationSlidingSharedExperts
	HybridValidationAlternatingShortConvolution
	HybridValidationAlternatingShortConvolutionExperts
)

// ValidationPolicy: architecture-owned metadata invariants.
type ValidationPolicy struct {
	AttentionFree         bool
	ClampQKV              bool
	RequireLogitScale     bool
	RelativeAttention     bool
	RequireDecoder        bool
	MultiAxisRoPE         bool
	BoundDeepstack        bool
	RopeFrequencyOptional bool
	QLoRARankOptional     bool
	RequiredBlockCount    uint32
	AlternateBlockCount   uint32
	SlidingPeriod         uint32
	ExpertMetadata        ExpertMetadataPolicy
	BaseRotary            BaseRotaryValidationPolicy
	Encoder               EncoderValidationPolicy
	Attention             AttentionValidationPolicy
	MLA                   MLAValidationPolicy
	Recurrent             RecurrentValidationPolicy
	Hybrid                HybridValidationPolicy
}

func (p ValidationPolicy) attentionOneOf(policies ...AttentionValidationPolicy) bool {
	return policyOneOf(p.Attention, policies...)
}

func (p ValidationPolicy) hybridOneOf(policies ...HybridValidationPolicy) bool {
	return policyOneOf(p.Hybrid, policies...)
}

func (p ValidationPolicy) encoderOneOf(policies ...EncoderValidationPolicy) bool {
	return policyOneOf(p.Encoder, policies...)
}

func (p ValidationPolicy) optionalRopeBase() bool {
	return p.attentionOneOf(
		AttentionValidationOptionalRotaryBase, AttentionValidationOptionalRotaryBaseGQA,
		AttentionValidationSlidingRotaryEmbeddingProjection, AttentionValidationFullHeadSlidingRotary,
	) || p.Hybrid == HybridValidationFullRotaryVision ||
		p.encoderOneOf(
			EncoderValidationRotaryOptionalExperts, EncoderValidationFullRotary,
			EncoderValidationRotary, EncoderValidationRotaryPeriodicExperts,
		) || p.MLA == MLAValidationScaledLatent
}

func (p ValidationPolicy) supportsYaRN() bool {
	return p.hybridOneOf(
		HybridValidationPerLayerYaRNExperts, HybridValidationSharedExpertNorm, HybridValidationRequiredExpertFeedForward,
		HybridValidationOptionalExperts, HybridValidationExtendedRotary,
	)
}

func policyOneOf[T comparable](policy T, candidates ...T) bool {
	return slices.Contains(candidates, policy)
}
