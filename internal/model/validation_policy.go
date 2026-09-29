package model

import "slices"

// BaseRotaryValidationPolicy: base rotary invariant.
type BaseRotaryValidationPolicy uint8

const (
	baseRotaryValidationDefault BaseRotaryValidationPolicy = iota
	baseRotaryValidationPaired
)

// ExpertMetadataPolicy: expert metadata admission.
type ExpertMetadataPolicy uint8

const (
	expertMetadataNone ExpertMetadataPolicy = iota
	expertMetadataRequired
	expertMetadataWhenDeclared
)

// EncoderValidationPolicy: encoder metadata contract.
type EncoderValidationPolicy uint8

const (
	encoderValidationNone EncoderValidationPolicy = iota
	encoderValidationTokenTypesMatchingHeads
	encoderValidationTokenTypesALiBi
	encoderValidationRotaryOptionalExperts
	encoderValidationFullRotary
	encoderValidationRotary
	encoderValidationRotaryPeriodicExperts
)

// AttentionValidationPolicy: attention metadata contract.
type AttentionValidationPolicy uint8

const (
	attentionValidationNone AttentionValidationPolicy = iota
	attentionValidationQKNormEpsilon
	attentionValidationMatchingAttentionKVHeads
	attentionValidationPerLayerAttentionAndFeedForward
	attentionValidationPerLayerSlidingAttention
	attentionValidationSparseLayerAttention
	attentionValidationScaledSlidingAttention
	attentionValidationSharedKVAlternatingState
	attentionValidationPerLayerDualRotaryAttention
	attentionValidationTargetHiddenDualRotaryAttention
	attentionValidationRequiredSlidingFrequency
	attentionValidationOptionalSlidingFrequency
	attentionValidationRequiredSlidingRotary
	attentionValidationRequiredSlidingRotaryExperts
	attentionValidationPeriodicExperts
	attentionValidationPartialRotaryRequired
	attentionValidationPartialRotaryFixed
	attentionValidationScaledPartialRotary
	attentionValidationFullScaledRotary
	attentionValidationFullHeadSlidingRotary
	attentionValidationSlidingRotaryEmbeddingProjection
	attentionValidationFullRotary
	attentionValidationFullScaledRotaryXIELU
	attentionValidationOptionalRotaryBase
	attentionValidationHalvedFeedForward
	attentionValidationMultiAxisAttention
	attentionValidationVisualExpertAttention
	attentionValidationLayerwiseQKNorm
	attentionValidationOptionalRopeSections
	attentionValidationOptionalRopeSectionsExperts
	attentionValidationSharedKVAttention
	attentionValidationOptionalRotaryBaseGQA
	attentionValidationOptionalExperts
)

// MLAValidationPolicy: MLA/DSA metadata relation.
type MLAValidationPolicy uint8

const (
	mlaValidationNone MLAValidationPolicy = iota
	mlaValidationSparseLatentIndexer
	mlaValidationCompressedHyper
	mlaValidationHybridLinearAttention
	mlaValidationOptionalExpertsLatent
	mlaValidationScaledLatent
)

// RecurrentValidationPolicy: recurrent metadata relation.
type RecurrentValidationPolicy uint8

const (
	recurrentValidationNone RecurrentValidationPolicy = iota
	recurrentValidationAudioDecoder
	recurrentValidationTargetLayerBlock
	recurrentValidationSingleBlockTarget
	recurrentValidationUngroupedStateSpace
	recurrentValidationGroupedStateSpace
	recurrentValidationGroupedStateSpaceAttention
	recurrentValidationTimeMixV6
	recurrentValidationTimeMixV6SharedKV
	recurrentValidationTimeMixV7Gated
	recurrentValidationTimeMixV7
	recurrentValidationStateSpaceAttentionExperts
	recurrentValidationGroupedStateSpaceOptionalExperts
	recurrentValidationUngroupedScheduledStateSpace
	recurrentValidationScheduledStateSpaceDense
	recurrentValidationScheduledStateSpaceExperts
)

// HybridValidationPolicy: hybrid/draft metadata contract.
type HybridValidationPolicy uint8

const (
	hybridValidationNone HybridValidationPolicy = iota
	hybridValidationAlternatingGatedDelta
	hybridValidationAlternatingGatedDeltaHybrid
	hybridValidationAlternatingGatedDeltaExperts
	hybridValidationSharedExperts
	hybridValidationSparseSharedExperts
	hybridValidationSigmoidExperts
	hybridValidationCompressedHyperDraft
	hybridValidationNonCausalExperts
	hybridValidationNormalizedSharedExperts
	hybridValidationRoutedExperts
	hybridValidationProductSharedExperts
	hybridValidationProductExperts
	hybridValidationScaledDense
	hybridValidationScaledExperts
	hybridValidationScaledStateSpace
	hybridValidationModelFeedForwardExperts
	hybridValidationSharedExpertNorm
	hybridValidationRequiredExpertFeedForward
	hybridValidationSharedExpertProduct
	hybridValidationMultiHeadDraft
	hybridValidationFullRotaryVision
	hybridValidationDualExpertProduct
	hybridValidationWeightedExpertProduct
	hybridValidationScaledSigmoidExperts
	hybridValidationScaledSharedExperts
	hybridValidationFullHeadExperts
	hybridValidationOptionalExperts
	hybridValidationExtendedRotary
	hybridValidationChunkedExperts
	hybridValidationSelectedSoftmaxExperts
	hybridValidationBasicScaledExperts
	hybridValidationPerLayerYaRNExperts
	hybridValidationSlidingSigmoidExperts
	hybridValidationSlidingSharedExperts
	hybridValidationAlternatingShortConvolution
	hybridValidationAlternatingShortConvolutionExperts
)

// validationPolicy: architecture-owned metadata invariants.
type validationPolicy struct {
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

func (p validationPolicy) attentionOneOf(policies ...AttentionValidationPolicy) bool {
	return policyOneOf(p.Attention, policies...)
}

func (p validationPolicy) hybridOneOf(policies ...HybridValidationPolicy) bool {
	return policyOneOf(p.Hybrid, policies...)
}

func (p validationPolicy) encoderOneOf(policies ...EncoderValidationPolicy) bool {
	return policyOneOf(p.Encoder, policies...)
}

func (p validationPolicy) optionalRopeBase() bool {
	return p.attentionOneOf(
		attentionValidationOptionalRotaryBase, attentionValidationOptionalRotaryBaseGQA,
		attentionValidationSlidingRotaryEmbeddingProjection, attentionValidationFullHeadSlidingRotary,
	) || p.Hybrid == hybridValidationFullRotaryVision ||
		p.encoderOneOf(
			encoderValidationRotaryOptionalExperts, encoderValidationFullRotary,
			encoderValidationRotary, encoderValidationRotaryPeriodicExperts,
		) || p.MLA == mlaValidationScaledLatent
}

func (p validationPolicy) supportsYaRN() bool {
	return p.hybridOneOf(
		hybridValidationPerLayerYaRNExperts, hybridValidationSharedExpertNorm, hybridValidationRequiredExpertFeedForward,
		hybridValidationOptionalExperts, hybridValidationExtendedRotary,
	)
}

func policyOneOf[T comparable](policy T, candidates ...T) bool {
	return slices.Contains(candidates, policy)
}
