package model

// BaseRotaryValidationPolicy: base rotary invariant.
type BaseRotaryValidationPolicy uint8

const (
	BaseRotaryValidationDefault BaseRotaryValidationPolicy = iota
	BaseRotaryValidationHalfWidth
)

// EncoderValidationPolicy: encoder metadata contract.
type EncoderValidationPolicy uint8

const (
	EncoderValidationNone EncoderValidationPolicy = iota
	EncoderValidationBERT
	EncoderValidationJinaV2
	EncoderValidationJinaV3
	EncoderValidationNeoBERT
	EncoderValidationNomicBERT
	EncoderValidationNomicBERTMoE
)

// AttentionValidationPolicy: attention metadata contract.
type AttentionValidationPolicy uint8

const (
	AttentionValidationNone AttentionValidationPolicy = iota
	AttentionValidationChameleon
	AttentionValidationJais2
	AttentionValidationOpenELM
	AttentionValidationPLaMo3
	AttentionValidationDeci
	AttentionValidationGemma3
	AttentionValidationGemma3N
	AttentionValidationGemma4
	AttentionValidationGemma4Assistant
	AttentionValidationGemma2
	AttentionValidationOLMo2
	AttentionValidationCohere2
	AttentionValidationCohere2MoE
	AttentionValidationErnie45MoE
	AttentionValidationStableLM
	AttentionValidationPhi2
	AttentionValidationPhi3
	AttentionValidationPanguEmbedded
	AttentionValidationModernBERT
	AttentionValidationGemmaEmbedding
	AttentionValidationTalkie
	AttentionValidationApertus
	AttentionValidationGPTNeoX
	AttentionValidationQwen
	AttentionValidationChatGLM
	AttentionValidationCogVLM
	AttentionValidationHunyuan
	AttentionValidationGLM4
	AttentionValidationGLM4MoE
	AttentionValidationEXAOne4
	AttentionValidationFalcon
	AttentionValidationRefact
)

// MLAValidationPolicy: MLA/DSA family metadata contract.
type MLAValidationPolicy uint8

const (
	MLAValidationNone MLAValidationPolicy = iota
	MLAValidationDeepSeek32
	MLAValidationDeepSeek4
	MLAValidationKimiLinear
	MLAValidationMistral3
	MLAValidationMiniCPM3
)

// RecurrentValidationPolicy: recurrent-family metadata contract.
type RecurrentValidationPolicy uint8

const (
	RecurrentValidationNone RecurrentValidationPolicy = iota
	RecurrentValidationWavTokenizer
	RecurrentValidationDFlash
	RecurrentValidationEagle3
	RecurrentValidationMamba
	RecurrentValidationMamba2
	RecurrentValidationFalconH1
	RecurrentValidationRWKV6
	RecurrentValidationRWKV6Qwen2
	RecurrentValidationRWKV7
	RecurrentValidationARWKV7
	RecurrentValidationJamba
	RecurrentValidationGraniteHybrid
	RecurrentValidationPLaMo2
	RecurrentValidationNemotronH
	RecurrentValidationNemotronHMoE
)

// HybridValidationPolicy: hybrid/draft metadata contract.
type HybridValidationPolicy uint8

const (
	HybridValidationNone HybridValidationPolicy = iota
	HybridValidationQwen3Next
	HybridValidationQwen35
	HybridValidationQwen35MoE
	HybridValidationQwen3MoE
	HybridValidationGroveMoE
	HybridValidationMiMo2
	HybridValidationStep35
	HybridValidationLLaDAMoE
	HybridValidationQwen2MoE
	HybridValidationArctic
	HybridValidationBailingMoE
	HybridValidationDeepSeek
	HybridValidationGranite
	HybridValidationGraniteMoE
	HybridValidationGraniteHybrid
	HybridValidationDBRX
	HybridValidationGrok
	HybridValidationMellum
	HybridValidationHunyuanMoE
	HybridValidationHYV3
	HybridValidationDeepSeek2OCR
	HybridValidationSmallThinker
	HybridValidationDOTS1
	HybridValidationMiniMaxM2
	HybridValidationBailingMoE2
	HybridValidationOLMoE
	HybridValidationLlama
	HybridValidationRopeScaling
	HybridValidationLlama4
	HybridValidationGPTOSS
	HybridValidationPhiMoE
	HybridValidationLaguna
	HybridValidationAFMoE
	HybridValidationEXAOneMoE
	HybridValidationLFM2
	HybridValidationLFM2MoE
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
	ProbeExpertCount      bool
	RopeFrequencyOptional bool
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

func (p ValidationPolicy) recurrentOneOf(policies ...RecurrentValidationPolicy) bool {
	return policyOneOf(p.Recurrent, policies...)
}

func (p ValidationPolicy) encoderOneOf(policies ...EncoderValidationPolicy) bool {
	return policyOneOf(p.Encoder, policies...)
}

func (p ValidationPolicy) requiresExpertMetadata(declaredExperts bool) bool {
	if p.Encoder == EncoderValidationNomicBERTMoE ||
		p.attentionOneOf(
			AttentionValidationCohere2MoE, AttentionValidationErnie45MoE,
			AttentionValidationGLM4MoE,
		) || p.MLA == MLAValidationKimiLinear || p.Recurrent == RecurrentValidationJamba {
		return true
	}
	if p.Hybrid == HybridValidationGranite {
		return declaredExperts
	}
	return p.hybridOneOf(
		HybridValidationQwen3Next, HybridValidationQwen35MoE,
		HybridValidationQwen3MoE, HybridValidationGroveMoE,
		HybridValidationMiMo2, HybridValidationStep35, HybridValidationLLaDAMoE,
		HybridValidationQwen2MoE, HybridValidationArctic, HybridValidationBailingMoE,
		HybridValidationDeepSeek, HybridValidationGraniteMoE, HybridValidationDBRX,
		HybridValidationGrok, HybridValidationMellum, HybridValidationHunyuanMoE,
		HybridValidationHYV3, HybridValidationDeepSeek2OCR,
		HybridValidationSmallThinker, HybridValidationDOTS1,
		HybridValidationMiniMaxM2, HybridValidationBailingMoE2,
		HybridValidationOLMoE, HybridValidationLlama4, HybridValidationGPTOSS,
		HybridValidationPhiMoE, HybridValidationLaguna, HybridValidationAFMoE,
		HybridValidationEXAOneMoE, HybridValidationLFM2MoE,
	)
}

func (p ValidationPolicy) optionalRopeBase() bool {
	return p.attentionOneOf(
		AttentionValidationGPTNeoX, AttentionValidationFalcon,
		AttentionValidationGemmaEmbedding, AttentionValidationModernBERT,
	) || p.Hybrid == HybridValidationDeepSeek2OCR ||
		p.encoderOneOf(
			EncoderValidationJinaV3, EncoderValidationNeoBERT,
			EncoderValidationNomicBERT, EncoderValidationNomicBERTMoE,
		) || p.MLA == MLAValidationMiniCPM3
}

func (p ValidationPolicy) supportsYaRN() bool {
	return p.hybridOneOf(
		HybridValidationLaguna, HybridValidationGrok, HybridValidationMellum,
		HybridValidationLlama, HybridValidationRopeScaling,
	)
}

func policyOneOf[T comparable](policy T, candidates ...T) bool {
	for _, candidate := range candidates {
		if policy == candidate {
			return true
		}
	}
	return false
}
