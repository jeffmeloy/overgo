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
	BaseRotary            BaseRotaryValidationPolicy
	Encoder               EncoderValidationPolicy
	Attention             AttentionValidationPolicy
	MLA                   MLAValidationPolicy
}
