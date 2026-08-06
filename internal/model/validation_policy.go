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
}
