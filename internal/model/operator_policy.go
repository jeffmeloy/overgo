package model

// EncoderOperatorPolicy: encoder leaf semantics.
type EncoderOperatorPolicy uint8

const (
	encoderOperatorNone EncoderOperatorPolicy = iota
	encoderOperatorPostNormAbsolute
	encoderOperatorPostNormALiBi
	encoderOperatorPreNormRoPEExperts
	encoderOperatorPreNormRoPEGated
	encoderOperatorPreNormRoPEGatedExperts
	encoderOperatorFusedQKVSliding
	encoderOperatorQKNormProjection
	encoderOperatorRelativeEncoderDecoder
	encoderOperatorRelativeEncoder
)

func (p EncoderOperatorPolicy) bidirectionalProjection() bool {
	return p >= encoderOperatorPostNormAbsolute && p <= encoderOperatorPreNormRoPEGatedExperts
}

func (p EncoderOperatorPolicy) usesRoPE() bool {
	return p >= encoderOperatorPreNormRoPEExperts && p <= encoderOperatorPreNormRoPEGatedExperts
}

func (p EncoderOperatorPolicy) usesExperts() bool {
	return p == encoderOperatorPreNormRoPEExperts || p == encoderOperatorPreNormRoPEGatedExperts
}

func (p EncoderOperatorPolicy) usesALiBiQKNorm() bool { return p == encoderOperatorPostNormALiBi }

type mlaVariantPolicy uint8

const (
	mlaVariantDefault mlaVariantPolicy = iota
	mlaVariantPLM
	mlaVariantMiniCPM3
	mlaVariantDeepSeek32
	mlaVariantKimi
)
