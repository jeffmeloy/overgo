package model

type encoderGraphKind uint8

const (
	encoderGraphNone encoderGraphKind = iota
	encoderGraphBERT
	encoderGraphJinaV2
	encoderGraphJinaV3
	encoderGraphNomic
	encoderGraphNomicMoE
	encoderGraphModernBERT
	encoderGraphGemmaEmbedding
	encoderGraphT5
	encoderGraphT5Encoder
)

// EncoderGraphPolicy: encoder leaf semantics.
type EncoderGraphPolicy struct {
	Kind encoderGraphKind
}

func (p EncoderGraphPolicy) bertFamily() bool {
	return p.Kind >= encoderGraphBERT && p.Kind <= encoderGraphNomicMoE
}

func (p EncoderGraphPolicy) usesRoPE() bool {
	return p.Kind == encoderGraphJinaV3 || p.Kind == encoderGraphNomic || p.Kind == encoderGraphNomicMoE
}

func (p EncoderGraphPolicy) usesExperts() bool {
	return p.Kind == encoderGraphJinaV3 || p.Kind == encoderGraphNomicMoE
}

type mlaVariantPolicy uint8

const (
	mlaVariantDefault mlaVariantPolicy = iota
	mlaVariantPLM
	mlaVariantMiniCPM3
	mlaVariantDeepSeek32
	mlaVariantKimi
)
