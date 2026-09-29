package model

type positionEmbeddingCatalogPolicy uint8

const (
	positionEmbeddingAbsent positionEmbeddingCatalogPolicy = iota
	positionEmbeddingRequired
	positionEmbeddingOptional
)

type tokenNormCatalogPolicy uint8

const (
	tokenNormAbsent tokenNormCatalogPolicy = iota
	tokenNormWeight
	tokenNormAffine
)

// WeightCatalogPolicy: root tensor-catalog topology.
type WeightCatalogPolicy uint8

const (
	weightCatalogLayered WeightCatalogPolicy = iota
	weightCatalogCompressedHyper
	weightCatalogTargetFeatures
	weightCatalogHiddenFusion
	weightCatalogPairedProjection
	weightCatalogEncoder
	weightCatalogAudioDecoder
	weightCatalogEncoderDecoder
)

// modelCatalogPolicy: model-level tensor inventory.
type modelCatalogPolicy struct {
	Weights                 WeightCatalogPolicy
	PositionEmbedding       positionEmbeddingCatalogPolicy
	TokenNorm               tokenNormCatalogPolicy
	TiedTokenEmbedding      bool
	RequireTokenTypes       bool
	SkipOutput              bool
	RequireOutputBias       bool
	OptionalOutputNormBias  bool
	SkipAttentionOutputBias bool
	DraftLayerOutputNorm    bool
}
