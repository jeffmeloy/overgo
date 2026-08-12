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
	WeightCatalogLayered WeightCatalogPolicy = iota
	WeightCatalogCompressedHyper
	WeightCatalogTargetFeatures
	WeightCatalogHiddenFusion
	WeightCatalogPairedProjection
	WeightCatalogEncoder
	WeightCatalogAudioDecoder
	WeightCatalogEncoderDecoder
)

// ModelCatalogPolicy: model-level tensor inventory.
type ModelCatalogPolicy struct {
	Weights                 WeightCatalogPolicy
	PositionEmbedding       positionEmbeddingCatalogPolicy
	TokenNorm               tokenNormCatalogPolicy
	TokenEmbeddingFallback  bool
	RequireTokenTypes       bool
	SkipOutput              bool
	RequireOutputBias       bool
	OptionalOutputNormBias  bool
	SkipAttentionOutputBias bool
	DraftLayerOutputNorm    bool
}
