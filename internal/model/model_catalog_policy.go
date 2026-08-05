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

// ModelCatalogPolicy: model-level tensor inventory.
type ModelCatalogPolicy struct {
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
