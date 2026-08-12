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

// DraftWeightCatalogPolicy: draft model-level tensor topology.
type DraftWeightCatalogPolicy uint8

const (
	DraftWeightCatalogNone DraftWeightCatalogPolicy = iota
	DraftWeightCatalogTargetFeatures
	DraftWeightCatalogHiddenFusion
	DraftWeightCatalogPairedProjection
)

// ModelCatalogPolicy: model-level tensor inventory.
type ModelCatalogPolicy struct {
	Draft                   DraftWeightCatalogPolicy
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
