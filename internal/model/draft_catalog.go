package model

import "llamacpp2go/internal/gguf"

// DraftWeightCatalog: family-neutral draft tensor view.
type DraftWeightCatalog struct {
	Kind            DraftKind
	Layer           LayerWeights
	EHProjection    gguf.TensorInfo
	EmbeddingNorm   gguf.TensorInfo
	HiddenNorm      gguf.TensorInfo
	TokenEmbedding  *gguf.TensorInfo
	LayerOutputNorm *gguf.TensorInfo
	OutputNorm      *gguf.TensorInfo
	Output          *gguf.TensorInfo
}

// DraftCatalogs: ordered speculative-head catalogs.
func (w Weights) DraftCatalogs() []DraftWeightCatalog {
	count := len(w.Step35MTP) + len(w.HYV3MTP) + len(w.NextNMTP)
	if w.Qwen35MTP != nil {
		count++
	}
	if w.Cohere2MTP != nil {
		count++
	}
	result := make([]DraftWeightCatalog, 0, count)
	appendCommon := func(kind DraftKind, layer LayerWeights, eh, embedding, hidden gguf.TensorInfo,
		token, layerOutput, outputNorm, output *gguf.TensorInfo,
	) {
		result = append(result, DraftWeightCatalog{
			Kind: kind, Layer: layer, EHProjection: eh, EmbeddingNorm: embedding, HiddenNorm: hidden,
			TokenEmbedding: token, LayerOutputNorm: layerOutput, OutputNorm: outputNorm, Output: output,
		})
	}
	if item := w.Qwen35MTP; item != nil {
		appendCommon(DraftQwen35MTP, item.Layer, item.EHProjection, item.EmbeddingNorm, item.HiddenNorm,
			item.TokenEmbedding, nil, item.OutputNorm, item.Output)
	}
	for _, item := range w.Step35MTP {
		appendCommon(DraftStep35MTP, item.Layer, item.EHProjection, item.EmbeddingNorm, item.HiddenNorm,
			item.TokenEmbedding, item.LayerOutputNorm, item.OutputNorm, item.Output)
	}
	for _, item := range w.HYV3MTP {
		appendCommon(DraftHYV3MTP, item.Layer, item.EHProjection, item.EmbeddingNorm, item.HiddenNorm,
			item.TokenEmbedding, item.LayerOutputNorm, item.OutputNorm, item.Output)
	}
	for _, item := range w.NextNMTP {
		appendCommon(DraftNextNMTP, item.Layer, item.EHProjection, item.EmbeddingNorm, item.HiddenNorm,
			item.TokenEmbedding, item.LayerOutputNorm, item.OutputNorm, item.Output)
	}
	if item := w.Cohere2MTP; item != nil {
		appendCommon(DraftCohere2MTP, item.Layer, item.EHProjection, item.EmbeddingNorm, item.HiddenNorm,
			item.TokenEmbedding, nil, item.OutputNorm, item.Output)
	}
	return result
}

// LayerCatalog: encoder, trunk, and draft layers.
func (w Weights) LayerCatalog() []LayerWeights {
	drafts := w.DraftCatalogs()
	result := make([]LayerWeights, 0, len(w.EncoderLayers)+len(w.Layers)+len(drafts))
	result = append(result, w.EncoderLayers...)
	result = append(result, w.Layers...)
	for _, draft := range drafts {
		result = append(result, draft.Layer)
	}
	return result
}
