package model

import "overgo/internal/gguf"

// DraftWeightCatalog: family-neutral draft tensor view.
type DraftWeightCatalog struct {
	Kind            DraftKind
	MTPOnly         bool
	Layer           LayerWeights
	EHProjection    gguf.TensorInfo
	EmbeddingNorm   gguf.TensorInfo
	HiddenNorm      gguf.TensorInfo
	TokenEmbedding  *gguf.TensorInfo
	LayerOutputNorm *gguf.TensorInfo
	OutputNorm      *gguf.TensorInfo
	Output          *gguf.TensorInfo
}

func draftWeightCatalog(
	kind DraftKind,
	mtpOnly bool,
	layer LayerWeights,
	eh, embedding, hidden gguf.TensorInfo,
	token, layerOutput, outputNorm, output *gguf.TensorInfo,
) DraftWeightCatalog {
	return DraftWeightCatalog{
		Kind: kind, MTPOnly: mtpOnly, Layer: layer,
		EHProjection: eh, EmbeddingNorm: embedding, HiddenNorm: hidden,
		TokenEmbedding: token, LayerOutputNorm: layerOutput, OutputNorm: outputNorm, Output: output,
	}
}

// DraftCatalog returns one indexed compiled draft tensor catalog.
func (w Weights) DraftCatalog(kind DraftKind, offset uint32) (DraftWeightCatalog, bool) {
	switch kind {
	case DraftQwen35MTP:
		if offset == 0 && w.Qwen35MTP != nil {
			item := w.Qwen35MTP
			return draftWeightCatalog(kind, item.MTPOnly, item.Layer, item.EHProjection, item.EmbeddingNorm,
				item.HiddenNorm, item.TokenEmbedding, nil, item.OutputNorm, item.Output), true
		}
	case DraftStep35MTP:
		if offset < uint32(len(w.Step35MTP)) {
			item := w.Step35MTP[offset]
			return draftWeightCatalog(kind, false, item.Layer, item.EHProjection, item.EmbeddingNorm,
				item.HiddenNorm, item.TokenEmbedding, item.LayerOutputNorm, item.OutputNorm, item.Output), true
		}
	case DraftHYV3MTP:
		if offset < uint32(len(w.HYV3MTP)) {
			item := w.HYV3MTP[offset]
			return draftWeightCatalog(kind, false, item.Layer, item.EHProjection, item.EmbeddingNorm,
				item.HiddenNorm, item.TokenEmbedding, item.LayerOutputNorm, item.OutputNorm, item.Output), true
		}
	case DraftNextNMTP:
		if offset < uint32(len(w.NextNMTP)) {
			item := w.NextNMTP[offset]
			return draftWeightCatalog(kind, false, item.Layer, item.EHProjection, item.EmbeddingNorm,
				item.HiddenNorm, item.TokenEmbedding, item.LayerOutputNorm, item.OutputNorm, item.Output), true
		}
	case DraftCohere2MTP:
		if offset == 0 && w.Cohere2MTP != nil {
			item := w.Cohere2MTP
			return draftWeightCatalog(kind, item.MTPOnly, item.Layer, item.EHProjection, item.EmbeddingNorm,
				item.HiddenNorm, item.TokenEmbedding, nil, item.OutputNorm, item.Output), true
		}
	}
	return DraftWeightCatalog{}, false
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
	for _, kind := range []DraftKind{DraftQwen35MTP, DraftStep35MTP, DraftHYV3MTP, DraftNextNMTP, DraftCohere2MTP} {
		for offset := uint32(0); ; offset++ {
			catalog, ok := w.DraftCatalog(kind, offset)
			if !ok {
				break
			}
			result = append(result, catalog)
		}
	}
	return result
}
