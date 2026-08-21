package model

import (
	"overgo/internal/gguf"
	"overgo/internal/tensor"
)

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
	case DraftSingleCatalog:
		if offset == tensor.FirstOffset && w.SingleCatalogDraft != nil {
			item := w.SingleCatalogDraft
			return draftWeightCatalog(kind, item.MTPOnly, item.Layer, item.EHProjection, item.EmbeddingNorm,
				item.HiddenNorm, item.TokenEmbedding, nil, item.OutputNorm, item.Output), true
		}
	case DraftAppendedMultiCarry:
		if offset < uint32(len(w.AppendedMultiCarryDraft)) {
			item := w.AppendedMultiCarryDraft[offset]
			return draftWeightCatalog(kind, false, item.Layer, item.EHProjection, item.EmbeddingNorm,
				item.HiddenNorm, item.TokenEmbedding, item.LayerOutputNorm, item.OutputNorm, item.Output), true
		}
	case DraftAppendedMulti:
		if offset < uint32(len(w.AppendedMultiDraft)) {
			item := w.AppendedMultiDraft[offset]
			return draftWeightCatalog(kind, false, item.Layer, item.EHProjection, item.EmbeddingNorm,
				item.HiddenNorm, item.TokenEmbedding, item.LayerOutputNorm, item.OutputNorm, item.Output), true
		}
	case DraftAppendedSingle:
		if offset < uint32(len(w.AppendedSingleDraft)) {
			item := w.AppendedSingleDraft[offset]
			return draftWeightCatalog(kind, false, item.Layer, item.EHProjection, item.EmbeddingNorm,
				item.HiddenNorm, item.TokenEmbedding, item.LayerOutputNorm, item.OutputNorm, item.Output), true
		}
	case DraftOptionalSingleCatalog:
		if offset == tensor.FirstOffset && w.OptionalCatalogDraft != nil {
			item := w.OptionalCatalogDraft
			return draftWeightCatalog(kind, item.MTPOnly, item.Layer, item.EHProjection, item.EmbeddingNorm,
				item.HiddenNorm, item.TokenEmbedding, nil, item.OutputNorm, item.Output), true
		}
	}
	return DraftWeightCatalog{}, false
}

// DraftCatalogs: ordered speculative-head catalogs.
func (w Weights) DraftCatalogs() []DraftWeightCatalog {
	count := len(w.AppendedMultiCarryDraft) + len(w.AppendedMultiDraft) + len(w.AppendedSingleDraft)
	if w.SingleCatalogDraft != nil {
		count++
	}
	if w.OptionalCatalogDraft != nil {
		count++
	}
	result := make([]DraftWeightCatalog, 0, count)
	for _, kind := range []DraftKind{DraftSingleCatalog, DraftAppendedMultiCarry, DraftAppendedMulti, DraftAppendedSingle, DraftOptionalSingleCatalog} {
		for offset := uint32(tensor.FirstOffset); ; offset++ {
			catalog, ok := w.DraftCatalog(kind, offset)
			if !ok {
				break
			}
			result = append(result, catalog)
		}
	}
	return result
}
