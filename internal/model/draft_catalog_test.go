package model

import (
	"testing"

	"overgo/internal/gguf"
)

func TestWeightsDraftCatalogs(t *testing.T) {
	info := func(name string) gguf.TensorInfo { return gguf.TensorInfo{Name: name} }
	weights := Weights{
		Layers: []LayerWeights{{AttentionNorm: pointerTensorInfo(info("trunk"))}},
		SingleCatalogDraft: &SingleDraftWeights{
			Layer: LayerWeights{AttentionNorm: pointerTensorInfo(info("qwen_layer"))}, EHProjection: info("qwen_eh"),
		},
		AppendedSingleDraft: []AppendedDraftWeights{{
			Layer: LayerWeights{AttentionNorm: pointerTensorInfo(info("next_layer"))}, EHProjection: info("next_eh"),
			LayerOutputNorm: pointerTensorInfo(info("next_layer_norm")),
		}},
	}
	catalogs := weights.DraftCatalogs()
	if len(catalogs) != 2 || catalogs[0].Kind != DraftSingleCatalog || catalogs[1].Kind != DraftAppendedSingle ||
		catalogs[1].LayerOutputNorm == nil || catalogs[1].LayerOutputNorm.Name != "next_layer_norm" {
		t.Fatalf("draft catalogs = %+v", catalogs)
	}
}

func pointerTensorInfo(value gguf.TensorInfo) *gguf.TensorInfo { return &value }
