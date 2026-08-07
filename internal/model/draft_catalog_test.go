package model

import (
	"testing"

	"overgo/internal/gguf"
)

func TestWeightsDraftCatalogs(t *testing.T) {
	info := func(name string) gguf.TensorInfo { return gguf.TensorInfo{Name: name} }
	weights := Weights{
		Layers: []LayerWeights{{AttentionNorm: info("trunk")}},
		Qwen35MTP: &Qwen35MTPWeights{
			Layer: LayerWeights{AttentionNorm: info("qwen_layer")}, EHProjection: info("qwen_eh"),
		},
		NextNMTP: []Step35MTPWeights{{
			Layer: LayerWeights{AttentionNorm: info("next_layer")}, EHProjection: info("next_eh"),
			LayerOutputNorm: pointerTensorInfo(info("next_layer_norm")),
		}},
	}
	catalogs := weights.DraftCatalogs()
	if len(catalogs) != 2 || catalogs[0].Kind != DraftQwen35MTP || catalogs[1].Kind != DraftNextNMTP ||
		catalogs[1].LayerOutputNorm == nil || catalogs[1].LayerOutputNorm.Name != "next_layer_norm" {
		t.Fatalf("draft catalogs = %+v", catalogs)
	}
	layers := weights.LayerCatalog()
	if len(layers) != 3 || layers[0].AttentionNorm.Name != "trunk" ||
		layers[1].AttentionNorm.Name != "qwen_layer" || layers[2].AttentionNorm.Name != "next_layer" {
		t.Fatalf("layer catalog = %+v", layers)
	}
}

func pointerTensorInfo(value gguf.TensorInfo) *gguf.TensorInfo { return &value }
