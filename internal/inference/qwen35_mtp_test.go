package inference

import (
	"testing"

	"llamacpp2go/internal/gguf"
	"llamacpp2go/internal/model"
)

func TestSelectedModelTensorsIncludesQwen35MTP(t *testing.T) {
	info := func(name string) gguf.TensorInfo { return gguf.TensorInfo{Name: name} }
	mtp := &model.Qwen35MTPWeights{
		Layer: model.LayerWeights{
			AttentionNorm: info("blk.1.attn_norm.weight"),
			AttentionQ:    info("blk.1.attn_q.weight"),
		},
		EHProjection:  info("blk.1.nextn.eh_proj.weight"),
		EmbeddingNorm: info("blk.1.nextn.enorm.weight"),
		HiddenNorm:    info("blk.1.nextn.hnorm.weight"),
	}
	privateHead := info("blk.1.nextn.shared_head_head.weight")
	mtp.Output = &privateHead
	file := &gguf.File{Tensors: []gguf.TensorInfo{
		info("token_embd.weight"), mtp.Layer.AttentionNorm, mtp.Layer.AttentionQ,
		mtp.EHProjection, mtp.EmbeddingNorm, mtp.HiddenNorm, privateHead,
	}}
	selected := selectedModelTensors(file, model.Weights{
		TokenEmbedding: info("token_embd.weight"), Qwen35MTP: mtp,
	})
	names := make(map[string]bool, len(selected))
	for _, item := range selected {
		names[item.Name] = true
	}
	for _, item := range file.Tensors {
		if !names[item.Name] {
			t.Fatalf("Qwen3.5 MTP tensor %q was not selected", item.Name)
		}
	}
}

func TestValidateQwen35MTP(t *testing.T) {
	runner := &Runner{
		spec:    model.Spec{Architecture: "qwen35", NextNPredictLayers: 1},
		weights: model.Weights{Qwen35MTP: &model.Qwen35MTPWeights{}},
	}
	if err := runner.validateQwen35MTP(); err != nil {
		t.Fatal(err)
	}
	runner.spec.NextNPredictLayers = 0
	if err := runner.validateQwen35MTP(); err == nil {
		t.Fatal("missing Qwen3.5 MTP metadata was accepted")
	}
}
