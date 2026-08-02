package inference

import (
	"context"
	"testing"

	"llamacpp2go/internal/gguf"
	"llamacpp2go/internal/model"
	"llamacpp2go/internal/tokenizer"
)

func TestSelectedModelTensorsIncludesCohere2MTP(t *testing.T) {
	info := func(name string) gguf.TensorInfo { return gguf.TensorInfo{Name: name} }
	privateHead := info("blk.2.nextn.shared_head_head.weight")
	mtp := &model.Cohere2MTPWeights{
		Layer: model.LayerWeights{
			AttentionNorm:     info("blk.2.attn_norm.weight"),
			AttentionQ:        info("blk.2.attn_q.weight"),
			FeedForwardRouter: pointerTensorInfo(info("blk.2.ffn_gate_inp.weight")),
		},
		EHProjection:  info("blk.2.nextn.eh_proj.weight"),
		EmbeddingNorm: info("blk.2.nextn.enorm.weight"),
		HiddenNorm:    info("blk.2.nextn.hnorm.weight"), Output: &privateHead,
	}
	file := &gguf.File{Tensors: []gguf.TensorInfo{
		info("token_embd.weight"), mtp.Layer.AttentionNorm, mtp.Layer.AttentionQ,
		*mtp.Layer.FeedForwardRouter, mtp.EHProjection, mtp.EmbeddingNorm, mtp.HiddenNorm, privateHead,
	}}
	selected := selectedModelTensors(file, model.Weights{
		TokenEmbedding: info("token_embd.weight"), Cohere2MTP: mtp,
	})
	names := make(map[string]bool, len(selected))
	for _, item := range selected {
		names[item.Name] = true
	}
	for _, item := range file.Tensors {
		if !names[item.Name] {
			t.Fatalf("Cohere2-MoE MTP tensor %q was not selected", item.Name)
		}
	}
}

func TestCohere2MTPOnlyRequiresCompatibleTarget(t *testing.T) {
	spec := model.Spec{CommonSpec: model.CommonSpec{Architecture: "cohere2moe", BlockCount: 2, NextNPredictLayers: 1,
		EmbeddingLength: 8, VocabularySize: 2}, AttentionSpec: model.AttentionSpec{HeadCount: 2, HeadCountKV: 1,
		KeyLength: 4, ValueLength: 4, RopeDimensionCount: 4},
	}
	vocab := &tokenizer.Vocab{Tokens: []tokenizer.Token{{Text: "a"}, {Text: "b"}}}
	draft := &Runner{
		path: "draft", spec: spec, vocab: vocab,
		weights: model.Weights{Cohere2MTP: &model.Cohere2MTPWeights{MTPOnly: true}},
	}
	target := &Runner{
		path: "target", spec: spec, vocab: vocab,
		weights: model.Weights{Layers: make([]model.LayerWeights, spec.BlockCount)},
	}
	if err := draft.validateCohere2MTPTarget(target); err != nil {
		t.Fatal(err)
	}
	if _, _, err := draft.forwardCachedLocked(context.Background(), []tokenizer.TokenID{0}, nil); err == nil {
		t.Fatal("ordinary forward accepted Cohere2-MoE MTP-only model")
	}
	target.vocab = &tokenizer.Vocab{Tokens: []tokenizer.Token{{Text: "a"}, {Text: "c"}}}
	if err := draft.validateCohere2MTPTarget(target); err == nil {
		t.Fatal("Cohere2-MoE MTP sidecar accepted mismatched target vocabulary")
	}
}

func pointerTensorInfo(info gguf.TensorInfo) *gguf.TensorInfo {
	return &info
}
