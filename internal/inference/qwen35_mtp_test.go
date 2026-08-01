package inference

import (
	"context"
	"math"
	"testing"

	"llamacpp2go/internal/gguf"
	"llamacpp2go/internal/model"
	"llamacpp2go/internal/tensor"
	"llamacpp2go/internal/tensor/reference"
	"llamacpp2go/internal/tokenizer"
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

func TestGreedyLogitProbability(t *testing.T) {
	token, probability, err := greedyLogit([]float32{0, 1, 1})
	if err != nil {
		t.Fatal(err)
	}
	want := math.Exp(1) / (1 + 2*math.Exp(1))
	if token != 1 || math.Abs(probability-want) > 1e-12 {
		t.Fatalf("greedy result = %d/%g, want 1/%g", token, probability, want)
	}
	if _, _, err := greedyLogit([]float32{0, float32(math.NaN())}); err == nil {
		t.Fatal("NaN greedy logits were accepted")
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

func TestQwen35MTPOnlyRequiresCompatibleTarget(t *testing.T) {
	spec := model.Spec{
		Architecture: "qwen35", BlockCount: 2, NextNPredictLayers: 1,
		EmbeddingLength: 8, VocabularySize: 2, HeadCount: 2, HeadCountKV: 1,
		KeyLength: 4, ValueLength: 4, RopeDimensionCount: 4,
		RopeSections:  [4]int32{1, 1, 0, 0},
		SSMConvKernel: 3, SSMInnerSize: 4, SSMStateSize: 2,
		SSMTimeStepRank: 2, SSMGroupCount: 1, FullAttentionInterval: 2,
	}
	vocab := &tokenizer.Vocab{Tokens: []tokenizer.Token{{Text: "a"}, {Text: "b"}}}
	draft := &Runner{
		path: "draft", spec: spec, vocab: vocab,
		weights: model.Weights{Qwen35MTP: &model.Qwen35MTPWeights{MTPOnly: true}},
	}
	target := &Runner{
		path: "target", spec: spec, vocab: vocab,
		weights: model.Weights{Layers: make([]model.LayerWeights, spec.BlockCount)},
	}
	if err := draft.validateQwen35MTPTarget(target); err != nil {
		t.Fatal(err)
	}
	cache := &KVCache{
		Tokens: 1, Position: 1,
		Layers: []LayerCache{
			{
				Key:   reference.Value{Shape: tensor.MustShape(2, 8), Data: make([]float32, 16)},
				Value: reference.Value{Shape: tensor.MustShape(2, 2, 2, 1), Data: make([]float32, 8)},
			},
			{
				Key:   reference.Value{Shape: tensor.MustShape(4, 1, 1), Data: make([]float32, 4)},
				Value: reference.Value{Shape: tensor.MustShape(4, 1, 1), Data: make([]float32, 4)},
			},
		},
	}
	if err := draft.validateCache(cache); err != nil {
		t.Fatalf("MTP-only runner rejected compatible target hybrid cache: %v", err)
	}
	if _, _, err := draft.forwardCachedLocked(context.Background(), []tokenizer.TokenID{0}, nil); err == nil {
		t.Fatal("ordinary forward accepted Qwen3.5 MTP-only model")
	}
	target.vocab = &tokenizer.Vocab{Tokens: []tokenizer.Token{{Text: "a"}, {Text: "c"}}}
	if err := draft.validateQwen35MTPTarget(target); err == nil {
		t.Fatal("Qwen3.5 MTP sidecar accepted mismatched target vocabulary")
	}
}
