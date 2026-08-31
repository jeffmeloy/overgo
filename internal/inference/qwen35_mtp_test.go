package inference

import (
	"math"
	"testing"

	"overgo/internal/gguf"
	"overgo/internal/model"
	"overgo/internal/modeltest"
	"overgo/internal/tensor"
	"overgo/internal/tensor/reference"
	"overgo/internal/tokenizer"
)

func TestSelectedModelTensorsIncludesQwen35MTP(t *testing.T) {
	info := func(name string) gguf.TensorInfo { return gguf.TensorInfo{Name: name} }
	mtp := &model.SingleDraftWeights{
		Layer: model.LayerWeights{
			AttentionNorm: new(info("blk.1.attn_norm.weight")),
			AttentionQ:    new(info("blk.1.attn_q.weight")),
		},
		EHProjection:  info("blk.1.nextn.eh_proj.weight"),
		EmbeddingNorm: info("blk.1.nextn.enorm.weight"),
		HiddenNorm:    info("blk.1.nextn.hnorm.weight"),
	}
	privateHead := info("blk.1.nextn.shared_head_head.weight")
	mtp.Output = &privateHead
	file := &gguf.File{Tensors: []gguf.TensorInfo{
		info("token_embd.weight"), *mtp.Layer.AttentionNorm, *mtp.Layer.AttentionQ,
		mtp.EHProjection, mtp.EmbeddingNorm, mtp.HiddenNorm, privateHead,
	}}
	selected := selectedModelTensors(file, model.Weights{
		TokenEmbedding: info("token_embd.weight"), SingleCatalogDraft: mtp,
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

func TestSelectedModelTensorsIncludesStep35MTP(t *testing.T) {
	info := func(name string) gguf.TensorInfo { return gguf.TensorInfo{Name: name} }
	mtp := model.AppendedDraftWeights{
		Layer: model.LayerWeights{
			AttentionNorm: new(info("blk.1.attn_norm.weight")),
			AttentionQ:    new(info("blk.1.attn_q.weight")),
		},
		EHProjection:   info("blk.1.nextn.eh_proj.weight"),
		EmbeddingNorm:  info("blk.1.nextn.enorm.weight"),
		HiddenNorm:     info("blk.1.nextn.hnorm.weight"),
		TokenEmbedding: new(info("blk.1.nextn.embed_tokens.weight")),
		OutputNorm:     new(info("blk.1.nextn.shared_head_norm.weight")),
		Output:         new(info("blk.1.nextn.shared_head_head.weight")),
	}
	weights := model.Weights{
		TokenEmbedding:          info("token_embd.weight"),
		AppendedMultiCarryDraft: []model.AppendedDraftWeights{mtp},
	}
	file := &gguf.File{Tensors: []gguf.TensorInfo{
		weights.TokenEmbedding, *mtp.Layer.AttentionNorm, *mtp.Layer.AttentionQ,
		mtp.EHProjection, mtp.EmbeddingNorm, mtp.HiddenNorm,
		*mtp.TokenEmbedding, *mtp.OutputNorm, *mtp.Output,
	}}
	selected := selectedModelTensors(file, weights)
	names := make(map[string]bool, len(selected))
	for _, info := range selected {
		names[info.Name] = true
	}
	for _, name := range []string{
		"blk.1.attn_norm.weight", "blk.1.attn_q.weight",
		"blk.1.nextn.eh_proj.weight", "blk.1.nextn.enorm.weight", "blk.1.nextn.hnorm.weight",
		"blk.1.nextn.embed_tokens.weight", "blk.1.nextn.shared_head_norm.weight",
		"blk.1.nextn.shared_head_head.weight",
	} {
		if !names[name] {
			t.Fatalf("Step3.5 MTP tensor %q was not selected", name)
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

func TestSingleHeadMTPRequiresCompiledCatalog(t *testing.T) {
	runner := &Runner{preparedModel: preparedModel{spec: model.Spec{CommonSpec: model.CommonSpec{Architecture: "qwen35", NextNPredictLayers: 1}},
		weights: model.Weights{SingleCatalogDraft: &model.SingleDraftWeights{}}},
	}
	runner = attachFixtureProgram(runner)
	if _, _, err := runner.singleHeadMTP(); err != nil {
		t.Fatal(err)
	}
	runner.weights.SingleCatalogDraft = nil
	if _, _, err := runner.singleHeadMTP(); err == nil {
		t.Fatal("missing Qwen3.5 MTP metadata was accepted")
	}
}

func TestQwen35MTPOnlyRequiresCompatibleTarget(t *testing.T) {
	fixture := modeltest.Qwen35DenseRecurrentPair()
	spec := fixture.Spec
	spec.NextNPredictLayers = 1
	spec.VocabularySize = 2
	vocab := &tokenizer.Vocab{Tokens: []tokenizer.Token{{Text: "a"}, {Text: "b"}}}
	draft := fixtureRunner(spec, model.Weights{SingleCatalogDraft: &model.SingleDraftWeights{MTPOnly: true}})
	draft.path, draft.vocab = "draft", vocab
	target := fixtureRunner(spec, model.Weights{Layers: make([]model.LayerWeights, spec.BlockCount)})
	target.path, target.vocab = "target", vocab
	if err := draft.validateMTPTarget(target); err != nil {
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
	if _, _, err := draft.forwardCachedLocked(t.Context(), []tokenizer.TokenID{0}, nil); err == nil {
		t.Fatal("ordinary forward accepted Qwen3.5 MTP-only model")
	}
	target.vocab = &tokenizer.Vocab{Tokens: []tokenizer.Token{{Text: "a"}, {Text: "c"}}}
	if err := draft.validateMTPTarget(target); err == nil {
		t.Fatal("Qwen3.5 MTP sidecar accepted mismatched target vocabulary")
	}
}
