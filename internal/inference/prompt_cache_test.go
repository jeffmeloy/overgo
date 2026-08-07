package inference

import (
	"context"
	"reflect"
	"testing"

	"overgo/internal/cuda/driver"
	"overgo/internal/cuda/executor"
	"overgo/internal/model"
	"overgo/internal/tensor"
	"overgo/internal/tensor/reference"
	"overgo/internal/tokenizer"
)

func TestReusablePromptPrefix(t *testing.T) {
	tests := []struct {
		name      string
		cached    []tokenizer.TokenID
		requested []tokenizer.TokenID
		minimum   int
		want      int
	}{
		{"exact", []tokenizer.TokenID{1, 2}, []tokenizer.TokenID{1, 2}, 0, 2},
		{"extension", []tokenizer.TokenID{1, 2}, []tokenizer.TokenID{1, 2, 3}, 2, 2},
		{"below threshold", []tokenizer.TokenID{1, 2}, []tokenizer.TokenID{1, 2, 3}, 3, 0},
		{"different", []tokenizer.TokenID{1, 2}, []tokenizer.TokenID{1, 9}, 0, 1},
		{"trim suffix", []tokenizer.TokenID{1, 2, 3}, []tokenizer.TokenID{1, 2}, 0, 2},
		{"partial below threshold", []tokenizer.TokenID{1, 2, 3}, []tokenizer.TokenID{1, 9}, 2, 0},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			if got := reusablePromptPrefix(test.cached, test.requested, test.minimum); got != test.want {
				t.Fatalf("prefix = %d, want %d", got, test.want)
			}
		})
	}
}

func TestClearPromptCachesPreservesPreparedModel(t *testing.T) {
	runner := &Runner{
		preparedModel: preparedModel{spec: model.Spec{CommonSpec: model.CommonSpec{Architecture: "llama"}}},
		runnerState:   runnerState{promptCaches: []*cachedPrompt{{Tokens: []tokenizer.TokenID{1}}}},
	}
	if err := runner.ClearPromptCaches(context.Background()); err != nil {
		t.Fatal(err)
	}
	if len(runner.promptCaches) != 0 || runner.Spec().Architecture != "llama" {
		t.Fatalf("runner after clear = %+v", runner)
	}
	runner.closed = true
	if err := runner.ClearPromptCaches(context.Background()); err == nil {
		t.Fatal("closed runner accepted prompt-cache clear")
	}
}

func TestHasRecurrentCache(t *testing.T) {
	runner := &Runner{preparedModel: preparedModel{weights: model.Weights{Layers: []model.LayerWeights{{}, {Recurrent: true}}}}}
	if !runner.hasRecurrentCache() {
		t.Fatal("recurrent cache was not detected")
	}
	runner.weights.Layers[1].Recurrent = false
	if runner.hasRecurrentCache() {
		t.Fatal("dense cache was classified as recurrent")
	}
}

func TestMultiplePromptCacheSelectionAndEviction(t *testing.T) {
	first := &cachedPrompt{
		Tokens: []tokenizer.TokenID{1, 2},
		Cache:  &KVCache{},
	}
	second := &cachedPrompt{
		Tokens: []tokenizer.TokenID{3, 4},
		Cache:  &KVCache{},
	}
	runner := &Runner{preparedModel: preparedModel{promptCacheCapacity: 2}, runnerState: runnerState{promptCaches: []*cachedPrompt{first, second}}}
	selected, common := runner.selectPromptCache(
		[]tokenizer.TokenID{3, 4, 5},
		2,
		false,
	)
	if selected != second || common != 2 || runner.promptCaches[0] != second {
		t.Fatalf("selected/common/cache order = %p/%d/%p",
			selected, common, runner.promptCaches[0])
	}
	third := &cachedPrompt{
		Tokens: []tokenizer.TokenID{6, 7},
		Cache:  &KVCache{},
	}
	if err := runner.storePromptCache(context.Background(), third); err != nil {
		t.Fatal(err)
	}
	if len(runner.promptCaches) != 2 ||
		runner.promptCaches[0] != third ||
		runner.promptCaches[1] != second {
		t.Fatalf("cache order after eviction = %+v", runner.promptCaches)
	}
}

func TestProjectedPromptCacheRequiresExactMediaSignature(t *testing.T) {
	signature := projectedInputsSignature(ProjectedInputs{EmbeddingOverrides: []EmbeddingOverride{{
		TokenIndex: 1, Embedding: []float32{1, 2},
	}}})
	entry := &cachedPrompt{
		Tokens: []tokenizer.TokenID{1, 2}, Cache: &KVCache{},
		HasProjection: true, ProjectionSignature: signature,
	}
	runner := &Runner{runnerState: runnerState{promptCaches: []*cachedPrompt{entry}}}
	selected, cached := runner.selectProjectedPromptCache(
		[]tokenizer.TokenID{1, 2}, signature, 2,
	)
	if selected != entry || cached != 2 {
		t.Fatalf("selected/cached = %p/%d", selected, cached)
	}
	other := projectedInputsSignature(ProjectedInputs{EmbeddingOverrides: []EmbeddingOverride{{
		TokenIndex: 1, Embedding: []float32{1, 3},
	}}})
	if selected, cached = runner.selectProjectedPromptCache(
		[]tokenizer.TokenID{1, 2}, other, 2,
	); selected != nil || cached != 0 {
		t.Fatalf("different media reused cache = %p/%d", selected, cached)
	}
	if selected, cached = runner.selectProjectedPromptCache(
		[]tokenizer.TokenID{1, 2, 3}, signature, 2,
	); selected != nil || cached != 0 {
		t.Fatalf("projected prefix reused cache = %p/%d", selected, cached)
	}
}

func TestT5SourceCacheRequiresExactSequence(t *testing.T) {
	hidden, err := reference.NewValue(tensor.MustShape(2, 2), []float32{1, 2, 3, 4})
	if err != nil {
		t.Fatal(err)
	}
	first := &cachedPrompt{Tokens: []tokenizer.TokenID{1, 2}, Hidden: hidden}
	second := &cachedPrompt{Tokens: []tokenizer.TokenID{3, 4}, Hidden: hidden}
	runner := &Runner{runnerState: runnerState{promptCaches: []*cachedPrompt{first, second}}}
	selected, reused := runner.selectT5SourceCache([]tokenizer.TokenID{3, 4}, 2)
	if selected != second || reused != 2 || runner.promptCaches[0] != second {
		t.Fatalf("selected/reused/cache order = %p/%d/%p", selected, reused, runner.promptCaches[0])
	}
	if selected, reused = runner.selectT5SourceCache([]tokenizer.TokenID{3, 4, 5}, 2); selected != nil || reused != 0 {
		t.Fatalf("prefix-only source cache was reused: %p/%d", selected, reused)
	}
	if selected, reused = runner.selectT5SourceCache([]tokenizer.TokenID{3, 4}, 3); selected != nil || reused != 0 {
		t.Fatalf("below-threshold source cache was reused: %p/%d", selected, reused)
	}
}

func TestTrimHostPromptCache(t *testing.T) {
	runner := cacheTestRunner()
	key, err := reference.NewValue(
		tensor.MustShape(2, 1, 4),
		[]float32{1, 2, 3, 4, 5, 6, 7, 8},
	)
	if err != nil {
		t.Fatal(err)
	}
	value, err := reference.NewValue(
		tensor.MustShape(3, 1, 4),
		[]float32{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12},
	)
	if err != nil {
		t.Fatal(err)
	}
	hidden, err := reference.NewValue(
		tensor.MustShape(2, 4),
		[]float32{10, 11, 20, 21, 30, 31, 40, 41},
	)
	if err != nil {
		t.Fatal(err)
	}
	trimmedHidden, trimmedCache, err := runner.trimHostPromptCache(
		hidden,
		&KVCache{
			Layers:   []LayerCache{{Key: key, Value: value}},
			Tokens:   4,
			Position: 4,
		},
		2,
	)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(trimmedHidden.Data, []float32{10, 11, 20, 21}) ||
		trimmedHidden.Shape.Dims[1] != 2 ||
		trimmedCache.Tokens != 2 ||
		trimmedCache.Position != 2 {
		t.Fatalf("trimmed hidden/cache = %+v / %+v",
			trimmedHidden, trimmedCache)
	}
}

func TestTrimDeviceCacheSuffix(t *testing.T) {
	shape := tensor.MustShape(2, 3, 4)
	cache := &deviceKVCache{
		Keys: []executor.DeviceValue{{
			Pointer: 1000,
			Shape:   shape,
		}},
		Values: []executor.DeviceValue{{
			Pointer: 2000,
			Shape:   shape,
		}},
		Tokens:   4,
		Position: 4,
		Logits:   []float32{1, 2},
	}
	if err := trimDeviceCacheSuffix(cache, 2); err != nil {
		t.Fatal(err)
	}
	if cache.Tokens != 2 ||
		cache.Position != 2 ||
		cache.Keys[0].Pointer != 1000 ||
		cache.Values[0].Pointer != 2000 ||
		cache.Keys[0].Shape.Dims[2] != 2 ||
		cache.Values[0].Shape.Dims[2] != 2 ||
		cache.Logits != nil {
		t.Fatalf("trimmed device cache = %+v", cache)
	}
}

func TestShiftDeviceCacheForAppendUsesPointerView(t *testing.T) {
	runner := &Runner{preparedModel: preparedModel{spec: model.Spec{CommonSpec: model.CommonSpec{ContextLength: 2}},
		weights: model.Weights{Layers: make([]model.LayerWeights, 1)}},
	}
	shape := tensor.MustShape(2, 3, 2)
	cache := &deviceKVCache{
		Keys: []executor.DeviceValue{{
			Pointer: driver.DevicePtr(1000),
			Shape:   shape,
		}},
		Values: []executor.DeviceValue{{
			Pointer: driver.DevicePtr(2000),
			Shape:   shape,
		}},
		Tokens:   2,
		Position: 5,
	}
	if err := runner.shiftDeviceCacheForAppend(cache, 1); err != nil {
		t.Fatal(err)
	}
	if cache.Tokens != 1 ||
		cache.Position != 5 ||
		cache.Keys[0].Pointer != 1024 ||
		cache.Values[0].Pointer != 2024 ||
		cache.Keys[0].Shape.Dims[2] != 1 ||
		cache.Values[0].Shape.Dims[2] != 1 {
		t.Fatalf("shifted cache = %+v", cache)
	}
}

func TestShiftDeviceCachePreservesRecurrentState(t *testing.T) {
	runner := &Runner{preparedModel: preparedModel{spec: model.Spec{CommonSpec: model.CommonSpec{Architecture: "qwen35", ContextLength: 2}},
		weights: model.Weights{Layers: []model.LayerWeights{{
			Recurrent: true,
		}}}},
	}
	stateShape := tensor.MustShape(4, 4)
	cache := &deviceKVCache{
		Keys:     []executor.DeviceValue{{Pointer: 1000, Shape: stateShape}},
		Values:   []executor.DeviceValue{{Pointer: 2000, Shape: stateShape}},
		Tokens:   2,
		Position: 5,
	}
	if err := runner.shiftDeviceCacheForAppend(cache, 1); err != nil {
		t.Fatal(err)
	}
	if cache.Tokens != 1 ||
		cache.Keys[0].Pointer != 1000 ||
		cache.Values[0].Pointer != 2000 ||
		!cache.Keys[0].Shape.Equal(stateShape) {
		t.Fatalf("shifted recurrent cache = %+v", cache)
	}
}
