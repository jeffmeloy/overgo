package inference

import (
	"context"
	"errors"
	"strings"
	"testing"

	"overgo/internal/cuda/executor"
	"overgo/internal/model"
	"overgo/internal/modelrecipe"
	"overgo/internal/tensor"
	"overgo/internal/tensor/reference"
	"overgo/internal/tokenizer"
)

func TestContinuousBatchSequenceLifecycle(t *testing.T) {
	t.Parallel()
	value, err := reference.NewValue(
		tensor.MustShape(1, 1, 5),
		[]float32{1, 2, 3, 4, 5},
	)
	if err != nil {
		t.Fatal(err)
	}
	runner := attachFixtureProgram(&Runner{preparedModel: preparedModel{spec: model.Spec{
		CommonSpec: model.CommonSpec{Architecture: "llama", ContextLength: 8},
	}}})
	runner.program.Decode.Session = modelrecipe.DecodeSessionCapacity
	batch := &ContinuousBatch{
		runner: runner, options: ContinuousBatchOptions{MaxSequences: 3},
		sequences: map[SequenceID]*continuousSequence{
			7: {host: &KVCache{
				Layers: []LayerCache{{Key: value, Value: value}},
				Tokens: 5, Position: 9,
			}},
			8: {host: &KVCache{
				Layers: []LayerCache{{Key: value, Value: value}},
				Tokens: 2, Position: 2,
			}},
		},
	}
	states := batch.Snapshot()
	if len(states) != 2 || states[0].ID != 7 || states[0].Tokens != 5 ||
		states[0].Position != 9 || len(states[0].Pages) != 2 ||
		states[0].Pages[1] != (SequenceCachePage{Start: 4, Tokens: 1}) ||
		states[1].ID != 8 || states[1].Tokens != 2 || len(states[1].Pages) != 1 {
		t.Fatalf("states = %+v", states)
	}
	if err := batch.Remove(t.Context(), 7); err != nil {
		t.Fatal(err)
	}
	if len(batch.Snapshot()) != 1 || batch.Snapshot()[0].ID != 8 {
		t.Fatalf("post-remove states = %+v", batch.Snapshot())
	}
	if err := batch.Close(t.Context()); err != nil {
		t.Fatal(err)
	}
	if len(batch.Snapshot()) != 0 {
		t.Fatal("close retained sequences")
	}
}

func TestContinuousBatchCloseReportsDeferredCleanup(t *testing.T) {
	want := errors.New("deferred device cleanup")
	batch := &ContinuousBatch{
		sequences:  make(map[SequenceID]*continuousSequence),
		cleanupErr: want,
	}
	if err := batch.Close(t.Context()); !errors.Is(err, want) {
		t.Fatalf("close cleanup error = %v, want %v", err, want)
	}
	if err := batch.Close(t.Context()); err != nil {
		t.Fatalf("repeated close error = %v", err)
	}
}

func TestContinuousBatchCloseRetriesDeferredDeviceCleanup(t *testing.T) {
	cache := &deviceKVCache{owner: newDeviceCacheOwner(&executor.RetainedOutputs{}, 1)}
	firstErr := cache.Release(t.Context())
	if firstErr == nil {
		t.Fatal("invalid retained owner cleanup succeeded")
	}
	batch := &ContinuousBatch{
		sequences:  make(map[SequenceID]*continuousSequence),
		deferred:   []*deviceKVCache{cache},
		cleanupErr: firstErr,
	}
	if err := batch.Close(t.Context()); err == nil {
		t.Fatal("deferred cleanup failure was not reported")
	}
	if len(batch.deferred) != 0 || cache.owner != nil {
		t.Fatalf("deferred cleanup was not retried: %d pending", len(batch.deferred))
	}
	if err := batch.Close(t.Context()); err != nil {
		t.Fatalf("repeated close error = %v", err)
	}
}

func TestDeviceCacheReleaseCancellationIsAtomic(t *testing.T) {
	owner := newDeviceCacheOwner(&executor.RetainedOutputs{}, 1)
	cache := &deviceKVCache{owner: owner}
	ctx, cancel := context.WithCancel(t.Context())
	cancel()
	if err := cache.Release(ctx); !errors.Is(err, context.Canceled) {
		t.Fatalf("canceled release error = %v", err)
	}
	if cache.owner != owner || owner.refs != 1 {
		t.Fatalf("canceled release mutated ownership: cache=%+v refs=%d", cache.owner, owner.refs)
	}
	cache.owner = nil
}

func TestDeviceCacheStorageReleaseCancellationIsAtomic(t *testing.T) {
	storage := &deviceCacheStorage{refs: 1}
	ctx, cancel := context.WithCancel(t.Context())
	cancel()
	if err := storage.release(ctx); !errors.Is(err, context.Canceled) {
		t.Fatalf("canceled release error = %v", err)
	}
	if storage.refs != 1 || storage.releasing {
		t.Fatalf("canceled release mutated storage: refs=%d releasing=%t", storage.refs, storage.releasing)
	}
}

func TestContinuousBatchAdmission(t *testing.T) {
	runner := &Runner{preparedModel: preparedModel{spec: model.Spec{
		CommonSpec: model.CommonSpec{
			Architecture: "llama", ContextLength: 8,
		},
	}},
	}
	runner = attachFixtureProgram(runner)
	batch, err := runner.NewContinuousBatch(ContinuousBatchOptions{
		MaxSequences: 2,
	})
	if err != nil {
		t.Fatal(err)
	}
	_, err = batch.Step(t.Context(), []SequenceBatchInput{
		{ID: 1, Tokens: []tokenizer.TokenID{1}},
		{ID: 1, Tokens: []tokenizer.TokenID{2}},
	})
	if err == nil || !strings.Contains(err.Error(), "duplicated") {
		t.Fatalf("duplicate error = %v", err)
	}
	if _, err := runner.NewContinuousBatch(ContinuousBatchOptions{}); err == nil {
		t.Fatal("zero sequence capacity accepted")
	}
	if _, err := runner.NewContinuousBatch(ContinuousBatchOptions{
		MaxSequences: 1, Device: true,
	}); err == nil || !strings.Contains(err.Error(), "preloaded") {
		t.Fatalf("device admission error = %v", err)
	}
	if _, err := batch.StepGreedy(t.Context(), []SequenceBatchInput{{
		ID: 2, Tokens: []tokenizer.TokenID{1},
	}}); err == nil || !strings.Contains(err.Error(), "device batch") {
		t.Fatalf("host greedy feedback error = %v", err)
	}
}

func TestPersistentDeviceCacheCapabilityProfile(t *testing.T) {
	tests := map[string]bool{
		"llama": true, "qwen3next": true, "qwen35": true,
		"mamba": true, "mamba2": true, "jamba": true,
		"rwkv6": true, "rwkv7": true, "falcon-h1": true,
		"granitehybrid": true, "kimi-linear": true, "plamo2": true,
		"nemotron_h": true, "lfm2": true, "lfm2moe": true,
		"deepseek32": false, "gemma3n": false,
	}
	for architecture, want := range tests {
		spec := model.Spec{CommonSpec: model.CommonSpec{Architecture: architecture}}
		plan, err := fixtureModelPlan(spec, model.Weights{})
		if err != nil {
			t.Fatal(err)
		}
		if got := plan.Forward().PersistentDeviceCache(); got != want {
			t.Fatalf("%s support = %t, want %t", architecture, got, want)
		}
	}
}

func TestRecurrentPrimaryStateShapesCoverFusedFamilies(t *testing.T) {
	tests := []struct {
		architecture string
		recurrent    bool
	}{
		{"mamba", true}, {"mamba2", true}, {"jamba", true},
		{"granitehybrid", true}, {"plamo2", true}, {"kimi-linear", true},
		{"rwkv6", true}, {"rwkv6qwen2", true}, {"rwkv7", true},
		{"arwkv7", true}, {"falcon-h1", false}, {"nemotron_h", true},
		{"lfm2", true}, {"lfm2moe", true},
	}
	for _, test := range tests {
		t.Run(test.architecture, func(t *testing.T) {
			spec := model.Spec{CommonSpec: model.CommonSpec{
				Architecture: test.architecture, EmbeddingLength: 8, BlockCount: 1,
			}, AttentionSpec: model.AttentionSpec{
				HeadCount: 2, HeadCountKV: 2, KeyLength: 4, ValueLength: 4,
			},
				RecurrentSpec: model.RecurrentSpec{
					SSMInnerSize: 6, SSMStateSize: 3, SSMGroupCount: 1,
					SSMConvKernel: 4, WKVHeadSize: 4, TokenShiftCount: 2,
					KDAHeadDim: 3, ShortConvCacheLength: 4,
					RecurrentLayers: []bool{test.recurrent},
				},
			}
			info := model.LayerWeights{Recurrent: test.recurrent}
			plan, err := fixtureModelPlan(spec, model.Weights{Layers: []model.LayerWeights{info}})
			if err != nil {
				t.Fatal(err)
			}
			schema, err := plan.CacheSchema(0, 1)
			if err != nil {
				t.Fatal(err)
			}
			first, second := schema.Primary.Key.Value.Shape, schema.Primary.Value.Value.Shape
			if test.architecture == "falcon-h1" {
				convolution, _ := schema.State(model.CacheStateConvolution)
				ssm, _ := schema.State(model.CacheStateSSM)
				first = convolution.Value.Shape
				second = ssm.Value.Shape
			}
			if first.Rank == 0 || second.Rank == 0 {
				t.Fatalf("state shapes = %v, %v", first.Slice(), second.Slice())
			}
		})
	}
}
