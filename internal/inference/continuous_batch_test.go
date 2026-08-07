package inference

import (
	"context"
	"errors"
	"strings"
	"testing"

	"overgo/internal/cuda/executor"
	"overgo/internal/model"
	"overgo/internal/tensor"
	"overgo/internal/tensor/reference"
	"overgo/internal/tokenizer"
)

func TestContinuousBatchSequenceLifecycle(t *testing.T) {
	value, err := reference.NewValue(
		tensor.MustShape(1, 1, 5),
		[]float32{1, 2, 3, 4, 5},
	)
	if err != nil {
		t.Fatal(err)
	}
	batch := &ContinuousBatch{
		options: ContinuousBatchOptions{MaxSequences: 3, PageTokens: 2},
		sequences: map[SequenceID]*continuousSequence{
			7: {host: &KVCache{
				Layers: []LayerCache{{Key: value, Value: value}},
				Tokens: 5, Position: 9,
			}},
		},
	}
	states := batch.Snapshot()
	if len(states) != 1 || states[0].ID != 7 || states[0].Tokens != 5 ||
		states[0].Position != 9 || len(states[0].Pages) != 3 ||
		states[0].Pages[2] != (SequenceCachePage{Start: 4, Tokens: 1}) {
		t.Fatalf("states = %+v", states)
	}
	if err := batch.Fork(7, 8); err != nil {
		t.Fatal(err)
	}
	batch.sequences[7].host.Layers[0].Key.Data[0] = 99
	if batch.sequences[8].host.Layers[0].Key.Data[0] != 1 {
		t.Fatal("fork shares host cache storage")
	}
	if err := batch.Remove(context.Background(), 7); err != nil {
		t.Fatal(err)
	}
	if len(batch.Snapshot()) != 1 || batch.Snapshot()[0].ID != 8 {
		t.Fatalf("post-remove states = %+v", batch.Snapshot())
	}
	if err := batch.Close(context.Background()); err != nil {
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
	if err := batch.Close(context.Background()); !errors.Is(err, want) {
		t.Fatalf("close cleanup error = %v, want %v", err, want)
	}
	if err := batch.Close(context.Background()); err != nil {
		t.Fatalf("repeated close error = %v", err)
	}
}

func TestContinuousBatchCloseRetriesDeferredDeviceCleanup(t *testing.T) {
	cache := &deviceKVCache{owner: newDeviceCacheOwner(&executor.RetainedOutputs{}, 1)}
	firstErr := cache.Release(context.Background())
	if firstErr == nil {
		t.Fatal("invalid retained owner cleanup succeeded")
	}
	batch := &ContinuousBatch{
		sequences:  make(map[SequenceID]*continuousSequence),
		deferred:   []*deviceKVCache{cache},
		cleanupErr: firstErr,
	}
	if err := batch.Close(context.Background()); err == nil {
		t.Fatal("deferred cleanup failure was not reported")
	}
	if len(batch.deferred) != 0 || cache.owner != nil {
		t.Fatalf("deferred cleanup was not retried: %d pending", len(batch.deferred))
	}
	if err := batch.Close(context.Background()); err != nil {
		t.Fatalf("repeated close error = %v", err)
	}
}

func TestDeviceCacheReleaseCancellationIsAtomic(t *testing.T) {
	owner := newDeviceCacheOwner(&executor.RetainedOutputs{}, 1)
	cache := &deviceKVCache{owner: owner}
	ctx, cancel := context.WithCancel(context.Background())
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
	ctx, cancel := context.WithCancel(context.Background())
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
	batch, err := runner.NewContinuousBatch(ContinuousBatchOptions{
		MaxSequences: 2,
		PageTokens:   4,
	})
	if err != nil {
		t.Fatal(err)
	}
	_, err = batch.Step(context.Background(), []SequenceBatchInput{
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
	if _, err := batch.StepGreedy(context.Background(), []SequenceBatchInput{{
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
		if got := supportsPersistentDeviceCache(spec); got != want {
			t.Fatalf("%s support = %t, want %t", architecture, got, want)
		}
	}
}

func TestCloneDeviceCacheSharesOwnerAndCopiesMetadata(t *testing.T) {
	shape := tensor.MustShape(2, 1, 3)
	owner := newDeviceCacheOwner(&executor.RetainedOutputs{}, 1)
	source := &deviceKVCache{
		owner: owner,
		session: &deviceDecodeSession{program: decodeSessionPlan{
			identity: decodeSessionIdentity{capacity: 4},
		}},
		Keys:   []executor.DeviceValue{{Shape: shape}},
		Values: []executor.DeviceValue{{Shape: shape}},
		States: []deviceLayerStates{{
			"fixed": {Mode: CacheStateFixed, Value: executor.DeviceValue{Shape: tensor.MustShape(2)}},
		}},
		Tokens: 3, Position: 7, Logits: []float32{1, 2},
	}
	fork, err := cloneDeviceCache(source)
	if err != nil {
		t.Fatal(err)
	}
	if owner.refs != 2 || fork.owner != owner || fork.session != nil ||
		fork.Tokens != 3 || fork.Position != 7 {
		t.Fatalf("fork = %+v, owner refs = %d", fork, owner.refs)
	}
	fork.Keys[0].Shape = tensor.MustShape(1)
	fork.States[0]["other"] = deviceLayerState{Mode: CacheStateFixed}
	fork.Logits[0] = 9
	if source.Keys[0].Shape.Equal(fork.Keys[0].Shape) ||
		len(source.States[0]) != 1 || source.Logits[0] != 1 {
		t.Fatal("device fork shares mutable metadata")
	}
	source.owner = nil
	fork.owner = nil
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
			schema, err := model.CacheSchema(spec, 0, info, 1)
			if err != nil {
				t.Fatal(err)
			}
			first, second := schema.Primary.Key.Value.Shape, schema.Primary.Value.Value.Shape
			if test.architecture == "falcon-h1" {
				first = schema.States[model.CacheStateConvolution].Value.Shape
				second = schema.States[model.CacheStateSSM].Value.Shape
			}
			if first.Rank == 0 || second.Rank == 0 {
				t.Fatalf("state shapes = %v, %v", first.Slice(), second.Slice())
			}
		})
	}
}
