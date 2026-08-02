package inference

import (
	"context"
	"strings"
	"testing"

	"llamacpp2go/internal/model"
	"llamacpp2go/internal/tensor"
	"llamacpp2go/internal/tensor/reference"
	"llamacpp2go/internal/tokenizer"
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

func TestContinuousBatchAdmission(t *testing.T) {
	runner := &Runner{
		spec: model.Spec{
			CommonSpec: model.CommonSpec{
				Architecture: "llama", ContextLength: 8,
			},
		},
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
}

func TestPersistentDeviceCacheCapabilityProfile(t *testing.T) {
	tests := map[string]bool{
		"llama": true, "qwen3next": true, "mamba": false,
		"deepseek32": false, "gemma3n": false,
	}
	for architecture, want := range tests {
		spec := model.Spec{CommonSpec: model.CommonSpec{Architecture: architecture}}
		if got := supportsPersistentDeviceCache(spec); got != want {
			t.Fatalf("%s support = %t, want %t", architecture, got, want)
		}
	}
}
