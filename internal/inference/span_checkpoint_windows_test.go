//go:build windows

package inference

import (
	"context"
	"math"
	"slices"
	"testing"

	cudatest "overgo/internal/cuda/testutil"
	"overgo/internal/recipe"
	"overgo/internal/tensor/reference"
	"overgo/internal/tokenizer"
)

// spanCheckpointTolerance is how far a committed span's cache may sit from one that read only the kept tokens: the
// span computes the kept token among two, at rounding level from one computed alone.
const spanCheckpointTolerance = 2e-4

// TestSpanCheckpointKeepsThePendingTokens: a verify span committed to its checkpoint holds what a span of its pending
// tokens alone leaves, on the hermetic Flash-Next (gated-delta, attention and n-gram layers): its token count and
// position, every layer's keys, values and states, and the lexical history of the tokens kept.
func TestSpanCheckpointKeepsThePendingTokens(t *testing.T) {
	// Serial: the spans share the one CUDA device.
	requireIntegration(t)
	cudatest.Require(t)
	runner, err := openFixtureRunnerWithResidency(writeHermeticFlashNextGGUF(t), OpenOptions{}, recipe.ResidencyDeviceNative)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := runner.Close(); err != nil {
			t.Error(err)
		}
	})
	ctx := t.Context()
	span := func(tokens []tokenizer.TokenID, past *deviceKVCache, checkpoint uint32) *deviceKVCache {
		t.Helper()
		cache, err := runner.forwardDeviceCachedModeLocked(ctx, deviceOutputGreedySpan, tokens, past, checkpoint)
		if err != nil {
			t.Fatal(err)
		}
		t.Cleanup(func() {
			if err := cache.Release(context.Background()); err != nil {
				t.Error(err)
			}
		})
		return cache
	}
	boundary := span([]tokenizer.TokenID{4, 9, 17}, nil, 0)
	pending, draft := []tokenizer.TokenID{21}, tokenizer.TokenID(30)
	verified := span(append(slices.Clone(pending), draft), boundary, uint32(len(pending)))
	if verified.checkpoint == nil {
		t.Fatal("a span with a checkpoint keeps none")
	}
	if err := runner.commitSpanCheckpoint(ctx, verified, boundary, pending); err != nil {
		t.Fatal(err)
	}
	alone := span(pending, boundary, 0)
	if verified.Tokens != alone.Tokens || verified.Position != alone.Position || !slices.Equal(verified.ngramHistory, alone.ngramHistory) {
		t.Fatalf("committed %d tokens at %d with history %v, alone %d at %d with %v",
			verified.Tokens, verified.Position, verified.ngramHistory, alone.Tokens, alone.Position, alone.ngramHistory)
	}
	committed, err := runner.downloadDeviceCache(ctx, verified)
	if err != nil {
		t.Fatal(err)
	}
	aloneCache, err := runner.downloadDeviceCache(ctx, alone)
	if err != nil {
		t.Fatal(err)
	}
	near := func(a, b float32) bool { return math.Abs(float64(a-b)) <= spanCheckpointTolerance }
	for layer, got := range committed.Layers {
		want := aloneCache.Layers[layer]
		pairs := map[string][2]reference.Value{"key": {got.Key, want.Key}, "value": {got.Value, want.Value}}
		for name, state := range got.States {
			pairs[string(name)] = [2]reference.Value{state.Value, want.States[name].Value}
		}
		for name, pair := range pairs {
			if !pair[0].Shape.Equal(pair[1].Shape) || !slices.EqualFunc(pair[0].Data, pair[1].Data, near) {
				t.Fatalf("layer %d %s: the committed span differs from the pending tokens alone (shapes %v and %v)", layer, name, pair[0].Shape, pair[1].Shape)
			}
		}
	}
}
