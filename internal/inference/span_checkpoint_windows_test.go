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
// span computes the kept tokens among more, at rounding level from them computed alone.
const spanCheckpointTolerance = 2e-4

// TestSpanCheckpointKeepsThePendingTokens: a verify span committed to a checkpoint holds what a span of the tokens it
// keeps alone leaves, its pending token and each prefix of its drafts, on the hermetic Flash-Next (gated-delta, attention
// and n-gram layers): its token count and position, every layer's keys, values and states, and the lexical history of
// the tokens kept.
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
	span := func(tokens []tokenizer.TokenID, past *deviceKVCache, checkpoints ...uint32) *deviceKVCache {
		t.Helper()
		cache, err := runner.forwardDeviceCachedModeLocked(ctx, deviceOutputGreedySpan, tokens, past, checkpoints...)
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
	boundary := span([]tokenizer.TokenID{4, 9, 17}, nil)
	// A pending token and three drafts keep a checkpoint after the pending token and after each of the first two drafts.
	pending, drafts := []tokenizer.TokenID{21}, []tokenizer.TokenID{30, 29, 27}
	checkpoints := []uint32{1, 2, 3}
	near := func(a, b float32) bool { return math.Abs(float64(a-b)) <= spanCheckpointTolerance }
	for accepted := range len(drafts) {
		verified := span(append(slices.Clone(pending), drafts...), boundary, checkpoints...)
		if len(verified.checkpoints) != len(checkpoints) {
			t.Fatalf("a span with %d checkpoints keeps %d", len(checkpoints), len(verified.checkpoints))
		}
		kept := append(slices.Clone(pending), drafts[:accepted]...)
		if err := runner.commitSpanCheckpoint(ctx, verified, boundary, kept); err != nil {
			t.Fatal(err)
		}
		alone := span(kept, boundary)
		if verified.Tokens != alone.Tokens || verified.Position != alone.Position || !slices.Equal(verified.ngramHistory, alone.ngramHistory) {
			t.Fatalf("%d drafts kept: committed %d tokens at %d with history %v, alone %d at %d with %v", accepted,
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
		for layer, got := range committed.Layers {
			want := aloneCache.Layers[layer]
			pairs := map[string][2]reference.Value{"key": {got.Key, want.Key}, "value": {got.Value, want.Value}}
			for name, state := range got.States {
				pairs[string(name)] = [2]reference.Value{state.Value, want.States[name].Value}
			}
			for name, pair := range pairs {
				if !pair[0].Shape.Equal(pair[1].Shape) || !slices.EqualFunc(pair[0].Data, pair[1].Data, near) {
					t.Fatalf("%d drafts kept: layer %d %s differs from the kept tokens alone (shapes %v and %v)",
						accepted, layer, name, pair[0].Shape, pair[1].Shape)
				}
			}
		}
	}
}
