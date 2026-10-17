//go:build windows

package inference

import (
	"context"
	"errors"
	"math"
	"slices"
	"testing"

	cudatest "overgo/internal/cuda/testutil"
	"overgo/internal/tensor/reference"
	"overgo/internal/testutil"
	"overgo/internal/tokenizer"
)

func TestDeviceBoundedPrefill(t *testing.T) {
	cudatest.Require(t)
	for _, fixture := range []struct {
		name string
		path func(*testing.T) string
	}{{"dense", func(t *testing.T) string { return testutil.HermeticLlamaGGUF(t, hermeticContext) }}, {"recurrent", writeHermeticQwen35GGUF}} {
		t.Run(fixture.name, func(t *testing.T) {
			runner, err := openF32FixtureRunner(fixture.path(t), OpenOptions{})
			if err != nil {
				t.Fatal(err)
			}
			defer runner.Close()
			ctx := t.Context()
			plan, err := compileDeviceOutputPlan(deviceOutputGreedy, 0, runner.spec.VocabularySize)
			if err != nil {
				t.Fatal(err)
			}
			tokens := []tokenizer.TokenID{1, 4, 5, 6, 7}
			whole, err := runner.forwardDeviceCachedBranchedBatchLocked(ctx, []deviceBatchAppend{{Tokens: tokens}}, plan)
			if err != nil {
				t.Fatal(err)
			}
			defer whole[0].Release(context.Background())
			split, err := runner.prefillDeviceCachedLocked(ctx, deviceBatchAppend{Tokens: tokens}, plan, 2)
			if err != nil {
				t.Fatal(err)
			}
			defer split.Release(context.Background())
			compare := func(left, right *deviceKVCache) {
				t.Helper()
				if left.Tokens != right.Tokens || left.Position != right.Position {
					t.Fatalf("cache positions differ: %d/%d and %d/%d", left.Tokens, left.Position, right.Tokens, right.Position)
				}
				l, err := runner.downloadDeviceCache(ctx, left)
				if err != nil {
					t.Fatal(err)
				}
				r, err := runner.downloadDeviceCache(ctx, right)
				if err != nil {
					t.Fatal(err)
				}
				for i, layer := range l.Layers {
					pairs := [][2]reference.Value{{layer.Key, r.Layers[i].Key}, {layer.Value, r.Layers[i].Value}}
					for name, state := range layer.States {
						pairs = append(pairs, [2]reference.Value{state.Value, r.Layers[i].States[name].Value})
					}
					for _, pair := range pairs {
						if !pair[0].Shape.Equal(pair[1].Shape) || !slices.EqualFunc(pair[0].Data, pair[1].Data, func(a, b float32) bool { return math.Abs(float64(a-b)) <= 2e-4 }) {
							t.Fatalf("layer %d split state differs from whole prefill", i)
						}
					}
				}
			}
			compare(split, whole[0])
			if split.Selected != whole[0].Selected {
				t.Fatalf("split selection = %d, whole = %d", split.Selected, whole[0].Selected)
			}
			// A refused later chunk releases its intermediate cache and preserves the supplied prefix.
			refs := split.storage.refs
			_, err = runner.prefillDeviceCachedLocked(ctx, deviceBatchAppend{Tokens: []tokenizer.TokenID{4, 5, 9999}, Past: split}, plan, 2)
			if err == nil || split.storage.refs != refs {
				t.Fatalf("late refusal = %v, prefix references = %d, want %d", err, split.storage.refs, refs)
			}
			compare(split, whole[0])
			canceled, cancel := context.WithCancel(ctx)
			cancel()
			if _, err = runner.prefillDeviceCachedLocked(canceled, deviceBatchAppend{Tokens: tokens, Past: split}, plan, 2); !errors.Is(err, context.Canceled) {
				t.Fatalf("cancellation = %v", err)
			}
			var continued [2]*deviceKVCache
			for i, past := range []*deviceKVCache{split, whole[0]} {
				_, continued[i], err = runner.forwardDeviceCachedLocked(ctx, []tokenizer.TokenID{3}, past)
				if err != nil {
					t.Fatal(err)
				}
				defer continued[i].Release(context.Background())
			}
			compare(continued[0], continued[1])
			if !slices.EqualFunc(continued[0].Logits, continued[1].Logits, func(a, b float32) bool { return math.Abs(float64(a-b)) <= 2e-4 }) {
				t.Fatal("split continuation logits differ")
			}
		})
	}
}
