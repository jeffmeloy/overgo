package main

import (
	"context"
	"errors"
	"slices"
	"testing"
	"time"

	"overgo/internal/artifact"
	"overgo/internal/clioptions"
	"overgo/internal/cuda/device"
	"overgo/internal/cuda/driver"
	cudatest "overgo/internal/cuda/testutil"
	"overgo/internal/dataroot"
	"overgo/internal/inference"
	"overgo/internal/longform"
	"overgo/internal/overgodb"
	"overgo/internal/testutil"
	"overgo/internal/tokenizer"
)

// TestE4BResourceRecovery exercises the plan's 32768-token text resource
// scope on the admitted physical E4B artifact. Three complete load/close cycles
// each repeat growth, shrinkage, cancellation and consumer-error recovery.
// The first complete cycle establishes a live-allocation ceiling; later cycles
// cannot add a byte or allocation, and every identical prompt keeps its tokens.
// This is not a projector-resource or dataset-quality acceptance.
func TestE4BResourceRecovery(t *testing.T) {
	cudatest.Require(t)
	root := testutil.RepoRoot(t)
	roots, err := dataroot.Resolve(root)
	if err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithTimeoutCause(t.Context(), 10*time.Minute, errors.New("E4B resource recovery budget exhausted"))
	defer cancel()
	observer, err := device.New(0)
	if err != nil {
		t.Fatal(err)
	}
	defer func() {
		if err := observer.Close(); err != nil {
			t.Error(err)
		}
	}()
	freeDevice := func() uint64 {
		t.Helper()
		var free uint64
		if err := observer.Do(ctx, func(state *device.State) error {
			var err error
			free, _, err = state.Driver.MemInfo()
			return err
		}); err != nil {
			t.Fatal(err)
		}
		return free
	}
	store, err := overgodb.OpenReadOnly(roots.Store)
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	weights, err := artifact.ParseID("tensor-set:sha256:cd4ada4703c2b76a84a10da94f09b9199b6d4dad7e3dcabbe79d8cee745f4501")
	if err != nil {
		t.Fatal(err)
	}
	path, err := artifact.AvailablePath(ctx, store, weights, artifact.LocationFile)
	if err != nil {
		t.Fatal(err)
	}
	const requiredContext = 32768 // Explicit E4B resource acceptance scope in the plan.
	corpus, err := longform.Corpus(root, corpusBytesPerToken*(requiredContext+longform.DeclaredFloors().ScoreTokens))
	if err != nil {
		t.Fatal(err)
	}
	expected := make(map[int][]tokenizer.TokenID)
	var ceiling driver.MemoryStats
	for reload := range 3 {
		runner, err := clioptions.OpenRunner(ctx, roots.Store, path, inference.OpenOptions{})
		if err != nil {
			t.Fatal(err)
		}
		var owned, beforeClose uint64
		func() {
			defer func() {
				if err := runner.Close(); err != nil {
					t.Error(err)
				}
			}()
			ids, err := runner.TokenizeText(corpus, true, false)
			if err != nil || len(ids) < requiredContext {
				t.Fatalf("resource corpus: %d tokens, %v", len(ids), err)
			}
			generate := func(callCtx context.Context, length int, callback func(inference.TokenEvent) error) ([]tokenizer.TokenID, error) {
				var emitted []tokenizer.TokenID
				_, _, err := runner.Generate(callCtx, "", inference.GenerateOptions{
					PromptTokenIDs: ids[:length], MaxNewTokens: 8, ContinueAfterEOG: true, DeviceGreedy: true,
					OnToken: func(event inference.TokenEvent) error {
						emitted = append(emitted, event.ID)
						if callback != nil {
							return callback(event)
						}
						return nil
					},
				})
				return emitted, err
			}
			for cycle := range 3 {
				for _, length := range []int{1024, requiredContext, 1024, requiredContext} {
					started := time.Now()
					got, err := generate(ctx, length, nil)
					if err != nil || len(got) != 8 {
						t.Fatalf("reload %d cycle %d length %d: %d tokens, %v", reload, cycle, length, len(got), err)
					}
					if want, found := expected[length]; found {
						if !slices.Equal(got, want) {
							t.Fatalf("reuse/reload changed %d-token prompt output: %v != %v", length, got, want)
						}
					} else {
						expected[length] = slices.Clone(got)
					}
					t.Logf("reload=%d cycle=%d prompt=%d output=8 wall=%s", reload, cycle, length, time.Since(started))
				}
				canceled, stop := context.WithCancelCause(ctx)
				_, err := generate(canceled, 1024, func(inference.TokenEvent) error { stop(context.Canceled); return nil })
				stop(context.Canceled)
				if !errors.Is(err, context.Canceled) {
					t.Fatalf("cancellation did not propagate: %v", err)
				}
				injected := errors.New("injected consumer failure")
				_, err = generate(ctx, 1024, func(inference.TokenEvent) error { return injected })
				if !errors.Is(err, injected) {
					t.Fatalf("consumer failure did not propagate: %v", err)
				}
				got, err := generate(ctx, 1024, nil)
				if err != nil || !slices.Equal(got, expected[1024]) {
					t.Fatalf("post-error recovery changed output: %v, %v", got, err)
				}
				memory, err := runner.DeviceMemoryStats(ctx)
				if err != nil {
					t.Fatal(err)
				}
				if reload == 0 && cycle == 0 {
					ceiling = memory
				} else if memory.CurrentBytes > ceiling.CurrentBytes || memory.Allocations > ceiling.Allocations {
					t.Fatalf("retention grew: %d -> %d bytes, %d -> %d allocations", ceiling.CurrentBytes, memory.CurrentBytes, ceiling.Allocations, memory.Allocations)
				}
				t.Logf("reload=%d cycle=%d recovery=passed retained=%d allocations=%d peak=%d", reload, cycle, memory.CurrentBytes, memory.Allocations, memory.PeakBytes)
				owned = memory.CurrentBytes
			}
			beforeClose = freeDevice()
		}()
		afterClose := freeDevice()
		if afterClose < beforeClose || afterClose-beforeClose < owned {
			t.Fatalf("release did not return owned device bytes: free %d -> %d, owned %d", beforeClose, afterClose, owned)
		}
		t.Logf("reload=%d released=%d owned=%d", reload, afterClose-beforeClose, owned)
		if err := runner.Close(); err != nil {
			t.Fatal(err)
		}
		if _, _, err := runner.Generate(ctx, "closed", inference.GenerateOptions{MaxNewTokens: 1}); err == nil {
			t.Fatal("closed runner accepted generation")
		}
	}
	t.Log("3 reloads x 3 grow/shrink/grow cycles; 36 complete generations, 9 cancellations, 9 consumer errors, 9 exact recovery generations; text resources only, no dataset quality or projector-resource claim")
}
