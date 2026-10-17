//go:build windows

package inference

import (
	"context"
	"math"
	"reflect"
	"slices"
	"testing"

	cudatest "overgo/internal/cuda/testutil"
	"overgo/internal/sampling"
	"overgo/internal/tensor"
	"overgo/internal/tensor/reference"
	"overgo/internal/testutil"
	"overgo/internal/tokenizer"
)

const (
	hermeticVocab   = uint64(8)
	hermeticContext = uint32(16)
)

func TestHermeticCUDAContinuousCacheParity(t *testing.T) {
	requireIntegration(t)
	cudatest.Require(t)
	path := testutil.HermeticLlamaGGUF(t, hermeticContext)
	runner, err := openF32FixtureRunner(path, OpenOptions{})
	if err != nil {
		t.Fatal(err)
	}
	defer runner.Close()
	deviceBatch, err := runner.NewContinuousBatch(ContinuousBatchOptions{
		MaxSequences: 2, Device: true,
	})
	if err != nil {
		t.Fatal(err)
	}
	defer deviceBatch.Close(t.Context())
	hostBatch, err := runner.NewContinuousBatch(ContinuousBatchOptions{
		MaxSequences: 2,
	})
	if err != nil {
		t.Fatal(err)
	}
	defer hostBatch.Close(t.Context())

	// A decode session is keyed by the past page it was compiled to read as
	// well as its target page, so replays happen only within one page
	// class. The replay counter is cumulative over a sequence's session
	// lineage: these nine decode steps compile at pasts 2, 3, 4, 5, 8, and
	// 9 (each a new page class) and replay at pasts 6 and 7 on the 8-token
	// page and at past 10 on the 16-token page, three replays; the
	// two-sequence cohort below then compiles at past 11 and replays at 12,
	// both on the 16-token page.
	const singleSequenceReplays = 3
	steps := [][]tokenizer.TokenID{{1, 4}, {5}, {6}, {7}, {4}, {5}, {6}, {7}, {4}, {5}}
	for step, tokens := range steps {
		device, deviceErr := deviceBatch.Step(t.Context(), []SequenceBatchInput{{ID: 1, Tokens: tokens}})
		host, hostErr := hostBatch.Step(t.Context(), []SequenceBatchInput{{ID: 1, Tokens: tokens}})
		if deviceErr != nil || hostErr != nil {
			t.Fatalf("step %d device/host errors = %v/%v", step, deviceErr, hostErr)
		}
		if len(device) != 1 || len(host) != 1 || device[0].Tokens != host[0].Tokens ||
			device[0].Position != host[0].Position || len(device[0].Logits) != len(host[0].Logits) {
			t.Fatalf("step %d states = %+v/%+v", step, device, host)
		}
		for index := range device[0].Logits {
			if delta := math.Abs(float64(device[0].Logits[index] - host[0].Logits[index])); delta > 2e-4 {
				t.Fatalf("step %d logit %d delta = %g", step, index, delta)
			}
		}
	}
	deviceBatch.mu.Lock()
	session := deviceBatch.sequences[1].device.session
	deviceBatch.mu.Unlock()
	if session == nil || session.replays != singleSequenceReplays || session.program.identity.output.mode != deviceOutputLogits {
		t.Fatalf("generic decode session = %+v", session)
	}
	beforeFailure := deviceBatch.Snapshot()
	canceled, cancel := context.WithCancel(t.Context())
	cancel()
	if _, err := deviceBatch.Step(canceled, []SequenceBatchInput{{
		ID: 1, Tokens: []tokenizer.TokenID{7},
	}}); err == nil {
		t.Fatal("canceled device append succeeded")
	}
	if afterFailure := deviceBatch.Snapshot(); !reflect.DeepEqual(afterFailure, beforeFailure) {
		t.Fatalf("canceled append changed cache state: before=%+v after=%+v", beforeFailure, afterFailure)
	}
	// Sequence 2 appends the whole history of sequence 1 in one step.
	//   - both then stand at past 11, and sequence 2 holds no decode session
	if _, err := deviceBatch.Step(t.Context(), []SequenceBatchInput{
		{ID: 2, Tokens: slices.Concat(steps...)},
	}); err != nil {
		t.Fatal(err)
	}
	branched, err := deviceBatch.Step(t.Context(), []SequenceBatchInput{
		{ID: 1, Tokens: []tokenizer.TokenID{2}},
		{ID: 2, Tokens: []tokenizer.TokenID{3}},
	})
	if err != nil {
		t.Fatal(err)
	}
	if len(branched) != 2 || branched[0].Tokens != branched[1].Tokens ||
		branched[0].Position != branched[1].Position {
		t.Fatalf("cohort states = %+v", branched)
	}
	deviceBatch.mu.Lock()
	cohortSession := deviceBatch.sequences[1].device.session
	var cohortReplays uint64
	if cohortSession != nil {
		cohortReplays = cohortSession.replays
	}
	deviceBatch.mu.Unlock()
	if cohortSession == nil {
		t.Fatal("two-sequence cohort has no reusable session")
	}
	if _, err = deviceBatch.Step(t.Context(), []SequenceBatchInput{
		{ID: 1, Tokens: []tokenizer.TokenID{4}},
		{ID: 2, Tokens: []tokenizer.TokenID{5}},
	}); err != nil {
		t.Fatal(err)
	}
	deviceBatch.mu.Lock()
	first := deviceBatch.sequences[1].device
	second := deviceBatch.sequences[2].device
	deviceBatch.mu.Unlock()
	if first.session == nil || first.session != second.session || first.session != cohortSession ||
		first.session.replays != cohortReplays+1 ||
		first.sessionBranch != 0 || second.sessionBranch != 1 ||
		first.session.program.identity.branches != 2 {
		t.Fatalf("multi-branch decode sessions = %+v/%+v", first.session, second.session)
	}
}

// TestHermeticCUDADeviceCacheRoundTrip reads a prefilled device cache back to
// the host and puts it on the device again, for a dense model and for one whose
// last layers share earlier layers' KV. The read-back matches the host cache;
// save, load and upload return it bit for bit; the uploaded cache appends
// across its 8-token page with the original's logits bit for bit; and a prompt
// state saved from one runner and restored into a fresh one serves its prefix
// from the cache with the greedy tokens of an uncached run.
func TestHermeticCUDADeviceCacheRoundTrip(t *testing.T) {
	// Serial: each case holds the device for runners it opens one after another.
	requireIntegration(t)
	cudatest.Require(t)
	prompt := []tokenizer.TokenID{1, 4, 5, 6, 7}
	appended := []tokenizer.TokenID{4, 5, 6, 7}
	const generated = 3
	for _, fixture := range []struct {
		name string
		path string
	}{
		{"llama", testutil.HermeticLlamaGGUF(t, hermeticContext)},
		{"shared-kv", testutil.HermeticGemma4SharedKVGGUF(t, sharedKVFixtureContext)},
	} {
		t.Run(fixture.name, func(t *testing.T) {
			open := func() *Runner {
				t.Helper()
				runner, err := openF32FixtureRunner(fixture.path, OpenOptions{})
				if err != nil {
					t.Fatal(err)
				}
				t.Cleanup(func() { _ = runner.Close() })
				return runner
			}
			runner := open()
			deviceCacheRoundTrip(t, runner, prompt, appended)

			greedy, err := sampling.New(sampling.Config{Temperature: 0})
			if err != nil {
				t.Fatal(err)
			}
			generate := func(runner *Runner, ids []tokenizer.TokenID, evaluation *PromptEvaluation) []tokenizer.TokenID {
				t.Helper()
				result, _, err := runner.Generate(t.Context(), "", GenerateOptions{
					MaxNewTokens: generated, Sampler: greedy, PromptTokenIDs: ids, CachePrompt: true, ContinueAfterEOG: true,
					OnPromptEvaluated: func(measured PromptEvaluation) { *evaluation = measured },
				})
				if err != nil {
					t.Fatal(err)
				}
				return result
			}
			var evaluation PromptEvaluation
			served := generate(runner, prompt, &evaluation)
			payload, count, err := runner.SavePromptState(t.Context())
			if err != nil || count != len(served)-1 {
				t.Fatalf("saved %d tokens of %v: %v", count, served, err)
			}
			restoredRunner := open()
			if restored, err := restoredRunner.RestorePromptState(t.Context(), payload); err != nil || restored != count {
				t.Fatalf("restored %d tokens, saved %d: %v", restored, count, err)
			}
			extended := slices.Concat(served[:count], appended)
			restoredIDs := generate(restoredRunner, extended, &evaluation)
			if evaluation.Cached != count {
				t.Fatalf("the restored run cached %d tokens, want %d", evaluation.Cached, count)
			}
			if freshIDs := generate(open(), extended, &evaluation); evaluation.Cached != 0 || !slices.Equal(restoredIDs, freshIDs) {
				t.Fatalf("restored run %v, uncached run %v cached %d", restoredIDs, freshIDs, evaluation.Cached)
			}
		})
	}
}

// deviceCacheRoundTrip prefills the prompt on the device and on the host, and
// holds the read-back, the saved round trip and the appends across a page.
func deviceCacheRoundTrip(t *testing.T, runner *Runner, prompt, appended []tokenizer.TokenID) {
	t.Helper()
	ctx := t.Context()
	runner.mu.Lock()
	defer runner.mu.Unlock()
	_, original, err := runner.forwardDeviceCachedLocked(ctx, prompt, nil)
	if err != nil {
		t.Fatal(err)
	}
	_, host, err := runner.forwardCachedLocked(ctx, prompt, nil)
	if err != nil {
		t.Fatal(err)
	}
	downloaded, err := runner.downloadDeviceCache(ctx, original)
	if err != nil {
		t.Fatal(err)
	}
	if downloaded.Tokens != host.Tokens || downloaded.Position != host.Position || len(downloaded.Layers) != len(host.Layers) {
		t.Fatalf("read-back holds %d tokens at %d in %d layers, host %d at %d in %d",
			downloaded.Tokens, downloaded.Position, len(downloaded.Layers), host.Tokens, host.Position, len(host.Layers))
	}
	for layer := range host.Layers {
		for _, pair := range [][2]reference.Value{
			{downloaded.Layers[layer].Key, host.Layers[layer].Key},
			{downloaded.Layers[layer].Value, host.Layers[layer].Value},
		} {
			if !pair[0].Shape.Equal(pair[1].Shape) || len(pair[0].Data) != len(pair[1].Data) {
				t.Fatalf("layer %d read-back shape %v, host %v", layer, pair[0].Shape.Slice(), pair[1].Shape.Slice())
			}
			for index := range pair[1].Data {
				if delta := math.Abs(float64(pair[0].Data[index] - pair[1].Data[index])); delta > 2e-4 {
					t.Fatalf("layer %d element %d read-back %g host %g", layer, index, pair[0].Data[index], pair[1].Data[index])
				}
			}
		}
	}
	saved, err := runner.SaveCache(downloaded)
	if err != nil {
		t.Fatal(err)
	}
	loaded, err := runner.LoadCache(saved)
	if err != nil {
		t.Fatal(err)
	}
	uploaded, err := runner.uploadDeviceCache(ctx, loaded)
	if err != nil {
		t.Fatal(err)
	}
	capacity := uint64(cachePageCapacity(uint32(len(prompt)), runner.spec.ContextLength, runner.program.Decode.Session))
	if extent, _, _ := tensor.FinalExtent(uploaded.storage.keys[0].Shape); extent != capacity {
		t.Fatalf("uploaded key storage holds %d tokens, the page holds %d", extent, capacity)
	}
	again, err := runner.downloadDeviceCache(ctx, uploaded)
	if err != nil {
		t.Fatal(err)
	}
	if resaved, err := runner.SaveCache(again); err != nil || !slices.Equal(resaved, saved) {
		t.Fatalf("the uploaded cache reads back other bytes: %v", err)
	}
	for step, token := range appended {
		var next [2]*deviceKVCache
		for index, past := range []*deviceKVCache{original, uploaded} {
			if _, next[index], err = runner.forwardDeviceCachedLocked(ctx, []tokenizer.TokenID{token}, past); err != nil {
				t.Fatalf("append %d: %v", step, err)
			}
			_ = past.Release(ctx)
		}
		original, uploaded = next[0], next[1]
		if !slices.EqualFunc(original.Logits, uploaded.Logits, func(left, right float32) bool {
			return math.Float32bits(left) == math.Float32bits(right)
		}) || len(original.Logits) == 0 {
			t.Fatalf("append %d at %d tokens: uploaded logits %v, original %v", step, original.Tokens, uploaded.Logits, original.Logits)
		}
	}
	if original.Tokens <= uint32(capacity) {
		t.Fatalf("the appends end at %d tokens, inside the %d-token page", original.Tokens, capacity)
	}
	_ = original.Release(ctx)
	_ = uploaded.Release(ctx)
}

// TestHermeticCUDACapacityCachePageBoundary drives the capacity decode session
// across a KV-cache page boundary (page=4: boundary at pastTokens=4) and beyond,
// comparing the device append path against the proven-correct host concat path at
// every step. The steps reach pastTokens 2..7 so the session is replayed while the
// storage lives in a GROWN page (Cap=8) -- the exact condition that exposed the
// keyShape/keyCapacity off-by-one on the 12B.
func TestHermeticCUDACapacityCachePageBoundary(t *testing.T) {
	requireIntegration(t)
	cudatest.Require(t)
	path := testutil.HermeticLlamaGGUF(t, hermeticContext)
	runner, err := openF32FixtureRunner(path, OpenOptions{})
	if err != nil {
		t.Fatal(err)
	}
	defer runner.Close()
	deviceBatch, err := runner.NewContinuousBatch(ContinuousBatchOptions{
		MaxSequences: 1, Device: true,
	})
	if err != nil {
		t.Fatal(err)
	}
	defer deviceBatch.Close(t.Context())
	hostBatch, err := runner.NewContinuousBatch(ContinuousBatchOptions{
		MaxSequences: 1,
	})
	if err != nil {
		t.Fatal(err)
	}
	defer hostBatch.Close(t.Context())

	// prefill 2 -> decode single tokens; cache.Tokens after each step = 2,3,4,5,6,7,8.
	// pastTokens=4 crosses page 0->1 (Cap 4->8); pastTokens=5,6,7 replay inside the
	// grown page -- where the stale source capacity struck.
	steps := [][]tokenizer.TokenID{{1, 4}, {5}, {6}, {7}, {4}, {5}, {6}}
	for step, tokens := range steps {
		device, deviceErr := deviceBatch.Step(t.Context(), []SequenceBatchInput{{ID: 1, Tokens: tokens}})
		host, hostErr := hostBatch.Step(t.Context(), []SequenceBatchInput{{ID: 1, Tokens: tokens}})
		if deviceErr != nil || hostErr != nil {
			t.Fatalf("step %d device/host errors = %v/%v", step, deviceErr, hostErr)
		}
		if len(device) != 1 || len(host) != 1 || device[0].Tokens != host[0].Tokens ||
			device[0].Position != host[0].Position || len(device[0].Logits) != len(host[0].Logits) {
			t.Fatalf("step %d states = %+v/%+v", step, device, host)
		}
		for index := range device[0].Logits {
			if delta := math.Abs(float64(device[0].Logits[index] - host[0].Logits[index])); delta > 2e-4 {
				t.Fatalf("step %d (tokens=%d) logit %d device=%g host=%g delta=%g",
					step, device[0].Tokens, index, device[0].Logits[index], host[0].Logits[index], delta)
			}
		}
	}
}
