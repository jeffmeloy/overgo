//go:build windows

package inference

import (
	"context"
	"math"
	"reflect"
	"testing"

	cudatest "overgo/internal/cuda/testutil"
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
	// page and at past 10 on the 16-token page, three replays; the forked
	// cohort below then compiles at past 11 and replays at 12, both on the
	// 16-token page.
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
	if err := deviceBatch.Fork(1, 2); err != nil {
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
		t.Fatalf("forked states = %+v", branched)
	}
	deviceBatch.mu.Lock()
	cohortSession := deviceBatch.sequences[1].device.session
	var cohortReplays uint64
	if cohortSession != nil {
		cohortReplays = cohortSession.replays
	}
	deviceBatch.mu.Unlock()
	if cohortSession == nil {
		t.Fatal("forked cohort has no reusable session")
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
