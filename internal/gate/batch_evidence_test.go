package gate

import (
	"errors"
	"path/filepath"
	"strings"
	"sync"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/automationcheck"
	"overgo/internal/overgodb"
)

func TestBatchEvidenceRetainsReplanObligations(t *testing.T) {
	t.Parallel()
	g, batch, tree := verificationBatchFixture(t, "failure")
	checks := persistenceInvocations(t, g, batch, tree)
	cache := g.loadRetryCache()
	results, err := g.executeChecks(checks, nil, map[artifact.ID]artifact.ID{}, &cache, nil, nil)
	if err != nil || results[0].Err != nil || results[1].Err == nil {
		t.Fatalf("expected independent producer pass and consumer failure: %+v %v", results, err)
	}
	ledger, err := g.openBatchEvidence(checks, nil, &cache)
	if err != nil {
		t.Fatal(err)
	}
	old := ledger.state.Checks["acceptance-consumer"].Obligation
	if !ledger.state.Checks["acceptance-producer"].Result.Evidence.Valid() || ledger.state.Checks["acceptance-consumer"].Result.Evidence.Valid() {
		t.Fatal("failed sibling erased a pass or received credit")
	}
	if err := g.closeStore(); err != nil {
		t.Fatal(err)
	}
	batch.Checkpoints[1].Verify += " -timeout=1m"
	checks = persistenceInvocations(t, g, batch, tree)
	cache = automationcheck.NewEvidenceCache(g.environment.ID)
	ledger, err = g.openBatchEvidence(checks, nil, &cache)
	if err != nil {
		t.Fatal(err)
	}
	if ledger.state.Checks["acceptance-consumer"].Obligation == old || len(cache.Entries) != 1 {
		t.Fatal("changed verifier retained its obligation or lost the independent pass")
	}
	requireStoredPackageObligation(t, filepath.Join(g.repo, g.storePath), old)
	if err := g.closeStore(); err != nil {
		t.Fatal(err)
	}
	batch.Checkpoints = batch.Checkpoints[:1]
	checks = persistenceInvocations(t, g, batch, tree)
	if _, err := g.openBatchEvidence(checks, nil, &cache); err == nil {
		g.closeStore()
		t.Fatal("replan silently dropped an outstanding checkpoint")
	} else if !strings.Contains(err.Error(), "drops outstanding checkpoint acceptance-consumer") {
		t.Fatal(err)
	}
	g.verificationBatch = nil
	if _, err := g.openBatchEvidence(checks, nil, &cache); err == nil {
		g.closeStore()
		t.Fatal("removing the entire declaration erased outstanding obligations")
	}
}

func TestBatchEvidencePublicationFailure(t *testing.T) {
	t.Parallel()
	for _, conflict := range []bool{false, true} {
		t.Run(map[bool]string{false: "closed store", true: "competing writer"}[conflict], func(t *testing.T) {
			g, batch, tree := verificationBatchFixture(t, "pass")
			checks := persistenceInvocations(t, g, batch, tree)
			cache := g.loadRetryCache()
			ledger, err := g.openBatchEvidence(checks, nil, &cache)
			if err != nil {
				t.Fatal(err)
			}
			t.Cleanup(func() { _ = g.closeStore() })
			before := ledger.state.ID
			evidence, err := automationcheck.Run(t.Context(), checks[0])
			if err != nil {
				t.Fatal(err)
			}
			want := overgodb.ErrClosed
			if conflict {
				// A second gate context over the same store and plan, built field
				// by field: the context carries locks and is never copied.
				competitor := cloneGateContext(g)
				other, err := competitor.openBatchEvidence(checks, nil, &cache)
				if err != nil {
					t.Fatal(err)
				}
				if err := other.store.Close(); err != nil {
					t.Fatal(err)
				}
				want = overgodb.ErrAliasConflict
			} else if err := g.closeStore(); err != nil {
				t.Fatal(err)
			}
			if err := ledger.record(t.Context(), checks[0].Check.Name, evidence); !errors.Is(err, want) {
				t.Fatalf("publication=%v, want %v", err, want)
			}
			if ledger.state.ID != before || len(cache.Entries) != 0 {
				t.Fatal("failed publication advanced memory or received cache credit")
			}
			restored, err := g.openBatchEvidence(checks, nil, &cache)
			if err != nil {
				t.Fatal(err)
			}
			defer restored.store.Close()
			if len(cache.Entries) != 0 || restored.state.Checks[checks[0].Check.Name].Resolution.Valid() {
				t.Fatal("failed publication received restart credit")
			}
		})
	}
}

func TestBatchEvidenceConcurrentTerminalPublication(t *testing.T) {
	t.Parallel()
	g, batch, tree := verificationBatchFixture(t, "pass")
	checks := persistenceInvocations(t, g, batch, tree)
	cache := g.loadRetryCache()
	ledger, err := g.openBatchEvidence(checks, nil, &cache)
	if err != nil {
		t.Fatal(err)
	}
	defer g.closeStore()
	var results []automationcheck.Evidence
	for _, check := range checks[:len(batch.Checkpoints)] {
		result, err := automationcheck.Run(t.Context(), check)
		if err != nil {
			t.Fatal(err)
		}
		results = append(results, result)
	}
	var workers sync.WaitGroup
	for _, result := range results {
		workers.Go(func() {
			if err := ledger.record(t.Context(), result.Name, result); err != nil {
				t.Error(err)
			}
		})
	}
	workers.Wait()
	if t.Failed() {
		return
	}
	cache = automationcheck.NewEvidenceCache(g.environment.ID)
	restarted, err := g.openBatchEvidence(checks, nil, &cache)
	if err != nil {
		t.Fatal(err)
	}
	defer restarted.store.Close()
	if len(cache.Entries) != len(results) {
		t.Fatalf("concurrent terminal publication retained %d/%d results", len(cache.Entries), len(results))
	}
	for _, check := range restarted.state.Checks {
		if !check.Resolution.Valid() {
			t.Fatal("concurrent publication lost an independent resolution")
		}
	}
	// Passing checkpoints have not promoted the parent. Replanning cannot
	// remove their acceptance and thereby evade later input invalidation.
	batch.Checkpoints = batch.Checkpoints[:1]
	checks = persistenceInvocations(t, g, batch, tree)
	if removed, err := g.openBatchEvidence(checks, nil, &cache); err == nil {
		removed.store.Close()
		t.Fatal("a checkpoint pass authorized removal before parent promotion")
	}
}
