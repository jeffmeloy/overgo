package gate

import (
	"context"
	"errors"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"overgo/internal/overgodb"
	"overgo/internal/processlock"
	"overgo/internal/processmeasure"
	"overgo/internal/runrecord"
)

func TestGateStoreLifetime(t *testing.T) {
	t.Parallel()
	repo := newLifecycleRepo(t)
	runGitFixture(t, repo, "init", "-q")
	runGitFixture(t, repo, "config", "user.email", "gate-store@example.invalid")
	runGitFixture(t, repo, "config", "user.name", "Gate Store Test")
	runGitFixture(t, repo, "commit", "--allow-empty", "-q", "-m", "baseline")
	g, inputs := terminalEvidenceFixture(t, repo)
	t.Cleanup(func() { _ = g.closeStore() })
	g.start = time.Now()
	g.clock = processmeasure.NewStopwatch()
	g.planRef = "fixture/do"
	head, err := command(repo, "git", "rev-parse", "HEAD")
	if err != nil {
		t.Fatal(err)
	}
	g.planHead = strings.TrimSpace(head)
	store, err := g.openStore()
	if err != nil {
		t.Fatal(err)
	}
	if err := g.prepareWithStore(store); err != nil {
		t.Fatal(err)
	}
	// A different process may own a transaction while the gate stays open.
	lock, err := processlock.Acquire(filepath.Join(repo, g.storePath, "overgodb.lock"), 0o600)
	if err != nil {
		t.Fatal(err)
	}
	defer lock.Close()
	ledger, err := g.openPackageEvidence()
	if err != nil || ledger.store != store {
		t.Fatalf("evidence reopened admission store: %v", err)
	}
	ctx, cancel := context.WithCancelCause(t.Context())
	done := make(chan error, 1)
	// The foreign owner leaves the publication no early success; an early
	// refusal would answer the receive below with its error.
	go func() { done <- ledger.prepare(ctx, []string{"fixture/good"}, "complete", inputs, g.retryCache) }()
	cause := errors.New("operator cancelled queued publication")
	cancel(cause)
	if err := <-done; !errors.Is(err, context.Canceled) && !errors.Is(err, cause) {
		t.Fatalf("cancelled publication: %v", err)
	}
	if len(g.retryCache.Entries) != 0 {
		t.Fatal("cancelled publication received evidence credit")
	}
	if other, err := processlock.Acquire(filepath.Join(repo, g.storePath, "overgodb.lock"), 0o600); !errors.Is(err, processlock.ErrBusy) {
		if other != nil {
			other.Close()
		}
		t.Fatalf("cancellation disturbed foreign owner: %v", err)
	}
	go func() {
		done <- ledger.prepare(t.Context(), []string{"fixture/good"}, "complete", inputs, g.retryCache)
	}()
	if err := lock.Close(); err != nil {
		t.Fatal(err)
	}
	if err := <-done; err != nil {
		t.Fatal(err)
	}
	if err := ledger.record(t.Context(), "fixture/good", true); err != nil {
		t.Fatal(err)
	}
	// Terminal recording must reuse the same handle even on a failed gate.
	g.steps = []runrecord.GateStep{{Name: "fixture-refusal", Phase: runrecord.PhaseValidate, Outcome: runrecord.StepFailed, DurationNS: 1}}
	if err := g.record(runrecord.OutcomeFailed, "fixture-refusal"); err != nil {
		t.Fatal(err)
	}
	if g.store != store {
		t.Fatal("terminal recording replaced the store owner")
	}
	if err := g.closeStore(); err != nil {
		t.Fatal(err)
	}
	if _, _, err := store.Artifact(t.Context(), g.preparation.ID); !errors.Is(err, overgodb.ErrClosed) {
		t.Fatalf("store remained open: %v", err)
	}
	restarted, err := g.openPackageEvidence()
	if err != nil {
		t.Fatal(err)
	}
	if err := restarted.prepare(t.Context(), []string{"fixture/good"}, "complete", inputs, g.retryCache); err != nil {
		t.Fatal(err)
	}
	if len(g.retryCache.Entries) != 1 {
		t.Fatal("restart lost completed package receipt")
	}
	final, found, err := runrecord.GateFinalizationForPreparation(t.Context(), restarted.store, g.preparation.ID)
	if err != nil || !found || final.Outcome != runrecord.OutcomeFailed {
		t.Fatalf("terminal record = %+v, %t, %v", final, found, err)
	}
}
