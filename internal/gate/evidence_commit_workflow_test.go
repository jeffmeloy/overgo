package gate

import (
	"errors"
	"os"
	"path/filepath"
	"testing"

	"overgo/internal/authoritylock"
	"overgo/internal/processlock"
	"overgo/internal/runrecord"
)

// TestEvidenceCommitWorkflow certifies the complete producer-to-commit evidence
// workflow in one scenario, on retained captures and counted acquisition stubs,
// independently of the full modality report. It threads the whole chain through
// the crash points the narrower recovery and restart tests own in isolation:
// writer contention before a receipt publication, a producer death that leaves
// one package without a durable result, a restart that resumes the exact durable
// receipts and re-executes only the interrupted package, and an interruption
// between the committed final publication and its recording that a supervisor
// restart recovers without a second commit or a repeated acquisition.
func TestEvidenceCommitWorkflow(t *testing.T) {
	t.Parallel()
	fixture := newInterruptedCommitFixture(t)
	// Recovery admits the canonical store name; the fixture is otherwise wholly
	// inside this isolated repository.
	if err := os.Rename(filepath.Join(fixture.repo, fixture.storePath), filepath.Join(fixture.repo, StorePath)); err != nil {
		t.Fatal(err)
	}
	fixture.storePath = StorePath

	// Producer: acquire durable package receipts through writer contention.
	producer, inputs := terminalEvidenceFixture(t, fixture.repo)
	ledger, err := producer.openPackageEvidence()
	if err != nil {
		t.Fatal(err)
	}
	packages := []string{"fixture/good", "fixture/pending"}
	if err := ledger.prepare(t.Context(), packages, "complete", inputs, producer.retryCache); err != nil {
		t.Fatal(err)
	}
	acquisitions := map[string]int{}

	// A live transaction holds the catalog: the receipt waits for it and still
	// lands durably once it releases. This is the one acquisition of good.
	lock, err := processlock.Acquire(filepath.Join(fixture.repo, StorePath, "overgodb.lock"), 0o644)
	if err != nil {
		t.Fatal(err)
	}
	contended := make(chan error, 1)
	acquisitions["fixture/good"]++
	go func() { contended <- ledger.record(t.Context(), "fixture/good", true) }()
	if err := lock.Close(); err != nil {
		t.Fatal(err)
	}
	if err := <-contended; err != nil {
		t.Fatalf("contended receipt did not wait for the live transaction: %v", err)
	}

	// The producer dies before the second package reaches a durable result.
	if err := producer.closeStore(); err != nil {
		t.Fatal(err)
	}

	// Restart resumes the exact durable receipt and re-executes only the
	// interrupted package; the durable acquisition never repeats.
	restarted, _ := terminalEvidenceFixture(t, fixture.repo)
	ledger, err = restarted.openPackageEvidence()
	if err != nil {
		t.Fatal(err)
	}
	if err := ledger.prepare(t.Context(), packages, "complete", inputs, restarted.retryCache); err != nil {
		t.Fatal(err)
	}
	pending, reused, err := restarted.packageCachePartition(packages, "complete", inputs)
	if err != nil || reused != 1 || len(pending) != 1 || pending[0] != "fixture/pending" {
		t.Fatalf("restart lost exact receipts: pending=%v reused=%d err=%v", pending, reused, err)
	}
	acquisitions["fixture/pending"]++
	if err := ledger.record(t.Context(), "fixture/pending", true); err != nil {
		t.Fatal(err)
	}
	if acquisitions["fixture/good"] != 1 || acquisitions["fixture/pending"] != 1 {
		t.Fatalf("workflow repeated a durable acquisition: %v", acquisitions)
	}
	if err := restarted.closeStore(); err != nil {
		t.Fatal(err)
	}

	// Commit: the final publication landed but its recording acknowledgement was
	// lost. A supervisor restart recovers without a second commit or publication.
	attempt, batch := successfulInterruptedAttemptBatch(t, fixture, true, "gate/final/"+fixture.preparation.ID.String())
	store := openRecoveryStore(t, fixture)
	defer store.Close()
	if _, err := store.CommitAs(t.Context(), gateProducer, batch); err != nil {
		t.Fatal(err)
	}
	before, sequence := store.Head()
	g := gateContext{repo: fixture.repo, preparation: fixture.preparation}
	if err := g.oweRecord(batch, errors.New("supervisor lost the final store recording")); err == nil {
		t.Fatal("interrupted recording retained no debt")
	}
	lock2, err := authoritylock.Acquire(fixture.repo)
	if err != nil {
		t.Fatal(err)
	}
	defer lock2.Close()
	ref, err := admitPendingGateState(fixture.repo, fixture.storePath, store)
	if err != nil || ref != "ratchet/recover" {
		t.Fatalf("supervisor recovery resume = %q, %v", ref, err)
	}
	if head, count := store.Head(); head != before || count != sequence {
		t.Fatal("recovery duplicated the final publication")
	}
	if head := recoveryGit(t, fixture.repo, "rev-parse", "HEAD"); head != fixture.commit {
		t.Fatalf("recovery made another commit: %s", head)
	}
	if count := recoveryGit(t, fixture.repo, "rev-list", "--count", fixture.parent+"..HEAD"); count != "1" {
		t.Fatalf("commit count=%s", count)
	}
	if _, err := os.Stat(filepath.Join(fixture.repo, gateDebtFile)); !errors.Is(err, os.ErrNotExist) {
		t.Fatalf("recording debt remains after recovery: %v", err)
	}
	if _, err := runrecord.VerifyAttemptGate(t.Context(), store, attempt); err != nil {
		t.Fatal(err)
	}
	t.Logf("producer-to-commit workflow: contended+restarted acquisitions=%v (each durable once), one commit, one final publication, recording recovered; excludes the full modality report", acquisitions)
}
