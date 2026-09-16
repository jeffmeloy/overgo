package gate

import (
	"context"
	"errors"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"slices"
	"testing"
	"time"

	"overgo/internal/artifact"
	"overgo/internal/overgodb"
	"overgo/internal/processcontrol"
	"overgo/internal/processlock"
	"overgo/internal/runrecord"
	"overgo/internal/testevidence"
	"overgo/internal/testutil"
)

const terminalEvidenceRepo = "OVERGO_TEST_TERMINAL_EVIDENCE_REPO"

// terminalReadyAnnouncement prefixes the line the child writes once it has
// published a terminal package result.
const terminalReadyAnnouncement = "terminal-ready "

func terminalEvidenceFixture(t *testing.T, root string) (*gateContext, map[string]artifact.ID) {
	t.Helper()
	g := &gateContext{repo: root, storePath: StorePath, environment: lifecycleTestEnvironment(t)}
	t.Cleanup(func() { _ = g.closeStore() })
	return g, map[string]artifact.ID{
		"fixture/good":    testutil.ArtifactID(t, artifact.KindEvidence, "good source"),
		"fixture/pending": testutil.ArtifactID(t, artifact.KindEvidence, "pending source"),
	}
}

func TestTerminalEvidenceProcess(t *testing.T) {
	root := os.Getenv(terminalEvidenceRepo)
	if root == "" {
		return
	}
	g, inputs := terminalEvidenceFixture(t, root)
	ledger, err := g.openPackageEvidence()
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = ledger.store.Close() })
	if err := ledger.prepare(t.Context(), []string{"fixture/good", "fixture/pending"}, "complete", inputs, g.retryCache); err != nil {
		t.Fatal(err)
	}
	reader, writer := io.Pipe()
	defer reader.Close()
	defer writer.Close()
	go func() {
		_, _ = io.WriteString(writer, `{"Action":"start","Package":"fixture/good"}
{"Action":"run","Package":"fixture/good","Test":"TestWorks"}
{"Action":"pass","Package":"fixture/good","Test":"TestWorks"}
{"Action":"pass","Package":"fixture/good"}
{"Action":"start","Package":"fixture/pending"}
`)
	}()
	observe := g.packagePassObserver(t.Context(), ledger, "complete", inputs)
	_, err = testevidence.GoTestJSONReader(reader, false, 1024, func(pkg string, passed bool) error {
		if err := observe(pkg, passed); err != nil {
			return err
		}
		// The parent reads this announcement from the child's output.
		fmt.Println(terminalReadyAnnouncement + pkg)
		return nil
	})
	t.Fatalf("process returned before kill: %v", err)
}

func TestTerminalEvidenceRestart(t *testing.T) {
	t.Parallel()
	t.Run("publication cost and identical durable results", testPackageLedgerCost)
	for _, failure := range []string{"closed store", "competing revocation"} {
		t.Run("publication refuses "+failure, func(t *testing.T) {
			root := t.TempDir()
			g, inputs := terminalEvidenceFixture(t, root)
			ledger, err := g.openPackageEvidence()
			if err != nil {
				t.Fatal(err)
			}
			t.Cleanup(func() { _ = ledger.store.Close() })
			packages := []string{"fixture/good"}
			if err := ledger.prepare(t.Context(), packages, "complete", inputs, g.retryCache); err != nil {
				t.Fatal(err)
			}
			want := overgodb.ErrClosed
			if failure == "closed store" {
				if err := ledger.store.Close(); err != nil {
					t.Fatal(err)
				}
			} else {
				other, _ := terminalEvidenceFixture(t, root)
				competitor, err := other.openPackageEvidence()
				if err != nil {
					t.Fatal(err)
				}
				t.Cleanup(func() { _ = competitor.store.Close() })
				if err := competitor.prepare(t.Context(), packages, "complete", inputs, other.retryCache); err != nil {
					t.Fatal(err)
				}
				if err := competitor.record(t.Context(), "fixture/good", false); err != nil {
					t.Fatal(err)
				}
				want = overgodb.ErrAliasConflict
			}
			if err := g.packagePassObserver(t.Context(), ledger, "complete", inputs)("fixture/good", true); !errors.Is(err, want) {
				t.Fatalf("publication error=%v want=%v", err, want)
			}
			pending, reused, err := g.packageCachePartition(packages, "complete", inputs)
			if err != nil || reused != 0 || !slices.Equal(pending, packages) {
				t.Fatalf("failed publication received credit: %v, %d, %v", pending, reused, err)
			}
			requireStoredPackageObligation(t, filepath.Join(root, StorePath), ledger.obligations["fixture/good"].ID)
		})
	}
	t.Run("receipt waits for a live transaction", func(t *testing.T) {
		root := t.TempDir()
		g, inputs := terminalEvidenceFixture(t, root)
		ledger, err := g.openPackageEvidence()
		if err != nil {
			t.Fatal(err)
		}
		t.Cleanup(func() { _ = ledger.store.Close() })
		if err := ledger.prepare(t.Context(), []string{"fixture/good"}, "complete", inputs, g.retryCache); err != nil {
			t.Fatal(err)
		}
		lock, err := processlock.Acquire(filepath.Join(root, StorePath, "overgodb.lock"), 0o644)
		if err != nil {
			t.Fatal(err)
		}
		defer lock.Close()
		ctx := t.Context()
		done := make(chan error, 1)
		// The live writer leaves the publication no early success; an early
		// refusal answers the receive below with its error.
		go func() { done <- ledger.record(ctx, "fixture/good", true) }()
		if err := lock.Close(); err != nil {
			t.Fatal(err)
		}
		if err := <-done; err != nil {
			t.Fatal(err)
		}
	})
	root := t.TempDir()
	initial, initialInputs := terminalEvidenceFixture(t, root)
	store, err := overgodb.Open(filepath.Join(root, StorePath))
	if err != nil {
		t.Fatal(err)
	}
	environmentBatch, err := initial.environment.Batch("terminal-environment")
	if err != nil {
		t.Fatal(err)
	}
	_, publishErr := artifact.CommitBatch(t.Context(), store, environmentBatch)
	if err := errors.Join(publishErr, store.Close()); err != nil {
		t.Fatal(err)
	}
	initialLedger, err := initial.openPackageEvidence()
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = initialLedger.store.Close() })
	if err := initialLedger.prepare(t.Context(), []string{"fixture/good", "fixture/pending"}, "complete", initialInputs, initial.retryCache); err != nil {
		t.Fatalf("published typed environment must retain its descriptor: %v", err)
	}
	ctx := t.Context()
	child := exec.CommandContext(ctx, os.Args[0], "-test.run=^TestTerminalEvidenceProcess$", "-test.timeout=1m")
	child.Env = append(os.Environ(), terminalEvidenceRepo+"="+root)
	ready := processcontrol.WatchLine(nil, terminalReadyAnnouncement)
	child.Stdout = ready
	if err := child.Start(); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = child.Process.Kill() })
	// The child announces its terminal publication on its own output.
	select {
	case <-ready.Line():
	case <-ctx.Done():
		_ = child.Process.Kill()
		_ = child.Wait()
		t.Fatal(context.Cause(ctx))
	}
	// A terminal writer must release the catalog while an unfinished sibling runs.
	store, err = overgodb.Open(filepath.Join(root, StorePath))
	if err != nil {
		t.Fatalf("unfinished verification retained the store writer: %v", err)
	}
	if err := store.Close(); err != nil {
		t.Fatal(err)
	}
	if err := child.Process.Kill(); err != nil {
		t.Fatal(err)
	}
	if err := child.Wait(); err == nil {
		t.Fatal("killed process reported success")
	}
	// Store receipts, not the disposable projection, must carry restart.
	if err := os.Remove(filepath.Join(root, filepath.FromSlash(gateRetryFile))); err != nil {
		t.Fatal(err)
	}
	g, inputs := terminalEvidenceFixture(t, root)
	ledger, err := g.openPackageEvidence()
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = ledger.store.Close() })
	packages := []string{"fixture/good", "fixture/pending"}
	if err := ledger.prepare(t.Context(), packages, "complete", inputs, g.retryCache); err != nil {
		t.Fatal(err)
	}
	pending, reused, err := g.packageCachePartition(packages, "complete", inputs)
	if err != nil || reused != 1 || !slices.Equal(pending, []string{"fixture/pending"}) {
		t.Fatalf("restart pending=%v reused=%d error=%v", pending, reused, err)
	}
	requireStoredPackageObligation(t, filepath.Join(root, StorePath), ledger.obligations["fixture/pending"].ID)
	if err := g.packagePassObserver(t.Context(), ledger, "complete", inputs)("fixture/good", false); err != nil {
		t.Fatal(err)
	}
	if err := ledger.prepare(t.Context(), packages, "complete", inputs, g.retryCache); err != nil {
		t.Fatal(err)
	}
	if pending, reused, err = g.packageCachePartition(packages, "complete", inputs); err != nil || reused != 0 || !slices.Equal(pending, packages) {
		t.Fatalf("revocation pending=%v reused=%d error=%v", pending, reused, err)
	}
	// Expanded scope and changed inputs retain old obligations without crediting them.
	old := ledger.obligations["fixture/good"].ID
	inputs["fixture/good"] = testutil.ArtifactID(t, artifact.KindEvidence, "changed source")
	inputs["fixture/new"] = testutil.ArtifactID(t, artifact.KindEvidence, "new source")
	packages = append(packages, "fixture/new")
	if err := ledger.prepare(t.Context(), packages, "complete", inputs, g.retryCache); err != nil {
		t.Fatal(err)
	}
	if old == ledger.obligations["fixture/good"].ID {
		t.Fatal("changed inputs retained obligation identity")
	}
	requireStoredPackageObligation(t, filepath.Join(root, StorePath), old)
	if pending, reused, err = g.packageCachePartition(packages, "complete", inputs); err != nil || reused != 0 || !slices.Equal(pending, packages) {
		t.Fatalf("expanded scope pending=%v reused=%d error=%v", pending, reused, err)
	}
	cancelled, stop := context.WithCancelCause(t.Context())
	stop(errors.New("test interrupted after terminal evidence"))
	if err := g.packagePassObserver(cancelled, ledger, "complete", inputs)("fixture/new", true); err != nil {
		t.Fatalf("cancellation discarded an observed terminal fact: %v", err)
	}
	t.Log("kill/restart: one terminal receipt recovered without tmp cache; incomplete, revoked, changed and new obligations remain uncredited")
}

func testPackageLedgerCost(t *testing.T) {
	const receipts, seedArtifacts = 32, 1024
	var expected artifact.CommitID
	for _, reopen := range []bool{true, false} {
		root := t.TempDir()
		store, err := overgodb.Open(filepath.Join(root, StorePath))
		if err != nil {
			t.Fatal(err)
		}
		seed := artifact.Batch{Key: "ledger-cost/seed"}
		for index := range seedArtifacts {
			seed.Artifacts = append(seed.Artifacts, artifact.Descriptor{ID: testutil.ArtifactID(t, artifact.KindEvidence, fmt.Sprintf("seed/%d", index))})
		}
		_, err = artifact.CommitBatch(t.Context(), store, seed)
		if err := errors.Join(err, store.Close()); err != nil {
			t.Fatal(err)
		}
		g, inputs := terminalEvidenceFixture(t, root)
		ledger, err := g.openPackageEvidence()
		if err != nil {
			t.Fatal(err)
		}
		t.Cleanup(func() { _ = ledger.store.Close() })
		var packages []string
		for index := range receipts {
			pkg := fmt.Sprintf("fixture/package/%d", index)
			packages = append(packages, pkg)
			inputs[pkg] = testutil.ArtifactID(t, artifact.KindEvidence, pkg)
		}
		if err := ledger.prepare(t.Context(), packages, "complete", inputs, g.retryCache); err != nil {
			t.Fatal(err)
		}
		opened := 1
		started := time.Now()
		for _, pkg := range packages {
			if reopen {
				if err := ledger.store.Close(); err != nil {
					t.Fatal(err)
				}
				ledger.store, err = overgodb.Open(filepath.Join(root, StorePath))
				if err != nil {
					t.Fatal(err)
				}
				opened++
			}
			if err := ledger.record(t.Context(), pkg, true); err != nil {
				t.Fatal(err)
			}
		}
		wall := time.Since(started)
		head, sequence := ledger.store.Head()
		if sequence != receipts+2 {
			t.Fatalf("publication denominator=%d want=%d", sequence, receipts+2)
		}
		if expected.Valid() && head != expected {
			t.Fatalf("retained handle changed durable results: %s != %s", head, expected)
		}
		expected = head
		t.Logf("reopen_per_receipt=%t seeded_artifacts=%d receipts=%d store_opens=%d publication_wall=%s head=%s", reopen, seedArtifacts, receipts, opened, wall, head)
	}
}

func requireStoredPackageObligation(t *testing.T, root string, id artifact.ID) {
	t.Helper()
	store, err := overgodb.Open(root)
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	if _, err := runrecord.RequireAgentObligation(t.Context(), store, id); err != nil {
		t.Fatal(err)
	}
}
