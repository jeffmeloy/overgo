package gate

import (
	"context"
	"errors"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"slices"
	"testing"
	"time"

	"overgo/internal/artifact"
	"overgo/internal/overgodb"
	"overgo/internal/runrecord"
	"overgo/internal/testevidence"
	"overgo/internal/testutil"
)

const terminalEvidenceRepo = "OVERGO_TEST_TERMINAL_EVIDENCE_REPO"

func terminalEvidenceFixture(t *testing.T, root string) (*gateContext, map[string]artifact.ID) {
	t.Helper()
	g := &gateContext{repo: root, storePath: StorePath, environment: lifecycleTestEnvironment(t)}
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
		return os.WriteFile(filepath.Join(root, "terminal-ready"), []byte(pkg), 0o600)
	})
	t.Fatalf("process returned before kill: %v", err)
}

func TestTerminalEvidenceRestart(t *testing.T) {
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
	if err := initialLedger.prepare(t.Context(), []string{"fixture/good", "fixture/pending"}, "complete", initialInputs, initial.retryCache); err != nil {
		t.Fatalf("published typed environment must retain its descriptor: %v", err)
	}
	ctx, cancel := context.WithTimeoutCause(t.Context(), 30*time.Second, errors.New("terminal receipt did not become durable"))
	defer cancel()
	child := exec.CommandContext(ctx, os.Args[0], "-test.run=^TestTerminalEvidenceProcess$", "-test.timeout=1m")
	child.Env = append(os.Environ(), terminalEvidenceRepo+"="+root)
	if err := child.Start(); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = child.Process.Kill() })
	ticker := time.Tick(10 * time.Millisecond)
	for {
		if _, err := os.Stat(filepath.Join(root, "terminal-ready")); err == nil {
			break
		}
		select {
		case <-ctx.Done():
			_ = child.Process.Kill()
			_ = child.Wait()
			t.Fatal(context.Cause(ctx))
		case <-ticker:
		}
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
	packages := []string{"fixture/good", "fixture/pending"}
	if err := ledger.prepare(t.Context(), packages, "complete", inputs, g.retryCache); err != nil {
		t.Fatal(err)
	}
	pending, reused, err := g.packageCachePartition(packages, "complete", inputs)
	if err != nil || reused != 1 || !slices.Equal(pending, []string{"fixture/pending"}) {
		t.Fatalf("restart pending=%v reused=%d error=%v", pending, reused, err)
	}
	requireStoredPackageObligation(t, ledger.root, ledger.obligations["fixture/pending"].ID)
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
	requireStoredPackageObligation(t, ledger.root, old)
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
