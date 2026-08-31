package gate

import (
	"bytes"
	"io/fs"
	"os"
	"path/filepath"
	"testing"
	"time"

	"overgo/internal/plan"
)

// TestRecoverInterruptedCommitSurvivesStatRefreshedIndex reproduces the
// concurrent-observer incident: another session's `git status` refreshes
// the stat cache between the interrupted commit and its recovery,
// rewriting the index bytes without changing the tree they resolve to.
// Recovery must normalize such an index back to the exact write-ahead
// state under its own lock and complete, instead of refusing a rollback
// that is still provably safe.
func TestRecoverInterruptedCommitSurvivesStatRefreshedIndex(t *testing.T) {
	fixture := newInterruptedCommitFixture(t)
	var intent gateCommitIntent
	if err := readJSON(fixture.repo, gateCommitIntentFile, &intent); err != nil {
		t.Fatal(err)
	}
	indexPath, err := gateIndexPath(fixture.repo)
	if err != nil {
		t.Fatal(err)
	}
	// A read-only observer's status walk opportunistically rewrites the
	// stat cache. Force the same effect deterministically: move the tracked
	// file's timestamps so the cached stat is stale, refresh, and require
	// that the bytes really moved away from every captured state.
	past := time.Now().Add(-time.Hour)
	if err := os.Chtimes(filepath.Join(fixture.repo, fixture.sourcePath), past, past); err != nil {
		t.Fatal(err)
	}
	recoveryGit(t, fixture.repo, "update-index", "--refresh")
	recoveryGit(t, fixture.repo, "status", "--porcelain")
	refreshed, err := os.ReadFile(indexPath)
	if err != nil {
		t.Fatal(err)
	}
	if bytes.Equal(refreshed, intent.IndexBefore) || bytes.Equal(refreshed, intent.IndexAfter) ||
		bytes.Equal(refreshed, intent.IndexRestore) {
		t.Fatal("stat refresh left the index byte-identical; the fixture cannot exercise normalization")
	}
	preparation, err := recoverInterruptedCommit(fixture.repo, fixture.storePath)
	if err != nil {
		t.Fatal(err)
	}
	if preparation != fixture.preparation.ID {
		t.Fatalf("recovered preparation = %s, want %s", preparation, fixture.preparation.ID)
	}
	if head := recoveryGit(t, fixture.repo, "rev-parse", "HEAD"); head != fixture.parent {
		t.Fatalf("HEAD = %s, want parent %s", head, fixture.parent)
	}
	if got, err := os.ReadFile(filepath.Join(fixture.repo, filepath.FromSlash(plan.Path))); err != nil {
		t.Fatal(err)
	} else if string(got) != string(fixture.planBefore) {
		t.Fatalf("restored plan differs from the write-ahead plan")
	}
	assertNoCommitIntent(t, fixture.repo)
}

// TestNormalizeGateIndexRefusesForeignTree pins the normalization
// boundary: an index whose tree matches no captured write-ahead state is
// genuine drift, stays byte-identical on disk, and the byte comparison
// refuses recovery exactly as before.
func TestNormalizeGateIndexRefusesForeignTree(t *testing.T) {
	fixture := newInterruptedCommitFixture(t)
	var intent gateCommitIntent
	if err := readJSON(fixture.repo, gateCommitIntentFile, &intent); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(
		filepath.Join(fixture.repo, "foreign.go"), []byte("package candidate\n\nconst foreign = 2\n"), 0o644,
	); err != nil {
		t.Fatal(err)
	}
	recoveryGit(t, fixture.repo, "add", "foreign.go")
	indexPath, err := gateIndexPath(fixture.repo)
	if err != nil {
		t.Fatal(err)
	}
	drifted, err := os.ReadFile(indexPath)
	if err != nil {
		t.Fatal(err)
	}
	if err := normalizeGateIndexToStates(
		fixture.repo, indexPath, fs.FileMode(intent.IndexMode),
		intent.IndexBefore, intent.IndexAfter, intent.IndexRestore,
	); err != nil {
		t.Fatal(err)
	}
	after, err := os.ReadFile(indexPath)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(drifted, after) {
		t.Fatal("normalization rewrote an index with genuinely foreign staged work")
	}
	if _, err := recoverInterruptedCommit(fixture.repo, fixture.storePath); err == nil {
		t.Fatal("recovery accepted an index with genuinely foreign staged work")
	}
}
