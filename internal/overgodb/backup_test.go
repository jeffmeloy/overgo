package overgodb

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"overgo/internal/artifact"
)

// TestStreamingSnapshotRoundTrip pins replay identity and publication refusal.
func TestStreamingSnapshotRoundTrip(t *testing.T) {
	root := t.TempDir()
	store, err := Open(root)
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	batch := fixtureBatch(t)
	commit, err := store.Commit(t.Context(), batch)
	if err != nil {
		t.Fatal(err)
	}

	destination := filepath.Join(t.TempDir(), "backup")
	backupReport, err := store.Backup(t.Context(), destination)
	head, sequence := backupReport.Head, backupReport.Sequence
	if err != nil {
		t.Fatal(err)
	}
	if head != commit || sequence == 0 {
		t.Fatalf("backup identity = %s@%d, want %s", head, sequence, commit)
	}

	replica, err := OpenReadOnly(destination)
	if err != nil {
		t.Fatal(err)
	}
	defer replica.Close()
	replicaHead, replicaSequence := replica.Head()
	if replicaHead != head || replicaSequence != sequence {
		t.Fatalf("replica head = %s@%d, want %s@%d", replicaHead, replicaSequence, head, sequence)
	}
	resolved, ok, err := replica.ResolveAlias(t.Context(), fixtureAlias)
	if err != nil || !ok || resolved != batch.Aliases[0].Target {
		t.Fatalf("replica resolve = (%s, %v, %v)", resolved, ok, err)
	}

	if _, err := store.Backup(t.Context(), destination); err == nil || !strings.Contains(err.Error(), "already exists") {
		t.Fatalf("existing destination not refused: %v", err)
	}

	partialTarget := filepath.Join(t.TempDir(), "second")
	if err := os.MkdirAll(partialTarget+".partial", 0o755); err != nil {
		t.Fatal(err)
	}
	if _, err := store.Backup(t.Context(), partialTarget); err != nil {
		t.Fatalf("empty partial did not resume: %v", err)
	}

	empty, err := Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer empty.Close()
	if _, err := empty.Backup(t.Context(), filepath.Join(t.TempDir(), "empty-backup")); err == nil || !strings.Contains(err.Error(), "no commits") {
		t.Fatalf("empty store not refused: %v", err)
	}
}

func TestConcurrentBackupExtent(t *testing.T) {
	ctx := t.Context()
	store, err := Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	first, err := store.Commit(ctx, fixtureBatch(t))
	if err != nil {
		t.Fatal(err)
	}
	store.mu.RLock()
	head, sequence, extent, root := store.head, store.sequence, store.replayEnd, store.root
	store.mu.RUnlock()
	tail := fixtureDescriptor(t, artifact.KindOutput, "concurrent-backup-tail")
	if _, err := store.Commit(ctx, artifact.Batch{
		Key: "backup/concurrent-tail", Artifacts: []artifact.Descriptor{tail},
	}); err != nil {
		t.Fatal(err)
	}
	destination := t.TempDir()
	var copied BackupReport
	if err := copied.copyFile(ctx,
		filepath.Join(root, storeFilename), filepath.Join(destination, storeFilename), extent, "", false,
	); err != nil {
		t.Fatal(err)
	}
	if head != first || sequence == 0 {
		t.Fatalf("captured identity = %s@%d, want %s", head, sequence, first)
	}
	replica, err := OpenReadOnly(destination)
	if err != nil {
		t.Fatal(err)
	}
	defer replica.Close()
	if _, found, err := replica.Artifact(ctx, tail.ID); err != nil || found {
		t.Fatalf("post-capture tail = (%v, %v)", found, err)
	}
}
