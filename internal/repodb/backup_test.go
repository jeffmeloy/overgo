package repodb

import (
	"context"
	"os"
	"path/filepath"
	"strings"
	"testing"
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
	commit, err := store.Commit(context.Background(), batch)
	if err != nil {
		t.Fatal(err)
	}

	destination := filepath.Join(t.TempDir(), "backup")
	head, sequence, err := store.Backup(destination)
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
	resolved, ok, err := replica.ResolveAlias(context.Background(), fixtureAlias)
	if err != nil || !ok || resolved != batch.Aliases[0].Target {
		t.Fatalf("replica resolve = (%s, %v, %v)", resolved, ok, err)
	}

	if _, _, err := store.Backup(destination); err == nil || !strings.Contains(err.Error(), "already exists") {
		t.Fatalf("existing destination not refused: %v", err)
	}

	partialTarget := filepath.Join(t.TempDir(), "second")
	if err := os.MkdirAll(partialTarget+".partial", 0o755); err != nil {
		t.Fatal(err)
	}
	if _, _, err := store.Backup(partialTarget); err == nil || !strings.Contains(err.Error(), "stale partial") {
		t.Fatalf("stale partial not refused: %v", err)
	}

	empty, err := Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer empty.Close()
	if _, _, err := empty.Backup(filepath.Join(t.TempDir(), "empty-backup")); err == nil || !strings.Contains(err.Error(), "no commits") {
		t.Fatalf("empty store not refused: %v", err)
	}
}
