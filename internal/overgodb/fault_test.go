package overgodb

import (
	"os"
	"path/filepath"
	"testing"

	"overgo/internal/artifact"
)

// The journal's own faults are refused through the crash recorder (crash_faults_test.go); a snapshot's fault needs none.

func TestStreamingSnapshotFaultPreservesStore(t *testing.T) {
	t.Parallel()
	root := t.TempDir()
	store, err := Create(root)
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	if _, err := store.Commit(t.Context(), fixtureBatch(t)); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(root, checkpointDirectory), []byte("blocked"), storeFileMode); err != nil {
		t.Fatal(err)
	}
	if _, err := store.Snapshot(t.Context()); err == nil {
		t.Fatal("blocked checkpoint directory accepted")
	}
	next := fixtureDescriptor(t, artifact.KindOutput, "after-snapshot-fault")
	if _, err := store.Commit(t.Context(), artifact.Batch{
		Key: "fixture/fault/snapshot-followup", Artifacts: []artifact.Descriptor{next},
	}); err != nil {
		t.Fatalf("snapshot failure faulted commit path: %v", err)
	}
}
