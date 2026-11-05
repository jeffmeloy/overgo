package overgodb

import (
	"errors"
	"os"
	"path/filepath"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/testcheck"
)

func TestSnapshotAnchorsReplayAndRetainsTail(t *testing.T) {
	t.Parallel()
	root := t.TempDir()
	store, err := Create(root)
	testcheck.NoError(t, err)
	base := fixtureBatch(t)
	first, err := store.Commit(t.Context(), base)
	testcheck.NoError(t, err)
	snapshot, err := store.Snapshot(t.Context())
	testcheck.NoError(t, err)
	if snapshot.Sequence != 1 || snapshot.Head != first {
		t.Fatalf("snapshot = %+v", snapshot)
	}
	again, err := store.Snapshot(t.Context())
	if err != nil || again.Path != snapshot.Path {
		t.Fatalf("repeated snapshot = (%+v, %v)", again, err)
	}
	tail := fixtureDescriptor(t, artifact.KindOutput, "snapshot-tail")
	second, err := store.Commit(t.Context(), artifact.Batch{
		Key: "snapshot/tail", Artifacts: []artifact.Descriptor{tail},
	})
	testcheck.NoError(t, err)
	testcheck.NoError(t, store.Close())
	store, err = OpenReadOnly(root)
	testcheck.NoError(t, err)
	defer store.Close()
	head, headSequence := store.Head()
	if head != second || headSequence != 2 {
		t.Fatalf("replayed head = (%s, %d)", head, headSequence)
	}
	if _, ok, err := store.Artifact(t.Context(), tail.ID); err != nil || !ok {
		t.Fatalf("tail artifact = (%v, %v)", ok, err)
	}
}

func TestForeignSnapshotFallsBackToLog(t *testing.T) {
	t.Parallel()
	sourceRoot := t.TempDir()
	source, err := Create(sourceRoot)
	testcheck.NoError(t, err)
	if _, err := source.Commit(t.Context(), fixtureBatch(t)); err != nil {
		t.Fatal(err)
	}
	snapshot, err := source.Snapshot(t.Context())
	testcheck.NoError(t, err)
	testcheck.NoError(t, source.Close())
	targetRoot := t.TempDir()
	target, err := Create(targetRoot)
	testcheck.NoError(t, err)
	targetArtifact := fixtureDescriptor(t, artifact.KindModel, "target-model")
	if _, err := target.Commit(t.Context(), artifact.Batch{
		Key: "snapshot/target", Artifacts: []artifact.Descriptor{targetArtifact},
	}); err != nil {
		t.Fatal(err)
	}
	testcheck.NoError(t, target.Close())
	// A generation of another store carries anchors this journal does not
	// hold; it must cost a replay, never answers from the wrong catalog.
	testcheck.NoError(t, os.CopyFS(filepath.Join(targetRoot, checkpointDirectory, filepath.Base(snapshot.Path)), os.DirFS(snapshot.Path)))
	target, err = OpenReadOnly(targetRoot)
	testcheck.NoError(t, err)
	defer target.Close()
	if _, ok, err := target.Artifact(t.Context(), targetArtifact.ID); err != nil || !ok {
		t.Fatalf("target artifact = (%v, %v)", ok, err)
	}
}

func TestSnapshotRejectsInvalidStoreModeAndState(t *testing.T) {
	t.Parallel()
	root := t.TempDir()
	store, err := Create(root)
	testcheck.NoError(t, err)
	if _, err := store.Snapshot(t.Context()); err == nil {
		t.Fatal("empty snapshot accepted")
	}
	if _, err := store.Commit(t.Context(), fixtureBatch(t)); err != nil {
		t.Fatal(err)
	}
	testcheck.NoError(t, store.Close())
	store, err = OpenReadOnly(root)
	testcheck.NoError(t, err)
	defer store.Close()
	if _, err := store.Snapshot(t.Context()); !errors.Is(err, ErrReadOnly) {
		t.Fatalf("read-only snapshot error = %v", err)
	}
}
