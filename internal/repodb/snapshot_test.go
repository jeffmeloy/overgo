package repodb

import (
	"context"
	"errors"
	"os"
	"path/filepath"
	"testing"

	"overgo/internal/artifact"
)

func TestSnapshotAnchorsReplayAndRetainsTail(t *testing.T) {
	root := t.TempDir()
	store, err := Open(root)
	if err != nil {
		t.Fatal(err)
	}
	base := fixtureBatch(t)
	first, err := store.Commit(context.Background(), base)
	if err != nil {
		t.Fatal(err)
	}
	snapshot, err := store.Snapshot(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if snapshot.Sequence != 1 || snapshot.Head != first {
		t.Fatalf("snapshot = %+v", snapshot)
	}
	again, err := store.Snapshot(context.Background())
	if err != nil || again.Path != snapshot.Path {
		t.Fatalf("repeated snapshot = (%+v, %v)", again, err)
	}
	tail := fixtureDescriptor(t, artifact.KindOutput, "snapshot-tail")
	second, err := store.Commit(context.Background(), artifact.Batch{
		Key: "snapshot/tail", Artifacts: []artifact.Descriptor{tail},
	})
	if err != nil {
		t.Fatal(err)
	}
	if err := store.Close(); err != nil {
		t.Fatal(err)
	}
	store, err = OpenReadOnly(root)
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	head, headSequence := store.Head()
	if head != second || headSequence != 2 {
		t.Fatalf("replayed head = (%s, %d)", head, headSequence)
	}
	if _, ok, err := store.Artifact(context.Background(), tail.ID); err != nil || !ok {
		t.Fatalf("tail artifact = (%v, %v)", ok, err)
	}
}

func TestSnapshotFallbackReport(t *testing.T) {
	root := t.TempDir()
	store, err := Open(root)
	if err != nil {
		t.Fatal(err)
	}
	batch := fixtureBatch(t)
	if _, err := store.Commit(context.Background(), batch); err != nil {
		t.Fatal(err)
	}
	snapshot, err := store.Snapshot(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if err := store.Close(); err != nil {
		t.Fatal(err)
	}
	file, err := os.OpenFile(snapshot.Path, os.O_RDWR, storeFileMode)
	if err != nil {
		t.Fatal(err)
	}
	info, err := file.Stat()
	if err != nil {
		t.Fatal(err)
	}
	value := []byte{0}
	if _, err := file.ReadAt(value, info.Size()-1); err != nil {
		t.Fatal(err)
	}
	value[0] ^= 0xff
	if _, err := file.WriteAt(value, info.Size()-1); err != nil {
		t.Fatal(err)
	}
	_ = file.Close()
	store, err = OpenReadOnly(root)
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	status := store.SnapshotReplay()
	if status.Loaded || status.Path != snapshot.Path || status.Fallback == "" {
		t.Fatalf("snapshot fallback = %+v", status)
	}
	if _, ok, err := store.Artifact(context.Background(), batch.Artifacts[0].ID); err != nil || !ok {
		t.Fatalf("rebuilt artifact = (%v, %v)", ok, err)
	}
}

func TestForeignSnapshotFallsBackToLog(t *testing.T) {
	sourceRoot := t.TempDir()
	source, err := Open(sourceRoot)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := source.Commit(context.Background(), fixtureBatch(t)); err != nil {
		t.Fatal(err)
	}
	snapshot, err := source.Snapshot(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if err := source.Close(); err != nil {
		t.Fatal(err)
	}
	targetRoot := t.TempDir()
	target, err := Open(targetRoot)
	if err != nil {
		t.Fatal(err)
	}
	targetArtifact := fixtureDescriptor(t, artifact.KindModel, "target-model")
	if _, err := target.Commit(context.Background(), artifact.Batch{
		Key: "snapshot/target", Artifacts: []artifact.Descriptor{targetArtifact},
	}); err != nil {
		t.Fatal(err)
	}
	if err := target.Close(); err != nil {
		t.Fatal(err)
	}
	directory := filepath.Join(targetRoot, snapshotDirectory)
	if err := os.MkdirAll(directory, storeDirectoryMode); err != nil {
		t.Fatal(err)
	}
	data, err := os.ReadFile(snapshot.Path)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(directory, filepath.Base(snapshot.Path)), data, storeFileMode); err != nil {
		t.Fatal(err)
	}
	target, err = OpenReadOnly(targetRoot)
	if err != nil {
		t.Fatal(err)
	}
	defer target.Close()
	if _, ok, err := target.Artifact(context.Background(), targetArtifact.ID); err != nil || !ok {
		t.Fatalf("target artifact = (%v, %v)", ok, err)
	}
}

func TestSnapshotRejectsInvalidStoreModeAndState(t *testing.T) {
	root := t.TempDir()
	store, err := Open(root)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := store.Snapshot(context.Background()); err == nil {
		t.Fatal("empty snapshot accepted")
	}
	if _, err := store.Commit(context.Background(), fixtureBatch(t)); err != nil {
		t.Fatal(err)
	}
	if err := store.Close(); err != nil {
		t.Fatal(err)
	}
	store, err = OpenReadOnly(root)
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	if _, err := store.Snapshot(context.Background()); !errors.Is(err, ErrReadOnly) {
		t.Fatalf("read-only snapshot error = %v", err)
	}
}
