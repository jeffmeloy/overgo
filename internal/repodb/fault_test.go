package repodb

import (
	"context"
	"errors"
	"io"
	"os"
	"path/filepath"
	"testing"

	"overgo/internal/artifact"
)

var errInjectedStoreFault = errors.New("injected store fault")

type faultWriter struct {
	file      *os.File
	remaining int
	failSync  bool
}

func (w *faultWriter) Write(data []byte) (int, error) {
	if w.remaining <= 0 {
		return 0, errInjectedStoreFault
	}
	if len(data) > w.remaining {
		data = data[:w.remaining]
	}
	written, err := w.file.Write(data)
	w.remaining -= written
	return written, err
}

func (w *faultWriter) Sync() error {
	if w.failSync {
		return errInjectedStoreFault
	}
	return w.file.Sync()
}

func TestAppendFaultClosesEveryStoreSurfaceUntilReopen(t *testing.T) {
	root := t.TempDir()
	store, err := Open(root)
	if err != nil {
		t.Fatal(err)
	}
	base := fixtureBatch(t)
	if _, err := store.Commit(context.Background(), base); err != nil {
		t.Fatal(err)
	}
	head, sequence := store.Head()
	store.log.writer = &faultWriter{file: store.log.file, remaining: frameHeaderBytes / 2}
	pending := fixtureDescriptor(t, artifact.KindOutput, "faulted-output")
	if _, err := store.Commit(context.Background(), artifact.Batch{
		Key: "fixture/fault/partial", Artifacts: []artifact.Descriptor{pending},
	}); !errors.Is(err, ErrStoreFaulted) || !errors.Is(err, errInjectedStoreFault) {
		t.Fatalf("partial append error = %v", err)
	}
	if current, currentSequence := store.Head(); current != head || currentSequence != sequence {
		t.Fatalf("faulted head = (%s, %d), want (%s, %d)", current, currentSequence, head, sequence)
	}
	assertFaultedStoreSurfaces(t, store, pending)
	if err := store.Close(); err != nil {
		t.Fatal(err)
	}

	store, err = Open(root)
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	if _, ok, err := store.Artifact(context.Background(), pending.ID); err != nil || ok {
		t.Fatalf("partial artifact after recovery = (%v, %v)", ok, err)
	}
	if _, err := store.Commit(context.Background(), artifact.Batch{
		Key: "fixture/fault/retry", Artifacts: []artifact.Descriptor{pending},
	}); err != nil {
		t.Fatal(err)
	}
}

func TestSyncFaultReplaysCompleteUncertainCommit(t *testing.T) {
	root := t.TempDir()
	store, err := Open(root)
	if err != nil {
		t.Fatal(err)
	}
	store.log.writer = &faultWriter{file: store.log.file, remaining: int(^uint(0) >> 1), failSync: true}
	pending := fixtureDescriptor(t, artifact.KindEvidence, "sync-uncertain")
	id, err := store.Commit(context.Background(), artifact.Batch{
		Key: "fixture/fault/sync", Artifacts: []artifact.Descriptor{pending},
	})
	if !id.Valid() || !errors.Is(err, ErrStoreFaulted) || !errors.Is(err, errInjectedStoreFault) {
		t.Fatalf("sync append = (%s, %v)", id, err)
	}
	if err := store.Close(); err != nil {
		t.Fatal(err)
	}
	store, err = Open(root)
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	if descriptor, ok, err := store.Artifact(context.Background(), pending.ID); err != nil || !ok || descriptor.ID != pending.ID {
		t.Fatalf("uncertain commit replay = (%+v, %v, %v)", descriptor, ok, err)
	}
}

func TestSnapshotFailureLeavesCommitPathHealthy(t *testing.T) {
	root := t.TempDir()
	store, err := Open(root)
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	if _, err := store.Commit(context.Background(), fixtureBatch(t)); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(root, snapshotDirectory), []byte("blocked"), storeFileMode); err != nil {
		t.Fatal(err)
	}
	if _, err := store.Snapshot(context.Background()); err == nil {
		t.Fatal("blocked snapshot directory accepted")
	}
	next := fixtureDescriptor(t, artifact.KindOutput, "after-snapshot-fault")
	if _, err := store.Commit(context.Background(), artifact.Batch{
		Key: "fixture/fault/snapshot-followup", Artifacts: []artifact.Descriptor{next},
	}); err != nil {
		t.Fatalf("snapshot failure faulted commit path: %v", err)
	}
}

func assertFaultedStoreSurfaces(t *testing.T, store *Store, pending artifact.Descriptor) {
	t.Helper()
	ctx := context.Background()
	if _, _, err := store.Artifact(ctx, pending.ID); !errors.Is(err, ErrStoreFaulted) {
		t.Fatalf("faulted artifact read error = %v", err)
	}
	if _, err := store.Query(ctx, Query{MaxResults: 1, Projection: ProjectArtifacts}); !errors.Is(err, ErrStoreFaulted) {
		t.Fatalf("faulted query error = %v", err)
	}
	if _, err := store.Snapshot(ctx); !errors.Is(err, ErrStoreFaulted) {
		t.Fatalf("faulted snapshot error = %v", err)
	}
	if _, err := store.Commit(ctx, artifact.Batch{
		Key: "fixture/fault/rejected", Artifacts: []artifact.Descriptor{pending},
	}); !errors.Is(err, ErrStoreFaulted) {
		t.Fatalf("faulted commit error = %v", err)
	}
	if _, err := store.log.file.Seek(0, io.SeekCurrent); err != nil {
		t.Fatalf("faulted log was unexpectedly closed: %v", err)
	}
}
