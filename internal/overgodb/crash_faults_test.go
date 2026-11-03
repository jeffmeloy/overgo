//go:build crashpoints

package overgodb

import (
	"bytes"
	"errors"
	"io"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"overgo/internal/artifact"
)

// The journal's own faults, each the refusal of one durable operation of an append through the crash recorder, as a
// full volume or a failed flush refuses it, run by the outer tests of durable_operations_test.go.

// crashRefuseJournal attaches to root, until t ends, a recorder that refuses the journal's next operation of kind, once.
func crashRefuseJournal(t *testing.T, root string, kind crashOpKind) {
	t.Helper()
	recorder := newCrashRecorder(root)
	recorder.target = crashOp{kind: kind, path: storeFilename}
	t.Cleanup(crashAttach(recorder))
}

// TestLogAppendStreamsFrame holds an append to streaming its frame: the header, the payload and the trailer reach the
// journal as three writes, with no copy that joins them, and one flush follows.
func TestLogAppendStreamsFrame(t *testing.T) {
	t.Parallel()
	root := t.TempDir()
	recorder := newCrashRecorder(root)
	t.Cleanup(crashAttach(recorder))
	file, err := os.OpenFile(filepath.Join(root, storeFilename), os.O_CREATE|os.O_RDWR, storeFileMode)
	if err != nil {
		t.Fatal(err)
	}
	defer file.Close()
	payload := []byte(`{"streamed":true}`)
	log := recordLog{file: file}
	if id, _, _, err := log.append(1, artifact.CommitID{}, payload); err != nil {
		t.Fatalf("append = (%s, %v)", id, err)
	}
	_, header, trailer := encodeFrame(1, artifact.CommitID{}, payload)
	want := []crashOp{{kind: crashWriteOp, data: header}, {kind: crashWriteOp, data: payload}, {kind: crashWriteOp, data: trailer[:]}, {kind: crashFsync}}
	if len(recorder.ops) != len(want) {
		t.Fatalf("an append made %d durable operations, want %d", len(recorder.ops), len(want))
	}
	for index, op := range recorder.ops {
		if op.kind != want[index].kind || op.path != storeFilename || !bytes.Equal(op.data, want[index].data) {
			t.Fatalf("operation %d = %s %s of %d bytes, want %s of %d", index, op.kind, op.path, len(op.data), want[index].kind, len(want[index].data))
		}
	}
}

// TestAppendFaultClosesEveryStoreSurfaceUntilReopen refuses an append part-way, after its blob was made durable: the
// catalog's facets and head stay as they were, every surface of the store refuses until a reopen, and the blob stays
// uncommitted, so the reopened store holds no artifact for it and takes the same batch again.
func TestAppendFaultClosesEveryStoreSurfaceUntilReopen(t *testing.T) {
	t.Parallel()
	root := t.TempDir()
	store, err := Create(root)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := store.Commit(t.Context(), fixtureBatch(t)); err != nil {
		t.Fatal(err)
	}
	head, sequence := store.Head()
	digest := catalogDigest(t, store.state)
	// Blob-sized: smaller content rides in the frame and has no blob to orphan.
	payload := []byte(strings.Repeat("orphaned by append fault ", inlineContentLimit))
	pending := crashContent(t, artifact.KindTensorSet, payload)
	batch := artifact.Batch{Key: "fixture/fault/partial", Contents: []artifact.Content{pending}}
	crashRefuseJournal(t, root, crashWriteOp)
	if _, err := store.Commit(t.Context(), batch); !errors.Is(err, ErrStoreFaulted) || !errors.Is(err, errCrashInjected) {
		t.Fatalf("partial append error = %v", err)
	}
	if current, currentSequence := store.Head(); current != head || currentSequence != sequence {
		t.Fatalf("faulted head = (%s, %d), want (%s, %d)", current, currentSequence, head, sequence)
	}
	if catalogDigest(t, store.state) != digest {
		t.Fatal("a faulted append changed the catalog's facets")
	}
	if !store.blobs.has(pending.Descriptor.ID) {
		t.Fatal("the blob was not durable before the append")
	}
	assertFaultedStoreSurfaces(t, store, pending.Descriptor)
	if err := store.Close(); err != nil {
		t.Fatal(err)
	}

	store, err = Create(root)
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	if _, ok, err := store.Artifact(t.Context(), pending.Descriptor.ID); err != nil || ok {
		t.Fatalf("partial artifact after recovery = (%v, %v)", ok, err)
	}
	if _, err := store.Commit(t.Context(), batch); err != nil {
		t.Fatal(err)
	}
}

// TestSyncFaultReplaysCompleteUncertainCommit refuses an append's flush after its frame was written whole: the commit is
// uncertain, the store faults, and a reopen replays the whole commit.
func TestSyncFaultReplaysCompleteUncertainCommit(t *testing.T) {
	t.Parallel()
	root := t.TempDir()
	store, err := Create(root)
	if err != nil {
		t.Fatal(err)
	}
	crashRefuseJournal(t, root, crashFsync)
	pending := fixtureDescriptor(t, artifact.KindEvidence, "sync-uncertain")
	id, err := store.Commit(t.Context(), artifact.Batch{
		Key: "fixture/fault/sync", Artifacts: []artifact.Descriptor{pending},
	})
	if !id.Valid() || !errors.Is(err, ErrStoreFaulted) || !errors.Is(err, errCrashInjected) {
		t.Fatalf("sync append = (%s, %v)", id, err)
	}
	if err := store.Close(); err != nil {
		t.Fatal(err)
	}
	store, err = Create(root)
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	if descriptor, ok, err := store.Artifact(t.Context(), pending.ID); err != nil || !ok || descriptor.ID != pending.ID {
		t.Fatalf("uncertain commit replay = (%+v, %v, %v)", descriptor, ok, err)
	}
}

func assertFaultedStoreSurfaces(t *testing.T, store *Store, pending artifact.Descriptor) {
	t.Helper()
	ctx := t.Context()
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
