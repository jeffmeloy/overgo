package overgodb

import (
	"context"
	"errors"
	"io"
	"os"
	"strings"
	"testing"

	"overgo/internal/artifact"
)

// TestExternalBlobCommitAtomicity holds the external-blob commit to
// its contract: the journal frame is a small envelope while the bytes
// live in the blob store; a committed descriptor whose required blob
// is missing refuses content access with exact evidence instead of
// degrading; an append fault after blob preparation leaves only an
// unreachable blob; and the whole corpus replays from the envelope
// journal plus blobs.
func TestExternalBlobCommitAtomicity(t *testing.T) {
	ctx := context.Background()
	root := t.TempDir()
	store, err := Open(root)
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()

	payload := make([]byte, 8192)
	for index := range payload {
		payload[index] = byte(index)
	}
	id, err := artifact.IdentifyBytes(artifact.KindTensorSet, payload)
	if err != nil {
		t.Fatal(err)
	}
	descriptor := artifact.Descriptor{ID: id, Size: uint64(len(payload)), MediaType: "application/octet-stream"}
	before, err := os.Stat(store.log.file.Name())
	if err != nil {
		t.Fatal(err)
	}
	if _, err := store.Commit(ctx, artifact.Batch{
		Key:      "blob-commit/large",
		Contents: []artifact.Content{{Descriptor: descriptor, Data: payload}},
	}); err != nil {
		t.Fatal(err)
	}
	after, err := os.Stat(store.log.file.Name())
	if err != nil {
		t.Fatal(err)
	}
	if envelope := after.Size() - before.Size(); envelope >= int64(len(payload)) {
		t.Fatalf("journal grew %d bytes for a %d-byte content; envelope embeds bytes", envelope, len(payload))
	}
	if !store.blobs.has(id) {
		t.Fatal("committed content has no blob")
	}
	_, reader, found, err := store.OpenContent(ctx, id)
	if err != nil || !found {
		t.Fatalf("content = (%v, %v)", found, err)
	}
	data, err := io.ReadAll(reader)
	if err != nil || len(data) != len(payload) {
		t.Fatalf("round trip = (%d bytes, %v)", len(data), err)
	}

	path, err := store.blobs.path(id)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.Remove(path); err != nil {
		t.Fatal(err)
	}
	if _, _, _, err := store.OpenContent(ctx, id); err == nil ||
		!strings.Contains(err.Error(), "requires an absent blob") || !strings.Contains(err.Error(), id.String()) {
		t.Fatalf("absent blob access = %v", err)
	}
	if err := store.blobs.prepare(id, payload); err != nil {
		t.Fatal(err)
	}

	store.log.writer = &faultWriter{file: store.log.file, remaining: frameHeaderBytes / 2}
	orphanPayload := []byte("orphaned by append fault")
	orphanID, err := artifact.IdentifyBytes(artifact.KindTensorSet, orphanPayload)
	if err != nil {
		t.Fatal(err)
	}
	orphanDescriptor := artifact.Descriptor{ID: orphanID, Size: uint64(len(orphanPayload)), MediaType: "text/plain"}
	if _, err := store.Commit(ctx, artifact.Batch{
		Key:      "blob-commit/orphan",
		Contents: []artifact.Content{{Descriptor: orphanDescriptor, Data: orphanPayload}},
	}); !errors.Is(err, ErrStoreFaulted) {
		t.Fatalf("append fault error = %v", err)
	}
	if !store.blobs.has(orphanID) {
		t.Fatal("blob was not durable before the append")
	}
	if err := store.Close(); err != nil {
		t.Fatal(err)
	}

	reopened, err := Open(root)
	if err != nil {
		t.Fatal(err)
	}
	defer reopened.Close()
	if _, found, err := reopened.Artifact(ctx, orphanID); err != nil || found {
		t.Fatalf("orphaned blob became a committed artifact: found=%v err=%v", found, err)
	}
	_, reader, found, err = reopened.OpenContent(ctx, id)
	if err != nil || !found {
		t.Fatalf("replayed content = (%v, %v)", found, err)
	}
	replayed, err := io.ReadAll(reader)
	if err != nil || len(replayed) != len(payload) {
		t.Fatalf("replayed round trip = (%d bytes, %v)", len(replayed), err)
	}
}
