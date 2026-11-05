package overgodb

import (
	"io"
	"os"
	"strings"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/testcheck"
)

// TestExternalBlobCommitAtomicity holds the external-blob commit to
// its contract: the journal frame is a small envelope while the bytes
// live in the blob store; a committed descriptor whose required blob
// is missing refuses content access with exact evidence instead of
// degrading; and the whole corpus replays from the envelope journal plus
// blobs. An append fault after blob preparation, which leaves only an
// unreachable blob, is TestAppendFaultClosesEveryStoreSurfaceUntilReopen.
func TestExternalBlobCommitAtomicity(t *testing.T) {
	t.Parallel()
	ctx := t.Context()
	root := t.TempDir()
	store, err := Create(root)
	testcheck.NoError(t, err)
	defer store.Close()

	payload := make([]byte, 8192)
	for index := range payload {
		payload[index] = byte(index)
	}
	id, err := artifact.IdentifyBytes(artifact.KindTensorSet, payload)
	testcheck.NoError(t, err)
	descriptor := artifact.Descriptor{ID: id, Size: uint64(len(payload)), MediaType: "application/octet-stream"}
	before, err := os.Stat(store.log.file.Name())
	testcheck.NoError(t, err)
	if _, err := store.Commit(ctx, artifact.Batch{
		Key:      "blob-commit/large",
		Contents: []artifact.Content{{Descriptor: descriptor, Data: payload}},
	}); err != nil {
		t.Fatal(err)
	}
	after, err := os.Stat(store.log.file.Name())
	testcheck.NoError(t, err)
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
	testcheck.NoError(t, err)
	testcheck.NoError(t, os.Remove(path))
	if _, _, _, err := store.OpenContent(ctx, id); err == nil ||
		!strings.Contains(err.Error(), "requires an absent blob") || !strings.Contains(err.Error(), id.String()) {
		t.Fatalf("absent blob access = %v", err)
	}
	testcheck.NoError(t, store.blobs.prepare(id, payload))

	testcheck.NoError(t, store.Close())

	reopened, err := Create(root)
	testcheck.NoError(t, err)
	defer reopened.Close()
	_, reader, found, err = reopened.OpenContent(ctx, id)
	if err != nil || !found {
		t.Fatalf("replayed content = (%v, %v)", found, err)
	}
	replayed, err := io.ReadAll(reader)
	if err != nil || len(replayed) != len(payload) {
		t.Fatalf("replayed round trip = (%d bytes, %v)", len(replayed), err)
	}
}
