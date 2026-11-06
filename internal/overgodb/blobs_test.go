package overgodb

import (
	"io"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/testcheck"
)

// TestBlobStorePublicationContract holds the blob owner to its
// publication contract: identity-addressed placement, refusal of a
// payload that does not hash to its claimed identity, idempotent
// re-preparation, byte-exact open, deep verification catching
// corruption, and orphan safety -- an interrupted staging file is
// never visible as a published blob.
func TestBlobStorePublicationContract(t *testing.T) {
	t.Parallel()
	blobs := newBlobStore(t.TempDir())
	payload := []byte("blob-owner contract payload")
	id := testcheck.Must(artifact.IdentifyBytes(artifact.KindTensorSet, payload))(t)

	foreign := testcheck.Must(artifact.IdentifyBytes(artifact.KindTensorSet, []byte("different payload")))(t)
	if err := blobs.prepare(foreign, payload); err == nil {
		t.Fatal("payload published under an identity it does not hash to")
	}
	if blobs.has(foreign) {
		t.Fatal("refused payload became visible")
	}

	testcheck.NoError(t, blobs.prepare(id, payload))
	if err := blobs.prepare(id, payload); err != nil {
		t.Fatalf("idempotent re-preparation refused: %v", err)
	}
	if !blobs.has(id) {
		t.Fatal("published blob is not visible")
	}

	reader := testcheck.Must(blobs.open(id, uint64(len(payload))))(t)
	data, err := io.ReadAll(reader)
	if closeErr := reader.Close(); err != nil || closeErr != nil || string(data) != string(payload) {
		t.Fatalf("open round trip = (%q, %v, %v)", data, err, closeErr)
	}
	if _, err := blobs.open(id, uint64(len(payload))+1); err == nil {
		t.Fatal("size mismatch opened")
	}
	testcheck.NoError(t, blobs.verify(t.Context(), id))

	path := testcheck.Must(blobs.path(id))(t)
	staging := filepath.Join(filepath.Dir(path), ".staging-interrupted")
	testcheck.NoError(t, os.WriteFile(staging, []byte("torn"), storeFileMode))
	tornID := testcheck.Must(artifact.IdentifyBytes(artifact.KindTensorSet, []byte("torn")))(t)
	if blobs.has(tornID) {
		t.Fatal("staging file visible as a published blob")
	}

	corrupted := testcheck.Must(os.ReadFile(path))(t)
	corrupted[0] ^= 0xFF
	testcheck.NoError(t, os.WriteFile(path, corrupted, storeFileMode))
	if err := blobs.verify(t.Context(), id); err == nil || !strings.Contains(err.Error(), "do not hash") {
		t.Fatalf("corrupted blob verified: %v", err)
	}

	fileID := testcheck.Must(artifact.IdentifyBytes(artifact.KindFile, payload))(t)
	for range 2 {
		if err := blobs.prepare(fileID, payload); err != nil {
			t.Fatalf("intact file publication or retry refused: %v", err)
		}
	}
	filePath := testcheck.Must(blobs.path(fileID))(t)
	testcheck.NoError(t, os.WriteFile(filePath, corrupted, storeFileMode))
	if err := blobs.prepare(fileID, payload); err == nil || !strings.Contains(err.Error(), "do not hash") {
		t.Fatalf("retry accepted same-sized corrupt orphan: %v", err)
	}
}
