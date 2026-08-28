package overgodb

import (
	"io"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"overgo/internal/artifact"
)

// TestBlobStorePublicationContract holds the blob owner to its
// publication contract: identity-addressed placement, refusal of a
// payload that does not hash to its claimed identity, idempotent
// re-preparation, byte-exact open, deep verification catching
// corruption, and orphan safety -- an interrupted staging file is
// never visible as a published blob.
func TestBlobStorePublicationContract(t *testing.T) {
	blobs := newBlobStore(t.TempDir())
	payload := []byte("blob-owner contract payload")
	id, err := artifact.IdentifyBytes(artifact.KindTensorSet, payload)
	if err != nil {
		t.Fatal(err)
	}

	foreign, err := artifact.IdentifyBytes(artifact.KindTensorSet, []byte("different payload"))
	if err != nil {
		t.Fatal(err)
	}
	if err := blobs.prepare(foreign, payload); err == nil {
		t.Fatal("payload published under an identity it does not hash to")
	}
	if blobs.has(foreign) {
		t.Fatal("refused payload became visible")
	}

	if err := blobs.prepare(id, payload); err != nil {
		t.Fatal(err)
	}
	if err := blobs.prepare(id, payload); err != nil {
		t.Fatalf("idempotent re-preparation refused: %v", err)
	}
	if !blobs.has(id) {
		t.Fatal("published blob is not visible")
	}

	reader, err := blobs.open(id, uint64(len(payload)))
	if err != nil {
		t.Fatal(err)
	}
	data, err := io.ReadAll(reader)
	if closeErr := reader.Close(); err != nil || closeErr != nil || string(data) != string(payload) {
		t.Fatalf("open round trip = (%q, %v, %v)", data, err, closeErr)
	}
	if _, err := blobs.open(id, uint64(len(payload))+1); err == nil {
		t.Fatal("size mismatch opened")
	}
	if err := blobs.verify(id); err != nil {
		t.Fatal(err)
	}

	path, err := blobs.path(id)
	if err != nil {
		t.Fatal(err)
	}
	staging := filepath.Join(filepath.Dir(path), ".staging-interrupted")
	if err := os.WriteFile(staging, []byte("torn"), storeFileMode); err != nil {
		t.Fatal(err)
	}
	tornID, err := artifact.IdentifyBytes(artifact.KindTensorSet, []byte("torn"))
	if err != nil {
		t.Fatal(err)
	}
	if blobs.has(tornID) {
		t.Fatal("staging file visible as a published blob")
	}

	corrupted, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	corrupted[0] ^= 0xFF
	if err := os.WriteFile(path, corrupted, storeFileMode); err != nil {
		t.Fatal(err)
	}
	if err := blobs.verify(id); err == nil || !strings.Contains(err.Error(), "do not hash") {
		t.Fatalf("corrupted blob verified: %v", err)
	}
}

// TestObservationChunkPublication proves retrying publication cannot bless a
// same-sized corrupt orphan while an intact orphan remains idempotent.
func TestObservationChunkPublication(t *testing.T) {
	t.Run("same-sized corrupt orphan refuses retry", func(t *testing.T) {
		blobs := newBlobStore(t.TempDir())
		payload := []byte("bounded observation chunk")
		id, err := artifact.IdentifyBytes(artifact.KindFile, payload)
		if err != nil {
			t.Fatal(err)
		}
		if err := blobs.prepare(id, payload); err != nil {
			t.Fatal(err)
		}

		path, err := blobs.path(id)
		if err != nil {
			t.Fatal(err)
		}
		corrupted := append([]byte(nil), payload...)
		corrupted[0] ^= 0xFF
		if err := os.WriteFile(path, corrupted, storeFileMode); err != nil {
			t.Fatal(err)
		}
		if err := blobs.prepare(id, payload); err == nil || !strings.Contains(err.Error(), "do not hash") {
			t.Fatalf("retry accepted same-sized corrupt orphan: %v", err)
		}
	})

	t.Run("intact orphan retry is idempotent", func(t *testing.T) {
		blobs := newBlobStore(t.TempDir())
		payload := []byte("intact observation chunk")
		id, err := artifact.IdentifyBytes(artifact.KindFile, payload)
		if err != nil {
			t.Fatal(err)
		}
		if err := blobs.prepare(id, payload); err != nil {
			t.Fatal(err)
		}
		if err := blobs.prepare(id, payload); err != nil {
			t.Fatalf("intact orphan retry refused: %v", err)
		}
	})
}
