package overgodb

import (
	"bytes"
	"io"
	"os"
	"path/filepath"
	"testing"

	"overgo/internal/artifact"
)

// TestVerifyBlobsCoversThePack holds the check a candidate passes after its
// blobs have been moved: it reads every live blob, packed and loose, against
// its identity, and a single changed byte in a packed payload -- which the
// index digest does not cover -- fails it.
func TestVerifyBlobsCoversThePack(t *testing.T) {
	t.Parallel()
	root := filepath.Join(t.TempDir(), "store")
	store, err := Open(root)
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	contents := []artifact.Content{
		retentionContent(t, artifact.KindEvidence, "packed"),
		retentionContent(t, artifact.KindEvidence, bytes.Repeat([]byte("loose "), inlineContentLimit)),
	}
	if _, err := store.Commit(t.Context(), artifact.Batch{Key: "verify/fixture", Contents: contents}); err != nil {
		t.Fatal(err)
	}
	if _, err := packBlobsUnder(t.Context(), store, int64(contents[0].Descriptor.Size)+1); err != nil {
		t.Fatal(err)
	}
	if checked, err := store.VerifyBlobs(t.Context()); err != nil || checked != len(contents) {
		t.Fatalf("verified %d blobs, %v; want both", checked, err)
	}
	pack := filepath.Join(root, blobDirectory, blobPackFilename)
	document, err := os.ReadFile(pack)
	if err != nil {
		t.Fatal(err)
	}
	document[len(document)-1] ^= 0xFF
	if err := os.WriteFile(pack, document, storeFileMode); err != nil {
		t.Fatal(err)
	}
	if _, err := store.VerifyBlobs(t.Context()); err == nil {
		t.Fatal("a changed byte in a packed payload passed verification")
	}
}

// TestSmallBlobsArePacked holds the pack to moving files without moving
// facts. Two blobs under the limit leave their loose files for one pack and a
// larger one stays loose; every content reads back byte for byte, from this
// handle and from a fresh open; preparing a packed identity again writes no
// loose file; packing again reads the members from the pack it replaces; a
// backup carries the pack and serves its members; and a pack whose index no
// longer matches its digest serves nothing rather than the wrong bytes.
func TestSmallBlobsArePacked(t *testing.T) {
	t.Parallel()
	root := filepath.Join(t.TempDir(), "store")
	store, err := Open(root)
	if err != nil {
		t.Fatal(err)
	}
	contents := []artifact.Content{
		retentionContent(t, artifact.KindEvidence, "first small"),
		retentionContent(t, artifact.KindProfile, "second small"),
		retentionContent(t, artifact.KindEvidence, bytes.Repeat([]byte("large "), inlineContentLimit)),
	}
	if _, err := store.Commit(t.Context(), artifact.Batch{Key: "pack/fixture", Contents: contents}); err != nil {
		t.Fatal(err)
	}
	limit := int64(contents[0].Descriptor.Size) + int64(inlineContentLimit)
	reads := func(store *Store) {
		t.Helper()
		for _, content := range contents {
			_, reader, found, err := store.OpenContent(t.Context(), content.Descriptor.ID)
			if err != nil || !found {
				t.Fatalf("content %s: found=%v err=%v", content.Descriptor.ID, found, err)
			}
			data, err := io.ReadAll(reader)
			if err != nil || !bytes.Equal(data, content.Data) {
				t.Fatalf("content %s read back %d bytes, %v", content.Descriptor.ID, len(data), err)
			}
		}
	}
	loose := func(content artifact.Content) bool {
		path, err := store.blobs.path(content.Descriptor.ID)
		if err != nil {
			t.Fatal(err)
		}
		_, statErr := os.Stat(path)
		return statErr == nil
	}

	report, err := packBlobsUnder(t.Context(), store, limit)
	if err != nil || report.Packed != 2 || report.LooseRemoved != 2 {
		t.Fatalf("pack = %+v, %v; want two blobs packed and their loose files gone", report, err)
	}
	if loose(contents[0]) || loose(contents[1]) || !loose(contents[2]) {
		t.Fatal("the pack left a small blob loose or took the large one")
	}
	reads(store)
	if err := store.blobs.prepare(contents[0].Descriptor.ID, contents[0].Data); err != nil || loose(contents[0]) {
		t.Fatalf("preparing a packed identity: %v, loose=%v", err, loose(contents[0]))
	}
	if again, err := packBlobsUnder(t.Context(), store, limit); err != nil || again.Packed != 2 || again.LooseRemoved != 0 {
		t.Fatalf("second pack = %+v, %v; want the same members read from the pack", again, err)
	}
	reads(store)

	backup := filepath.Join(t.TempDir(), "backup")
	if _, err := store.Backup(t.Context(), backup); err != nil {
		t.Fatal(err)
	}
	if err := store.Close(); err != nil {
		t.Fatal(err)
	}
	for _, location := range []string{root, backup} {
		reopened, err := OpenReadOnly(location)
		if err != nil {
			t.Fatal(err)
		}
		reads(reopened)
		if err := reopened.Close(); err != nil {
			t.Fatal(err)
		}
	}

	pack := filepath.Join(root, blobDirectory, blobPackFilename)
	document, err := os.ReadFile(pack)
	if err != nil {
		t.Fatal(err)
	}
	document[packHeaderBytes+packKeyBytes] ^= 0xFF
	if err := os.WriteFile(pack, document, storeFileMode); err != nil {
		t.Fatal(err)
	}
	damaged, err := OpenReadOnly(root)
	if err != nil {
		t.Fatal(err)
	}
	defer damaged.Close()
	if _, _, _, err := damaged.OpenContent(t.Context(), contents[0].Descriptor.ID); err == nil {
		t.Fatal("a pack whose index does not match its digest still served a blob")
	}
}
