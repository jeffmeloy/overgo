package overgodb

import (
	"bytes"
	"crypto/sha256"
	"encoding/binary"
	"io"
	"math"
	"os"
	"path/filepath"
	"slices"
	"testing"

	"overgo/internal/artifact"
)

// TestPackIndexRejectsImpossibleCounts holds the pack reader to its file: a
// header whose entry count cannot fit in the file -- a damaged count, the
// largest count, one entry too many -- is refused before it sizes an
// allocation, and a count that fits still reads its verified index.
func TestPackIndexRejectsImpossibleCounts(t *testing.T) {
	t.Parallel()
	entries := bytes.Repeat([]byte{1}, 2*packEntryBytes)
	digest := sha256.Sum256(entries)
	write := func(count uint64) string {
		path := filepath.Join(t.TempDir(), blobPackFilename)
		header := append([]byte(blobPackMagic), binary.BigEndian.AppendUint64(nil, count)...)
		if err := os.WriteFile(path, slices.Concat(header, entries, digest[:]), 0o600); err != nil {
			t.Fatal(err)
		}
		return path
	}
	for _, count := range []uint64{math.MaxUint64, math.MaxUint64 / packEntryBytes, 3} {
		if index := readPackIndex(write(count)); index != nil {
			t.Fatalf("count %d read an index of %d bytes from a two-entry file", count, len(index))
		}
	}
	if index := readPackIndex(write(2)); !bytes.Equal(index, entries) {
		t.Fatalf("a count that fits read %d bytes, want the two entries", len(index))
	}
}

// TestMergeBlobTreesCarriesThePack holds the merge a rebuild uses to the one
// thing a pack changes. Loose blobs are named for their content, so a merge
// moves them by name; a pack is not, so a source that holds one is refused
// rather than dropped as a duplicate of the destination's, and the
// destination's pack still serves its members after a merge of loose blobs.
func TestMergeBlobTreesCarriesThePack(t *testing.T) {
	t.Parallel()
	packedRoot, looseRoot := filepath.Join(t.TempDir(), "packed"), filepath.Join(t.TempDir(), "loose")
	packedContent := retentionContent(t, artifact.KindEvidence, "held in the pack")
	looseContent := retentionContent(t, artifact.KindEvidence, "held loose")
	for root, content := range map[string]artifact.Content{packedRoot: packedContent, looseRoot: looseContent} {
		store, err := Open(root)
		if err != nil {
			t.Fatal(err)
		}
		if _, err := store.Commit(t.Context(), artifact.Batch{Key: "merge/" + filepath.Base(root), Contents: []artifact.Content{content}}); err != nil {
			t.Fatal(err)
		}
		if root == packedRoot {
			if _, err := packBlobsUnder(t.Context(), store, int64(content.Descriptor.Size)+1); err != nil {
				t.Fatal(err)
			}
		}
		if err := store.Close(); err != nil {
			t.Fatal(err)
		}
	}
	if err := MergeBlobTrees(packedRoot, looseRoot); err == nil {
		t.Fatal("a source holding a pack was merged by file name")
	}
	if err := MergeBlobTrees(looseRoot, packedRoot); err != nil {
		t.Fatal(err)
	}
	destination := newBlobStore(packedRoot)
	for _, content := range []artifact.Content{packedContent, looseContent} {
		if err := destination.verify(content.Descriptor.ID); err != nil {
			t.Fatalf("after the merge the destination does not serve %s: %v", content.Descriptor.ID, err)
		}
	}
}

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
