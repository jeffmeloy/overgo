//go:build windows

package overgodb

import (
	"crypto/sha256"
	"os"
	"path/filepath"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/testcheck"
)

// TestBlobRetryPersistsTheNameItFinds holds a retry to the rule its bytes do not
// prove: a blob found after a publication that may not have reached the disk is
// flushed through its own link, which is what names it on this platform. A file
// that cannot be flushed fails the retry rather than passing on being readable.
func TestBlobRetryPersistsTheNameItFinds(t *testing.T) {
	t.Parallel()
	blobs := blobStore{root: filepath.Join(t.TempDir(), blobDirectory), pack: new(blobPack)}
	payload := []byte("a blob an earlier attempt published")
	id := testcheck.Must(artifact.NewID(artifact.KindFile, sha256.Sum256(payload)))(t)
	testcheck.NoError(t, blobs.prepare(id, payload))
	if err := blobs.prepare(id, payload); err != nil {
		t.Fatalf("a retry over a sound blob: %v", err)
	}
	destination := testcheck.Must(blobs.path(id))(t)
	testcheck.NoError(t, os.Chmod(destination, 0o444))
	t.Cleanup(func() { _ = os.Chmod(destination, 0o644) })
	if err := blobs.prepare(id, payload); err == nil {
		t.Fatal("a retry passed without persisting the name it found")
	}
}
