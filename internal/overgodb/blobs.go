package overgodb

import (
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"

	"overgo/internal/artifact"
	"overgo/internal/fsatomic"
)

// blobStore owns immutable content bytes outside the metadata journal,
// addressed by artifact identity: the ID's digest IS the blob key, so
// preparation is idempotent, a payload that does not hash to its
// claimed identity refuses, and a crash between prepare and journal
// commit leaves only an unreachable file that no committed artifact
// can name. Publication order is the safety argument: bytes are fully
// durable (file and directory synced) before any journal frame may
// reference them.
type blobStore struct {
	root string
}

const (
	blobDirectory = "blobs"
	// blobFanout is the directory fan-out prefix length in hex digits;
	// two digits give 256 buckets, keeping directories small at the
	// corpus scales the store contract measures.
	blobFanout = 2
)

func newBlobStore(root string) blobStore {
	return blobStore{root: filepath.Join(root, blobDirectory)}
}

// path derives the blob's immutable location from its identity alone.
func (b blobStore) path(id artifact.ID) (string, error) {
	encoded := id.DigestHex()
	if !id.Valid() || len(encoded) <= blobFanout {
		return "", errors.New("overgodb: invalid blob identity")
	}
	kind := strings.ToLower(id.Kind().String())
	return filepath.Join(b.root, kind, encoded[:blobFanout], encoded), nil
}

// prepare makes the payload durable under its identity. An existing
// blob makes preparation an idempotent verify; a payload whose digest
// differs from the claimed identity refuses before any bytes land.
func (b blobStore) prepare(id artifact.ID, payload []byte) error {
	digest := sha256.Sum256(payload)
	if hex.EncodeToString(digest[:]) != id.DigestHex() {
		return fmt.Errorf("overgodb: blob payload does not hash to %s", id)
	}
	destination, err := b.path(id)
	if err != nil {
		return err
	}
	if info, statErr := os.Stat(destination); statErr == nil {
		if info.Size() != int64(len(payload)) {
			return fmt.Errorf("overgodb: blob %s exists with conflicting size", id)
		}
		return b.verify(id)
	}
	directory := filepath.Dir(destination)
	if err := os.MkdirAll(directory, storeDirectoryMode); err != nil {
		return fmt.Errorf("overgodb: create blob directory: %w", err)
	}
	staging, err := os.CreateTemp(directory, ".staging-*")
	if err != nil {
		return fmt.Errorf("overgodb: stage blob: %w", err)
	}
	stagingPath := staging.Name()
	if err := writeAll(staging, payload); err != nil {
		_ = staging.Close()
		_ = os.Remove(stagingPath)
		return fmt.Errorf("overgodb: write blob: %w", err)
	}
	if err := staging.Sync(); err != nil {
		_ = staging.Close()
		_ = os.Remove(stagingPath)
		return fmt.Errorf("overgodb: sync blob: %w", err)
	}
	if err := staging.Close(); err != nil {
		_ = os.Remove(stagingPath)
		return fmt.Errorf("overgodb: close blob: %w", err)
	}
	if err := os.Rename(stagingPath, destination); err != nil {
		_ = os.Remove(stagingPath)
		return fmt.Errorf("overgodb: publish blob: %w", err)
	}
	return fsatomic.SyncDirectory(directory)
}

// open returns the blob's bytes stream after a size check against the
// committed descriptor; identity-deep verification is verify's job.
func (b blobStore) open(id artifact.ID, size uint64) (io.ReadCloser, error) {
	destination, err := b.path(id)
	if err != nil {
		return nil, err
	}
	file, err := os.Open(destination)
	if err != nil {
		return nil, fmt.Errorf("overgodb: open blob %s: %w", id, err)
	}
	info, err := file.Stat()
	if err != nil || uint64(info.Size()) != size {
		_ = file.Close()
		return nil, fmt.Errorf("overgodb: blob %s size differs from descriptor", id)
	}
	return &blobReader{file: file, remaining: int64(size)}, nil
}

// blobReader closes its file the moment the stream is exhausted --
// at end of file OR after the final expected byte -- so callers that
// read exactly the descriptor size through a bare io.Reader never
// leak an OS handle; abandoning callers close through io.Closer.
type blobReader struct {
	file      *os.File
	remaining int64
	done      bool
}

// Read streams the blob and closes the file at end of stream.
func (r *blobReader) Read(buffer []byte) (int, error) {
	if r.done {
		return 0, io.EOF
	}
	count, err := r.file.Read(buffer)
	r.remaining -= int64(count)
	if (err == io.EOF || r.remaining <= 0) && !r.done {
		r.done = true
		_ = r.file.Close()
		if err == nil {
			return count, nil
		}
	}
	return count, err
}

// Close releases the file for callers that abandon the stream early.
func (r *blobReader) Close() error {
	if r.done {
		return nil
	}
	r.done = true
	return r.file.Close()
}

// verify re-derives the blob's digest from its bytes and compares it
// to the identity that addresses it.
func (b blobStore) verify(id artifact.ID) error {
	destination, err := b.path(id)
	if err != nil {
		return err
	}
	file, err := os.Open(destination)
	if err != nil {
		return fmt.Errorf("overgodb: verify blob %s: %w", id, err)
	}
	defer file.Close()
	hash := sha256.New()
	if _, err := io.Copy(hash, file); err != nil {
		return fmt.Errorf("overgodb: hash blob %s: %w", id, err)
	}
	if hex.EncodeToString(hash.Sum(nil)) != id.DigestHex() {
		return fmt.Errorf("overgodb: blob %s bytes do not hash to their identity", id)
	}
	return nil
}

// has reports whether a published blob exists for id; staging files
// are invisible, so an interrupted prepare can never look published.
func (b blobStore) has(id artifact.ID) bool {
	destination, err := b.path(id)
	if err != nil {
		return false
	}
	_, statErr := os.Stat(destination)
	return statErr == nil
}
