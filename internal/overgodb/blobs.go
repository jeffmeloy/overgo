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
	pack *blobPack
}

const (
	blobDirectory = "blobs"
	// blobFanout is the directory fan-out prefix length in hex digits;
	// two digits give 256 buckets, keeping directories small at the
	// corpus scales the store contract measures.
	blobFanout = 2
)

func newBlobStore(root string) blobStore {
	return blobStore{root: filepath.Join(root, blobDirectory), pack: new(blobPack)}
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
	if _, _, found := b.packed(id); found {
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
	reader, err := b.reader(id)
	if err != nil {
		return nil, err
	}
	if uint64(reader.remaining) != size {
		_ = reader.Close()
		return nil, fmt.Errorf("overgodb: blob %s size differs from descriptor", id)
	}
	return reader, nil
}

// reader streams the blob's bytes from its loose file, or from the pack when
// there is none.
func (b blobStore) reader(id artifact.ID) (*blobReader, error) {
	destination, err := b.path(id)
	if err != nil {
		return nil, err
	}
	file, err := os.Open(destination)
	offset, size, found := int64(0), int64(0), false
	if errors.Is(err, os.ErrNotExist) {
		if offset, size, found = b.packed(id); found {
			file, err = os.Open(filepath.Join(b.root, blobPackFilename))
		}
	}
	if err != nil {
		return nil, fmt.Errorf("overgodb: open blob %s: %w", id, err)
	}
	if !found {
		info, statErr := file.Stat()
		if statErr != nil {
			_ = file.Close()
			return nil, statErr
		}
		size = info.Size()
	}
	if _, err := file.Seek(offset, io.SeekStart); err != nil {
		_ = file.Close()
		return nil, err
	}
	return &blobReader{file: file, remaining: size}, nil
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

// Read streams the expected bytes and closes the file at end of stream. A
// blob's file ends with its content; a sealed segment runs on past inline
// content, so no read asks for more than is still expected.
func (r *blobReader) Read(buffer []byte) (int, error) {
	if r.done {
		return 0, io.EOF
	}
	count, err := r.file.Read(buffer[:min(int64(len(buffer)), r.remaining)])
	r.remaining -= int64(count)
	if (errors.Is(err, io.EOF) || r.remaining <= 0) && !r.done {
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
	_, err := b.copyVerified(io.Discard, id, nil)
	return err
}

// copyVerified streams the blob's bytes into destination through buffer and
// holds them to the identity that addresses it, returning how many it wrote:
// the one bounded read that verification and packing share, whatever the
// blob's size. A nil buffer takes io.Copy's own.
func (b blobStore) copyVerified(destination io.Writer, id artifact.ID, buffer []byte) (int64, error) {
	file, err := b.reader(id)
	if err != nil {
		return 0, fmt.Errorf("overgodb: read blob %s: %w", id, err)
	}
	defer file.Close()
	hash := sha256.New()
	written, err := io.CopyBuffer(io.MultiWriter(destination, hash), file, buffer)
	if err != nil {
		return written, fmt.Errorf("overgodb: copy blob %s: %w", id, err)
	}
	if hex.EncodeToString(hash.Sum(nil)) != id.DigestHex() {
		return written, fmt.Errorf("overgodb: blob %s bytes do not hash to their identity", id)
	}
	return written, nil
}

// remove deletes the published blob for id and reports whether a file
// went; an absent blob is not an error, since a release is idempotent.
// Packed bytes stay until the next pack, which leaves released content out.
func (b blobStore) remove(id artifact.ID) (bool, error) {
	destination, err := b.path(id)
	if err != nil {
		return false, err
	}
	err = os.Remove(destination)
	switch {
	case errors.Is(err, os.ErrNotExist):
		return false, nil
	case err != nil:
		return false, fmt.Errorf("overgodb: remove blob %s: %w", id, err)
	}
	return true, nil
}

// has reports whether a published blob exists for id; staging files
// are invisible, so an interrupted prepare can never look published.
func (b blobStore) has(id artifact.ID) bool {
	destination, err := b.path(id)
	if err != nil {
		return false
	}
	if _, statErr := os.Stat(destination); statErr == nil {
		return true
	}
	_, _, found := b.packed(id)
	return found
}

// MergeBlobTrees moves every content-addressed blob under sourceRoot's
// blobs directory into destinationRoot's blob tree. A journal swap after a
// rebuild needs this: the rebuilt journal references content the rebuild
// externalized into its own blobs directory, and a swap without the blobs
// strands every chain whose content the old journal carried inline. Blob
// paths are content hashes, so an already-present target is byte-identical
// and the source copy is simply dropped. A pack is not named for its content:
// two stores' packs share a name and hold different members, so a source that
// holds one is refused rather than dropped or moved over the destination's,
// which stays where it is.
func MergeBlobTrees(sourceRoot, destinationRoot string) error {
	source := filepath.Join(sourceRoot, blobDirectory)
	if _, err := os.Stat(source); errors.Is(err, os.ErrNotExist) {
		return nil
	}
	return filepath.WalkDir(source, func(path string, entry os.DirEntry, err error) error {
		if err != nil || entry.IsDir() {
			return err
		}
		relative, err := filepath.Rel(source, path)
		if err != nil {
			return err
		}
		if relative == blobPackFilename {
			return fmt.Errorf("overgodb: %s holds a blob pack; its members cannot be merged by file name", sourceRoot)
		}
		target := filepath.Join(destinationRoot, blobDirectory, relative)
		if _, err := os.Stat(target); err == nil {
			return os.Remove(path)
		}
		if err := os.MkdirAll(filepath.Dir(target), storeDirectoryMode); err != nil {
			return err
		}
		return os.Rename(path, target)
	})
}
