package overgodb

import (
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"

	"overgo/internal/artifact"
)

// Backup publishes one replay-verified committed log extent.
func (s *Store) Backup(destinationRoot string) (artifact.CommitID, uint64, error) {
	s.mu.RLock()
	if err := s.ready(false); err != nil {
		s.mu.RUnlock()
		return artifact.CommitID{}, 0, err
	}
	head, sequence, extent, sourceRoot := s.head, s.sequence, s.replayEnd, s.root
	requiredBlobs := make([]artifact.ID, 0, len(s.state.contents.locators))
	for id, locator := range s.state.contents.locators {
		if locator.blob {
			requiredBlobs = append(requiredBlobs, id)
		}
	}
	s.mu.RUnlock()
	if !head.Valid() {
		return artifact.CommitID{}, 0, errors.New("overgodb: refusing to back up a store with no commits")
	}
	if destinationRoot == "" {
		return artifact.CommitID{}, 0, errors.New("overgodb: empty backup destination")
	}
	if _, err := os.Stat(destinationRoot); err == nil {
		return artifact.CommitID{}, 0, fmt.Errorf("overgodb: backup destination %q already exists", destinationRoot)
	} else if !os.IsNotExist(err) {
		return artifact.CommitID{}, 0, err
	}
	partial := destinationRoot + ".partial"
	if _, err := os.Stat(partial); err == nil {
		return artifact.CommitID{}, 0, fmt.Errorf("overgodb: stale partial backup %q; inspect and remove it manually", partial)
	}
	if err := os.MkdirAll(partial, storeDirectoryMode); err != nil {
		return artifact.CommitID{}, 0, err
	}
	// The stable extent: every sealed segment is immutable, the active
	// file copies only to the captured extent, required blobs are
	// content-addressed and immutable, and checkpoint anchors ride along
	// -- a stale set is discarded by the anchored open, never trusted.
	if err := copyTreeSync(filepath.Join(sourceRoot, segmentDirectory), filepath.Join(partial, segmentDirectory)); err != nil {
		return artifact.CommitID{}, 0, err
	}
	if err := copyFileSync(filepath.Join(sourceRoot, storeFilename), filepath.Join(partial, storeFilename), extent); err != nil {
		return artifact.CommitID{}, 0, err
	}
	sourceBlobs, destinationBlobs := newBlobStore(sourceRoot), newBlobStore(partial)
	for _, id := range requiredBlobs {
		sourcePath, err := sourceBlobs.path(id)
		if err != nil {
			return artifact.CommitID{}, 0, err
		}
		destinationPath, err := destinationBlobs.path(id)
		if err != nil {
			return artifact.CommitID{}, 0, err
		}
		if err := os.MkdirAll(filepath.Dir(destinationPath), storeDirectoryMode); err != nil {
			return artifact.CommitID{}, 0, err
		}
		info, err := os.Stat(sourcePath)
		if err != nil {
			return artifact.CommitID{}, 0, fmt.Errorf("overgodb: backup blob %s: %w", id, err)
		}
		if err := copyFileSync(sourcePath, destinationPath, info.Size()); err != nil {
			return artifact.CommitID{}, 0, err
		}
	}
	if err := copyTreeSync(filepath.Join(sourceRoot, checkpointDirectory), filepath.Join(partial, checkpointDirectory)); err != nil {
		return artifact.CommitID{}, 0, err
	}
	copyHead, copySequence, err := replayedHead(partial)
	if err != nil {
		return artifact.CommitID{}, 0, fmt.Errorf("overgodb: backup verification: %w", err)
	}
	if copyHead != head || copySequence != sequence {
		return artifact.CommitID{}, 0, fmt.Errorf(
			"overgodb: backup replay head %s@%d does not match captured %s@%d",
			copyHead, copySequence, head, sequence)
	}
	if err := os.Rename(partial, destinationRoot); err != nil {
		return artifact.CommitID{}, 0, err
	}
	return head, sequence, nil
}

// copyTreeSync mirrors one directory tree with per-file durability; an
// absent source is an empty tree.
func copyTreeSync(source, destination string) error {
	entries, err := os.ReadDir(source)
	if os.IsNotExist(err) {
		return nil
	}
	if err != nil {
		return err
	}
	if err := os.MkdirAll(destination, storeDirectoryMode); err != nil {
		return err
	}
	for _, entry := range entries {
		sourcePath := filepath.Join(source, entry.Name())
		destinationPath := filepath.Join(destination, entry.Name())
		if entry.IsDir() {
			if err := copyTreeSync(sourcePath, destinationPath); err != nil {
				return err
			}
			continue
		}
		info, err := entry.Info()
		if err != nil {
			return err
		}
		if err := copyFileSync(sourcePath, destinationPath, info.Size()); err != nil {
			return err
		}
	}
	return nil
}

// replayedHead opens root read-only and returns its replayed head identity.
func replayedHead(root string) (artifact.CommitID, uint64, error) {
	store, err := OpenReadOnly(root)
	if err != nil {
		return artifact.CommitID{}, 0, err
	}
	head, sequence := store.Head()
	if err := store.Close(); err != nil {
		return artifact.CommitID{}, 0, err
	}
	return head, sequence, nil
}

func copyFileSync(sourcePath, destinationPath string, extent int64) error {
	source, err := os.Open(sourcePath)
	if err != nil {
		return err
	}
	defer func() { _ = source.Close() }()
	destination, err := os.OpenFile(destinationPath, os.O_WRONLY|os.O_CREATE|os.O_EXCL, storeFileMode)
	if err != nil {
		return err
	}
	if written, err := io.CopyN(destination, source, extent); err != nil || written != extent {
		_ = destination.Close()
		if err == nil {
			err = io.ErrUnexpectedEOF
		}
		return err
	}
	if err := destination.Sync(); err != nil {
		_ = destination.Close()
		return err
	}
	return destination.Close()
}
