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
	if err := copyFileSync(storeLogPath(sourceRoot), filepath.Join(partial, storeFilename), extent); err != nil {
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
