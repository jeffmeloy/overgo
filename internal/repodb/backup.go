package repodb

import (
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"

	"overgo/internal/artifact"
)

// Backup copies the store's committed log into destinationRoot and proves the
// copy replays to the same head commit and sequence before publishing it. The
// destination directory must not exist; the copy is written to a sibling
// ".partial" directory and renamed only after verification, so a crashed or
// unverified backup is never mistaken for a good one. A stale partial is
// refused rather than deleted: automation never removes store bytes.
//
// The source log is append-only with CRC-framed records, so a backup taken
// while a writer is mid-append can capture a torn tail; the verification open
// recovers by truncation and the head comparison then fails closed. Retry
// between commits.
func (s *Store) Backup(destinationRoot string) (artifact.CommitID, uint64, error) {
	if err := s.ready(false); err != nil {
		return artifact.CommitID{}, 0, err
	}
	head, sequence := s.Head()
	if !head.Valid() {
		return artifact.CommitID{}, 0, errors.New("repodb: refusing to back up a store with no commits")
	}
	if destinationRoot == "" {
		return artifact.CommitID{}, 0, errors.New("repodb: empty backup destination")
	}
	if _, err := os.Stat(destinationRoot); err == nil {
		return artifact.CommitID{}, 0, fmt.Errorf("repodb: backup destination %q already exists", destinationRoot)
	} else if !os.IsNotExist(err) {
		return artifact.CommitID{}, 0, err
	}
	partial := destinationRoot + ".partial"
	if _, err := os.Stat(partial); err == nil {
		return artifact.CommitID{}, 0, fmt.Errorf("repodb: stale partial backup %q; inspect and remove it manually", partial)
	}
	if err := os.MkdirAll(partial, 0o755); err != nil {
		return artifact.CommitID{}, 0, err
	}
	if err := copyFileSync(filepath.Join(s.root, storeFilename), filepath.Join(partial, storeFilename)); err != nil {
		return artifact.CommitID{}, 0, err
	}
	copyHead, copySequence, err := replayedHead(partial)
	if err != nil {
		return artifact.CommitID{}, 0, fmt.Errorf("repodb: backup verification: %w", err)
	}
	if copyHead != head || copySequence != sequence {
		return artifact.CommitID{}, 0, fmt.Errorf(
			"repodb: backup replay head %s@%d does not match source %s@%d (torn tail or concurrent write); retry between commits",
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

func copyFileSync(sourcePath, destinationPath string) error {
	source, err := os.Open(sourcePath)
	if err != nil {
		return err
	}
	defer func() { _ = source.Close() }()
	destination, err := os.OpenFile(destinationPath, os.O_WRONLY|os.O_CREATE|os.O_EXCL, 0o644)
	if err != nil {
		return err
	}
	if _, err := io.Copy(destination, source); err != nil {
		_ = destination.Close()
		return err
	}
	if err := destination.Sync(); err != nil {
		_ = destination.Close()
		return err
	}
	return destination.Close()
}
