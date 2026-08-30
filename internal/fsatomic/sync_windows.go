//go:build windows

package fsatomic

import (
	"errors"
	"fmt"
	"os"
)

// SyncDirectory is a no-op because Win32 does not document
// FlushFileBuffers for directory handles. Windows callers must publish
// namespace changes through Replace or flush a surviving hard link through
// SyncFile instead.
func SyncDirectory(string) error { return nil }

// SyncFile commits file data and metadata changes through FlushFileBuffers.
// Microsoft documents file metadata as cached and persisted by flushing the
// file; GENERIC_WRITE is therefore required here rather than a read-only open.
func SyncFile(path string) error {
	file, err := os.OpenFile(path, os.O_WRONLY, 0)
	if err != nil {
		return fmt.Errorf("fsatomic: open file for sync: %w", err)
	}
	err = file.Sync()
	return errors.Join(err, file.Close())
}
