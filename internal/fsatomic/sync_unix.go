//go:build !windows

package fsatomic

import (
	"errors"
	"fmt"
	"os"
)

// SyncDirectory makes a completed rename durable by flushing the
// directory that received it.
func SyncDirectory(path string) error {
	return syncPath(path, "directory")
}

// SyncFile commits file data and metadata changes to stable storage.
func SyncFile(path string) error {
	return syncPath(path, "file")
}

func syncPath(path, kind string) error {
	file, err := os.Open(path)
	if err != nil {
		return fmt.Errorf("fsatomic: open %s for sync: %w", kind, err)
	}
	err = file.Sync()
	return errors.Join(err, file.Close())
}
