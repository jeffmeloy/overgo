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
	directory, err := os.Open(path)
	if err != nil {
		return fmt.Errorf("fsatomic: open directory for sync: %w", err)
	}
	err = directory.Sync()
	return errors.Join(err, directory.Close())
}
