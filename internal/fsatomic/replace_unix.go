//go:build !windows

package fsatomic

import (
	"errors"
	"fmt"
	"os"
	"path/filepath"
)

// Replace moves oldPath over newPath on the same filesystem and persists both
// affected directory entries before returning.
func Replace(oldPath, newPath string) error {
	if err := os.Rename(oldPath, newPath); err != nil {
		return err
	}
	oldDirectory := filepath.Clean(filepath.Dir(oldPath))
	newDirectory := filepath.Clean(filepath.Dir(newPath))
	newErr := SyncDirectory(newDirectory)
	if oldDirectory == newDirectory {
		return newErr
	}
	if err := errors.Join(newErr, SyncDirectory(oldDirectory)); err != nil {
		return fmt.Errorf("fsatomic: sync renamed directories: %w", err)
	}
	return nil
}
