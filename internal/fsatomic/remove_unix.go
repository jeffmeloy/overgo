//go:build !windows

package fsatomic

import (
	"errors"
	"os"
	"path/filepath"
)

// Remove makes path's absence durable before returning. It is idempotent so a
// caller can repeat cleanup after an interrupted protocol.
func Remove(path string) error {
	if err := os.Remove(path); err != nil && !errors.Is(err, os.ErrNotExist) {
		return err
	}
	return SyncDirectory(filepath.Dir(path))
}

// RemoveTree removes the directory tree at path with everything in it, and
// leaves what a link in it names: a link is deleted and never followed. Like
// Remove it makes the absence durable, and a path that is not there is no
// error.
func RemoveTree(path string) error {
	if err := os.RemoveAll(path); err != nil {
		return err
	}
	return SyncDirectory(filepath.Dir(path))
}
