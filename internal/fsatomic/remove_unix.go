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
