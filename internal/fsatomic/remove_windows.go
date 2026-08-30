//go:build windows

package fsatomic

import (
	"crypto/sha256"
	"errors"
	"fmt"
	"os"
	"path/filepath"
)

// Remove makes path's absence durable before returning. Windows has no
// documented write-through unlink, so path is first write-through renamed to a
// deterministic sidecar and that sidecar is then deleted. A power loss may
// restore the harmless sidecar, but never the original name; repeating Remove
// cleans it without accumulating new names.
func Remove(path string) error {
	tombstone := removalTombstone(path)
	if _, err := os.Lstat(path); errors.Is(err, os.ErrNotExist) {
		_ = os.Remove(tombstone)
		return nil
	} else if err != nil {
		return err
	}
	if err := Replace(path, tombstone); err != nil {
		return fmt.Errorf("fsatomic: quarantine removed path: %w", err)
	}
	// The durable contract is absence at path. Tombstone cleanup is deliberately
	// best effort: its deletion has no documented write-through operation and a
	// failure or later power-loss resurrection cannot restore path.
	_ = os.Remove(tombstone)
	return nil
}

func removalTombstone(path string) string {
	digest := sha256.Sum256([]byte(filepath.Clean(path)))
	return filepath.Join(filepath.Dir(path), fmt.Sprintf(".fsatomic-remove-%x", digest))
}
