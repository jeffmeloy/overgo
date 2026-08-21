//go:build !windows

package objectstore

import (
	"errors"
	"fmt"
	"os"
)

func syncDirectory(path string) error {
	directory, err := os.Open(path)
	if err != nil {
		return fmt.Errorf("object store: open directory for sync: %w", err)
	}
	err = directory.Sync()
	return errors.Join(err, directory.Close())
}
