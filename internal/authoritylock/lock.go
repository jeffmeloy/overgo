// Package authoritylock serializes mutations that can change plan or gate
// authority. Readers remain lock-free; every writer reloads its inputs after
// acquiring this process-lifetime lock.
package authoritylock

import (
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"overgo/internal/processlock"
)

const (
	relativePath = "tmp/gate.lock"
	// authorityDirectoryMode keeps repository-local mutation state private to
	// the repository owner on newly created workspaces.
	authorityDirectoryMode os.FileMode = 0o700
	// authorityFileMode keeps the process lock private to the repository owner.
	authorityFileMode os.FileMode = 0o600
)

// Acquire takes the repository-local plan/gate mutation lock without waiting.
func Acquire(repository string) (*processlock.Lock, error) {
	rawRepository := strings.TrimSpace(repository)
	if rawRepository == "" {
		return nil, errors.New("authority lock: repository is required")
	}
	repository = filepath.Clean(rawRepository)
	directory := filepath.Join(repository, filepath.Dir(relativePath))
	if err := os.MkdirAll(directory, authorityDirectoryMode); err != nil {
		return nil, fmt.Errorf("authority lock: create directory: %w", err)
	}
	lock, err := processlock.Acquire(filepath.Join(repository, filepath.FromSlash(relativePath)), authorityFileMode)
	if err != nil {
		if errors.Is(err, processlock.ErrBusy) {
			return nil, fmt.Errorf("authority lock: another plan or gate mutation is active: %w", err)
		}
		return nil, fmt.Errorf("authority lock: acquire: %w", err)
	}
	return lock, nil
}
