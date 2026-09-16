//overgo:runtime-inputs caller

// Package authoritylock serializes mutations that can change plan or gate
// authority. Readers remain lock-free; every writer reloads its inputs after
// acquiring this process-lifetime lock.
package authoritylock

import (
	"context"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"overgo/internal/processlock"
)

// Wait waits for the current mutation owner to finish without stealing its
// lock or inferring abandonment from elapsed time.
func Wait(ctx context.Context, repository string) error {
	path := filepath.Join(repository, filepath.FromSlash(relativePath))
	if err := os.MkdirAll(filepath.Dir(path), authorityDirectoryMode); err != nil {
		return err
	}
	lock, err := processlock.AcquireContext(ctx, path, authorityFileMode)
	if err != nil {
		return err
	}
	return lock.Close()
}

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
	return acquire(repository, processlock.Acquire)
}

// AcquireContext takes the lock, waiting for its holder until ctx ends.
func AcquireContext(ctx context.Context, repository string) (*processlock.Lock, error) {
	return acquire(repository, func(path string, mode os.FileMode) (*processlock.Lock, error) {
		return processlock.AcquireContext(ctx, path, mode)
	})
}

func acquire(repository string, take func(string, os.FileMode) (*processlock.Lock, error)) (*processlock.Lock, error) {
	rawRepository := strings.TrimSpace(repository)
	if rawRepository == "" {
		return nil, errors.New("authority lock: repository is required")
	}
	repository = filepath.Clean(rawRepository)
	directory := filepath.Join(repository, filepath.Dir(relativePath))
	if err := os.MkdirAll(directory, authorityDirectoryMode); err != nil {
		return nil, fmt.Errorf("authority lock: create directory: %w", err)
	}
	lock, err := take(filepath.Join(repository, filepath.FromSlash(relativePath)), authorityFileMode)
	if err != nil {
		if errors.Is(err, processlock.ErrBusy) {
			return nil, fmt.Errorf("authority lock: another plan or gate mutation is active: %w", err)
		}
		return nil, fmt.Errorf("authority lock: acquire: %w", err)
	}
	return lock, nil
}
