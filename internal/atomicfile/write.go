// Package atomicfile owns durable whole-file replacement.
package atomicfile

import (
	"errors"
	"io/fs"
	"os"
	"path/filepath"

	"overgo/internal/fsatomic"
)

// Write replaces path with data through one same-volume namespace move. The
// temporary file and replacement move are durable before success. Unix gives
// atomic rename visibility; Windows gives MoveFileEx replacement semantics,
// whose write-through contract is durable but does not promise portable Unix
// reader-visibility semantics.
func Write(path string, data []byte, mode fs.FileMode) error {
	return write(path, data, mode, fsatomic.Replace)
}

func write(path string, data []byte, mode fs.FileMode, replace func(string, string) error) error {
	directory := filepath.Dir(path)
	temporaryPath, err := prepare(directory, ".atomic-write-*", data, mode)
	if err != nil {
		return err
	}
	defer os.Remove(temporaryPath)
	if err := replace(temporaryPath, path); err != nil {
		return err
	}
	return nil
}

func prepare(directory, pattern string, data []byte, mode fs.FileMode) (path string, err error) {
	return prepareWithHook(directory, pattern, data, mode, nil)
}

func prepareWithHook(
	directory, pattern string,
	data []byte,
	mode fs.FileMode,
	afterCreate func(string),
) (path string, err error) {
	temporary, err := os.CreateTemp(directory, pattern)
	if err != nil {
		return "", err
	}
	path = temporary.Name()
	if afterCreate != nil {
		afterCreate(path)
	}
	defer func() {
		if err == nil {
			return
		}
		_ = temporary.Chmod(mode | fs.ModePerm)
		err = errors.Join(err, temporary.Close())
		_ = os.Remove(path)
	}()
	if err := temporary.Chmod(mode); err != nil {
		return "", err
	}
	if _, err := temporary.Write(data); err != nil {
		return "", err
	}
	if err := temporary.Sync(); err != nil {
		return "", err
	}
	if err := temporary.Close(); err != nil {
		return "", err
	}
	return path, nil
}
