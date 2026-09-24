package processlock

import (
	"errors"
	"fmt"
	"io/fs"
	"os"
	"path/filepath"
)

const (
	// stateDirectoryMode keeps a checkout's process state private to its owner.
	stateDirectoryMode fs.FileMode = 0o700
	// stateIgnoreFile and stateIgnoreRule make the state directory ignore
	// everything in it, itself included.
	stateIgnoreFile = ".gitignore"
	stateIgnoreRule = "*\n"
)

// EnsureStateDirectory creates the state directory of the checkout at root
// on first use; every writer of process state calls it first. The directory
// ignores itself: a checkout of a commit older than the directory, or a
// repository with an ignore file of its own, must not see a lock as an
// untracked change, so no ignore file outside the directory is relied on.
func EnsureStateDirectory(root string) error {
	directory := filepath.Join(root, StateDirectory)
	if err := os.MkdirAll(directory, stateDirectoryMode); err != nil {
		return fmt.Errorf("process state: create directory: %w", err)
	}
	ignore := filepath.Join(directory, stateIgnoreFile)
	if _, err := os.Lstat(ignore); err == nil {
		return nil
	}
	err := placeIgnoreRule(directory, ignore)
	if err == nil {
		return nil
	}
	// Another process racing to the same first use may have placed the rule.
	if _, statErr := os.Lstat(ignore); statErr == nil {
		return nil
	}
	return fmt.Errorf("process state: write ignore rule: %w", err)
}

// StateFile is the path of a named file in the state directory of the
// checkout at root, the directory created on first use. A file an older
// binary left at its location before the directory existed is carried
// across once -- renamed into place, or dropped when the directory already
// holds a newer one -- so it is neither lost nor left in the tree as
// untracked dirt.
func StateFile(root, name, legacy string) (string, error) {
	if err := EnsureStateDirectory(root); err != nil {
		return "", err
	}
	path := filepath.Join(root, StateDirectory, name)
	old := filepath.Join(root, filepath.FromSlash(legacy))
	if _, err := os.Lstat(old); errors.Is(err, os.ErrNotExist) {
		return path, nil
	}
	if _, err := os.Lstat(path); err == nil {
		return path, os.Remove(old)
	}
	if err := os.Rename(old, path); err != nil && !errors.Is(err, os.ErrNotExist) {
		return "", fmt.Errorf("process state: carry %s across: %w", legacy, err)
	}
	return path, nil
}

// placeIgnoreRule stages the rule and renames it into place, so git never
// reads a rule file that is half written.
func placeIgnoreRule(directory, ignore string) error {
	staged, err := os.CreateTemp(directory, stateIgnoreFile+"-*")
	if err != nil {
		return err
	}
	_, err = staged.WriteString(stateIgnoreRule)
	if err = errors.Join(err, staged.Close()); err != nil {
		return errors.Join(err, os.Remove(staged.Name()))
	}
	if err := os.Rename(staged.Name(), ignore); err != nil {
		return errors.Join(err, os.Remove(staged.Name()))
	}
	return nil
}
