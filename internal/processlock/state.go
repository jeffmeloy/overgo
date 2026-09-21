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
