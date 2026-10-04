package fsatomic

import (
	"errors"
	"io/fs"
	"os"
	"path/filepath"
)

// PublishDirectory makes the directory name under parent, which must not
// exist, hold what populate writes into it, and returns its path. populate
// writes into a staged sibling that moves into place only once it is complete,
// so a refused or failed write leaves nothing, a killed one leaves no
// half-written directory at the name, and a writer of the same name that lost
// the race removes only its own stage, never the winner's directory. An
// existing name is an error that wraps fs.ErrExist.
func PublishDirectory(parent, name string, mode fs.FileMode, populate func(stage string) error) (string, error) {
	target := filepath.Join(parent, name)
	if _, err := os.Lstat(target); err == nil {
		return "", &fs.PathError{Op: "publish", Path: target, Err: fs.ErrExist}
	} else if !errors.Is(err, fs.ErrNotExist) {
		return "", err
	}
	if err := os.MkdirAll(parent, mode); err != nil {
		return "", err
	}
	stage, err := os.MkdirTemp(parent, "."+name+".stage-")
	if err != nil {
		return "", err
	}
	if err := errors.Join(os.Chmod(stage, mode), populate(stage)); err != nil {
		return "", errors.Join(err, os.RemoveAll(stage))
	}
	if err := os.Rename(stage, target); err != nil {
		if _, held := os.Lstat(target); held == nil {
			err = &fs.PathError{Op: "publish", Path: target, Err: fs.ErrExist}
		}
		return "", errors.Join(err, os.RemoveAll(stage))
	}
	return target, SyncDirectory(parent)
}
