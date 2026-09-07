//go:build !windows

package processlock

import (
	"errors"
	"fmt"
	"io/fs"
	"os"
	"syscall"
)

// Lock is an exclusive file lock released by Close or process exit.
type Lock struct {
	file *os.File
}

// Acquire takes an exclusive non-blocking lock on path.
func Acquire(path string, mode fs.FileMode) (*Lock, error) {
	file, err := os.OpenFile(path, os.O_CREATE|os.O_RDWR, mode)
	if err != nil {
		return nil, err
	}
	if err := syscall.Flock(int(file.Fd()), syscall.LOCK_EX|syscall.LOCK_NB); err != nil {
		_ = file.Close()
		if errors.Is(err, syscall.EWOULDBLOCK) || errors.Is(err, syscall.EAGAIN) {
			err = errors.Join(ErrBusy, err)
		}
		return nil, &os.PathError{Op: "lock", Path: path, Err: err}
	}
	return &Lock{file: file}, nil
}

// Close releases the lock and closes its file. Repeated calls are safe.
func (lock *Lock) Close() error {
	if lock == nil || lock.file == nil {
		return nil
	}
	unlockErr := syscall.Flock(int(lock.file.Fd()), syscall.LOCK_UN)
	closeErr := lock.file.Close()
	lock.file = nil
	if unlockErr != nil {
		return fmt.Errorf("unlock: %w", unlockErr)
	}
	return closeErr
}
