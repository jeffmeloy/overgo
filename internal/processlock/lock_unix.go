//go:build !windows

package processlock

import (
	"context"
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

// AcquireContext waits for exclusive access until ctx ends: a busy lock is
// waited for with a blocking flock on its own thread, and a wait the caller
// abandons releases the lock the moment it is finally granted.
func AcquireContext(ctx context.Context, path string, mode fs.FileMode) (*Lock, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	lock, err := Acquire(path, mode)
	if !errors.Is(err, ErrBusy) {
		return lock, err
	}
	file, err := os.OpenFile(path, os.O_CREATE|os.O_RDWR, mode)
	if err != nil {
		return nil, err
	}
	type outcome struct {
		lock *Lock
		err  error
	}
	granted := make(chan outcome, 1)
	go func() {
		if err := syscall.Flock(int(file.Fd()), syscall.LOCK_EX); err != nil {
			_ = file.Close()
			granted <- outcome{err: &os.PathError{Op: "lock", Path: path, Err: err}}
			return
		}
		granted <- outcome{lock: &Lock{file: file}}
	}()
	select {
	case result := <-granted:
		return result.lock, result.err
	case <-ctx.Done():
		go func() {
			if result := <-granted; result.lock != nil {
				_ = result.lock.Close()
			}
		}()
		return nil, errors.Join(ctx.Err(), ErrBusy)
	}
}
