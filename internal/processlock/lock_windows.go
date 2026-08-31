//go:build windows

package processlock

import (
	"fmt"
	"io/fs"
	"os"
	"syscall"
	"unsafe"
)

const (
	lockfileExclusiveLock   = 0x00000002
	lockfileFailImmediately = 0x00000001
	firstByteLockLength     = uintptr(1)
)

var (
	kernel32         = syscall.NewLazyDLL("kernel32.dll")
	procLockFileEx   = kernel32.NewProc("LockFileEx")
	procUnlockFileEx = kernel32.NewProc("UnlockFileEx")
)

// Lock is an exclusive file-range lock released by Close or process exit.
type Lock struct {
	file       *os.File
	overlapped syscall.Overlapped
}

// Acquire takes an exclusive non-blocking lock on the first byte of path.
func Acquire(path string, mode fs.FileMode) (*Lock, error) {
	file, err := os.OpenFile(path, os.O_CREATE|os.O_RDWR, mode)
	if err != nil {
		return nil, err
	}
	lock := &Lock{file: file}
	result, callErr := lockFirstByte(file, &lock.overlapped)
	if result == 0 {
		_ = file.Close()
		return nil, callErr
	}
	return lock, nil
}

// Close releases the lock and closes its file. Repeated calls are safe.
func (lock *Lock) Close() error {
	if lock == nil || lock.file == nil {
		return nil
	}
	result, callErr := unlockFirstByte(lock.file, &lock.overlapped)
	closeErr := lock.file.Close()
	lock.file = nil
	if result == 0 {
		return fmt.Errorf("unlock: %w", callErr)
	}
	return closeErr
}

func lockFirstByte(file *os.File, overlapped *syscall.Overlapped) (uintptr, error) {
	var reserved, rangeHigh uintptr
	result, _, err := procLockFileEx.Call(
		file.Fd(), uintptr(lockfileExclusiveLock|lockfileFailImmediately), reserved,
		firstByteLockLength, rangeHigh, uintptr(unsafe.Pointer(overlapped)),
	)
	return result, err
}

func unlockFirstByte(file *os.File, overlapped *syscall.Overlapped) (uintptr, error) {
	var reserved, rangeHigh uintptr
	result, _, err := procUnlockFileEx.Call(
		file.Fd(), reserved, firstByteLockLength, rangeHigh, uintptr(unsafe.Pointer(overlapped)),
	)
	return result, err
}
