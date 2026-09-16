//go:build windows

package processlock

import (
	"context"
	"errors"
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
	// ERROR_LOCK_VIOLATION is the Win32 byte-range contention result.
	errorLockViolation = 33
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
		if errors.Is(callErr, syscall.Errno(errorLockViolation)) {
			callErr = errors.Join(ErrBusy, callErr)
		}
		return nil, &os.PathError{Op: "lock", Path: path, Err: callErr}
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

// AcquireContext waits for exclusive access until ctx ends. Cancellation drains
// the pending OS request before releasing its handle and OVERLAPPED storage.
func AcquireContext(ctx context.Context, path string, mode fs.FileMode) (*Lock, error) {
	if err := context.Cause(ctx); err != nil {
		return nil, contendedCause(err, path, mode)
	}
	name, err := syscall.UTF16PtrFromString(path)
	if err != nil {
		return nil, err
	}
	handle, err := syscall.CreateFile(name, syscall.GENERIC_READ|syscall.GENERIC_WRITE,
		syscall.FILE_SHARE_READ|syscall.FILE_SHARE_WRITE, nil, syscall.OPEN_ALWAYS,
		syscall.FILE_ATTRIBUTE_NORMAL|syscall.FILE_FLAG_OVERLAPPED, 0)
	if err != nil {
		return nil, &os.PathError{Op: "open", Path: path, Err: err}
	}
	// Keep the asynchronous request outside os.File's completion port.
	lock := &Lock{}
	var null uintptr
	const manualReset = uintptr(1)
	event, _, eventErr := kernel32.NewProc("CreateEventW").Call(null, manualReset, null, null)
	if event == 0 {
		_ = syscall.CloseHandle(handle)
		return nil, eventErr
	}
	defer syscall.CloseHandle(syscall.Handle(event))
	lock.overlapped.HEvent = syscall.Handle(event)
	var reserved, high uintptr
	result, _, callErr := procLockFileEx.Call(uintptr(handle), lockfileExclusiveLock, reserved,
		firstByteLockLength, high, uintptr(unsafe.Pointer(&lock.overlapped)))
	contended := result == 0 && errors.Is(callErr, syscall.ERROR_IO_PENDING)
	if contended {
		cancelled := make(chan struct{})
		stop := context.AfterFunc(ctx, func() {
			_ = syscall.CancelIoEx(handle, &lock.overlapped)
			close(cancelled)
		})
		var transferred uint32
		const waitForCompletion = uintptr(1)
		result, _, callErr = kernel32.NewProc("GetOverlappedResult").Call(uintptr(handle),
			uintptr(unsafe.Pointer(&lock.overlapped)), uintptr(unsafe.Pointer(&transferred)), waitForCompletion)
		if !stop() {
			<-cancelled
		}
	}
	if result == 0 {
		_ = syscall.CloseHandle(handle)
		if err := context.Cause(ctx); err != nil {
			if contended {
				return nil, errors.Join(err, ErrBusy)
			}
			return nil, err
		}
		return nil, &os.PathError{Op: "lock", Path: path, Err: callErr}
	}
	lock.overlapped.HEvent = 0
	lock.file = os.NewFile(uintptr(handle), path)
	if err := context.Cause(ctx); err != nil {
		_ = lock.Close()
		return nil, err
	}
	return lock, nil
}
