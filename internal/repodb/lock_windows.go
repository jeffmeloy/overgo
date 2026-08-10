//go:build windows

package repodb

import (
	"fmt"
	"os"
	"syscall"
	"unsafe"
)

// File-range locking via kernel32 through the no-cgo LazyDLL path (overgo uses
// no cgo; C ABIs are reached with syscall.NewLazyDLL). Values are documented
// Win32 API dwFlags bits for LockFileEx.
const (
	lockfileExclusiveLock   = 0x00000002 // LOCKFILE_EXCLUSIVE_LOCK
	lockfileFailImmediately = 0x00000001 // LOCKFILE_FAIL_IMMEDIATELY
)

var (
	kernel32         = syscall.NewLazyDLL("kernel32.dll")
	procLockFileEx   = kernel32.NewProc("LockFileEx")
	procUnlockFileEx = kernel32.NewProc("UnlockFileEx")
)

type fileLock struct {
	file       *os.File
	overlapped syscall.Overlapped
}

func acquireFileLock(path string) (*fileLock, error) {
	file, err := os.OpenFile(path, os.O_CREATE|os.O_RDWR, storeFileMode)
	if err != nil {
		return nil, err
	}
	lock := &fileLock{file: file}
	// BOOL LockFileEx(hFile, dwFlags, dwReserved=0, nLow=1, nHigh=0, lpOverlapped)
	result, _, callErr := procLockFileEx.Call(
		file.Fd(),
		uintptr(lockfileExclusiveLock|lockfileFailImmediately),
		0,
		1,
		0,
		uintptr(unsafe.Pointer(&lock.overlapped)),
	)
	if result == 0 {
		_ = file.Close()
		return nil, callErr
	}
	return lock, nil
}

func (l *fileLock) Close() error {
	if l == nil || l.file == nil {
		return nil
	}
	// BOOL UnlockFileEx(hFile, dwReserved=0, nLow=1, nHigh=0, lpOverlapped)
	result, _, callErr := procUnlockFileEx.Call(
		l.file.Fd(),
		0,
		1,
		0,
		uintptr(unsafe.Pointer(&l.overlapped)),
	)
	closeErr := l.file.Close()
	l.file = nil
	if result == 0 {
		return fmt.Errorf("unlock: %w", callErr)
	}
	return closeErr
}
