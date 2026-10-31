//go:build windows

package processcontrol

import (
	"errors"
	"fmt"
	"runtime"
	"syscall"
	"unsafe"
)

const (
	// Win32 INFINITE and WAIT_ABANDONED. An abandoned mutex grants ownership.
	resourceInfiniteWait  = 0xffffffff
	resourceWaitAbandoned = 0x00000080
	// Win32 WAIT_TIMEOUT: a wait of no time found the mutex owned.
	resourceWaitTimeout = 0x00000102
)

var (
	procCreateResourceMutex  = kernel32.NewProc("CreateMutexW")
	procWaitResourceMutex    = kernel32.NewProc("WaitForSingleObject")
	procReleaseResourceMutex = kernel32.NewProc("ReleaseMutex")
)

func resourceTransactionHeld(name string) (held bool, err error) {
	encoded, err := syscall.UTF16PtrFromString("Global\\" + resourceObjectName(name) + ".Mutation")
	if err != nil {
		return false, err
	}
	handle, _, callErr := procCreateResourceMutex.Call(0, 0, uintptr(unsafe.Pointer(encoded)))
	if handle == 0 {
		return false, fmt.Errorf("processcontrol: create resource transaction: %w", callErr)
	}
	defer syscall.CloseHandle(syscall.Handle(handle))
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	switch result, _, callErr := procWaitResourceMutex.Call(handle, 0); result {
	case resourceWaitTimeout:
		return true, nil
	case 0, resourceWaitAbandoned:
		if ok, _, releaseErr := procReleaseResourceMutex.Call(handle); ok == 0 {
			return false, fmt.Errorf("processcontrol: release resource transaction: %w", releaseErr)
		}
		return false, nil
	default:
		return false, fmt.Errorf("processcontrol: probe resource transaction: %w", callErr)
	}
}

func resourceTransaction(name string, action func() error) (err error) {
	encoded, err := syscall.UTF16PtrFromString("Global\\" + resourceObjectName(name) + ".Mutation")
	if err != nil {
		return err
	}
	handle, _, callErr := procCreateResourceMutex.Call(0, 0, uintptr(unsafe.Pointer(encoded)))
	if handle == 0 {
		return fmt.Errorf("processcontrol: create resource transaction: %w", callErr)
	}
	defer syscall.CloseHandle(syscall.Handle(handle))
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	result, _, callErr := procWaitResourceMutex.Call(handle, resourceInfiniteWait)
	if result != 0 && result != resourceWaitAbandoned {
		return fmt.Errorf("processcontrol: enter resource transaction: %w", callErr)
	}
	defer func() {
		if ok, _, releaseErr := procReleaseResourceMutex.Call(handle); ok == 0 {
			err = errors.Join(err, fmt.Errorf("processcontrol: release resource transaction: %w", releaseErr))
		}
	}()
	return action()
}
