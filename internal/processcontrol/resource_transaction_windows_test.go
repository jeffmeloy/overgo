//go:build windows

package processcontrol

import (
	"errors"
	"runtime"
	"syscall"
	"testing"
	"unsafe"
)

func TestGPUCapacityTransactionRecovery(t *testing.T) {
	name := sharedAdmissionName(t)
	owner := holdSharedAdmission(t, sharedAdmissionProbe{Name: name, Transaction: true})
	encoded, err := syscall.UTF16PtrFromString("Global\\" + resourceObjectName(name) + ".Mutation")
	if err != nil {
		t.Fatal(err)
	}
	handle, _, callErr := procCreateResourceMutex.Call(0, 0, uintptr(unsafe.Pointer(encoded)))
	if handle == 0 {
		t.Fatal(callErr)
	}
	defer syscall.CloseHandle(syscall.Handle(handle))
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	// WAIT_TIMEOUT proves the independent live owner prevents a transaction.
	const waitTimeout = 258
	if result, _, err := procWaitResourceMutex.Call(handle, 0); result != waitTimeout {
		if result == 0 {
			_, _, _ = procReleaseResourceMutex.Call(handle)
		}
		t.Fatalf("live transaction was not exclusive: %d: %v", result, err)
	}
	owner.kill(t)
	want := errors.New("transaction callback failed")
	if err := ResourceTransaction(name, func() error { return want }); !errors.Is(err, want) {
		t.Fatalf("abandoned transaction recovery: %v", err)
	}
	entered := false
	if err := ResourceTransaction(name, func() error { entered = true; return nil }); err != nil || !entered {
		t.Fatalf("callback failure retained ownership: %v", err)
	}
	// An idle reference handle remains open throughout; it never grants ownership.
	t.Log("cross-process exclusion, abandoned-owner recovery, callback-failure release and idle-handle coexistence")
}
