//go:build windows

package processcontrol

import (
	"fmt"
	"unsafe"
)

var procGlobalMemoryStatusEx = kernel32.NewProc("GlobalMemoryStatusEx")

// MachineCommit reads the machine's commit left and its limit, in bytes (GlobalMemoryStatusEx): the one reading of
// it, which the driver's own reads and a job's memory bound (boundOf) take.
func MachineCommit() (left, limit uint64, err error) {
	// status is MEMORYSTATUSEX: its length, the memory load, then the physical, page-file and virtual totals and
	// what of each is available; the page-file pair is the commit limit and the commit left.
	var status struct {
		length, load                                          uint32
		totalPhysical, availablePhysical, commitLimit, commit uint64
		totalVirtual, availableVirtual, availableExtended     uint64
	}
	status.length = uint32(unsafe.Sizeof(status))
	if ok, _, callErr := procGlobalMemoryStatusEx.Call(uintptr(unsafe.Pointer(&status))); ok == 0 {
		return 0, 0, fmt.Errorf("GlobalMemoryStatusEx: %w", callErr)
	}
	return status.commit, status.commitLimit, nil
}

var procGetPhysicallyInstalledSystemMemory = kernel32.NewProc("GetPhysicallyInstalledSystemMemory")

// machineMemory is the memory installed in the machine, in bytes (GetPhysicallyInstalledSystemMemory), which a job's
// bound reads beside its commit limit (boundOf).
func machineMemory() (uint64, error) {
	var kilobytes uint64
	if ok, _, err := procGetPhysicallyInstalledSystemMemory.Call(uintptr(unsafe.Pointer(&kilobytes))); ok == 0 {
		return 0, fmt.Errorf("GetPhysicallyInstalledSystemMemory: %w", err)
	}
	return kilobytes << 10, nil
}
