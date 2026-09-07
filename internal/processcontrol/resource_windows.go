//go:build windows

package processcontrol

import (
	"crypto/sha256"
	"errors"
	"fmt"
	"syscall"
	"unsafe"
)

// duplicateSameAccess is the Win32 DUPLICATE_SAME_ACCESS option.
const duplicateSameAccess = 0x00000002

var (
	procIsProcessInJob    = kernel32.NewProc("IsProcessInJob")
	procGetCurrentProcess = kernel32.NewProc("GetCurrentProcess")
	procDuplicateHandle   = kernel32.NewProc("DuplicateHandle")
	resourceJobs          = map[string]uintptr{}
)

// claimResource retains the handle: membership alone does not retain the name.
// Every supervised child receives a handle before executing. Native descendants
// must use supervised launch or claim before consuming the physical resource.
func claimResource(name string) error {
	if resourceJobs[name] != 0 {
		return nil
	}
	jobName, err := syscall.UTF16PtrFromString(fmt.Sprintf("Global\\Overgo.Resource.%x", sha256.Sum256([]byte(name))))
	if err != nil {
		return err
	}
	job, _, callErr := procCreateJobObjectW.Call(0, uintptr(unsafe.Pointer(jobName)))
	if job == 0 {
		return fmt.Errorf("processcontrol: create resource reservation: %w", callErr)
	}
	retained := false
	defer func() {
		if !retained {
			_, _, _ = procCloseHandle.Call(job)
		}
	}()
	process, _, _ := procGetCurrentProcess.Call()
	var member uint32
	if ok, _, err := procIsProcessInJob.Call(process, job, uintptr(unsafe.Pointer(&member))); ok == 0 {
		return fmt.Errorf("processcontrol: inspect resource membership: %w", err)
	}
	if errors.Is(callErr, syscall.ERROR_ALREADY_EXISTS) && member == 0 {
		return fmt.Errorf("processcontrol: physical resource %q is already reserved", name)
	}
	if member == 0 {
		if ok, _, err := procAssignProcessToJob.Call(job, process); ok == 0 {
			return fmt.Errorf("processcontrol: reserve physical resource %q: %w", name, err)
		}
	}
	resourceJobs[name] = job
	retained = true
	return nil
}

// retainChildResources duplicates reservations into the suspended consumer.
// Closing the launcher's handles cannot release a surviving consumer's claim.
// Handles intentionally remain open until process exit; no heartbeat or release
// call can shorten the measured consumer lifetime.
func retainChildResources(process uintptr) error {
	current, _, _ := procGetCurrentProcess.Call()
	for name, job := range resourceJobs {
		var handle uintptr
		if ok, _, err := procDuplicateHandle.Call(current, job, process, uintptr(unsafe.Pointer(&handle)), 0, 0, duplicateSameAccess); ok == 0 {
			return fmt.Errorf("processcontrol: retain child resource %q: %w", name, err)
		}
	}
	return nil
}
