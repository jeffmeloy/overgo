//go:build windows

package processcontrol

import (
	"errors"
	"fmt"
	"path/filepath"
	"sync"
	"syscall"
	"unsafe"
)

const (
	// Win32 JOB_OBJECT_QUERY, CSIDL_COMMON_APPDATA and ERROR_SHARING_VIOLATION.
	resourceJobQuery         = 0x0004
	resourceCommonAppData    = 0x0023
	resourceSharingViolation = syscall.Errno(32)
)

var (
	procOpenResourceJob = kernel32.NewProc("OpenJobObjectW")
	procResourceFolder  = syscall.NewLazyDLL("shell32.dll").NewProc("SHGetFolderPathW")
)

func resourceAdmissionPath(name string) (string, error) {
	var path [syscall.MAX_PATH]uint16
	result, _, _ := procResourceFolder.Call(0, resourceCommonAppData, 0, 0, uintptr(unsafe.Pointer(&path[0])))
	if result != 0 {
		return "", fmt.Errorf("processcontrol: common resource directory: HRESULT %#x", result)
	}
	return filepath.Join(syscall.UTF16ToString(path[:]), resourceObjectName(name)+".lock"), nil
}

// Only an open handle owns admission. The empty file survives crashes harmlessly.
func openResourceAdmission(name string, shared bool) (syscall.Handle, error) {
	path, err := resourceAdmissionPath(name)
	if err != nil {
		return 0, err
	}
	encoded, err := syscall.UTF16PtrFromString(path)
	if err != nil {
		return 0, err
	}
	var share uint32
	if shared {
		share = syscall.FILE_SHARE_READ
	}
	handle, err := syscall.CreateFile(encoded, syscall.GENERIC_READ, share, nil, syscall.OPEN_ALWAYS, syscall.FILE_ATTRIBUTE_NORMAL, 0)
	if err != nil {
		if errors.Is(err, resourceSharingViolation) {
			err = errors.Join(&ResourceBusyError{Name: name}, err)
		}
		return 0, fmt.Errorf("processcontrol: resource admission %q: %w", name, err)
	}
	return handle, nil
}

func shareResource(name string) (func() error, error) {
	if resourceJobs[name] != 0 {
		return func() error { return nil }, nil
	}
	handle, err := openResourceAdmission(name, true)
	if err != nil {
		return nil, err
	}
	retained := false
	defer func() {
		if !retained {
			_ = syscall.CloseHandle(handle)
		}
	}()
	jobName, err := syscall.UTF16PtrFromString("Global\\" + resourceObjectName(name))
	if err != nil {
		return nil, err
	}
	job, _, callErr := procOpenResourceJob.Call(resourceJobQuery, 0, uintptr(unsafe.Pointer(jobName)))
	if job == 0 {
		if !errors.Is(callErr, syscall.ERROR_FILE_NOT_FOUND) {
			return nil, fmt.Errorf("processcontrol: inspect exclusive resource %q: %w", name, callErr)
		}
		retained = true
		// The release wakes the processes waiting on this resource.
		return sync.OnceValue(func() error {
			err := syscall.CloseHandle(handle)
			signalResourceRelease(name)
			return err
		}), nil
	}
	defer syscall.CloseHandle(syscall.Handle(job))
	process, _, _ := procGetCurrentProcess.Call()
	var member uint32
	if ok, _, err := procIsProcessInJob.Call(process, job, uintptr(unsafe.Pointer(&member))); ok == 0 {
		return nil, fmt.Errorf("processcontrol: inspect resource membership: %w", err)
	}
	if member == 0 {
		return nil, fmt.Errorf("processcontrol: %w", &ResourceBusyError{Name: name})
	}
	if err := claimResource(name); err != nil {
		return nil, err
	}
	claimedResources[name] = true
	return func() error { return nil }, nil
}
