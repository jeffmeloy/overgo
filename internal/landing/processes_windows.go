package landing

import (
	"errors"
	"fmt"
	"syscall"
	"unsafe"
)

// Processes lists every process that runs on this machine, from one snapshot
// the kernel takes, so a parent and its child are read at the same moment.
func Processes() (listed []Process, err error) {
	snapshot, err := syscall.CreateToolhelp32Snapshot(syscall.TH32CS_SNAPPROCESS, 0)
	if err != nil {
		return nil, fmt.Errorf("the process list: %w", err)
	}
	defer func() { err = errors.Join(err, syscall.CloseHandle(snapshot)) }()
	var entry syscall.ProcessEntry32
	entry.Size = uint32(unsafe.Sizeof(entry))
	for err = syscall.Process32First(snapshot, &entry); err == nil; err = syscall.Process32Next(snapshot, &entry) {
		listed = append(listed, Process{Name: syscall.UTF16ToString(entry.ExeFile[:]), ID: int(entry.ProcessID), Parent: int(entry.ParentProcessID)})
	}
	// The walk ends with the kernel's answer that no entry is left, and any other end is a failure.
	if !errors.Is(err, syscall.ERROR_NO_MORE_FILES) {
		return nil, fmt.Errorf("the process list: %w", err)
	}
	return listed, nil
}
