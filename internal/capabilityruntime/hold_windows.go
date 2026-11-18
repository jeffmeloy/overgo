//go:build windows

package capabilityruntime

import (
	"io/fs"
	"os"
	"syscall"
)

// holdFile opens path for reading, sharing reading alone: until it is closed no process writes, replaces or removes
// the file, which the system refuses, and every process may still read it.
func holdFile(path string) (*os.File, error) {
	name, err := syscall.UTF16PtrFromString(path)
	if err != nil {
		return nil, &fs.PathError{Op: "open", Path: path, Err: err}
	}
	handle, err := syscall.CreateFile(name, syscall.GENERIC_READ, syscall.FILE_SHARE_READ, nil, syscall.OPEN_EXISTING, syscall.FILE_ATTRIBUTE_NORMAL, 0)
	if err != nil {
		return nil, &fs.PathError{Op: "open", Path: path, Err: err}
	}
	return os.NewFile(uintptr(handle), path), nil
}
