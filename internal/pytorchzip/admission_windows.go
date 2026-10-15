package pytorchzip

import (
	"os"
	"syscall"
)

// openAdmission excludes writers and replacement for the lifetime of the handle.
func openAdmission(path string) (*os.File, bool, error) {
	name, err := syscall.UTF16PtrFromString(path)
	if err != nil {
		return nil, false, err
	}
	handle, err := syscall.CreateFile(name, syscall.GENERIC_READ, syscall.FILE_SHARE_READ, nil, syscall.OPEN_EXISTING, syscall.FILE_ATTRIBUTE_NORMAL, 0)
	if err != nil {
		return nil, false, &os.PathError{Op: "admit archive", Path: path, Err: err}
	}
	return os.NewFile(uintptr(handle), path), true, nil
}
