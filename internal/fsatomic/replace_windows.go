//go:build windows

package fsatomic

import (
	"os"
	"path/filepath"
	"strings"
	"syscall"
	"unsafe"
)

const (
	moveFileReplaceExisting = 0x00000001
	moveFileWriteThrough    = 0x00000008
)

var moveFileExW = syscall.NewLazyDLL("kernel32.dll").NewProc("MoveFileExW")

// Replace moves oldPath over newPath on the same volume. MOVEFILE_WRITE_THROUGH
// makes MoveFileExW wait until the namespace move reaches disk before success.
func Replace(oldPath, newPath string) error {
	return replace(oldPath, newPath, callMoveFileEx)
}

type moveFileExFunc func(oldPath, newPath *uint16, flags uint32) error

func replace(oldPath, newPath string, move moveFileExFunc) error {
	oldPointer, err := windowsPath(oldPath)
	if err != nil {
		return &os.LinkError{Op: "rename", Old: oldPath, New: newPath, Err: err}
	}
	newPointer, err := windowsPath(newPath)
	if err != nil {
		return &os.LinkError{Op: "rename", Old: oldPath, New: newPath, Err: err}
	}
	if err := move(oldPointer, newPointer, moveFileReplaceExisting|moveFileWriteThrough); err != nil {
		return &os.LinkError{Op: "rename", Old: oldPath, New: newPath, Err: err}
	}
	return nil
}

func callMoveFileEx(oldPath, newPath *uint16, flags uint32) error {
	result, _, callErr := moveFileExW.Call(
		uintptr(unsafe.Pointer(oldPath)),
		uintptr(unsafe.Pointer(newPath)),
		uintptr(flags),
	)
	if result != 0 {
		return nil
	}
	if callErr == syscall.Errno(0) {
		return syscall.EINVAL
	}
	return callErr
}

func windowsPath(path string) (*uint16, error) {
	absolute, err := filepath.Abs(path)
	if err != nil {
		return nil, err
	}
	absolute = filepath.Clean(absolute)
	switch {
	case strings.HasPrefix(absolute, `\\?\`):
	case strings.HasPrefix(absolute, `\\`):
		absolute = `\\?\UNC\` + strings.TrimPrefix(absolute, `\\`)
	default:
		absolute = `\\?\` + absolute
	}
	return syscall.UTF16PtrFromString(absolute)
}
