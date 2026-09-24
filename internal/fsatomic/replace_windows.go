//go:build windows

package fsatomic

import (
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"syscall"
	"unsafe"
)

const (
	moveFileReplaceExisting = 0x00000001
	moveFileWriteThrough    = 0x00000008
	// Win32 and NT values: the file's process-ids-using-file class, the NT
	// statuses a short buffer earns, FILE_READ_ATTRIBUTES, the holder slots a
	// first listing offers, and the Win32 errors a held destination or an
	// exited holder returns.
	fileProcessIDsUsingFileInformation               = 47
	statusInfoLengthMismatch                         = 0xC0000004
	statusBufferOverflow                             = 0x80000005
	fileReadAttributes                               = 0x0080
	statusSuccess                                    = 0
	initialHolderSlots                               = 64
	errorSharingViolation              syscall.Errno = 32
	errorUserMappedFile                syscall.Errno = 1224
	// SetFileInformationByHandle's FileRenameInfoEx class, its replace and
	// POSIX-semantics flags, the DELETE access right a rename needs, and the
	// byte offsets of FILE_RENAME_INFO's name length and name on a 64-bit
	// handle layout.
	fileRenameInfoEx       = 22
	renameReplaceIfExists  = 0x1
	renamePosixSemantics   = 0x2
	deleteAccess           = 0x00010000
	renameInfoLengthOffset = 16
	renameInfoNameOffset   = 20
)

var (
	moveFileExW                = syscall.NewLazyDLL("kernel32.dll").NewProc("MoveFileExW")
	setFileInformationByHandle = syscall.NewLazyDLL("kernel32.dll").NewProc("SetFileInformationByHandle")
	ntQueryInformationFile     = syscall.NewLazyDLL("ntdll.dll").NewProc("NtQueryInformationFile")
)

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
	err = move(oldPointer, newPointer, moveFileReplaceExisting|moveFileWriteThrough)
	if err == nil {
		return nil
	}
	// A destination another process holds open or mapped refuses MoveFileExW
	// with one of these; anything else is the move's own error.
	if !errors.Is(err, syscall.ERROR_ACCESS_DENIED) && !errors.Is(err, errorSharingViolation) && !errors.Is(err, errorUserMappedFile) {
		return &os.LinkError{Op: "rename", Old: oldPath, New: newPath, Err: err}
	}
	// A holder that shares delete access -- git hashing the worktree --
	// admits a POSIX-semantics rename: the name moves to the new file while
	// the holder keeps reading the old one until it closes it.
	posixErr := posixRename(oldPointer, newPointer)
	if posixErr == nil {
		return nil
	}
	// A holder that denies deletion keeps the name for as long as it holds
	// the file. The replace names it rather than waiting on its lifetime: a
	// holder that releases the file and stays alive lets the next replace
	// through, which a wait on its exit never would.
	holders, listErr := FileUsers(newPath)
	names := make([]string, 0, len(holders))
	for _, holder := range holders {
		names = append(names, fmt.Sprintf("process %d", holder))
	}
	held := fmt.Errorf("held by %s", strings.Join(names, ", "))
	if len(names) == 0 {
		held = errors.New("no process holds the destination")
	}
	return &os.LinkError{Op: "rename", Old: oldPath, New: newPath, Err: errors.Join(err, posixErr, listErr, held)}
}

// posixRename replaces newPath with oldPath under POSIX semantics, then
// flushes the moved file so the replace is as durable as a write-through
// MoveFileExW.
func posixRename(oldPath, newPath *uint16) error {
	var noTemplate int32
	handle, err := syscall.CreateFile(oldPath, deleteAccess|syscall.GENERIC_WRITE|syscall.SYNCHRONIZE,
		syscall.FILE_SHARE_READ|syscall.FILE_SHARE_WRITE|syscall.FILE_SHARE_DELETE, nil, syscall.OPEN_EXISTING, syscall.FILE_ATTRIBUTE_NORMAL, noTemplate)
	if err != nil {
		return err
	}
	defer syscall.CloseHandle(handle)
	length := 0
	for *(*uint16)(unsafe.Add(unsafe.Pointer(newPath), uintptr(length)*unsafe.Sizeof(*newPath))) != 0 {
		length++
	}
	name := unsafe.Slice(newPath, length)
	// FILE_RENAME_INFO: flags, the root directory handle, the name's byte
	// length, then the name itself.
	info := make([]byte, renameInfoNameOffset+len(name)*int(unsafe.Sizeof(name[0])))
	*(*uint32)(unsafe.Pointer(&info[0])) = renameReplaceIfExists | renamePosixSemantics
	*(*uint32)(unsafe.Pointer(&info[renameInfoLengthOffset])) = uint32(len(name) * int(unsafe.Sizeof(name[0])))
	copy(unsafe.Slice((*uint16)(unsafe.Pointer(&info[renameInfoNameOffset])), len(name)), name)
	if ok, _, callErr := setFileInformationByHandle.Call(uintptr(handle), fileRenameInfoEx, uintptr(unsafe.Pointer(&info[0])), uintptr(len(info))); ok == 0 {
		return callErr
	}
	return syscall.FlushFileBuffers(handle)
}

// FileUsers asks the kernel which processes hold path open, none when it does
// not exist. Attribute access joins no sharing check, so the query sees an
// exclusive holder too.
func FileUsers(path string) ([]uint32, error) {
	encoded, err := syscall.UTF16PtrFromString(path)
	if err != nil {
		return nil, err
	}
	var noTemplate int32
	handle, err := syscall.CreateFile(encoded, fileReadAttributes,
		syscall.FILE_SHARE_READ|syscall.FILE_SHARE_WRITE|syscall.FILE_SHARE_DELETE, nil, syscall.OPEN_EXISTING, syscall.FILE_ATTRIBUTE_NORMAL, noTemplate)
	if errors.Is(err, syscall.ERROR_FILE_NOT_FOUND) || errors.Is(err, syscall.ERROR_PATH_NOT_FOUND) {
		return nil, nil
	}
	if err != nil {
		return nil, fmt.Errorf("fsatomic: inspect %s: %w", path, err)
	}
	defer syscall.CloseHandle(handle)
	// The list is a count word followed by one process id per slot.
	buffer := make([]uintptr, initialHolderSlots)
	for {
		var status [2]uintptr
		result, _, _ := ntQueryInformationFile.Call(uintptr(handle), uintptr(unsafe.Pointer(&status[0])),
			uintptr(unsafe.Pointer(&buffer[0])), uintptr(len(buffer))*unsafe.Sizeof(buffer[0]), fileProcessIDsUsingFileInformation)
		switch uint32(result) {
		case statusSuccess:
			ids := buffer[1:]
			pids := make([]uint32, 0, min(int(buffer[0]), len(ids)))
			for _, id := range ids[:cap(pids)] {
				pids = append(pids, uint32(id))
			}
			return pids, nil
		case statusInfoLengthMismatch, statusBufferOverflow:
			buffer = make([]uintptr, len(buffer)*2)
		default:
			return nil, fmt.Errorf("fsatomic: list holders of %s: NTSTATUS %#x", path, uint32(result))
		}
	}
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
	if errors.Is(callErr, syscall.Errno(0)) {
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
