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
	errorInvalidParameter              syscall.Errno = 87
	errorUserMappedFile                syscall.Errno = 1224
)

var (
	moveFileExW            = syscall.NewLazyDLL("kernel32.dll").NewProc("MoveFileExW")
	ntQueryInformationFile = syscall.NewLazyDLL("ntdll.dll").NewProc("NtQueryInformationFile")
	// waitingOnHolder observes each holder a replace waits on.
	waitingOnHolder = func(uint32) {}
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
	for {
		err := move(oldPointer, newPointer, moveFileReplaceExisting|moveFileWriteThrough)
		if err == nil {
			return nil
		}
		// A destination another process holds open or mapped refuses the
		// move with one of these; a holder is waited out, anything else --
		// a denial with no holder among them -- is the move's own error.
		if !errors.Is(err, syscall.ERROR_ACCESS_DENIED) && !errors.Is(err, errorSharingViolation) && !errors.Is(err, errorUserMappedFile) {
			return &os.LinkError{Op: "rename", Old: oldPath, New: newPath, Err: err}
		}
		if waitErr := awaitHolders(newPath); waitErr != nil {
			return &os.LinkError{Op: "rename", Old: oldPath, New: newPath, Err: errors.Join(err, waitErr)}
		}
	}
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

// awaitHolders waits until every other process holding path open has exited,
// and refuses, naming them, when none does or a holder cannot be waited on:
// this process itself, or one it may not synchronize with.
func awaitHolders(path string) error {
	holders, err := FileUsers(path)
	if err != nil {
		return err
	}
	if len(holders) == 0 {
		return errors.New("no process holds the destination")
	}
	self := uint32(os.Getpid())
	var unwaitable []string
	for _, holder := range holders {
		process, err := syscall.OpenProcess(syscall.SYNCHRONIZE, false, holder)
		if errors.Is(err, errorInvalidParameter) {
			// The holder has already exited and released the file.
			continue
		}
		if holder == self || err != nil {
			unwaitable = append(unwaitable, fmt.Sprintf("process %d", holder))
			if err == nil {
				_ = syscall.CloseHandle(process)
			}
			continue
		}
		waitingOnHolder(holder)
		_, waitErr := syscall.WaitForSingleObject(process, syscall.INFINITE)
		if err := errors.Join(waitErr, syscall.CloseHandle(process)); err != nil {
			return err
		}
	}
	if len(unwaitable) != 0 {
		return fmt.Errorf("held by %s", strings.Join(unwaitable, ", "))
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
