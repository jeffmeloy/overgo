package landing

import (
	"errors"
	"fmt"
	"path/filepath"
	"syscall"
	"unsafe"
)

// The kernel's calls free space is read and its fall watched with: a volume's free space, its changes reported to a
// handle, and an event that ends the watch.
var (
	kernel                     = syscall.NewLazyDLL("kernel32.dll")
	freeSpaceCall              = kernel.NewProc("GetDiskFreeSpaceExW")
	firstChangeCall            = kernel.NewProc("FindFirstChangeNotificationW")
	nextChangeCall             = kernel.NewProc("FindNextChangeNotification")
	closeChangeCall            = kernel.NewProc("FindCloseChangeNotification")
	createEventCall            = kernel.NewProc("CreateEventW")
	setEventCall               = kernel.NewProc("SetEvent")
	waitForMultipleObjectsCall = kernel.NewProc("WaitForMultipleObjects")
)

const (
	// changeFilter is the changes that report a write: a file or directory made, renamed or removed, or a file's size
	// changed (FILE_NOTIFY_CHANGE_FILE_NAME, DIR_NAME and SIZE).
	changeFilter = 0x1 | 0x2 | 0x8
	// waitForever is INFINITE, and waitFailed WAIT_FAILED.
	waitForever = 0xFFFFFFFF
	waitFailed  = 0xFFFFFFFF
)

// VolumeFree reads how many bytes this process may still write on the volume
// that holds directory, which is less than the volume's free bytes under a quota.
func VolumeFree(directory string) (uint64, error) {
	name, err := syscall.UTF16PtrFromString(directory)
	if err != nil {
		return 0, err
	}
	var free uint64
	// The call's later results, the volume's size and its free bytes for every user, are not asked for.
	if read, _, failure := freeSpaceCall.Call(uintptr(unsafe.Pointer(name)), uintptr(unsafe.Pointer(&free)), 0, 0); read == 0 {
		return 0, fmt.Errorf("free space of %s: %w", directory, failure)
	}
	return free, nil
}

// watchChanges calls changed each time the volume that holds directory reports a change under its root, until stop is
// called; stop waits for the watch to end.
func watchChanges(directory string, changed func()) (stop func() error, err error) {
	root, err := syscall.UTF16PtrFromString(filepath.VolumeName(directory) + `\`)
	if err != nil {
		return nil, err
	}
	change, _, failure := firstChangeCall.Call(uintptr(unsafe.Pointer(root)), 1, changeFilter)
	if syscall.Handle(change) == syscall.InvalidHandle {
		return nil, fmt.Errorf("watch the changes of the volume of %s: %w", directory, failure)
	}
	ended, _, failure := createEventCall.Call(0, 1, 0, 0)
	if ended == 0 {
		_, _, _ = closeChangeCall.Call(change)
		return nil, fmt.Errorf("make the event that ends the watch of %s: %w", directory, failure)
	}
	handles := [2]uintptr{change, ended}
	done := make(chan error, 1)
	go func() {
		var watched error
		for {
			woke, _, failure := waitForMultipleObjectsCall.Call(2, uintptr(unsafe.Pointer(&handles[0])), 0, waitForever)
			if woke != 0 {
				if woke == waitFailed {
					watched = fmt.Errorf("wait for the changes of the volume of %s: %w", directory, failure)
				}
				break
			}
			changed()
			if next, _, failure := nextChangeCall.Call(change); next == 0 {
				watched = fmt.Errorf("watch the next change of the volume of %s: %w", directory, failure)
				break
			}
		}
		_, _, _ = closeChangeCall.Call(change)
		done <- watched
	}()
	return func() error {
		if set, _, failure := setEventCall.Call(ended); set == 0 {
			return fmt.Errorf("end the watch of %s: %w", directory, failure)
		}
		watched := <-done
		return errors.Join(watched, syscall.CloseHandle(syscall.Handle(ended)))
	}, nil
}
