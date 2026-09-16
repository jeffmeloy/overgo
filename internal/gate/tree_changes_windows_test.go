//go:build windows

package gate

import (
	"fmt"
	"syscall"
	"unsafe"
)

var (
	testKernel32                    = syscall.NewLazyDLL("kernel32.dll")
	procFindFirstChangeNotification = testKernel32.NewProc("FindFirstChangeNotificationW")
	procFindNextChangeNotification  = testKernel32.NewProc("FindNextChangeNotificationW")
	procFindCloseChangeNotification = testKernel32.NewProc("FindCloseChangeNotification")
	procWaitForSingleObject         = testKernel32.NewProc("WaitForSingleObject")
)

const (
	// Win32 FILE_NOTIFY_CHANGE_FILE_NAME, TRUE for the subtree flag,
	// INFINITE and WAIT_OBJECT_0.
	notifyChangeFileName = 0x00000001
	watchSubtree         = 1
	waitInfinite         = 0xFFFFFFFF
	waitObjectZero       = 0
)

// The Windows notification binds itself as the package's tree watcher.
func init() { treeChangeWatch = watchTreeChanges }

// treeWatch delivers the file system's own notification of a name change
// anywhere below a directory, so a test waits on a lock file's removal
// instead of polling for it.
type treeWatch struct{ handle uintptr }

func watchTreeChanges(directory string) (treeWatcher, error) {
	encoded, err := syscall.UTF16PtrFromString(directory)
	if err != nil {
		return nil, err
	}
	handle, _, callErr := procFindFirstChangeNotification.Call(uintptr(unsafe.Pointer(encoded)), watchSubtree, notifyChangeFileName)
	if handle == uintptr(syscall.InvalidHandle) {
		return nil, fmt.Errorf("watch %s: %w", directory, callErr)
	}
	return &treeWatch{handle: handle}, nil
}

// next returns once a name below the directory has changed and re-arms.
func (w *treeWatch) next() error {
	if result, _, callErr := procWaitForSingleObject.Call(w.handle, waitInfinite); result != waitObjectZero {
		return fmt.Errorf("wait for tree change: %w", callErr)
	}
	if ok, _, callErr := procFindNextChangeNotification.Call(w.handle); ok == 0 {
		return fmt.Errorf("re-arm tree change: %w", callErr)
	}
	return nil
}

func (w *treeWatch) close() {
	_, _, _ = procFindCloseChangeNotification.Call(w.handle)
}
