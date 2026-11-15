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

// turnWatchChanges are the changes to the turn's directory a waiter reads the turn's record again on: a file made or
// gone, written or resized, as a holder's record is when the turn passes to it or its count moves.
const turnWatchChanges = syscall.FILE_NOTIFY_CHANGE_FILE_NAME | syscall.FILE_NOTIFY_CHANGE_LAST_WRITE | syscall.FILE_NOTIFY_CHANGE_SIZE

// turnWatchBuffer is what one read of the directory's changes holds; a change past it still ends the read, which is
// all the watch reads it for.
const turnWatchBuffer = 4096

// waitObject0 is what WaitForMultipleObjects returns when the first handle it waits on is signalled; windowsTrue is
// the system's true.
const (
	waitObject0 = 0
	windowsTrue = 1
)

var procGetOverlappedResult = kernel32.NewProc("GetOverlappedResult")

// watchTurn calls changed with the turn's record each time the directory the record is kept in changes while the
// record's process holds it open: a waiter reads who holds the turn as it passes, woken by the change and never by a
// clock (turn-holder/do). stop ends the watch, cancelling the read it waits on, and returns once nothing more is
// called.
func watchTurn(name string, changed func(turnRecord)) (stop func(), err error) {
	path, err := turnFile(name, turnRecordFile)
	if err != nil {
		return nil, err
	}
	encoded, err := syscall.UTF16PtrFromString(filepath.Dir(path))
	if err != nil {
		return nil, err
	}
	directory, err := syscall.CreateFile(encoded, syscall.FILE_LIST_DIRECTORY, syscall.FILE_SHARE_READ|syscall.FILE_SHARE_WRITE|syscall.FILE_SHARE_DELETE,
		nil, syscall.OPEN_EXISTING, syscall.FILE_FLAG_BACKUP_SEMANTICS|syscall.FILE_FLAG_OVERLAPPED, 0)
	if err != nil {
		return nil, fmt.Errorf("processcontrol: watch the turn %q: %w", name, err)
	}
	// The first event marks a change read, the second the watch's end.
	events := make([]uintptr, 2)
	for index := range events {
		event, _, callErr := procCreateEventW.Call(0, manualResetEvent, 0, 0)
		if event == 0 {
			closeTurnWatch(directory, events)
			return nil, fmt.Errorf("processcontrol: watch the turn %q: %w", name, callErr)
		}
		events[index] = event
	}
	ended := make(chan struct{})
	go func() {
		defer close(ended)
		overlapped := syscall.Overlapped{HEvent: syscall.Handle(events[0])}
		buffer := make([]byte, turnWatchBuffer)
		for {
			_, _, _ = procResetEvent.Call(events[0])
			if err := syscall.ReadDirectoryChanges(directory, &buffer[0], uint32(len(buffer)), false, turnWatchChanges, nil, &overlapped, 0); err != nil && !errors.Is(err, syscall.ERROR_IO_PENDING) {
				return
			}
			signalled, _, _ := procWaitForMultipleObjects.Call(uintptr(len(events)), uintptr(unsafe.Pointer(&events[0])), windowsFalse, resourceInfiniteWait)
			var transferred uint32
			if signalled != waitObject0 {
				// The read is drained before its buffer and record go.
				_ = syscall.CancelIoEx(directory, &overlapped)
				_, _, _ = procGetOverlappedResult.Call(uintptr(directory), uintptr(unsafe.Pointer(&overlapped)), uintptr(unsafe.Pointer(&transferred)), windowsTrue)
				return
			}
			if read, _, _ := procGetOverlappedResult.Call(uintptr(directory), uintptr(unsafe.Pointer(&overlapped)), uintptr(unsafe.Pointer(&transferred)), windowsFalse); read == 0 {
				return
			}
			if record, held, err := turnHolder(name); err == nil && held {
				changed(record)
			}
		}
	}()
	return sync.OnceFunc(func() {
		_, _, _ = procSetEvent.Call(events[1])
		<-ended
		closeTurnWatch(directory, events)
	}), nil
}

// closeTurnWatch closes a watch's directory and the events it made.
func closeTurnWatch(directory syscall.Handle, events []uintptr) {
	for _, event := range events {
		if event != 0 {
			_ = syscall.CloseHandle(syscall.Handle(event))
		}
	}
	_ = syscall.CloseHandle(directory)
}
