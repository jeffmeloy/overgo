//go:build windows

package processcontrol

import (
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"slices"
	"sync"
	"syscall"

	"overgo/internal/fsatomic"
)

// A turn's files beside the resource admissions: the record its holder writes and keeps open, and the file its
// waiters hold open while they wait.
const (
	turnRecordFile  = ".turn.json"
	turnWaitingFile = ".waiting"
)

// turnFile is the path of the turn's file of the kind named.
func turnFile(name, kind string) (string, error) {
	admission, err := resourceAdmissionPath(name)
	if err != nil {
		return "", err
	}
	return filepath.Join(filepath.Dir(admission), resourceObjectName(name)+kind), nil
}

// openTurnFile opens the turn's file of the kind named for access, shared with every other opener, as disposition
// creates it.
func openTurnFile(name, kind string, access, disposition uint32) (syscall.Handle, error) {
	path, err := turnFile(name, kind)
	if err != nil {
		return 0, err
	}
	encoded, err := syscall.UTF16PtrFromString(path)
	if err != nil {
		return 0, err
	}
	handle, err := syscall.CreateFile(encoded, access, syscall.FILE_SHARE_READ|syscall.FILE_SHARE_WRITE|syscall.FILE_SHARE_DELETE,
		nil, disposition, syscall.FILE_ATTRIBUTE_NORMAL, 0)
	if err != nil {
		return 0, fmt.Errorf("processcontrol: open %s of the turn %q: %w", kind, name, err)
	}
	return handle, nil
}

// markTurnWaiter holds the turn's waiting file open until unmark runs, so its holder reads that a process waits
// (TurnWaited); the system closes it when the waiter ends.
func markTurnWaiter(name string) (unmark func() error, err error) {
	handle, err := openTurnFile(name, turnWaitingFile, syscall.GENERIC_READ, syscall.OPEN_ALWAYS)
	if err != nil {
		return nil, err
	}
	return sync.OnceValue(func() error { return syscall.CloseHandle(handle) }), nil
}

// turnWaiters lists the other processes waiting for the turn.
func turnWaiters(name string) ([]uint32, error) {
	path, err := turnFile(name, turnWaitingFile)
	if err != nil {
		return nil, err
	}
	users, err := fsatomic.FileUsers(path)
	if err != nil {
		return nil, fmt.Errorf("processcontrol: read the turn's waiters %q: %w", name, err)
	}
	return slices.DeleteFunc(users, func(pid uint32) bool { return pid == uint32(os.Getpid()) }), nil
}

// recordTurn writes the holder's record and keeps it open until unrecord runs: a waiter believes the record only of
// a process that has it open, so a record left by a holder that has gone names no one.
func recordTurn(name string, record turnRecord) (unrecord func() error, err error) {
	data, err := json.Marshal(record)
	if err != nil {
		return nil, err
	}
	handle, err := openTurnFile(name, turnRecordFile, syscall.GENERIC_WRITE, syscall.CREATE_ALWAYS)
	if err != nil {
		return nil, err
	}
	if _, err := syscall.Write(handle, data); err != nil {
		return nil, errors.Join(fmt.Errorf("processcontrol: record the turn %q: %w", name, err), syscall.CloseHandle(handle))
	}
	return func() error { return syscall.CloseHandle(handle) }, nil
}

// turnHolder reads the turn's record, and whether the process it names holds it open, so holds the turn.
func turnHolder(name string) (turnRecord, bool, error) {
	path, err := turnFile(name, turnRecordFile)
	if err != nil {
		return turnRecord{}, false, err
	}
	var record turnRecord
	data, err := os.ReadFile(path)
	if errors.Is(err, os.ErrNotExist) || err == nil && json.Unmarshal(data, &record) != nil {
		return turnRecord{}, false, nil
	}
	if err != nil {
		return turnRecord{}, false, fmt.Errorf("processcontrol: read the turn %q: %w", name, err)
	}
	users, err := fsatomic.FileUsers(path)
	if err != nil {
		return turnRecord{}, false, fmt.Errorf("processcontrol: read the turn %q: %w", name, err)
	}
	return record, slices.Contains(users, record.Process), nil
}
