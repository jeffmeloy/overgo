//go:build windows

package processcontrol

import (
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"slices"
	"syscall"

	"overgo/internal/fsatomic"
)

// turnRecordPath is where a turn's holder writes its record, beside the resource admissions.
func turnRecordPath(name string) (string, error) {
	admission, err := resourceAdmissionPath(name)
	if err != nil {
		return "", err
	}
	return filepath.Join(filepath.Dir(admission), resourceObjectName(name)+".turn.json"), nil
}

// recordTurn writes the holder's record and keeps it open until unrecord runs: a waiter believes the record only of
// a process that has it open, so a record left by a holder that has gone names no one.
func recordTurn(name string, record turnRecord) (unrecord func() error, err error) {
	path, err := turnRecordPath(name)
	if err != nil {
		return nil, err
	}
	encoded, err := syscall.UTF16PtrFromString(path)
	if err != nil {
		return nil, err
	}
	data, err := json.Marshal(record)
	if err != nil {
		return nil, err
	}
	handle, err := syscall.CreateFile(encoded, syscall.GENERIC_WRITE, syscall.FILE_SHARE_READ|syscall.FILE_SHARE_WRITE|syscall.FILE_SHARE_DELETE,
		nil, syscall.CREATE_ALWAYS, syscall.FILE_ATTRIBUTE_NORMAL, 0)
	if err != nil {
		return nil, fmt.Errorf("processcontrol: record the turn %q: %w", name, err)
	}
	if _, err := syscall.Write(handle, data); err != nil {
		return nil, errors.Join(fmt.Errorf("processcontrol: record the turn %q: %w", name, err), syscall.CloseHandle(handle))
	}
	return func() error { return syscall.CloseHandle(handle) }, nil
}

// turnHolder reads the turn's record, and whether the process it names holds it open, so holds the turn.
func turnHolder(name string) (turnRecord, bool, error) {
	path, err := turnRecordPath(name)
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
