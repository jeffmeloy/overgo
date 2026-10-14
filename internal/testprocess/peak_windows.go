//go:build windows

package testprocess

import (
	"fmt"
	"os"
	"syscall"

	"overgo/internal/processmeasure"
)

// processQueryLimitedInformation is PROCESS_QUERY_LIMITED_INFORMATION, the
// access the memory counters need.
const processQueryLimitedInformation = 0x1000

// peakHandle retains a child's process object so its kernel-kept peak
// working set is readable after the child has exited.
type peakHandle struct{ handle syscall.Handle }

// retainPeak duplicates the running child's handle with query access.
func retainPeak(process *os.Process) (peakHandle, error) {
	var retained syscall.Handle
	var duplicateErr error
	err := process.WithHandle(func(handle uintptr) {
		current, err := syscall.GetCurrentProcess()
		if err != nil {
			duplicateErr = err
			return
		}
		duplicateErr = syscall.DuplicateHandle(current, syscall.Handle(handle), current, &retained, processQueryLimitedInformation, false, 0)
	})
	if err != nil {
		return peakHandle{}, err
	}
	if duplicateErr != nil {
		return peakHandle{}, fmt.Errorf("retain process handle: %w", duplicateErr)
	}
	return peakHandle{handle: retained}, nil
}

// read answers the process's peak working set from its memory counters.
func (p peakHandle) read() (uint64, error) { return processmeasure.PeakWorkingSet(p.handle) }

// close releases the retained handle.
func (p peakHandle) close() error { return syscall.CloseHandle(p.handle) }
