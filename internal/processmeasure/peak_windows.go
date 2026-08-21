//go:build windows

package processmeasure

import (
	"errors"
	"os"
	"syscall"
	"unsafe"
)

var getProcessMemoryInfo = syscall.NewLazyDLL("psapi.dll").NewProc("GetProcessMemoryInfo")

type processMemoryCounters struct {
	Size                       uint32
	PageFaultCount             uint32
	PeakWorkingSetSize         uintptr
	WorkingSetSize             uintptr
	QuotaPeakPagedPoolUsage    uintptr
	QuotaPagedPoolUsage        uintptr
	QuotaPeakNonPagedPoolUsage uintptr
	QuotaNonPagedPoolUsage     uintptr
	PagefileUsage              uintptr
	PeakPagefileUsage          uintptr
	PrivateUsage               uintptr
}

// SelfPeakWorkingSet reports the calling process's own peak working set (the
// RSS high-water), through the current-process pseudo handle.
func SelfPeakWorkingSet() (uint64, error) {
	handle, err := syscall.GetCurrentProcess()
	if err != nil {
		return 0, err
	}
	var counters processMemoryCounters
	counters.Size = uint32(unsafe.Sizeof(counters))
	result, _, callErr := getProcessMemoryInfo.Call(
		uintptr(handle), uintptr(unsafe.Pointer(&counters)), uintptr(counters.Size),
	)
	if result == 0 {
		if callErr == nil || errors.Is(callErr, syscall.Errno(0)) {
			callErr = errors.New("GetProcessMemoryInfo failed")
		}
		return 0, callErr
	}
	return uint64(counters.PeakWorkingSetSize), nil
}

func peakWorkingSet(process *os.Process) (uint64, error) {
	var counters processMemoryCounters
	counters.Size = uint32(unsafe.Sizeof(counters))
	var queryErr error
	err := process.WithHandle(func(handle uintptr) {
		result, _, callErr := getProcessMemoryInfo.Call(
			handle, uintptr(unsafe.Pointer(&counters)), uintptr(counters.Size),
		)
		if result == 0 {
			queryErr = callErr
			if queryErr == nil || errors.Is(queryErr, syscall.Errno(0)) {
				queryErr = errors.New("GetProcessMemoryInfo failed")
			}
		}
	})
	if err != nil {
		return 0, err
	}
	if queryErr != nil {
		return 0, queryErr
	}
	return uint64(counters.PeakWorkingSetSize), nil
}
