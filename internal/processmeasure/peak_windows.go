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
