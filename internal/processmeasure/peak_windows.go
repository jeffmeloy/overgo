//go:build windows

package processmeasure

import (
	"errors"
	"fmt"
	"os"
	"sync"
	"syscall"
	"time"
	"unsafe"
)

var getProcessMemoryInfo = syscall.NewLazyDLL("psapi.dll").NewProc("GetProcessMemoryInfo")

var performanceCounter = syscall.NewLazyDLL("kernel32.dll").NewProc("QueryPerformanceCounter")

var performanceFrequency = sync.OnceValues(func() (int64, error) {
	query := syscall.NewLazyDLL("kernel32.dll").NewProc("QueryPerformanceFrequency")
	var frequency int64
	if ok, _, err := query.Call(uintptr(unsafe.Pointer(&frequency))); ok == 0 {
		return 0, fmt.Errorf("measurement: performance frequency: %w", err)
	}
	return frequency, nil
})

// Counter returns a high-resolution monotonic timestamp for interval subtraction.
// Its origin is arbitrary; it is not a wall-clock time or a persisted identity.
func Counter() (time.Duration, error) {
	frequency, err := performanceFrequency()
	if err != nil {
		return 0, err
	}
	var ticks int64
	if ok, _, err := performanceCounter.Call(uintptr(unsafe.Pointer(&ticks))); ok == 0 {
		return 0, fmt.Errorf("measurement: performance counter: %w", err)
	}
	return counterDuration(ticks, frequency)
}

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
