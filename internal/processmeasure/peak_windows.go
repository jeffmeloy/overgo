//go:build windows

package processmeasure

import (
	"errors"
	"fmt"
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
	return PeakWorkingSet(handle)
}

// PeakWorkingSet reports the peak working set of the process a handle names, from its memory counters.
//   - the handle needs query access
//   - a handle held open across the process's exit keeps the counters readable.
func PeakWorkingSet(handle syscall.Handle) (uint64, error) {
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
