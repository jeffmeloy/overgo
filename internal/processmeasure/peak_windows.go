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

// getProcessIOCounters is GetProcessIoCounters, which fills an IO_COUNTERS.
var getProcessIOCounters = syscall.NewLazyDLL("kernel32.dll").NewProc("GetProcessIoCounters")

// ioCounters is IO_COUNTERS, in its order.
type ioCounters struct {
	readOperations, writeOperations, otherOperations, readBytes, writeBytes, otherBytes uint64
}

// ReadUse reads what this process has spent by now: its processor time, user and kernel (GetProcessTimes), and the
// bytes it has read through its files (GetProcessIoCounters). Measured says the platform reads them; Windows does.
func ReadUse() (use Use, measured bool, err error) {
	self, err := syscall.GetCurrentProcess()
	if err != nil {
		return Use{}, false, err
	}
	var creation, exit, kernel, user syscall.Filetime
	if err := syscall.GetProcessTimes(self, &creation, &exit, &kernel, &user); err != nil {
		return Use{}, false, fmt.Errorf("measurement: process times: %w", err)
	}
	var counters ioCounters
	if ok, _, err := getProcessIOCounters.Call(uintptr(self), uintptr(unsafe.Pointer(&counters))); ok == 0 {
		return Use{}, false, fmt.Errorf("measurement: process read bytes: %w", err)
	}
	// A FILETIME span counts 100-nanosecond intervals.
	span := func(at syscall.Filetime) time.Duration {
		return time.Duration(uint64(at.HighDateTime)<<32|uint64(at.LowDateTime)) * 100
	}
	return Use{CPU: span(kernel) + span(user), ReadBytes: counters.readBytes}, true, nil
}
