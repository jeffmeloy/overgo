package driver

import (
	"errors"
	"fmt"
	"strconv"

	"overgo/internal/processcontrol"
)

// Device: CUDA device ordinal
type Device int32

// Context: CUDA driver context handle
type Context uintptr

// Stream: CUDA stream handle
type Stream uintptr

// DevicePtr: address in CUDA device memory
type DevicePtr uint64

// Module: loaded CUDA module
type Module uintptr

// Function: CUDA kernel function handle
type Function uintptr

type Graph uintptr
type GraphExec uintptr

// Dim3: CUDA grid or block dimension
type Dim3 struct {
	X uint32
	Y uint32
	Z uint32
}

// DeviceInfo: stable subset of CUDA device properties used by
// runtime and diagnostics
type DeviceInfo struct {
	Ordinal                int    `json:"ordinal"`
	UUID                   string `json:"uuid"`
	Name                   string `json:"name"`
	TotalMemoryBytes       uint64 `json:"totalMemoryBytes"`
	ComputeCapabilityMajor int    `json:"computeCapabilityMajor"`
	ComputeCapabilityMinor int    `json:"computeCapabilityMinor"`
	MultiprocessorCount    int    `json:"multiprocessorCount"`
}

// ReserveDevices reserves visible physical devices for this process tree.
// Consumers inherit the same admission; unrelated process trees cannot enter.
func (l *Library) ReserveDevices() ([]DeviceInfo, error) {
	if err := l.Init(); err != nil {
		return nil, err
	}
	count, err := l.DeviceCount()
	if err != nil {
		return nil, err
	}
	if count == 0 {
		return nil, errors.New("CUDA resource admission requires a physical device")
	}
	var devices []DeviceInfo
	for ordinal := range count {
		info, err := l.DeviceInfo(ordinal)
		if err != nil {
			return nil, err
		}
		if err := processcontrol.ClaimResource(info.UUID); err != nil {
			return nil, err
		}
		devices = append(devices, info)
	}
	return devices, nil
}

// MemoryStats: reports allocations made through one Library instance; is
// runtime-owned accounting view, not total device usage by other processes
type MemoryStats struct {
	CurrentBytes uint64 `json:"currentBytes"`
	PeakBytes    uint64 `json:"peakBytes"`
	Allocations  uint64 `json:"allocations"`
	// PeakAllocationBytes is the size of the single allocation that last
	// raised PeakBytes -- the allocation "at the crossing." It turns a
	// peak-over-budget finding from a diffuse total into one nameable
	// buffer: the size class points at which allocation to chase.
	PeakAllocationBytes uint64 `json:"peakAllocationBytes"`
	// LargestLiveBytes is the largest single live allocation, a second
	// coordinate for locating an outlier buffer.
	LargestLiveBytes uint64 `json:"largestLiveBytes"`
	// PeakLedger is the live-allocation size histogram captured at the
	// moment the peak was last raised, largest total first. It is the
	// evidence a peak-over-budget diagnosis needs: a single oversized
	// class names one buffer, while an excess spread across many small
	// classes is genuinely diffuse.
	PeakLedger []AllocationSizeClass `json:"peakLedger,omitempty"`
}

// AllocationSizeClass counts the live allocations of one exact size at
// the peak crossing.
type AllocationSizeClass struct {
	Bytes uint64 `json:"bytes"`
	Count uint64 `json:"count"`
}

// ExecutionStats: reports successful CUDA Driver API work submitted through
// one Library instance; Counters are monotonic for library lifetime
type ExecutionStats struct {
	KernelLaunches          uint64 `json:"kernelLaunches"`
	StreamSynchronizations  uint64 `json:"streamSynchronizations"`
	ContextSynchronizations uint64 `json:"contextSynchronizations"`
	HostToDeviceCopies      uint64 `json:"hostToDeviceCopies"`
	HostToDeviceBytes       uint64 `json:"hostToDeviceBytes"`
	DeviceToHostCopies      uint64 `json:"deviceToHostCopies"`
	DeviceToHostBytes       uint64 `json:"deviceToHostBytes"`
	DeviceToDeviceCopies    uint64 `json:"deviceToDeviceCopies"`
	DeviceToDeviceBytes     uint64 `json:"deviceToDeviceBytes"`
	DeviceMemsets           uint64 `json:"deviceMemsets"`
	DeviceMemsetBytes       uint64 `json:"deviceMemsetBytes"`
	GraphInstantiations     uint64 `json:"graphInstantiations"`
	GraphUpdates            uint64 `json:"graphUpdates"`
	GraphLaunches           uint64 `json:"graphLaunches"`
}

// Version: CUDA's integer driver/toolkit version representation
type Version int

func (v Version) String() string {
	n := int(v)
	if n <= 0 {
		return "0"
	}
	major := n / 1000
	minor := (n % 1000) / 10
	return strconv.Itoa(major) + "." + strconv.Itoa(minor)
}

// ResultError: reports failed CUDA Driver API operation
type ResultError struct {
	Operation string
	Code      int32
	Name      string
	Message   string
}

// cudaErrorOutOfMemory: CUDA_ERROR_OUT_OF_MEMORY (CUDA Driver API result code).
const cudaErrorOutOfMemory = 2

// IsOutOfMemory: allocation-failure classification; derived-residency
// placement falls back to streaming on exactly this result.
func IsOutOfMemory(err error) bool {
	result, ok := errors.AsType[*ResultError](err)
	return ok && result.Code == cudaErrorOutOfMemory
}

func (e *ResultError) Error() string {
	detail := e.Name
	if detail == "" {
		detail = fmt.Sprintf("CUDA result %d", e.Code)
	}
	if e.Message != "" {
		detail += ": " + e.Message
	}
	if e.Operation == "" {
		return detail
	}
	return e.Operation + ": " + detail
}

const (
	attributeMultiprocessorCount    = 16
	attributeMaxThreadsPerMultiproc = 39
	attributeComputeCapabilityMajor = 75
	attributeComputeCapabilityMinor = 76
)
