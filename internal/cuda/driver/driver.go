package driver

import (
	"fmt"
	"strconv"
)

// Device is a CUDA device ordinal.
type Device int32

// Context is a CUDA driver context handle.
type Context uintptr

// Stream is a CUDA stream handle.
type Stream uintptr

// DevicePtr is an address in CUDA device memory.
type DevicePtr uint64

// Module is a loaded CUDA module.
type Module uintptr

// Function is a CUDA kernel function handle.
type Function uintptr

// Dim3 is a CUDA grid or block dimension.
type Dim3 struct {
	X uint32
	Y uint32
	Z uint32
}

// DeviceInfo is the stable subset of CUDA device properties used by the
// runtime and diagnostics.
type DeviceInfo struct {
	Ordinal                int    `json:"ordinal"`
	Name                   string `json:"name"`
	TotalMemoryBytes       uint64 `json:"totalMemoryBytes"`
	ComputeCapabilityMajor int    `json:"computeCapabilityMajor"`
	ComputeCapabilityMinor int    `json:"computeCapabilityMinor"`
	MultiprocessorCount    int    `json:"multiprocessorCount"`
}

// MemoryStats reports allocations made through one Library instance. It is a
// runtime-owned accounting view, not total device usage by other processes.
type MemoryStats struct {
	CurrentBytes uint64 `json:"currentBytes"`
	PeakBytes    uint64 `json:"peakBytes"`
	Allocations  uint64 `json:"allocations"`
}

// ExecutionStats reports successful CUDA Driver API work submitted through
// one Library instance. Counters are monotonic for the library lifetime.
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
}

// Version is CUDA's integer driver/toolkit version representation.
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

// ResultError reports a failed CUDA Driver API operation.
type ResultError struct {
	Operation string
	Code      int32
	Name      string
	Message   string
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
	attributeComputeCapabilityMajor = 75
	attributeComputeCapabilityMinor = 76
)
