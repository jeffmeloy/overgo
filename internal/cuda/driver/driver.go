package driver

import (
	"errors"
	"fmt"
	"strconv"
)

// CardTurn names the turn every user of the card whose identity is uuid queues for (processcontrol.TakeTurn): each
// context is created in it.
func CardTurn(uuid string) string { return "overgo-card-turn:" + uuid }

// CardTurnOf names the turn of the card of ordinal (CardTurn), for a job that takes it once for the processes it runs
// on the card, which ride on its place (processcontrol.TurnEnvironment): a landing's lanes and a measurement run's
// phases on the card, where each test had waited out a picture of a render beside them (card-handover/lane-turn).
func CardTurnOf(ordinal int) (name string, err error) {
	library, err := Open()
	if err != nil {
		return "", err
	}
	defer func() { err = errors.Join(err, library.Close()) }()
	if err := library.Init(); err != nil {
		return "", err
	}
	info, err := library.DeviceInfo(ordinal)
	if err != nil {
		return "", err
	}
	return CardTurn(info.UUID), nil
}

// Device: CUDA device ordinal
type Device int32

// Context: CUDA driver context handle
type Context uintptr

// Stream: CUDA stream handle
type Stream uintptr

// Event is a mark of a stream's work that another stream waits for once (RecordEvent, WaitEvent).
type Event uintptr

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
	Ordinal          int    `json:"ordinal"`
	UUID             string `json:"uuid"`
	Name             string `json:"name"`
	TotalMemoryBytes uint64 `json:"totalMemoryBytes"`
	// BudgetBytes is the declared device budget the card is read under where it holds more (devicebudget), none where
	// unset: TotalMemoryBytes stays the card's own, so a record names both.
	BudgetBytes            uint64 `json:"budgetBytes,omitzero"`
	ComputeCapabilityMajor int    `json:"computeCapabilityMajor"`
	ComputeCapabilityMinor int    `json:"computeCapabilityMinor"`
	MultiprocessorCount    int    `json:"multiprocessorCount"`
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
	// PinnedHostBytes and PeakPinnedHostBytes are the host memory this library holds pinned now and has at most: a
	// model whose experts live pinned on the host takes most of the machine's memory there (flash-next-load).
	PinnedHostBytes     uint64 `json:"pinnedHostBytes,omitzero"`
	PeakPinnedHostBytes uint64 `json:"peakPinnedHostBytes,omitzero"`
}

// Commit is the machine's commit charge: Windows backs every pinned host byte and every card byte a process allocates
// with commit (8 GiB on the card took 8 GiB of it and no memory, tmp/flash-next-load/cardcommit), whose limit is the
// memory and the page file together, so a pin or a card allocation fails once the machine's commit is spent however
// much memory is free (flash-next-load).
type Commit struct{ Left, Limit uint64 }

// String names the commit left of its limit, in GiB.
func (c Commit) String() string {
	return fmt.Sprintf("%.1f GiB of commit left of %.1f", float64(c.Left)/(1<<30), float64(c.Limit)/(1<<30))
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
	KernelLaunches         uint64 `json:"kernelLaunches"`
	StreamSynchronizations uint64 `json:"streamSynchronizations"`
	// ComputeStreamSynchronizations are the waits on the stream a worker computes on, without its side streams' (the
	// copies of experts landing, the host experts' inputs read), which depend on how fast the card moved them beside
	// its neighbours; a worker's ExecutionStats fills it (replay-waits/own-waits).
	ComputeStreamSynchronizations uint64 `json:"computeStreamSynchronizations"`
	ContextSynchronizations       uint64 `json:"contextSynchronizations"`
	HostToDeviceCopies            uint64 `json:"hostToDeviceCopies"`
	HostToDeviceBytes             uint64 `json:"hostToDeviceBytes"`
	DeviceToHostCopies            uint64 `json:"deviceToHostCopies"`
	DeviceToHostBytes             uint64 `json:"deviceToHostBytes"`
	DeviceToDeviceCopies          uint64 `json:"deviceToDeviceCopies"`
	DeviceToDeviceBytes           uint64 `json:"deviceToDeviceBytes"`
	DeviceMemsets                 uint64 `json:"deviceMemsets"`
	DeviceMemsetBytes             uint64 `json:"deviceMemsetBytes"`
	GraphInstantiations           uint64 `json:"graphInstantiations"`
	GraphUpdates                  uint64 `json:"graphUpdates"`
	GraphLaunches                 uint64 `json:"graphLaunches"`
}

// CardReading is one live reading of a card's load, by the name /metrics gives it after overgo_card_.
type CardReading struct {
	Name  string
	Value float64
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
