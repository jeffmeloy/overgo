//go:build windows

package hostexpert

import (
	"cmp"
	"encoding/binary"
	"math/bits"
	"runtime"
	"slices"
	"syscall"
	"unsafe"
)

var (
	kernel32                         = syscall.NewLazyDLL("kernel32.dll")
	getLogicalProcessorInformationEx = kernel32.NewProc("GetLogicalProcessorInformationEx")
	setThreadGroupAffinity           = kernel32.NewProc("SetThreadGroupAffinity")
	getCurrentThread                 = kernel32.NewProc("GetCurrentThread")
)

// The layout GetLogicalProcessorInformationEx writes for RelationProcessorCore: each entry a
// SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX, its relationship and size, then a PROCESSOR_RELATIONSHIP whose efficiency
// class, group count and GROUP_AFFINITY masks (a 64-bit mask and its group) name the core's logical processors.
const (
	relationProcessorCore = 0
	entrySizeAt           = 4
	entryEfficiencyAt     = 9
	entryGroupCountAt     = 30
	entryGroupMasksAt     = 32
	groupAffinityBytes    = 16
	groupAffinityGroupAt  = 8
	groupProcessors       = 64
)

// physicalCores answers each physical core's first logical processor as Strata's detect_cpu_topology finds them
// (src/kernels/cpu/pool.cpp): asked of the system, not assumed, since half the logical processors of an SMT machine are
// siblings that halve the bandwidth an expert's rows are bound by; on a hybrid processor only its most efficient
// class's cores. None where the system does not say.
func physicalCores() []int {
	var length uint32
	getLogicalProcessorInformationEx.Call(relationProcessorCore, 0, uintptr(unsafe.Pointer(&length)))
	if length == 0 {
		return nil
	}
	buffer := make([]byte, length)
	if done, _, _ := getLogicalProcessorInformationEx.Call(relationProcessorCore, uintptr(unsafe.Pointer(&buffer[0])),
		uintptr(unsafe.Pointer(&length))); done == 0 {
		return nil
	}
	type core struct {
		efficiency byte
		first      int
	}
	var cores []core
	for at := 0; at+entryGroupMasksAt <= int(length); {
		size := int(binary.LittleEndian.Uint32(buffer[at+entrySizeAt:]))
		if size < entryGroupMasksAt || at+size > int(length) {
			break
		}
		entry := buffer[at : at+size]
		at += size
		if binary.LittleEndian.Uint32(entry) != relationProcessorCore {
			continue
		}
		groups := int(binary.LittleEndian.Uint16(entry[entryGroupCountAt:]))
		for group := range groups {
			mask := entry[min(len(entry), entryGroupMasksAt+group*groupAffinityBytes):]
			if len(mask) < groupAffinityBytes {
				break
			}
			if processors := binary.LittleEndian.Uint64(mask); processors != 0 {
				number := int(binary.LittleEndian.Uint16(mask[groupAffinityGroupAt:]))
				cores = append(cores, core{entry[entryEfficiencyAt], number*groupProcessors + bits.TrailingZeros64(processors)})
				break
			}
		}
	}
	if len(cores) == 0 {
		return nil
	}
	best := slices.MaxFunc(cores, func(a, b core) int { return cmp.Compare(a.efficiency, b.efficiency) }).efficiency
	firsts := make([]int, 0, len(cores))
	for _, c := range cores {
		if c.efficiency == best {
			firsts = append(firsts, c.first)
		}
	}
	return firsts
}

// pinThread holds the calling goroutine to its thread and the thread to the logical processor, as Strata pins a pool
// worker (SetThreadGroupAffinity); a refusal leaves the thread where the system puts it.
func pinThread(processor int) {
	runtime.LockOSThread()
	var affinity [groupAffinityBytes]byte
	binary.LittleEndian.PutUint64(affinity[:], 1<<(processor%groupProcessors))
	binary.LittleEndian.PutUint16(affinity[groupAffinityGroupAt:], uint16(processor/groupProcessors))
	thread, _, _ := getCurrentThread.Call()
	setThreadGroupAffinity.Call(thread, uintptr(unsafe.Pointer(&affinity[0])), 0)
}
