//go:build windows

package driver

import (
	"errors"
	"runtime"
	"strings"
	"sync"
	"sync/atomic"
	"syscall"
	"unsafe"
)

const cudaDeviceNameBytes = 256

var (
	kernel32      = syscall.NewLazyDLL("kernel32.dll")
	lstrlenA      = kernel32.NewProc("lstrlenA")
	rtlMoveMemory = kernel32.NewProc("RtlMoveMemory")
)

// Library: dynamically loaded CUDA Driver API library
type Library struct {
	dll *syscall.DLL

	allocationMu        sync.Mutex
	allocations         map[DevicePtr]uint64
	currentBytes        uint64
	peakBytes           uint64
	peakAllocationBytes uint64

	kernelLaunches          atomic.Uint64
	streamSynchronizations  atomic.Uint64
	contextSynchronizations atomic.Uint64
	hostToDeviceCopies      atomic.Uint64
	hostToDeviceBytes       atomic.Uint64
	deviceToHostCopies      atomic.Uint64
	deviceToHostBytes       atomic.Uint64
	deviceToDeviceCopies    atomic.Uint64
	deviceToDeviceBytes     atomic.Uint64
	deviceMemsets           atomic.Uint64
	deviceMemsetBytes       atomic.Uint64
	graphInstantiations     atomic.Uint64
	graphUpdates            atomic.Uint64
	graphLaunches           atomic.Uint64

	cuInit               *syscall.Proc
	cuDriverGetVersion   *syscall.Proc
	cuDeviceGetCount     *syscall.Proc
	cuDeviceGet          *syscall.Proc
	cuDeviceGetName      *syscall.Proc
	cuDeviceTotalMem     *syscall.Proc
	cuDeviceGetAttribute *syscall.Proc
	cuGetErrorName       *syscall.Proc
	cuGetErrorString     *syscall.Proc
	cuCtxCreate          *syscall.Proc
	cuCtxDestroy         *syscall.Proc
	cuCtxSetCurrent      *syscall.Proc
	cuCtxSynchronize     *syscall.Proc
	cuStreamCreate       *syscall.Proc
	cuStreamDestroy      *syscall.Proc
	cuStreamSynchronize  *syscall.Proc
	cuMemAlloc           *syscall.Proc
	cuMemFree            *syscall.Proc
	cuMemGetInfo         *syscall.Proc
	cuMemcpyHtoD         *syscall.Proc
	cuMemcpyDtoH         *syscall.Proc
	cuMemcpyDtoD         *syscall.Proc
	cuMemsetD32Async     *syscall.Proc
	cuModuleLoadData     *syscall.Proc
	cuModuleUnload       *syscall.Proc
	cuModuleGetFunction  *syscall.Proc
	cuFuncSetAttribute   *syscall.Proc
	cuLaunchKernel       *syscall.Proc
	cuStreamBeginCapture *syscall.Proc
	cuStreamEndCapture   *syscall.Proc
	cuGraphInstantiate   *syscall.Proc
	cuGraphExecUpdate    *syscall.Proc
	cuGraphLaunch        *syscall.Proc
	cuGraphDestroy       *syscall.Proc
	cuGraphExecDestroy   *syscall.Proc
}

// Open: loads nvcuda.dll and resolves required entry points
func Open() (*Library, error) {
	dll, err := syscall.LoadDLL("nvcuda.dll")
	if err != nil {
		return nil, errors.New("load nvcuda.dll: " + err.Error())
	}

	lib := &Library{dll: dll}
	required := []struct {
		name string
		dst  **syscall.Proc
	}{
		{"cuInit", &lib.cuInit},
		{"cuDriverGetVersion", &lib.cuDriverGetVersion},
		{"cuDeviceGetCount", &lib.cuDeviceGetCount},
		{"cuDeviceGet", &lib.cuDeviceGet},
		{"cuDeviceGetName", &lib.cuDeviceGetName},
		{"cuDeviceTotalMem_v2", &lib.cuDeviceTotalMem},
		{"cuDeviceGetAttribute", &lib.cuDeviceGetAttribute},
		{"cuGetErrorName", &lib.cuGetErrorName},
		{"cuGetErrorString", &lib.cuGetErrorString},
		{"cuCtxCreate_v2", &lib.cuCtxCreate},
		{"cuCtxDestroy_v2", &lib.cuCtxDestroy},
		{"cuCtxSetCurrent", &lib.cuCtxSetCurrent},
		{"cuCtxSynchronize", &lib.cuCtxSynchronize},
		{"cuStreamCreate", &lib.cuStreamCreate},
		{"cuStreamDestroy_v2", &lib.cuStreamDestroy},
		{"cuStreamSynchronize", &lib.cuStreamSynchronize},
		{"cuMemAlloc_v2", &lib.cuMemAlloc},
		{"cuMemFree_v2", &lib.cuMemFree},
		{"cuMemGetInfo_v2", &lib.cuMemGetInfo},
		{"cuMemcpyHtoD_v2", &lib.cuMemcpyHtoD},
		{"cuMemcpyDtoH_v2", &lib.cuMemcpyDtoH},
		{"cuMemcpyDtoD_v2", &lib.cuMemcpyDtoD},
		{"cuMemsetD32Async", &lib.cuMemsetD32Async},
		{"cuModuleLoadData", &lib.cuModuleLoadData},
		{"cuModuleUnload", &lib.cuModuleUnload},
		{"cuModuleGetFunction", &lib.cuModuleGetFunction},
		{"cuFuncSetAttribute", &lib.cuFuncSetAttribute},
		{"cuLaunchKernel", &lib.cuLaunchKernel},
		{"cuStreamBeginCapture", &lib.cuStreamBeginCapture},
		{"cuStreamEndCapture", &lib.cuStreamEndCapture},
		{"cuGraphInstantiateWithFlags", &lib.cuGraphInstantiate},
		{"cuGraphExecUpdate_v2", &lib.cuGraphExecUpdate},
		{"cuGraphLaunch", &lib.cuGraphLaunch},
		{"cuGraphDestroy", &lib.cuGraphDestroy},
		{"cuGraphExecDestroy", &lib.cuGraphExecDestroy},
	}

	for _, item := range required {
		proc, findErr := dll.FindProc(item.name)
		if findErr != nil {
			_ = dll.Release()
			return nil, errors.New("resolve " + item.name + ": " + findErr.Error())
		}
		*item.dst = proc
	}

	return lib, nil
}

func (l *Library) StreamBeginCapture(stream Stream) error {
	result, _, _ := l.cuStreamBeginCapture.Call(uintptr(stream), 0)
	return l.result("cuStreamBeginCapture", result)
}

func (l *Library) StreamEndCapture(stream Stream) (Graph, error) {
	var graph Graph
	var pinned runtime.Pinner
	pinned.Pin(&graph)
	defer pinned.Unpin()
	result, _, _ := l.cuStreamEndCapture.Call(uintptr(stream), uintptr(unsafe.Pointer(&graph)))
	return graph, l.result("cuStreamEndCapture", result)
}

func (l *Library) GraphInstantiate(graph Graph) (GraphExec, error) {
	var execution GraphExec
	var pinned runtime.Pinner
	pinned.Pin(&execution)
	defer pinned.Unpin()
	result, _, _ := l.cuGraphInstantiate.Call(uintptr(unsafe.Pointer(&execution)), uintptr(graph), 0)
	err := l.result("cuGraphInstantiateWithFlags", result)
	if err == nil {
		l.graphInstantiations.Add(1)
	}
	return execution, err
}

func (l *Library) GraphExecUpdate(execution GraphExec, graph Graph) (bool, error) {
	var errorNode uintptr
	var updateResult int32
	var pinned runtime.Pinner
	pinned.Pin(&errorNode)
	pinned.Pin(&updateResult)
	defer pinned.Unpin()
	result, _, _ := l.cuGraphExecUpdate.Call(
		uintptr(execution), uintptr(graph), uintptr(unsafe.Pointer(&errorNode)), uintptr(unsafe.Pointer(&updateResult)),
	)
	err := l.result("cuGraphExecUpdate_v2", result)
	if err == nil && updateResult == 0 {
		l.graphUpdates.Add(1)
	}
	return updateResult == 0, err
}

func (l *Library) GraphLaunch(execution GraphExec, stream Stream) error {
	result, _, _ := l.cuGraphLaunch.Call(uintptr(execution), uintptr(stream))
	err := l.result("cuGraphLaunch", result)
	if err == nil {
		l.graphLaunches.Add(1)
	}
	return err
}

func (l *Library) GraphDestroy(graph Graph) error {
	if graph == 0 {
		return nil
	}
	result, _, _ := l.cuGraphDestroy.Call(uintptr(graph))
	return l.result("cuGraphDestroy", result)
}

func (l *Library) GraphExecDestroy(execution GraphExec) error {
	if execution == 0 {
		return nil
	}
	result, _, _ := l.cuGraphExecDestroy.Call(uintptr(execution))
	return l.result("cuGraphExecDestroy", result)
}

// Close releases process handle for nvcuda.dll
func (l *Library) Close() error {
	if l == nil || l.dll == nil {
		return nil
	}
	err := l.dll.Release()
	l.dll = nil
	return err
}

// Init initializes CUDA driver
func (l *Library) Init() error {
	result, _, _ := l.cuInit.Call(0)
	return l.result("cuInit", result)
}

// DriverVersion: returns highest CUDA version supported by driver
func (l *Library) DriverVersion() (Version, error) {
	var version int32
	var pinned runtime.Pinner
	pinned.Pin(&version)
	defer pinned.Unpin()
	result, _, _ := l.cuDriverGetVersion.Call(uintptr(unsafe.Pointer(&version)))
	if err := l.result("cuDriverGetVersion", result); err != nil {
		return 0, err
	}
	return Version(version), nil
}

// DeviceCount: returns number of CUDA devices visible to driver
func (l *Library) DeviceCount() (int, error) {
	var count int32
	var pinned runtime.Pinner
	pinned.Pin(&count)
	defer pinned.Unpin()
	result, _, _ := l.cuDeviceGetCount.Call(uintptr(unsafe.Pointer(&count)))
	if err := l.result("cuDeviceGetCount", result); err != nil {
		return 0, err
	}
	return int(count), nil
}

// Device: returns CUDA device handle for ordinal
func (l *Library) Device(ordinal int) (Device, error) {
	var device int32
	var pinned runtime.Pinner
	pinned.Pin(&device)
	defer pinned.Unpin()
	result, _, _ := l.cuDeviceGet.Call(
		uintptr(unsafe.Pointer(&device)),
		uintptr(ordinal),
	)
	if err := l.result("cuDeviceGet", result); err != nil {
		return 0, err
	}
	return Device(device), nil
}

// DeviceInfo: returns properties used by Go runtime
func (l *Library) DeviceInfo(ordinal int) (DeviceInfo, error) {
	device, err := l.Device(ordinal)
	if err != nil {
		return DeviceInfo{}, err
	}

	nameBytes := make([]byte, cudaDeviceNameBytes)
	var pinned runtime.Pinner
	pinned.Pin(&nameBytes[0])
	defer pinned.Unpin()
	result, _, _ := l.cuDeviceGetName.Call(
		uintptr(unsafe.Pointer(&nameBytes[0])),
		uintptr(len(nameBytes)),
		uintptr(device),
	)
	if err := l.result("cuDeviceGetName", result); err != nil {
		return DeviceInfo{}, err
	}

	var totalMemory uint64
	pinned.Pin(&totalMemory)
	result, _, _ = l.cuDeviceTotalMem.Call(
		uintptr(unsafe.Pointer(&totalMemory)),
		uintptr(device),
	)
	if err := l.result("cuDeviceTotalMem_v2", result); err != nil {
		return DeviceInfo{}, err
	}

	major, err := l.deviceAttribute(device, attributeComputeCapabilityMajor)
	if err != nil {
		return DeviceInfo{}, err
	}
	minor, err := l.deviceAttribute(device, attributeComputeCapabilityMinor)
	if err != nil {
		return DeviceInfo{}, err
	}
	multiprocessors, err := l.deviceAttribute(device, attributeMultiprocessorCount)
	if err != nil {
		return DeviceInfo{}, err
	}

	runtime.KeepAlive(nameBytes)
	return DeviceInfo{
		Ordinal:                ordinal,
		Name:                   strings.TrimRight(string(nameBytes), "\x00"),
		TotalMemoryBytes:       totalMemory,
		ComputeCapabilityMajor: major,
		ComputeCapabilityMinor: minor,
		MultiprocessorCount:    multiprocessors,
	}, nil
}

// ContextCreate creates CUDA context for device; calling goroutine
// must remain locked to its OS thread while it uses context
func (l *Library) ContextCreate(device Device, flags uint32) (Context, error) {
	var context Context
	var pinned runtime.Pinner
	pinned.Pin(&context)
	defer pinned.Unpin()
	result, _, _ := l.cuCtxCreate.Call(
		uintptr(unsafe.Pointer(&context)),
		uintptr(flags),
		uintptr(device),
	)
	if err := l.result("cuCtxCreate_v2", result); err != nil {
		return 0, err
	}
	return context, nil
}

// ContextDestroy destroys CUDA context
func (l *Library) ContextDestroy(context Context) error {
	result, _, _ := l.cuCtxDestroy.Call(uintptr(context))
	return l.result("cuCtxDestroy_v2", result)
}

// ContextSetCurrent makes context current on calling OS thread
func (l *Library) ContextSetCurrent(context Context) error {
	result, _, _ := l.cuCtxSetCurrent.Call(uintptr(context))
	return l.result("cuCtxSetCurrent", result)
}

// ContextSynchronize waits for all preceding work in current context
func (l *Library) ContextSynchronize() error {
	result, _, _ := l.cuCtxSynchronize.Call()
	if err := l.result("cuCtxSynchronize", result); err != nil {
		return err
	}
	l.contextSynchronizations.Add(1)
	return nil
}

// StreamCreate creates stream in current context
func (l *Library) StreamCreate(flags uint32) (Stream, error) {
	var stream Stream
	var pinned runtime.Pinner
	pinned.Pin(&stream)
	defer pinned.Unpin()
	result, _, _ := l.cuStreamCreate.Call(
		uintptr(unsafe.Pointer(&stream)),
		uintptr(flags),
	)
	if err := l.result("cuStreamCreate", result); err != nil {
		return 0, err
	}
	return stream, nil
}

// StreamDestroy destroys stream
func (l *Library) StreamDestroy(stream Stream) error {
	result, _, _ := l.cuStreamDestroy.Call(uintptr(stream))
	return l.result("cuStreamDestroy_v2", result)
}

// StreamSynchronize waits for preceding work in stream
func (l *Library) StreamSynchronize(stream Stream) error {
	result, _, _ := l.cuStreamSynchronize.Call(uintptr(stream))
	if err := l.result("cuStreamSynchronize", result); err != nil {
		return err
	}
	l.streamSynchronizations.Add(1)
	return nil
}

// MemAlloc: allocates bytes in current CUDA context
func (l *Library) MemAlloc(bytes uint64) (DevicePtr, error) {
	if bytes == 0 {
		return 0, errors.New("cuMemAlloc_v2: allocation size is zero")
	}
	var pointer DevicePtr
	var pinned runtime.Pinner
	pinned.Pin(&pointer)
	defer pinned.Unpin()
	result, _, _ := l.cuMemAlloc.Call(
		uintptr(unsafe.Pointer(&pointer)),
		uintptr(bytes),
	)
	if err := l.result("cuMemAlloc_v2", result); err != nil {
		return 0, err
	}
	l.accountAllocation(pointer, bytes)
	return pointer, nil
}

// MemInfo: returns free and total device memory of the current context
// (cuMemGetInfo_v2) -- driver-reported bytes, the basis for measured resident
// capacity and real peak sampling (never a synthetic accounting sum).
func (l *Library) MemInfo() (free, total uint64, err error) {
	var f, t uint64
	var pinned runtime.Pinner
	pinned.Pin(&f)
	pinned.Pin(&t)
	defer pinned.Unpin()
	result, _, _ := l.cuMemGetInfo.Call(
		uintptr(unsafe.Pointer(&f)),
		uintptr(unsafe.Pointer(&t)),
	)
	if e := l.result("cuMemGetInfo_v2", result); e != nil {
		return 0, 0, e
	}
	return f, t, nil
}

// MemFree: frees device allocation
func (l *Library) MemFree(pointer DevicePtr) error {
	if pointer == 0 {
		return nil
	}
	result, _, _ := l.cuMemFree.Call(uintptr(pointer))
	if err := l.result("cuMemFree_v2", result); err != nil {
		return err
	}
	l.accountFree(pointer)
	return nil
}

// accountFree removes one allocation from the ledger, lowering the live
// total; the peak and its crossing size are high-water marks and stay.
func (l *Library) accountFree(pointer DevicePtr) {
	l.allocationMu.Lock()
	if bytes, ok := l.allocations[pointer]; ok {
		delete(l.allocations, pointer)
		l.currentBytes -= bytes
	}
	l.allocationMu.Unlock()
}

// accountAllocation records one allocation in the ledger and raises the
// peak, capturing the size of the allocation that set a new high-water
// so a peak-over-budget finding names the crossing buffer.
func (l *Library) accountAllocation(pointer DevicePtr, bytes uint64) {
	l.allocationMu.Lock()
	if l.allocations == nil {
		l.allocations = make(map[DevicePtr]uint64)
	}
	l.allocations[pointer] = bytes
	l.currentBytes += bytes
	if l.currentBytes > l.peakBytes {
		l.peakBytes = l.currentBytes
		l.peakAllocationBytes = bytes
	}
	l.allocationMu.Unlock()
}

// ResetPeakBytes lowers the tracked allocation high-water to the current live
// bytes, so a caller can measure the peak of one bounded phase (e.g. one training
// step) rather than the library-lifetime high-water.
func (l *Library) ResetPeakBytes() {
	if l == nil {
		return
	}
	l.allocationMu.Lock()
	l.peakBytes = l.currentBytes
	l.peakAllocationBytes = 0
	l.allocationMu.Unlock()
}

func (l *Library) MemoryStats() MemoryStats {
	if l == nil {
		return MemoryStats{}
	}
	l.allocationMu.Lock()
	defer l.allocationMu.Unlock()
	var largest uint64
	for _, bytes := range l.allocations {
		if bytes > largest {
			largest = bytes
		}
	}
	return MemoryStats{
		CurrentBytes:        l.currentBytes,
		PeakBytes:           l.peakBytes,
		Allocations:         uint64(len(l.allocations)),
		PeakAllocationBytes: l.peakAllocationBytes,
		LargestLiveBytes:    largest,
	}
}

func (l *Library) ExecutionStats() ExecutionStats {
	if l == nil {
		return ExecutionStats{}
	}
	return ExecutionStats{
		KernelLaunches:          l.kernelLaunches.Load(),
		StreamSynchronizations:  l.streamSynchronizations.Load(),
		ContextSynchronizations: l.contextSynchronizations.Load(),
		HostToDeviceCopies:      l.hostToDeviceCopies.Load(),
		HostToDeviceBytes:       l.hostToDeviceBytes.Load(),
		DeviceToHostCopies:      l.deviceToHostCopies.Load(),
		DeviceToHostBytes:       l.deviceToHostBytes.Load(),
		DeviceToDeviceCopies:    l.deviceToDeviceCopies.Load(),
		DeviceToDeviceBytes:     l.deviceToDeviceBytes.Load(),
		DeviceMemsets:           l.deviceMemsets.Load(),
		DeviceMemsetBytes:       l.deviceMemsetBytes.Load(),
		GraphInstantiations:     l.graphInstantiations.Load(),
		GraphUpdates:            l.graphUpdates.Load(),
		GraphLaunches:           l.graphLaunches.Load(),
	}
}

// MemcpyHtoD: copies host byte slice into device memory
func (l *Library) MemcpyHtoD(destination DevicePtr, source []byte) error {
	if len(source) == 0 {
		return nil
	}
	if !l.ownsDeviceRange(destination, uint64(len(source))) {
		return errors.New("cuMemcpyHtoD_v2: destination range is not allocated")
	}
	var pinned runtime.Pinner
	pinned.Pin(&source[0])
	defer pinned.Unpin()
	result, _, _ := l.cuMemcpyHtoD.Call(
		uintptr(destination),
		uintptr(unsafe.Pointer(&source[0])),
		uintptr(len(source)),
	)
	runtime.KeepAlive(source)
	if err := l.result("cuMemcpyHtoD_v2", result); err != nil {
		return err
	}
	l.hostToDeviceCopies.Add(1)
	l.hostToDeviceBytes.Add(uint64(len(source)))
	return nil
}

// MemcpyDtoH: copies device memory into host byte slice
func (l *Library) MemcpyDtoH(destination []byte, source DevicePtr) error {
	if len(destination) == 0 {
		return nil
	}
	if !l.ownsDeviceRange(source, uint64(len(destination))) {
		return errors.New("cuMemcpyDtoH_v2: source range is not allocated")
	}
	var pinned runtime.Pinner
	pinned.Pin(&destination[0])
	defer pinned.Unpin()
	result, _, _ := l.cuMemcpyDtoH.Call(
		uintptr(unsafe.Pointer(&destination[0])),
		uintptr(source),
		uintptr(len(destination)),
	)
	runtime.KeepAlive(destination)
	if err := l.result("cuMemcpyDtoH_v2", result); err != nil {
		return err
	}
	l.deviceToHostCopies.Add(1)
	l.deviceToHostBytes.Add(uint64(len(destination)))
	return nil
}

// MemcpyDtoD: copies bytes between two allocations in current CUDA
// context; Source and destination ranges must not overlap
func (l *Library) MemcpyDtoD(
	destination, source DevicePtr,
	bytes uint64,
) error {
	if bytes == 0 {
		return nil
	}
	if !l.ownsDeviceRange(destination, bytes) || !l.ownsDeviceRange(source, bytes) {
		return errors.New("cuMemcpyDtoD_v2: source or destination range is not allocated")
	}
	result, _, _ := l.cuMemcpyDtoD.Call(
		uintptr(destination),
		uintptr(source),
		uintptr(bytes),
	)
	if err := l.result("cuMemcpyDtoD_v2", result); err != nil {
		return err
	}
	l.deviceToDeviceCopies.Add(1)
	l.deviceToDeviceBytes.Add(bytes)
	return nil
}

// MemsetD32Async: enqueues 32-bit device fill in stream
func (l *Library) MemsetD32Async(
	destination DevicePtr,
	value uint32,
	count uint64,
	stream Stream,
) error {
	if count == 0 {
		return nil
	}
	if count > ^uint64(0)/4 || !l.ownsDeviceRange(destination, count*4) {
		return errors.New("cuMemsetD32Async: destination range is not allocated")
	}
	result, _, _ := l.cuMemsetD32Async.Call(
		uintptr(destination),
		uintptr(value),
		uintptr(count),
		uintptr(stream),
	)
	if err := l.result("cuMemsetD32Async", result); err != nil {
		return err
	}
	l.deviceMemsets.Add(1)
	l.deviceMemsetBytes.Add(count * 4)
	return nil
}

func (l *Library) ownsDeviceRange(pointer DevicePtr, bytes uint64) bool {
	if l == nil || pointer == 0 || bytes == 0 {
		return false
	}
	l.allocationMu.Lock()
	defer l.allocationMu.Unlock()
	for base, capacity := range l.allocations {
		if pointer < base {
			continue
		}
		offset := uint64(pointer - base)
		if offset <= capacity && bytes <= capacity-offset {
			return true
		}
	}
	return false
}

// ModuleLoadData: loads PTX or CUDA binary module from memory
func (l *Library) ModuleLoadData(image []byte) (Module, error) {
	if len(image) == 0 {
		return 0, errors.New("cuModuleLoadData: empty module image")
	}
	data := image
	if image[len(image)-1] != 0 {
		data = make([]byte, len(image)+1)
		copy(data, image)
	}
	var module Module
	var pinned runtime.Pinner
	pinned.Pin(&module)
	pinned.Pin(&data[0])
	defer pinned.Unpin()
	result, _, _ := l.cuModuleLoadData.Call(
		uintptr(unsafe.Pointer(&module)),
		uintptr(unsafe.Pointer(&data[0])),
	)
	runtime.KeepAlive(data)
	if err := l.result("cuModuleLoadData", result); err != nil {
		return 0, err
	}
	return module, nil
}

// ModuleUnload: unloads CUDA module
func (l *Library) ModuleUnload(module Module) error {
	if module == 0 {
		return nil
	}
	result, _, _ := l.cuModuleUnload.Call(uintptr(module))
	return l.result("cuModuleUnload", result)
}

// ModuleFunction: resolves kernel function by name
func (l *Library) ModuleFunction(module Module, name string) (Function, error) {
	if strings.IndexByte(name, 0) >= 0 {
		return 0, errors.New("cuModuleGetFunction: function name contains NUL")
	}
	nameBytes := append([]byte(name), 0)
	var function Function
	var pinned runtime.Pinner
	pinned.Pin(&function)
	pinned.Pin(&nameBytes[0])
	defer pinned.Unpin()
	result, _, _ := l.cuModuleGetFunction.Call(
		uintptr(unsafe.Pointer(&function)),
		uintptr(module),
		uintptr(unsafe.Pointer(&nameBytes[0])),
	)
	runtime.KeepAlive(nameBytes)
	if err := l.result("cuModuleGetFunction", result); err != nil {
		return 0, err
	}
	return function, nil
}

// funcAttributeMaxDynamicSharedBytes: CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES.
const funcAttributeMaxDynamicSharedBytes = 8

// FuncSetMaxDynamicShared: opts a kernel into dynamic shared memory above
// the 48KB default (sm_89 permits up to 99KB per block).
func (l *Library) FuncSetMaxDynamicShared(function Function, bytes uint32) error {
	result, _, _ := l.cuFuncSetAttribute.Call(
		uintptr(function),
		uintptr(funcAttributeMaxDynamicSharedBytes),
		uintptr(bytes),
	)
	return l.result("cuFuncSetAttribute", result)
}

// LaunchKernel: launches CUDA kernel; Each argument must point to storage
// containing one kernel parameter value and must stay live until this method
// returns
func (l *Library) LaunchKernel(
	function Function,
	grid Dim3,
	block Dim3,
	sharedMemoryBytes uint32,
	stream Stream,
	arguments []unsafe.Pointer,
) error {
	if grid.X == 0 || grid.Y == 0 || grid.Z == 0 {
		return errors.New("cuLaunchKernel: grid dimensions must be nonzero")
	}
	if block.X == 0 || block.Y == 0 || block.Z == 0 {
		return errors.New("cuLaunchKernel: block dimensions must be nonzero")
	}
	var argumentPointer uintptr
	var pinned runtime.Pinner
	if len(arguments) > 0 {
		pinned.Pin(&arguments[0])
		defer pinned.Unpin()
		for _, argument := range arguments {
			if argument == nil {
				return errors.New("cuLaunchKernel: kernel argument is nil")
			}
			pinned.Pin(argument)
		}
		argumentPointer = uintptr(unsafe.Pointer(&arguments[0]))
	}
	result, _, _ := l.cuLaunchKernel.Call(
		uintptr(function),
		uintptr(grid.X),
		uintptr(grid.Y),
		uintptr(grid.Z),
		uintptr(block.X),
		uintptr(block.Y),
		uintptr(block.Z),
		uintptr(sharedMemoryBytes),
		uintptr(stream),
		argumentPointer,
		0,
	)
	runtime.KeepAlive(arguments)
	if err := l.result("cuLaunchKernel", result); err != nil {
		return err
	}
	l.kernelLaunches.Add(1)
	return nil
}

// DeviceProfile: returns the two device fields a torch.cuda distribution kernel
// needs to size its Philox launch -- SM count and max resident threads per SM.
// These bound the grid so the counter-advance geometry matches PyTorch's.
func (l *Library) DeviceProfile(device Device) (smCount, maxThreadsPerSM int, err error) {
	smCount, err = l.deviceAttribute(device, attributeMultiprocessorCount)
	if err != nil {
		return 0, 0, err
	}
	maxThreadsPerSM, err = l.deviceAttribute(device, attributeMaxThreadsPerMultiproc)
	if err != nil {
		return 0, 0, err
	}
	return smCount, maxThreadsPerSM, nil
}

func (l *Library) deviceAttribute(device Device, attribute int) (int, error) {
	var value int32
	var pinned runtime.Pinner
	pinned.Pin(&value)
	defer pinned.Unpin()
	result, _, _ := l.cuDeviceGetAttribute.Call(
		uintptr(unsafe.Pointer(&value)),
		uintptr(attribute),
		uintptr(device),
	)
	if err := l.result("cuDeviceGetAttribute", result); err != nil {
		return 0, err
	}
	return int(value), nil
}

func (l *Library) result(operation string, value uintptr) error {
	code := int32(value)
	if code == 0 {
		return nil
	}

	err := &ResultError{Operation: operation, Code: code}
	var namePtr uintptr
	var pinned runtime.Pinner
	pinned.Pin(&namePtr)
	defer pinned.Unpin()
	if result, _, _ := l.cuGetErrorName.Call(uintptr(code), uintptr(unsafe.Pointer(&namePtr))); int32(result) == 0 {
		err.Name = readCString(namePtr, cudaDeviceNameBytes)
	}
	var messagePtr uintptr
	pinned.Pin(&messagePtr)
	if result, _, _ := l.cuGetErrorString.Call(uintptr(code), uintptr(unsafe.Pointer(&messagePtr))); int32(result) == 0 {
		err.Message = readCString(messagePtr, 1024)
	}
	return err
}

// readCString: bounded copy from driver-owned C string
func readCString(pointer uintptr, limit int) string {
	if pointer == 0 || limit <= 0 {
		return ""
	}
	length, _, _ := lstrlenA.Call(pointer)
	length = min(length, uintptr(limit))
	if length == 0 {
		return ""
	}
	data := make([]byte, length)
	var pinned runtime.Pinner
	pinned.Pin(&data[0])
	defer pinned.Unpin()
	_, _, _ = rtlMoveMemory.Call(
		uintptr(unsafe.Pointer(&data[0])),
		pointer,
		length,
	)
	runtime.KeepAlive(data)
	return string(data)
}
