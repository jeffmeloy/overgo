//go:build !windows

package driver

import (
	"errors"
	"unsafe"
)

// Library: unavailable on initial non-Windows target
type Library struct{}

func Open() (*Library, error) {
	return nil, errors.New("CUDA driver loading is currently supported only on Windows")
}

func (l *Library) Close() error {
	return nil
}

func (l *Library) Init() error {
	return errors.New("CUDA driver loading is currently supported only on Windows")
}

func (l *Library) DriverVersion() (Version, error) {
	return 0, errors.New("CUDA driver loading is currently supported only on Windows")
}

func (l *Library) DeviceCount() (int, error) {
	return 0, errors.New("CUDA driver loading is currently supported only on Windows")
}

func (l *Library) Device(ordinal int) (Device, error) {
	return 0, errors.New("CUDA driver loading is currently supported only on Windows")
}

func (l *Library) DeviceInfo(ordinal int) (DeviceInfo, error) {
	return DeviceInfo{}, errors.New("CUDA driver loading is currently supported only on Windows")
}

// DeviceUUID is unavailable where no CUDA driver loads.
func (l *Library) DeviceUUID(Device) (string, error) {
	return "", errors.New("CUDA driver loading is currently supported only on Windows")
}

// CardReadings reads nothing where no card's management library is bound.
func CardReadings(string) []CardReading { return nil }

func (l *Library) DeviceProfile(device Device) (smCount, maxThreadsPerSM int, err error) {
	return 0, 0, errors.New("CUDA driver loading is currently supported only on Windows")
}

func (l *Library) ContextCreate(device Device, flags uint32) (Context, error) {
	return 0, errors.New("CUDA driver loading is currently supported only on Windows")
}

func (l *Library) ContextDestroy(context Context) error {
	return errors.New("CUDA driver loading is currently supported only on Windows")
}

func (l *Library) ContextSetCurrent(context Context) error {
	return errors.New("CUDA driver loading is currently supported only on Windows")
}

func (l *Library) StreamCreate(flags uint32) (Stream, error) {
	return 0, errors.New("CUDA driver loading is currently supported only on Windows")
}

func (l *Library) StreamDestroy(stream Stream) error {
	return errors.New("CUDA driver loading is currently supported only on Windows")
}

func (l *Library) StreamSynchronize(stream Stream) error {
	return errors.New("CUDA driver loading is currently supported only on Windows")
}

func (l *Library) MemAlloc(bytes uint64) (DevicePtr, error) {
	return 0, errors.New("CUDA driver loading is currently supported only on Windows")
}

func (l *Library) MemFree(pointer DevicePtr) error {
	return errors.New("CUDA driver loading is currently supported only on Windows")
}

func (l *Library) MemInfo() (free, total uint64, err error) {
	return 0, 0, errors.New("CUDA driver loading is currently supported only on Windows")
}

func (l *Library) ResetPeakBytes() {}

// ProcessPeakBytes is zero where no library allocates device memory.
func ProcessPeakBytes() uint64 { return 0 }

// ProcessCurrentBytes is zero where no library allocates device memory.
func ProcessCurrentBytes() uint64 { return 0 }

func (l *Library) MemoryStats() MemoryStats {
	return MemoryStats{}
}

func (l *Library) ExecutionStats() ExecutionStats {
	return ExecutionStats{}
}

func (l *Library) StreamBeginCapture(Stream) error        { return errors.New("CUDA unsupported") }
func (l *Library) StreamEndCapture(Stream) (Graph, error) { return 0, errors.New("CUDA unsupported") }
func (l *Library) GraphInstantiate(Graph) (GraphExec, error) {
	return 0, errors.New("CUDA unsupported")
}
func (l *Library) GraphExecUpdate(GraphExec, Graph) (bool, error) {
	return false, errors.New("CUDA unsupported")
}
func (l *Library) GraphLaunch(GraphExec, Stream) error { return errors.New("CUDA unsupported") }
func (l *Library) GraphDestroy(Graph) error            { return nil }
func (l *Library) GraphExecDestroy(GraphExec) error    { return nil }

func (l *Library) MemcpyHtoD(destination DevicePtr, source []byte) error {
	return errors.New("CUDA driver loading is currently supported only on Windows")
}

func (l *Library) MemcpyDtoH(destination []byte, source DevicePtr) error {
	return errors.New("CUDA driver loading is currently supported only on Windows")
}

func (l *Library) MemcpyDtoD(DevicePtr, DevicePtr, uint64) error {
	return errors.New("CUDA driver loading is currently supported only on Windows")
}

func (l *Library) MemsetD32Async(DevicePtr, uint32, uint64, Stream) error {
	return errors.New("CUDA driver loading is currently supported only on Windows")
}

// MemHostAlloc reports that pinned, device-mapped host memory needs the Windows driver.
func (l *Library) MemHostAlloc(uint64) ([]byte, DevicePtr, error) {
	return nil, 0, errors.New("CUDA driver loading is currently supported only on Windows")
}

// MemFreeHost reports that pinned host memory needs the Windows driver.
func (l *Library) MemFreeHost([]byte) error {
	return errors.New("CUDA driver loading is currently supported only on Windows")
}

func (l *Library) ModuleLoadData(image []byte) (Module, error) {
	return 0, errors.New("CUDA driver loading is currently supported only on Windows")
}

func (l *Library) ModuleUnload(module Module) error {
	return errors.New("CUDA driver loading is currently supported only on Windows")
}

func (l *Library) ModuleFunction(module Module, name string) (Function, error) {
	return 0, errors.New("CUDA driver loading is currently supported only on Windows")
}

func (l *Library) FuncSetMaxDynamicShared(function Function, bytes uint32) error {
	return errors.New("CUDA driver loading is currently supported only on Windows")
}

func (l *Library) LaunchKernel(
	function Function,
	grid Dim3,
	block Dim3,
	sharedMemoryBytes uint32,
	stream Stream,
	arguments []unsafe.Pointer,
) error {
	return errors.New("CUDA driver loading is currently supported only on Windows")
}
