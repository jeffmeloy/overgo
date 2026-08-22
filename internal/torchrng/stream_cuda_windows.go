//go:build windows

package torchrng

import (
	"errors"
	"fmt"
	"unsafe"

	"overgo/internal/cuda/device"
	"overgo/internal/cuda/driver"
	"overgo/internal/cuda/kernel"
)

const kernelName = "torch_cuda_randn"

// ensureLoaded lazily loads the torch_cuda_randn PTX module, resolves the
// kernel, and reads the device launch profile. The handles are cached on the
// Stream and are valid only within the worker/context that loaded them.
func (s *Stream) ensureLoaded(state *device.State) error {
	if s.loaded {
		return nil
	}
	if state == nil || state.Driver == nil {
		return errors.New("torchrng: nil device state")
	}
	if err := kernel.ValidateAssets(); err != nil {
		return err
	}
	module, err := state.Driver.ModuleLoadData(kernel.TorchCUDARandnPTX)
	if err != nil {
		return err
	}
	function, err := state.Driver.ModuleFunction(module, kernelName)
	if err != nil {
		_ = state.Driver.ModuleUnload(module)
		return err
	}
	smCount, threadsPerSM, err := state.Driver.DeviceProfile(state.Device)
	if err != nil {
		_ = state.Driver.ModuleUnload(module)
		return err
	}
	s.module = uintptr(module)
	s.function = uintptr(function)
	s.smCount = smCount
	s.threadsPerSM = threadsPerSM
	s.loaded = true
	return nil
}

// FillDevice writes `elements` seeded normals into device memory at dst and
// advances the stream's Philox counter. It does not synchronize: the launch is
// enqueued on state.Stream and the caller sequences it like any other kernel.
func (s *Stream) FillDevice(state *device.State, dst driver.DevicePtr, elements int) error {
	if s == nil {
		return errors.New("torchrng: nil stream")
	}
	if dst == 0 || elements <= 0 {
		return fmt.Errorf("torchrng: bad device fill dst=%#x elements=%d", uint64(dst), elements)
	}
	if err := s.ensureLoaded(state); err != nil {
		return err
	}
	grid, drawOffset, err := s.reserve(int64(elements), s.smCount, s.threadsPerSM)
	if err != nil {
		return err
	}
	// Kernel signature: (float* out, long long numel, unsigned long long seed,
	// unsigned long long offset). All four PTX params are .u64; each argument is
	// a pointer to its parameter value.
	destination := dst
	numel := int64(elements)
	seed := s.seed
	offset := drawOffset
	return state.Driver.LaunchKernel(
		driver.Function(s.function),
		driver.Dim3{X: uint32(grid), Y: 1, Z: 1},
		driver.Dim3{X: uint32(Block), Y: 1, Z: 1},
		0,
		state.Stream,
		[]unsafe.Pointer{
			unsafe.Pointer(&destination),
			unsafe.Pointer(&numel),
			unsafe.Pointer(&seed),
			unsafe.Pointer(&offset),
		},
	)
}

// FillHost fills dst with seeded normals via a device scratch buffer and copies
// the result back, advancing the stream's Philox counter. It synchronizes the
// stream before the device-to-host copy.
func (s *Stream) FillHost(state *device.State, dst []float32) error {
	if s == nil {
		return errors.New("torchrng: nil stream")
	}
	if len(dst) == 0 {
		return errors.New("torchrng: empty host destination")
	}
	if state == nil || state.Driver == nil {
		return errors.New("torchrng: nil device state")
	}
	scratch, err := state.Driver.MemAlloc(uint64(len(dst)) * 4)
	if err != nil {
		return err
	}
	defer func() { _ = state.Driver.MemFree(scratch) }()
	if err := s.FillDevice(state, scratch, len(dst)); err != nil {
		return err
	}
	if err := state.Driver.StreamSynchronize(state.Stream); err != nil {
		return err
	}
	return state.Driver.MemcpyDtoH(driver.Bytes(dst), scratch)
}

// Close unloads the cached CUDA module. Safe to call on an unused Stream.
func (s *Stream) Close(state *device.State) error {
	if s == nil || !s.loaded {
		return nil
	}
	s.loaded = false
	function := s.function
	module := s.module
	s.function = 0
	s.module = 0
	_ = function
	if state == nil || state.Driver == nil {
		return nil
	}
	return state.Driver.ModuleUnload(driver.Module(module))
}
