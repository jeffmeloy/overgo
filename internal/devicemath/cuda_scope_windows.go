//go:build windows

package devicemath

import (
	"context"
	"errors"
	"math"
	"unsafe"

	"overgo/internal/checked"
	"overgo/internal/cuda/cublas"
	"overgo/internal/cuda/device"
	"overgo/internal/cuda/driver"
	"overgo/internal/cuda/kernel"
)

const (
	f32Bytes           = uint64(4)
	deviceBlockThreads = uint32(256)
)

type cudaScope struct {
	state       *device.State
	module      driver.Module
	allocations []driver.DevicePtr
}

type cudaDownload struct {
	data    []float32
	pointer driver.DevicePtr
}

type cudaBLAS struct {
	*cudaScope
	library *cublas.Library
	handle  cublas.Handle
}

func withCUDA(worker *device.Worker, run func(*cudaScope) error) error {
	return worker.Do(context.Background(), func(state *device.State) (result error) {
		scope := &cudaScope{state: state}
		defer func() { result = errors.Join(result, scope.close()) }()
		return run(scope)
	})
}

func withCUDABLAS(
	worker *device.Worker,
	run func(*cudaBLAS) error,
) error {
	return withCUDA(worker, func(scope *cudaScope) (result error) {
		blas, err := cublas.Open()
		if err != nil {
			return err
		}
		defer func() { result = errors.Join(result, blas.Close()) }()
		handle, err := blas.Create()
		if err != nil {
			return err
		}
		defer func() { result = errors.Join(result, blas.Destroy(handle)) }()
		if err := blas.SetStream(handle, scope.state.Stream); err != nil {
			return err
		}
		return run(&cudaBLAS{cudaScope: scope, library: blas, handle: handle})
	})
}

func (s *cudaBLAS) gemm(
	transposeLeft, transposeRight bool,
	rows, inner, columns int,
	left, right, output driver.DevicePtr,
) error {
	if rows <= 0 || inner <= 0 || columns <= 0 ||
		rows > math.MaxInt32 || inner > math.MaxInt32 || columns > math.MaxInt32 {
		return errors.New("CUDA GEMM dimensions exceed the ABI")
	}
	return s.library.RowMajorGEMMExF32(
		s.handle, transposeLeft, transposeRight,
		int32(rows), int32(inner), int32(columns), left, right, output,
	)
}

func (s *cudaScope) alloc(count int) (driver.DevicePtr, error) {
	if count <= 0 {
		return 0, errors.New("CUDA f32 allocation count must be positive")
	}
	bytes, ok := checked.Bytes(uint64(count), f32Bytes)
	if !ok {
		return 0, errors.New("CUDA f32 allocation size overflow")
	}
	pointer, err := s.state.Driver.MemAlloc(bytes)
	if err == nil {
		s.allocations = append(s.allocations, pointer)
	}
	return pointer, err
}

func (s *cudaScope) upload(data []float32) (driver.DevicePtr, error) {
	pointer, err := s.alloc(len(data))
	if err != nil {
		return 0, err
	}
	return pointer, s.state.Driver.MemcpyHtoD(pointer, driver.Bytes(data))
}

func (s *cudaScope) download(data []float32, pointer driver.DevicePtr) error {
	return s.state.Driver.MemcpyDtoH(driver.Bytes(data), pointer)
}

func (s *cudaScope) function(name string) (driver.Function, error) {
	if s.module == 0 {
		module, err := s.state.Driver.ModuleLoadData(kernel.OpsF32PTX)
		if err != nil {
			return 0, err
		}
		s.module = module
	}
	return s.state.Driver.ModuleFunction(s.module, name)
}

func (s *cudaScope) launch1D(function driver.Function, count uint32, arguments ...unsafe.Pointer) error {
	if count == 0 {
		return nil
	}
	blocks := (count + deviceBlockThreads - 1) / deviceBlockThreads
	return s.state.Driver.LaunchKernel(
		function,
		driver.Dim3{X: blocks, Y: 1, Z: 1},
		driver.Dim3{X: deviceBlockThreads, Y: 1, Z: 1},
		0, s.state.Stream, arguments,
	)
}

func (s *cudaScope) launchVector3(
	function driver.Function,
	first, second, output driver.DevicePtr,
	count int,
) error {
	if count <= 0 || uint64(count) > math.MaxUint32 {
		return errors.New("CUDA vector launch count exceeds the ABI")
	}
	countU := uint32(count)
	return s.launch1D(function, countU,
		unsafe.Pointer(&first), unsafe.Pointer(&second), unsafe.Pointer(&output), unsafe.Pointer(&countU))
}

func (s *cudaScope) finish(downloads ...cudaDownload) error {
	if err := s.state.Driver.StreamSynchronize(s.state.Stream); err != nil {
		return err
	}
	for _, download := range downloads {
		if err := s.download(download.data, download.pointer); err != nil {
			return err
		}
	}
	return nil
}

func (s *cudaScope) close() error {
	var result error
	for index := len(s.allocations) - 1; index >= 0; index-- {
		result = errors.Join(result, s.state.Driver.MemFree(s.allocations[index]))
	}
	if s.module != 0 {
		result = errors.Join(result, s.state.Driver.ModuleUnload(s.module))
	}
	return result
}
