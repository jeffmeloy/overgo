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
	moduleOwned bool
	allocations []driver.DevicePtr
}

type cudaDownload struct {
	data    []float32
	pointer driver.DevicePtr
}

type cudaBLAS struct {
	*cudaScope
	library                             *cublas.Library
	handle                              cublas.Handle
	bf16Left, bf16Right                 driver.DevicePtr
	bf16LeftCapacity, bf16RightCapacity int
	bf16Operands                        bool
	frozenTail                          *frozenCausalTailBuffers
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
		session, err := openCUDABLAS(scope)
		if err != nil {
			return err
		}
		defer func() { result = errors.Join(result, session.closeBLAS()) }()
		return run(session)
	})
}

func openCUDABLAS(scope *cudaScope) (*cudaBLAS, error) {
	blas, err := cublas.Open()
	if err != nil {
		return nil, err
	}
	handle, err := blas.Create()
	if err != nil {
		_ = blas.Close()
		return nil, err
	}
	if err := blas.SetStream(handle, scope.state.Stream); err != nil {
		_ = blas.Destroy(handle)
		_ = blas.Close()
		return nil, err
	}
	return &cudaBLAS{cudaScope: scope, library: blas, handle: handle}, nil
}

func (s *cudaBLAS) close() error {
	if s == nil {
		return nil
	}
	return errors.Join(s.cudaScope.close(), s.closeBLAS())
}

func (s *cudaBLAS) closeBLAS() error {
	s.closeBF16()
	return errors.Join(s.library.Destroy(s.handle), s.library.Close())
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
	if s.bf16Operands {
		return s.gemmBF16(transposeLeft, transposeRight, rows, inner, columns, left, right, output)
	}
	return s.library.RowMajorGEMMExF32(
		s.handle, transposeLeft, transposeRight,
		int32(rows), int32(inner), int32(columns), left, right, output,
	)
}

func (s *cudaBLAS) gemmBF16(transposeLeft, transposeRight bool, rows, inner, columns int, left, right, output driver.DevicePtr) error {
	leftCount, valid := checkedProduct(rows, inner)
	if !valid {
		return errors.New("CUDA GEMM left operand size overflow")
	}
	rightCount, valid := checkedProduct(inner, columns)
	if !valid {
		return errors.New("CUDA GEMM right operand size overflow")
	}
	left16, err := s.ensureBF16(&s.bf16Left, &s.bf16LeftCapacity, leftCount)
	if err != nil {
		return err
	}
	right16, err := s.ensureBF16(&s.bf16Right, &s.bf16RightCapacity, rightCount)
	if err != nil {
		return err
	}
	convert, err := s.function("f32_to_bf16")
	if err != nil {
		return err
	}
	for _, item := range []struct {
		source, target driver.DevicePtr
		count          int
	}{{left, left16, leftCount}, {right, right16, rightCount}} {
		countU := uint32(item.count)
		if err := s.launch1D(convert, countU, unsafe.Pointer(&item.source), unsafe.Pointer(&item.target), unsafe.Pointer(&countU)); err != nil {
			return err
		}
	}
	return s.library.RowMajorGEMMExBF16(
		s.handle, transposeLeft, transposeRight,
		int32(rows), int32(inner), int32(columns), left16, right16, output,
	)
}

func checkedProduct(left, right int) (int, bool) {
	product, valid := checked.Mul64(uint64(left), uint64(right))
	if !valid {
		return 0, false
	}
	return checked.Int(product)
}

func (s *cudaBLAS) ensureBF16(pointer *driver.DevicePtr, capacity *int, count int) (driver.DevicePtr, error) {
	if count <= *capacity {
		return *pointer, nil
	}
	bytes, valid := checked.Bytes(uint64(count), 2)
	if !valid {
		return 0, errors.New("CUDA BF16 staging size overflow")
	}
	next, err := s.state.Driver.MemAlloc(bytes)
	if err != nil {
		return 0, err
	}
	if *pointer != 0 {
		if err := s.state.Driver.MemFree(*pointer); err != nil {
			s.state.Driver.MemFree(next)
			return 0, err
		}
	}
	*pointer, *capacity = next, count
	return next, nil
}

func (s *cudaBLAS) closeBF16() {
	for _, pointer := range []driver.DevicePtr{s.bf16Left, s.bf16Right} {
		if pointer != 0 {
			_ = s.state.Driver.MemFree(pointer)
		}
	}
}

func (s *cudaBLAS) gemmStrided(
	transposeLeft, transposeRight bool,
	rows, inner, columns, batch int,
	left driver.DevicePtr, leftStride int,
	right driver.DevicePtr, rightStride int,
	output driver.DevicePtr, outputStride int,
) error {
	for _, value := range []int{rows, inner, columns, batch} {
		if value <= 0 || value > math.MaxInt32 {
			return errors.New("CUDA batched GEMM dimensions exceed the ABI")
		}
	}
	return s.library.RowMajorGEMMStridedBatchedF32(
		s.handle, transposeLeft, transposeRight,
		int32(rows), int32(inner), int32(columns),
		left, int64(leftStride), right, int64(rightStride),
		output, int64(outputStride), int32(batch),
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
		s.moduleOwned = true
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
	if s.moduleOwned && s.module != 0 {
		result = errors.Join(result, s.state.Driver.ModuleUnload(s.module))
	}
	return result
}
