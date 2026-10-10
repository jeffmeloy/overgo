//go:build windows

package cudnn

import (
	"errors"
	"runtime"
	"syscall"
	"unsafe"

	"overgo/internal/cuda/cublas"
	"overgo/internal/cuda/driver"
)

// libraryName is the cuDNN library a process loads; the library loads its own
// parts from beside it.
const libraryName = "cudnn64_9.dll"

// Library is the dynamically loaded cuDNN library and one context of it.
type Library struct {
	dll    *syscall.DLL
	handle uintptr

	create             *syscall.Proc
	destroy            *syscall.Proc
	setStream          *syscall.Proc
	createTensor       *syscall.Proc
	setTensor          *syscall.Proc
	destroyTensor      *syscall.Proc
	createFilter       *syscall.Proc
	setFilter          *syscall.Proc
	destroyFilter      *syscall.Proc
	createConvolution  *syscall.Proc
	setConvolution     *syscall.Proc
	setMathType        *syscall.Proc
	destroyConvolution *syscall.Proc
	workspaceSize      *syscall.Proc
	forward            *syscall.Proc
}

// Open loads cuDNN from the search path or CUDA_PATH and makes a context in
// the calling thread's device context, which computes on stream. On a machine
// that has no cuDNN it returns ErrAbsent with the loader's error.
func Open(stream driver.Stream) (lib *Library, err error) {
	lib = &Library{}
	dll, err := cublas.Load(libraryName, []cublas.Binding{
		{Name: "cudnnCreate", Proc: &lib.create},
		{Name: "cudnnDestroy", Proc: &lib.destroy},
		{Name: "cudnnSetStream", Proc: &lib.setStream},
		{Name: "cudnnCreateTensorDescriptor", Proc: &lib.createTensor},
		{Name: "cudnnSetTensorNdDescriptorEx", Proc: &lib.setTensor},
		{Name: "cudnnDestroyTensorDescriptor", Proc: &lib.destroyTensor},
		{Name: "cudnnCreateFilterDescriptor", Proc: &lib.createFilter},
		{Name: "cudnnSetFilterNdDescriptor", Proc: &lib.setFilter},
		{Name: "cudnnDestroyFilterDescriptor", Proc: &lib.destroyFilter},
		{Name: "cudnnCreateConvolutionDescriptor", Proc: &lib.createConvolution},
		{Name: "cudnnSetConvolutionNdDescriptor", Proc: &lib.setConvolution},
		{Name: "cudnnSetConvolutionMathType", Proc: &lib.setMathType},
		{Name: "cudnnDestroyConvolutionDescriptor", Proc: &lib.destroyConvolution},
		{Name: "cudnnGetConvolutionForwardWorkspaceSize", Proc: &lib.workspaceSize},
		{Name: "cudnnConvolutionForward", Proc: &lib.forward},
	})
	if err != nil {
		return nil, errors.Join(ErrAbsent, err)
	}
	lib.dll = dll
	var pinned runtime.Pinner
	pinned.Pin(lib)
	defer pinned.Unpin()
	status, _, _ := lib.create.Call(uintptr(unsafe.Pointer(&lib.handle)))
	if err = result("cudnnCreate", status); err == nil {
		status, _, _ = lib.setStream.Call(lib.handle, uintptr(stream))
		err = result("cudnnSetStream", status)
	}
	if err != nil {
		return nil, errors.Join(err, lib.Close())
	}
	return lib, nil
}

// Close destroys the context and releases the library.
func (l *Library) Close() error {
	if l == nil || l.dll == nil {
		return nil
	}
	var failed error
	if l.handle != 0 {
		status, _, _ := l.destroy.Call(l.handle)
		failed, l.handle = result("cudnnDestroy", status), 0
	}
	failed = errors.Join(failed, l.dll.Release())
	l.dll = nil
	return failed
}

// Convolution is a planned forward convolution: the descriptors of its input,
// filter and output, and the bytes of workspace its algorithm needs.
type Convolution struct {
	input, filter, output, convolution uintptr
	Workspace                          uint64
}

// Plan describes the convolution of shape to the library, its tensors and its
// filter of the type data with channels before extents and its sums kept in
// the type compute, and asks what workspace it needs. The library refuses a
// shape or a pair of types it does not convolve.
func (l *Library) Plan(shape Shape, data, compute DataType) (plan Convolution, err error) {
	if !shape.valid() {
		return Convolution{}, errors.New("cudnn: a convolution needs positive channels, extents, kernel and strides")
	}
	defer func() {
		if err != nil {
			err = errors.Join(err, l.Release(plan))
			plan = Convolution{}
		}
	}()
	frames, height, width := shape.output()
	tensor := func(operation string, channels, frames, height, width int) (uintptr, error) {
		var descriptor uintptr
		dims := [dimensions]int32{1, int32(channels), int32(frames), int32(height), int32(width)}
		var pinned runtime.Pinner
		pinned.Pin(&descriptor)
		pinned.Pin(&dims)
		defer pinned.Unpin()
		status, _, _ := l.createTensor.Call(uintptr(unsafe.Pointer(&descriptor)))
		if err := result("cudnnCreateTensorDescriptor", status); err != nil {
			return 0, err
		}
		status, _, _ = l.setTensor.Call(descriptor, channelsFirst, uintptr(data), dimensions, uintptr(unsafe.Pointer(&dims)))
		return descriptor, result(operation, status)
	}
	if plan.input, err = tensor("cudnnSetTensorNdDescriptorEx input", shape.In, shape.Frames, shape.Height, shape.Width); err != nil {
		return plan, err
	}
	if plan.output, err = tensor("cudnnSetTensorNdDescriptorEx output", shape.Out, frames, height, width); err != nil {
		return plan, err
	}
	filter := [dimensions]int32{int32(shape.Out), int32(shape.In), int32(shape.Kernel[overFrames]), int32(shape.Kernel[overRows]), int32(shape.Kernel[overColumns])}
	var pad, stride, dilation [extents]int32
	for at := range extents {
		pad[at], stride[at], dilation[at] = int32(shape.Pad[at]), int32(shape.Stride[at]), 1
	}
	var pinned runtime.Pinner
	pinned.Pin(&plan)
	pinned.Pin(&filter)
	pinned.Pin(&pad)
	pinned.Pin(&stride)
	pinned.Pin(&dilation)
	defer pinned.Unpin()
	status, _, _ := l.createFilter.Call(uintptr(unsafe.Pointer(&plan.filter)))
	if err = result("cudnnCreateFilterDescriptor", status); err != nil {
		return plan, err
	}
	status, _, _ = l.setFilter.Call(plan.filter, uintptr(data), channelsFirst, dimensions, uintptr(unsafe.Pointer(&filter)))
	if err = result("cudnnSetFilterNdDescriptor", status); err != nil {
		return plan, err
	}
	status, _, _ = l.createConvolution.Call(uintptr(unsafe.Pointer(&plan.convolution)))
	if err = result("cudnnCreateConvolutionDescriptor", status); err != nil {
		return plan, err
	}
	status, _, _ = l.setConvolution.Call(plan.convolution, extents,
		uintptr(unsafe.Pointer(&pad)), uintptr(unsafe.Pointer(&stride)), uintptr(unsafe.Pointer(&dilation)), crossCorrelation, uintptr(compute))
	if err = result("cudnnSetConvolutionNdDescriptor", status); err != nil {
		return plan, err
	}
	status, _, _ = l.setMathType.Call(plan.convolution, tensorMathAllowConversion)
	if err = result("cudnnSetConvolutionMathType", status); err != nil {
		return plan, err
	}
	status, _, _ = l.workspaceSize.Call(l.handle, plan.input, plan.filter, plan.convolution, plan.output, implicitPrecomputedGEMM, uintptr(unsafe.Pointer(&plan.Workspace)))
	return plan, result("cudnnGetConvolutionForwardWorkspaceSize", status)
}

// Release destroys a plan's descriptors.
func (l *Library) Release(plan Convolution) error {
	var failed error
	for _, descriptor := range []struct {
		handle  uintptr
		destroy *syscall.Proc
		name    string
	}{
		{plan.input, l.destroyTensor, "cudnnDestroyTensorDescriptor"}, {plan.output, l.destroyTensor, "cudnnDestroyTensorDescriptor"},
		{plan.filter, l.destroyFilter, "cudnnDestroyFilterDescriptor"}, {plan.convolution, l.destroyConvolution, "cudnnDestroyConvolutionDescriptor"},
	} {
		if descriptor.handle != 0 {
			status, _, _ := descriptor.destroy.Call(descriptor.handle)
			failed = errors.Join(failed, result(descriptor.name, status))
		}
	}
	return failed
}

// Forward writes the planned convolution of input by filter to output.
// workspace holds at least the plan's Workspace bytes when that is not zero.
func (l *Library) Forward(plan Convolution, input, filter, workspace, output driver.DevicePtr) error {
	one, zero := float32(1), float32(0)
	var pinned runtime.Pinner
	pinned.Pin(&one)
	pinned.Pin(&zero)
	defer pinned.Unpin()
	status, _, _ := l.forward.Call(l.handle, uintptr(unsafe.Pointer(&one)), plan.input, uintptr(input), plan.filter, uintptr(filter),
		plan.convolution, implicitPrecomputedGEMM, uintptr(workspace), uintptr(plan.Workspace), uintptr(unsafe.Pointer(&zero)), plan.output, uintptr(output))
	return result("cudnnConvolutionForward", status)
}
