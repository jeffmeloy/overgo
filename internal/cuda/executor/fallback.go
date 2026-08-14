package executor

import (
	"errors"
	"fmt"
	"math"
	"reflect"
	"runtime"
	"unsafe"

	"overgo/internal/cuda/device"
	"overgo/internal/cuda/driver"
	"overgo/internal/tensor"
	"overgo/internal/tensor/dtype"
	"overgo/internal/tensor/reference"
)

// launchReferenceNode: synchronized bounded-host correctness bridge.
func launchReferenceNode(
	state *device.State,
	node *tensor.Tensor,
	pointers devicePointerTable,
) error {
	if state.Trace != nil {
		return errors.New("reference bridge cannot join a traced graph")
	}
	if err := state.Driver.StreamSynchronize(state.Stream); err != nil {
		return err
	}
	inputs := make([]reference.Value, len(node.Inputs))
	for index, input := range node.Inputs {
		if input.Type != dtype.F32 {
			return fmt.Errorf("reference bridge input %d has type %s", index, input.Type)
		}
		elements, err := input.Shape.Elements()
		if err != nil || elements > uint64(math.MaxInt) {
			return errors.New("reference bridge input is too large")
		}
		data := make([]float32, int(elements))
		if err := state.Driver.MemcpyDtoH(driver.Bytes(data), pointers.get(input)); err != nil {
			return err
		}
		inputs[index] = reference.Value{Shape: input.Shape, Data: data}
	}
	value, err := reference.ExecuteOperation(node, inputs)
	if err != nil {
		return err
	}
	return state.Driver.MemcpyHtoD(pointers.get(node), driver.Bytes(value.Data))
}

func nativeQuantizedType(value dtype.Type) bool {
	_, ok := quantKernels[value]
	return ok
}

func kernelBool(value bool) uint32 {
	if value {
		return 1
	}
	return 0
}

// launch1DABI: marshal typed scalar addresses; retain through launch.
func launch1DABI(
	state *device.State,
	function boundKernel,
	count uint32,
	arguments ...any,
) error {
	if count == 0 {
		return nil
	}
	const threads = uint32(256)
	blocks := (count + threads - 1) / threads
	return launchGridABI(
		state, function,
		driver.Dim3{X: blocks, Y: 1, Z: 1},
		driver.Dim3{X: threads, Y: 1, Z: 1},
		arguments...,
	)
}

func launchGridABI(
	state *device.State,
	function boundKernel,
	grid, block driver.Dim3,
	arguments ...any,
) error {
	return launchGridSharedABI(state, function, grid, block, 0, arguments...)
}

func launchGridSharedABI(
	state *device.State,
	function boundKernel,
	grid, block driver.Dim3,
	sharedBytes uint32,
	arguments ...any,
) error {
	if err := validateKernelArgumentCount(function, len(arguments)); err != nil {
		return err
	}
	pointers := make([]unsafe.Pointer, len(arguments))
	for index, argument := range arguments {
		value := reflect.ValueOf(argument)
		if value.Kind() != reflect.Pointer || value.IsNil() {
			panic("CUDA ABI argument must be a non-nil pointer")
		}
		pointers[index] = value.UnsafePointer()
	}
	if trace := state.Trace; trace != nil {
		// grid/block Y and Z are CUDA-bounded to 65535; X keeps 32 bits
		trace.Words(
			uint64(function.id),
			uint64(grid.X)|uint64(grid.Y)<<32|uint64(grid.Z)<<48,
			uint64(block.X)|uint64(block.Y)<<32|uint64(block.Z)<<48,
			uint64(sharedBytes),
		)
		for index, pointer := range pointers {
			var size uintptr
			switch arguments[index].(type) {
			case *uint32, *float32:
				size = 4
			case *driver.DevicePtr, *uint64:
				size = 8
			default:
				size = reflect.TypeOf(arguments[index]).Elem().Size()
			}
			trace.Bytes(unsafe.Slice((*byte)(pointer), size))
		}
		if trace.Probe {
			runtime.KeepAlive(arguments)
			return nil
		}
	}
	err := state.Driver.LaunchKernel(function.function, grid, block, sharedBytes, state.Stream, pointers)
	runtime.KeepAlive(arguments)
	return err
}

// traceExternalCall: records a non-kernel stream operation (cuBLAS) so trace
// equality still implies identical device work; true means probe-skip.
func traceExternalCall(state *device.State, tag uint64, words ...uint64) bool {
	trace := state.Trace
	if trace == nil {
		return false
	}
	trace.Words(tag)
	trace.Words(words...)
	return trace.Probe
}

const (
	traceTagSGEMM        = uint64(1) << 32
	traceTagGEMMEx       = uint64(2) << 32
	traceTagSGEMMStrided = uint64(3) << 32
)

func validateKernelArgumentCount(function boundKernel, count int) error {
	if count == int(function.argumentCount) {
		return nil
	}
	return fmt.Errorf(
		"CUDA kernel %q received %d ABI arguments; want %d",
		kernelFunctionNames[function.id], count, function.argumentCount,
	)
}

func elementCount32(shape tensor.Shape) (uint32, error) {
	elements, err := shape.Elements()
	if err != nil {
		return 0, err
	}
	return uint32Checked(elements, "tensor element count")
}

func rowDimensions32(shape tensor.Shape) (uint32, uint32, error) {
	width, err := uint32Checked(shape.Dims[0], "tensor row width")
	if err != nil {
		return 0, 0, err
	}
	elements, err := shape.Elements()
	if err != nil {
		return 0, 0, err
	}
	return width, uint32(elements / uint64(width)), nil
}

func uint32Checked(value uint64, name string) (uint32, error) {
	if value > math.MaxUint32 {
		return 0, fmt.Errorf("%s %d exceeds uint32", name, value)
	}
	return uint32(value), nil
}

func shapeDimensions32(shape tensor.Shape) ([tensor.MaxDimensions]uint32, error) {
	var result [tensor.MaxDimensions]uint32
	for axis := range tensor.MaxDimensions {
		value, err := uint32Checked(shape.Dims[axis], fmt.Sprintf("tensor dimension %d", axis))
		if err != nil {
			return result, err
		}
		result[axis] = value
	}
	return result, nil
}
