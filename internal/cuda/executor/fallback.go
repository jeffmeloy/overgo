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
	pointers launchPointerFrame,
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
		if err := state.Driver.MemcpyDtoH(driver.Bytes(data), pointers.input(index)); err != nil {
			return err
		}
		inputs[index] = reference.Value{Shape: input.Shape, Data: data}
	}
	value, err := reference.ExecuteOperation(node, inputs)
	if err != nil {
		return err
	}
	return state.Driver.MemcpyHtoD(pointers.output(), driver.Bytes(value.Data))
}

func nativeQuantizedType(value dtype.Type) bool {
	_, ok := quantKernel(value)
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
	blocks := (count-1)/threads + 1
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
	traceTagFP8Matmul    = uint64(4) << 32
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

// The largest values the words of a launch hold, which checkLaunchWords holds each launch's shape to when its graph compiles.
const (
	// launchWordLimit is the largest dimension or element count a kernel takes.
	//   - source = the kernels' arguments, which pass each dimension and count as a 32-bit unsigned word
	//   - scope = every launch checkLaunchWords reaches
	//   - reconsider when a kernel takes a 64-bit count
	launchWordLimit = math.MaxUint32

	// cublasDimensionLimit is the largest dimension of an operation of cublasLaunches.
	//   - source = the cuBLAS GEMM calls, which take each dimension and leading dimension as a signed 32-bit int
	//   - scope = the operations of cublasLaunches
	//   - reconsider when those calls take 64-bit dimensions
	cublasDimensionLimit = math.MaxInt32
)

// cublasLaunches are the operations whose launch may pass a dimension to cuBLAS (the F32 matrix products, conv_2d's im2col product and attention's batched products).
var cublasLaunches = map[tensor.Op]bool{
	tensor.OpMulMat: true, tensor.OpGroupedMulMat: true, tensor.OpConv2D: true, tensor.OpAttention: true,
}

// uncountedLaunches are the operations whose launch narrows no element count, so checkLaunchWords bounds their dimensions alone.
//   - their launches count rows, heads or sequences and bound those products themselves
//   - a count rule would refuse an output of more than launchWordLimit elements that these launches hold (a mul_mat over a vocabulary)
var uncountedLaunches = map[tensor.Op]bool{
	tensor.OpMulMat: true, tensor.OpGroupedMulMat: true, tensor.OpSSMScan: true, tensor.OpGatedDeltaNet: true,
	tensor.OpGatedLinearAttention: true, tensor.OpWKV6: true, tensor.OpWKV7: true, tensor.OpSumRows: true,
}

// inputCountedLaunches are the operations whose launch counts the rows of input 0 and not the elements of its output.
var inputCountedLaunches = map[tensor.Op]bool{tensor.OpFWHT: true, tensor.OpTopK: true, tensor.OpTopKPartials: true}

// checkLaunchWords refuses a launch whose shape the 32-bit words of its kernel cannot hold, naming the operation.
//   - each operand's shape counts its elements without overflow, and each of its dimensions <= launchWordLimit, or <= cublasDimensionLimit for an operation of cublasLaunches
//   - the counted operand's elements <= launchWordLimit, the counted operand = the output, or input 0 for an operation of inputCountedLaunches, and none for an operation of uncountedLaunches
//   - an input a launch counts beside these (concat's right input, cache_append's inputs) holds no more elements than the output, because the builder joins it into the output
//   - the host bridge passes no kernel word and the mixture-of-experts launch keeps its own checks, so both pass
//   - so a launch narrows its dimensions and counts with conversions that cannot fail
func checkLaunchWords(node *tensor.Tensor) error {
	if descriptor, _ := tensor.DescribeOperation(node.Op); descriptor.CUDA == tensor.CUDAProgramReference || descriptor.CUDA == tensor.CUDAProgramMoE {
		return nil
	}
	limit := uint64(launchWordLimit)
	if cublasLaunches[node.Op] {
		limit = cublasDimensionLimit
	}
	counted := node.Shape
	if inputCountedLaunches[node.Op] {
		counted = node.Inputs[0].Shape
	}
	operands := append([]*tensor.Tensor{node}, node.Inputs...)
	for index, operand := range operands {
		name := "output"
		if index > 0 {
			name = fmt.Sprintf("input %d", index-1)
		}
		if _, err := operand.Shape.Elements(); err != nil {
			return fmt.Errorf("CUDA %s (tensor %d) %s: %w", node.Op, node.ID, name, err)
		}
		for axis, dimension := range operand.Shape.Dims {
			if dimension > limit {
				return fmt.Errorf("CUDA %s (tensor %d) %s dimension %d is %d, past the %d a kernel word holds", node.Op, node.ID, name, axis, dimension, limit)
			}
		}
	}
	if elements, _ := counted.Elements(); !uncountedLaunches[node.Op] && elements > launchWordLimit {
		return fmt.Errorf("CUDA %s (tensor %d) counts %d elements, past the %d a kernel word holds", node.Op, node.ID, elements, uint64(launchWordLimit))
	}
	return nil
}

// shapeCount32 narrows a launched operand's element count to a kernel word.
//   - checkLaunchWords bounds the count when the graph compiles, so the narrowing cannot fail
func shapeCount32(shape tensor.Shape) uint32 {
	elements, _ := shape.Elements()
	return uint32(elements)
}

// rowWords narrows a launched operand's row width and row count to kernel words.
//   - checkLaunchWords bounds both when the graph compiles, so neither narrowing can fail
func rowWords(shape tensor.Shape) (uint32, uint32) {
	width := uint32(shape.Dims[0])
	return width, shapeCount32(shape) / width
}

// shapeDimensions32 narrows a launched operand's dimensions to kernel words.
//   - checkLaunchWords bounds each when the graph compiles, so no narrowing can fail
func shapeDimensions32(shape tensor.Shape) [tensor.MaxDimensions]uint32 {
	var result [tensor.MaxDimensions]uint32
	for axis, dimension := range shape.Dims {
		result[axis] = uint32(dimension)
	}
	return result
}
