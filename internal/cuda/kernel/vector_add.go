package kernel

import (
	_ "embed"
	"errors"
	"math"
	"runtime"
	"unsafe"

	"llamacpp2go/internal/cuda/driver"
)

// VectorAddPTX: contains Phase 1 CUDA smoke-test kernel
//
//go:embed vector_add.ptx
var VectorAddPTX []byte

//go:embed ops_f32.ptx
var OpsF32PTX []byte

// VectorAdd: executes Phase 1 smoke-test kernel; owns all
// allocations so complete module, launch, transfer, and cleanup path is
// exercised
func VectorAdd(
	lib *driver.Library,
	stream driver.Stream,
	inputA []float32,
	inputB []float32,
) ([]float32, error) {
	if len(inputA) != len(inputB) {
		return nil, errors.New("vector add: input lengths differ")
	}
	if len(inputA) == 0 {
		return []float32{}, nil
	}
	if uint64(len(inputA)) > math.MaxUint32 {
		return nil, errors.New("vector add: input is too large")
	}
	if err := ValidateAssets(); err != nil {
		return nil, err
	}

	bytes := uint64(len(inputA)) * uint64(unsafe.Sizeof(float32(0)))
	pointerA, err := lib.MemAlloc(bytes)
	if err != nil {
		return nil, err
	}
	defer lib.MemFree(pointerA)
	pointerB, err := lib.MemAlloc(bytes)
	if err != nil {
		return nil, err
	}
	defer lib.MemFree(pointerB)
	pointerOutput, err := lib.MemAlloc(bytes)
	if err != nil {
		return nil, err
	}
	defer lib.MemFree(pointerOutput)

	if err := lib.MemcpyHtoD(pointerA, driver.Bytes(inputA)); err != nil {
		return nil, err
	}
	if err := lib.MemcpyHtoD(pointerB, driver.Bytes(inputB)); err != nil {
		return nil, err
	}

	module, err := lib.ModuleLoadData(VectorAddPTX)
	if err != nil {
		return nil, err
	}
	defer lib.ModuleUnload(module)
	function, err := lib.ModuleFunction(module, "vector_add")
	if err != nil {
		return nil, err
	}

	count := uint32(len(inputA))
	arguments := []unsafe.Pointer{
		unsafe.Pointer(&pointerA),
		unsafe.Pointer(&pointerB),
		unsafe.Pointer(&pointerOutput),
		unsafe.Pointer(&count),
	}
	const threads = uint32(256)
	blocks := (count + threads - 1) / threads
	if err := lib.LaunchKernel(
		function,
		driver.Dim3{X: blocks, Y: 1, Z: 1},
		driver.Dim3{X: threads, Y: 1, Z: 1},
		0,
		stream,
		arguments,
	); err != nil {
		return nil, err
	}
	runtime.KeepAlive(pointerA)
	runtime.KeepAlive(pointerB)
	runtime.KeepAlive(pointerOutput)
	runtime.KeepAlive(count)

	if err := lib.StreamSynchronize(stream); err != nil {
		return nil, err
	}
	output := make([]float32, len(inputA))
	if err := lib.MemcpyDtoH(driver.Bytes(output), pointerOutput); err != nil {
		return nil, err
	}
	return output, nil
}
