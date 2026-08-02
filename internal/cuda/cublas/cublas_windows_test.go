//go:build windows

package cublas

import (
	"context"
	cudatest "llamacpp2go/internal/cuda/testutil"
	"math"
	"testing"
	"unsafe"

	"llamacpp2go/internal/cuda/device"
	"llamacpp2go/internal/cuda/driver"
)

func TestSGEMMIntegration(t *testing.T) {
	cudatest.Require(t)
	worker, err := device.New(0)
	if err != nil {
		t.Fatal(err)
	}
	defer worker.Close()

	left := []float32{
		1, 2, 3,
		4, 5, 6,
	}
	right := []float32{
		1, 0, 0,
		0, 1, 0,
	}
	output := make([]float32, 4)
	err = worker.Do(context.Background(), func(state *device.State) error {
		lib, err := Open()
		if err != nil {
			return err
		}
		defer lib.Close()
		handle, err := lib.Create()
		if err != nil {
			return err
		}
		defer lib.Destroy(handle)
		if err := lib.SetStream(handle, state.Stream); err != nil {
			return err
		}
		leftPtr, err := copyToDevice(state.Driver, left)
		if err != nil {
			return err
		}
		defer state.Driver.MemFree(leftPtr)
		rightPtr, err := copyToDevice(state.Driver, right)
		if err != nil {
			return err
		}
		defer state.Driver.MemFree(rightPtr)
		outputPtr, err := state.Driver.MemAlloc(uint64(len(output) * 4))
		if err != nil {
			return err
		}
		defer state.Driver.MemFree(outputPtr)

		if err := lib.SGEMM(
			handle,
			OperationTranspose,
			OperationNone,
			2,
			2,
			3,
			1,
			leftPtr,
			3,
			rightPtr,
			3,
			0,
			outputPtr,
			2,
		); err != nil {
			return err
		}
		if err := state.Driver.StreamSynchronize(state.Stream); err != nil {
			return err
		}
		return state.Driver.MemcpyDtoH(floatBytes(output), outputPtr)
	})
	if err != nil {
		t.Fatal(err)
	}
	want := []float32{1, 4, 2, 5}
	for index := range want {
		if math.Abs(float64(output[index]-want[index])) > 1e-6 {
			t.Fatalf("output[%d] = %v, want %v", index, output[index], want[index])
		}
	}
}

func copyToDevice(lib *driver.Library, values []float32) (driver.DevicePtr, error) {
	pointer, err := lib.MemAlloc(uint64(len(values) * 4))
	if err != nil {
		return 0, err
	}
	if err := lib.MemcpyHtoD(pointer, floatBytes(values)); err != nil {
		_ = lib.MemFree(pointer)
		return 0, err
	}
	return pointer, nil
}

func floatBytes(values []float32) []byte {
	return unsafe.Slice((*byte)(unsafe.Pointer(&values[0])), len(values)*4)
}
