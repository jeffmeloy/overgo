//go:build windows

package cublas

import (
	"context"
	"math"
	cudatest "overgo/internal/cuda/testutil"
	"testing"

	"overgo/internal/cuda/device"
	"overgo/internal/cuda/driver"
	"overgo/internal/tensor/dtype"
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
		return state.Driver.MemcpyDtoH(driver.Bytes(output), outputPtr)
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

// TestRowMajorGEMMF32Integration verifies the row-major convention on a
// non-square product (m != k != n), the case that exposes leading-dimension or
// transpose mistakes: A[2x3] · B[3x4] = C[2x4] against a host reference.
func TestRowMajorGEMMF32Integration(t *testing.T) {
	cudatest.Require(t)
	worker, err := device.New(0)
	if err != nil {
		t.Fatal(err)
	}
	defer worker.Close()

	const m, k, n = 2, 3, 4
	a := []float32{1, 2, 3, 4, 5, 6}
	b := []float32{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}
	want := make([]float32, m*n)
	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			var sum float32
			for p := 0; p < k; p++ {
				sum += a[i*k+p] * b[p*n+j]
			}
			want[i*n+j] = sum
		}
	}
	out := make([]float32, m*n)
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
		aPtr, err := copyToDevice(state.Driver, a)
		if err != nil {
			return err
		}
		defer state.Driver.MemFree(aPtr)
		bPtr, err := copyToDevice(state.Driver, b)
		if err != nil {
			return err
		}
		defer state.Driver.MemFree(bPtr)
		cPtr, err := state.Driver.MemAlloc(uint64(len(out) * 4))
		if err != nil {
			return err
		}
		defer state.Driver.MemFree(cPtr)
		if err := lib.RowMajorGEMMF32(handle, m, k, n, aPtr, bPtr, cPtr); err != nil {
			return err
		}
		if err := state.Driver.StreamSynchronize(state.Stream); err != nil {
			return err
		}
		return state.Driver.MemcpyDtoH(driver.Bytes(out), cPtr)
	})
	if err != nil {
		t.Fatal(err)
	}
	for index := range want {
		if math.Abs(float64(out[index]-want[index])) > 1e-4 {
			t.Fatalf("out[%d] = %v, want %v", index, out[index], want[index])
		}
	}
}

func TestGEMMExBF16Integration(t *testing.T) {
	cudatest.Require(t)
	worker, err := device.New(0)
	if err != nil {
		t.Fatal(err)
	}
	defer worker.Close()
	encode := func(values ...float32) []uint16 {
		result := make([]uint16, len(values))
		for index, value := range values {
			result[index] = dtype.Float32ToBF16(value)
		}
		return result
	}
	left := encode(1, 2, 3, 4, 5, 6)
	right := encode(1, 0, 0, 0, 1, 0)
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
		if err := lib.GEMMEx(
			handle, OperationTranspose, OperationNone, 2, 2, 3, 1,
			leftPtr, DataBF16, 3, rightPtr, DataBF16, 3, 0,
			outputPtr, DataF32, 2, ComputeF32, GemmDefault,
		); err != nil {
			return err
		}
		if err := state.Driver.StreamSynchronize(state.Stream); err != nil {
			return err
		}
		return state.Driver.MemcpyDtoH(driver.Bytes(output), outputPtr)
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

func copyToDevice[T ~uint16 | ~float32](lib *driver.Library, values []T) (driver.DevicePtr, error) {
	storage := driver.Bytes(values)
	pointer, err := lib.MemAlloc(uint64(len(storage)))
	if err != nil {
		return 0, err
	}
	if err := lib.MemcpyHtoD(pointer, storage); err != nil {
		_ = lib.MemFree(pointer)
		return 0, err
	}
	return pointer, nil
}
