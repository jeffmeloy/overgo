//go:build windows

package cublas

import (
	"context"
	"errors"
	"testing"

	"overgo/internal/cuda/device"
	"overgo/internal/cuda/driver"
	cudatest "overgo/internal/cuda/testutil"
	"overgo/internal/tensor/dtype"
	"overgo/internal/testutil"
)

const (
	fixtureDevice       = 0
	fixtureFloat32Bytes = uint64(4)
)

type blasFixture struct {
	t      *testing.T
	worker *device.Worker
}

type fixtureMemory struct {
	driver   *driver.Library
	pointers []driver.DevicePtr
}

func newBLASFixture(t *testing.T) *blasFixture {
	t.Helper()
	cudatest.Require(t)
	worker, err := device.New(fixtureDevice)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := worker.Close(); err != nil {
			t.Errorf("close CUDA worker: %v", err)
		}
	})
	return &blasFixture{t: t, worker: worker}
}

func (f *blasFixture) run(work func(*device.State, *Library, Handle, *fixtureMemory) error) {
	f.t.Helper()
	err := f.worker.Do(context.Background(), func(state *device.State) (resultErr error) {
		lib, err := Open()
		if err != nil {
			return err
		}
		defer func() { resultErr = errors.Join(resultErr, lib.Close()) }()
		handle, err := lib.Create()
		if err != nil {
			return err
		}
		defer func() { resultErr = errors.Join(resultErr, lib.Destroy(handle)) }()
		if err := lib.SetStream(handle, state.Stream); err != nil {
			return err
		}
		memory := &fixtureMemory{driver: state.Driver}
		defer func() { resultErr = errors.Join(resultErr, memory.close()) }()
		return work(state, lib, handle, memory)
	})
	if err != nil {
		f.t.Fatal(err)
	}
}

func (m *fixtureMemory) allocate(bytes uint64) (driver.DevicePtr, error) {
	pointer, err := m.driver.MemAlloc(bytes)
	if err == nil {
		m.pointers = append(m.pointers, pointer)
	}
	return pointer, err
}

func (m *fixtureMemory) copy(storage []byte) (driver.DevicePtr, error) {
	pointer, err := m.allocate(uint64(len(storage)))
	if err != nil {
		return 0, err
	}
	if err := m.driver.MemcpyHtoD(pointer, storage); err != nil {
		return 0, err
	}
	return pointer, nil
}

func (m *fixtureMemory) close() error {
	var result error
	for index := len(m.pointers) - 1; index >= 0; index-- {
		result = errors.Join(result, m.driver.MemFree(m.pointers[index]))
	}
	return result
}

func TestSGEMMIntegration(t *testing.T) {
	fixture := newBLASFixture(t)
	left := []float32{
		1, 2, 3,
		4, 5, 6,
	}
	right := []float32{
		1, 0, 0,
		0, 1, 0,
	}
	output := make([]float32, 4)
	fixture.run(func(state *device.State, lib *Library, handle Handle, memory *fixtureMemory) error {
		leftPtr, err := memory.copy(driver.Bytes(left))
		if err != nil {
			return err
		}
		rightPtr, err := memory.copy(driver.Bytes(right))
		if err != nil {
			return err
		}
		outputPtr, err := memory.allocate(uint64(len(output)) * fixtureFloat32Bytes)
		if err != nil {
			return err
		}

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
	want := []float32{1, 4, 2, 5}
	testutil.RequireSliceClose(t, "output", output, want, 1e-6)
}

// TestRowMajorGEMMF32Integration: non-square row-major convention.
func TestRowMajorGEMMF32Integration(t *testing.T) {
	fixture := newBLASFixture(t)
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
	fixture.run(func(state *device.State, lib *Library, handle Handle, memory *fixtureMemory) error {
		aPtr, err := memory.copy(driver.Bytes(a))
		if err != nil {
			return err
		}
		bPtr, err := memory.copy(driver.Bytes(b))
		if err != nil {
			return err
		}
		cPtr, err := memory.allocate(uint64(len(out)) * fixtureFloat32Bytes)
		if err != nil {
			return err
		}
		if err := lib.RowMajorGEMMF32(handle, m, k, n, aPtr, bPtr, cPtr); err != nil {
			return err
		}
		if err := state.Driver.StreamSynchronize(state.Stream); err != nil {
			return err
		}
		return state.Driver.MemcpyDtoH(driver.Bytes(out), cPtr)
	})
	testutil.RequireSliceClose(t, "out", out, want, 1e-4)
}

// TestRowMajorGEMMExF32TransposeIntegration: non-square Gram products.
func TestRowMajorGEMMExF32TransposeIntegration(t *testing.T) {
	fixture := newBLASFixture(t)
	const rows, cols = 3, 2
	x := []float32{1, 2, 3, 4, 5, 6} // row-major [3x2]
	// Host X^T*X [cols, cols].
	wantXtX := make([]float32, cols*cols)
	for i := 0; i < cols; i++ {
		for j := 0; j < cols; j++ {
			var s float32
			for r := 0; r < rows; r++ {
				s += x[r*cols+i] * x[r*cols+j]
			}
			wantXtX[i*cols+j] = s
		}
	}
	// Host X*X^T [rows, rows].
	wantXXt := make([]float32, rows*rows)
	for i := 0; i < rows; i++ {
		for j := 0; j < rows; j++ {
			var s float32
			for c := 0; c < cols; c++ {
				s += x[i*cols+c] * x[j*cols+c]
			}
			wantXXt[i*rows+j] = s
		}
	}

	gotXtX := make([]float32, cols*cols)
	gotXXt := make([]float32, rows*rows)
	fixture.run(func(state *device.State, lib *Library, handle Handle, memory *fixtureMemory) error {
		xPtr, err := memory.copy(driver.Bytes(x))
		if err != nil {
			return err
		}
		xtxPtr, err := memory.allocate(uint64(len(gotXtX)) * fixtureFloat32Bytes)
		if err != nil {
			return err
		}
		xxtPtr, err := memory.allocate(uint64(len(gotXXt)) * fixtureFloat32Bytes)
		if err != nil {
			return err
		}

		if err := lib.RowMajorGEMMExF32(handle, true, false, cols, rows, cols, xPtr, xPtr, xtxPtr); err != nil {
			return err
		}

		if err := lib.RowMajorGEMMExF32(handle, false, true, rows, cols, rows, xPtr, xPtr, xxtPtr); err != nil {
			return err
		}
		if err := state.Driver.StreamSynchronize(state.Stream); err != nil {
			return err
		}
		if err := state.Driver.MemcpyDtoH(driver.Bytes(gotXtX), xtxPtr); err != nil {
			return err
		}
		return state.Driver.MemcpyDtoH(driver.Bytes(gotXXt), xxtPtr)
	})
	testutil.RequireSliceClose(t, "XtX", gotXtX, wantXtX, 1e-4)
	testutil.RequireSliceClose(t, "XXt", gotXXt, wantXXt, 1e-4)
}

func TestGEMMExBF16Integration(t *testing.T) {
	fixture := newBLASFixture(t)
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
	fixture.run(func(state *device.State, lib *Library, handle Handle, memory *fixtureMemory) error {
		leftPtr, err := memory.copy(driver.Bytes(left))
		if err != nil {
			return err
		}
		rightPtr, err := memory.copy(driver.Bytes(right))
		if err != nil {
			return err
		}
		outputPtr, err := memory.allocate(uint64(len(output)) * fixtureFloat32Bytes)
		if err != nil {
			return err
		}
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
	want := []float32{1, 4, 2, 5}
	testutil.RequireSliceClose(t, "output", output, want, 1e-6)
}
