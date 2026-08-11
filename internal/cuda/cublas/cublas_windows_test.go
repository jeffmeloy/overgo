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

// TestRowMajorGEMMExF32TransposeIntegration verifies the transpose-capable
// path against host references for the two gram products device Newton-Schulz
// needs: XᵀX (transA) and XXᵀ (transB), with X[3x2] non-square.
func TestRowMajorGEMMExF32TransposeIntegration(t *testing.T) {
	cudatest.Require(t)
	worker, err := device.New(0)
	if err != nil {
		t.Fatal(err)
	}
	defer worker.Close()

	const rows, cols = 3, 2
	x := []float32{1, 2, 3, 4, 5, 6} // row-major [3x2]

	// Host gram XᵀX [cols x cols].
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
	// Host gram XXᵀ [rows x rows].
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
		xPtr, err := copyToDevice(state.Driver, x)
		if err != nil {
			return err
		}
		defer state.Driver.MemFree(xPtr)
		xtxPtr, err := state.Driver.MemAlloc(uint64(len(gotXtX) * 4))
		if err != nil {
			return err
		}
		defer state.Driver.MemFree(xtxPtr)
		xxtPtr, err := state.Driver.MemAlloc(uint64(len(gotXXt) * 4))
		if err != nil {
			return err
		}
		defer state.Driver.MemFree(xxtPtr)
		// XᵀX: op(A)=Xᵀ [cols x rows], op(B)=X [rows x cols] -> [cols x cols].
		if err := lib.RowMajorGEMMExF32(handle, true, false, cols, rows, cols, xPtr, xPtr, xtxPtr); err != nil {
			return err
		}
		// XXᵀ: op(A)=X [rows x cols], op(B)=Xᵀ [cols x rows] -> [rows x rows].
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
	if err != nil {
		t.Fatal(err)
	}
	for i := range wantXtX {
		if math.Abs(float64(gotXtX[i]-wantXtX[i])) > 1e-4 {
			t.Fatalf("XtX[%d] = %v, want %v", i, gotXtX[i], wantXtX[i])
		}
	}
	for i := range wantXXt {
		if math.Abs(float64(gotXXt[i]-wantXXt[i])) > 1e-4 {
			t.Fatalf("XXt[%d] = %v, want %v", i, gotXXt[i], wantXXt[i])
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
