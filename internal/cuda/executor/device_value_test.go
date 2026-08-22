package executor

import (
	"math"
	"testing"

	"overgo/internal/cuda/driver"
	"overgo/internal/tensor"
	"overgo/internal/tensor/dtype"
)

func TestDeviceValueSliceLastAxis(t *testing.T) {
	const (
		width       = uint64(2)
		heads       = uint64(3)
		tokens      = uint64(5)
		start       = uint64(2)
		count       = uint64(2)
		basePointer = driver.DevicePtr(1024)
	)
	shape := tensor.MustShape(width, heads, tokens)
	capacity, err := shape.Bytes(dtype.F32)
	if err != nil {
		t.Fatal(err)
	}
	value := DeviceValue{Pointer: basePointer, Shape: shape, CapacityBytes: capacity}
	view, err := value.SliceLastAxis(dtype.F32, start, count)
	if err != nil {
		t.Fatal(err)
	}
	unit := tensor.MustShape(width, heads, 1)
	stride, err := unit.Bytes(dtype.F32)
	if err != nil {
		t.Fatal(err)
	}
	if view.Pointer != basePointer+driver.DevicePtr(start*stride) ||
		!view.Shape.Equal(tensor.MustShape(width, heads, count)) ||
		view.CapacityBytes != capacity-start*stride {
		t.Fatalf("slice = %+v", view)
	}
}

func TestDeviceValueSliceLastAxisRejectsInvalidViews(t *testing.T) {
	shape := tensor.MustShape(2, 3)
	fixtures := []DeviceValue{
		{Pointer: 1, Shape: shape},
		{Pointer: driver.DevicePtr(math.MaxUint64), Shape: shape},
		{Pointer: 1, Shape: shape, CapacityBytes: 1},
	}
	starts := []uint64{shape.Dims[1], 1, 0}
	counts := []uint64{1, 1, 1}
	for index, fixture := range fixtures {
		if _, err := fixture.SliceLastAxis(dtype.F32, starts[index], counts[index]); err == nil {
			t.Fatalf("invalid view %d accepted", index)
		}
	}
}

func TestDeviceCopyRetainsTensorStorageContract(t *testing.T) {
	shape := tensor.MustShape(32)
	bytes, err := shape.Bytes(dtype.Q4_0)
	if err != nil {
		t.Fatal(err)
	}
	value := DeviceValue{Pointer: 256, Shape: shape}
	copySpec, err := value.Copy(dtype.Q4_0)
	if err != nil {
		t.Fatal(err)
	}
	if copySpec.Storage != dtype.Q4_0 || len(copySpec.Segments) != 1 || copySpec.Segments[0].Bytes != bytes {
		t.Fatalf("Q4_0 copy contract = %+v", copySpec)
	}

	packed, err := PackedCopy([]DeviceValue{value, value}, dtype.BF16)
	if err != nil {
		t.Fatal(err)
	}
	if packed.Storage != dtype.BF16 || !packed.Shape.Equal(tensor.MustShape(32, 2)) {
		t.Fatalf("BF16 packed copy contract = %+v", packed)
	}
}
