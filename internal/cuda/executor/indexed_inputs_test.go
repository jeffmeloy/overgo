package executor

import (
	"testing"

	"overgo/internal/cuda/driver"
	"overgo/internal/tensor"
	"overgo/internal/tensor/dtype"
)

// TestIndexedOperandSlotsOnly pins the operand contract after the map-layer
// deletion: DeviceInputs.Set binds each compiled input node into its indexed
// slot directly, non-compiled nodes are refused by name, and the tensor-keyed
// BindDeviceInputs conversion no longer exists (its callers all migrated;
// compilation of this package without it is the proof).
func TestIndexedOperandSlotsOnly(t *testing.T) {
	builder := tensor.NewBuilder()
	shape := tensor.MustShape(4, 2)
	left := builder.Input("left", dtype.F32, shape)
	right := builder.Input("right", dtype.F32, shape)
	output := builder.Add(left, right)
	compiled, err := Compile(output)
	if err != nil {
		t.Fatal(err)
	}
	inputs := compiled.NewDeviceInputs()
	if err := inputs.Set(left, driver.DevicePtr(4096)); err != nil {
		t.Fatal(err)
	}
	if err := inputs.Set(right, driver.DevicePtr(8192)); err != nil {
		t.Fatal(err)
	}
	leftSlot, ok := compiled.InputSlot(left)
	if !ok {
		t.Fatal("left input has no compiled slot")
	}
	rightSlot, ok := compiled.InputSlot(right)
	if !ok {
		t.Fatal("right input has no compiled slot")
	}
	if inputs.Pointers[leftSlot] != driver.DevicePtr(4096) || inputs.Pointers[rightSlot] != driver.DevicePtr(8192) {
		t.Fatalf("indexed slots = %v", inputs.Pointers)
	}
	if err := inputs.Set(output, driver.DevicePtr(1)); err == nil {
		t.Fatal("non-input node accepted into an operand slot")
	}
	var missing *DeviceInputs
	if err := missing.Set(left, driver.DevicePtr(1)); err == nil {
		t.Fatal("nil device inputs accepted a binding")
	}
}
