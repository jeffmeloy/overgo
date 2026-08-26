package executor

import (
	"testing"

	"overgo/internal/cuda/driver"
	"overgo/internal/tensor"
	"overgo/internal/tensor/dtype"
)

// TestIndexedOperandSlotsOnly pins direct compiled-slot binding.
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
	const (
		leftPointer  = driver.DevicePtr(4096)
		rightPointer = driver.DevicePtr(8192)
	)
	bindings := tensor.InputBindings[driver.DevicePtr]{
		{Node: left, Value: leftPointer},
		{Node: right, Value: rightPointer},
	}
	if err := inputs.Bind(bindings); err != nil {
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
	if inputs.Pointers[leftSlot] != leftPointer || inputs.Pointers[rightSlot] != rightPointer {
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
