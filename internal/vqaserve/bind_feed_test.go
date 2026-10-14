package vqaserve

import (
	"slices"
	"testing"

	"overgo/internal/cuda/device"
	"overgo/internal/cuda/driver"
	"overgo/internal/cuda/executor"
	"overgo/internal/tensor"
	"overgo/internal/tensor/dtype"
)

// TestBindFeedLeavesInputsUnchangedWhenTheUploadFails holds BindFeed to its refusal on the host.
//   - an allocation set without a device refuses every upload, and a fake device is out of reach because device.Worker owns a real CUDA context
//   - the bound path is held by the device modes of `validate -parity vqa`
func TestBindFeedLeavesInputsUnchangedWhenTheUploadFails(t *testing.T) {
	t.Parallel()
	builder := tensor.NewBuilder()
	weight := builder.Input("weight", dtype.F32, tensor.MustShape(tensor.SingletonExtent))
	output := builder.Add(weight, weight)
	if err := builder.Err(); err != nil {
		t.Fatal(err)
	}
	compiled, err := executor.Compile(output)
	if err != nil {
		t.Fatal(err)
	}
	inputs := compiled.NewDeviceInputs()
	allocations := device.NewAllocationSet(nil)
	if err := BindFeed(t.Context(), &allocations, inputs, weight, driver.Bytes([]float32{1})); err == nil {
		t.Fatal("BindFeed bound a payload through an allocation set without a device, want the upload refused")
	}
	if slices.ContainsFunc(inputs.Pointers, func(pointer driver.DevicePtr) bool { return pointer != 0 }) {
		t.Fatalf("a refused upload left device inputs %v, want every slot unbound", inputs.Pointers)
	}
}
