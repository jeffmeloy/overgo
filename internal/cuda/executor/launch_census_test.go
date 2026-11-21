package executor

import (
	"maps"
	"testing"

	"overgo/internal/cuda/device"
	"overgo/internal/cuda/driver"
	"overgo/internal/cuda/kernel"
	"overgo/internal/testcheck"
)

// TestALaunchCensusCountsEachKernelByName holds the launch census to its kernels' names: a probe's launches, one
// kernel twice and another once, are counted under each kernel's PTX entry name, which a test outside the executor
// reads to say which kernels a pass launched (moss-int8-epilogue's census of a MOSS pass). The probe issues no device
// work.
func TestALaunchCensusCountsEachKernelByName(t *testing.T) {
	t.Parallel()
	census := map[string]int{}
	state := &device.State{Trace: &device.LaunchTrace{Probe: true, Census: census}}
	var input, output driver.DevicePtr
	width, rows := uint32(4), uint32(1)
	softmax := boundKernel{id: kernelSoftmaxF32, argumentCount: 4}
	exponential := boundKernel{id: kernelExpF32, argumentCount: 3}
	for range 2 {
		testcheck.NoError(t, launchGridABI(state, softmax, kernel.Grid1D(int(rows)), kernel.DefaultBlock1D(), &input, &output, &width, &rows))
	}
	testcheck.NoError(t, launchGridABI(state, exponential, kernel.Grid1D(int(rows)), kernel.DefaultBlock1D(), &input, &output, &width))
	if want := map[string]int{"softmax_f32": 2, "exp_f32": 1}; !maps.Equal(census, want) {
		t.Fatalf("the census counted %v; want %v", census, want)
	}
}
