//go:build windows

package kernel

import (
	cudatest "overgo/internal/cuda/testutil"
	"testing"

	"overgo/internal/cuda/device"
)

func TestVectorAddIntegration(t *testing.T) {
	cudatest.Require(t)

	worker, err := device.New(0)
	if err != nil {
		t.Fatal(err)
	}
	defer worker.Close()

	inputA := []float32{1, 2, 3, -4, 100.5}
	inputB := []float32{9, -2, 0.25, 4, -0.5}
	want := []float32{10, 0, 3.25, 0, 100}
	var output []float32
	err = worker.Do(t.Context(), func(state *device.State) error {
		var launchErr error
		output, launchErr = VectorAdd(state.Driver, state.Stream, inputA, inputB)
		return launchErr
	})
	if err != nil {
		t.Fatal(err)
	}
	for i := range want {
		if output[i] != want[i] {
			t.Fatalf("output[%d] = %v, want %v", i, output[i], want[i])
		}
	}
}
