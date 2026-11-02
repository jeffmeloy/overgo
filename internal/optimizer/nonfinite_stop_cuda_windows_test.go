//go:build windows

package optimizer

import (
	"errors"
	"math"
	"testing"

	"overgo/internal/cuda/device"
	cudatest "overgo/internal/cuda/testutil"
)

// TestDeviceMuonStopsOnANonFiniteDirection: on the device a NaN gradient reads false at the Frobenius guard, and one so
// large its squared norm passes the f32 range gives a scale of zero and a silent update of nothing. Both stop the step
// with the error every device stepper returns, and a finite gradient steps.
func TestDeviceMuonStopsOnANonFiniteDirection(t *testing.T) {
	cudatest.Require(t)
	worker, err := device.New(0)
	if err != nil {
		t.Fatal(err)
	}
	defer worker.Close()
	const side, huge = 4, 1e30
	elements := side * side
	plan, err := CompilePlan(elements, []GroupSpec{{Name: "m", Start: 0, End: elements, Rows: side, Cols: side}})
	if err != nil {
		t.Fatal(err)
	}
	config := Config{BaseLearningRate: 0.1, Momentum: 0.9, Schedule: ScheduleConstant}
	step := func(planted float32) error {
		weights, gradients, momentum := make([]float32, elements), make([]float32, elements), make([]float32, elements)
		for index := range gradients {
			weights[index], gradients[index] = 1, float32(index+1)
		}
		gradients[0] = planted
		return DeviceMuonStepPlan(worker, weights, gradients, momentum, plan, 1, config)
	}
	if err := step(float32(math.NaN())); !errors.Is(err, ErrNonfiniteDirection) {
		t.Fatalf("a NaN gradient = %v, want the step stopped", err)
	}
	if err := step(huge); !errors.Is(err, ErrNonfiniteDirection) {
		t.Fatalf("a gradient past the f32 range when squared = %v, want the step stopped", err)
	}
	if err := step(1); err != nil {
		t.Fatalf("a finite gradient = %v", err)
	}
}
