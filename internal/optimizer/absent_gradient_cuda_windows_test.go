//go:build windows

package optimizer

import (
	"testing"

	cudatest "overgo/internal/cuda/testutil"
)

// TestAbsentGradientKeepsItsTensorOnTheDevice holds a pack's device stepper to the gathers it reads.
func TestAbsentGradientKeepsItsTensorOnTheDevice(t *testing.T) {
	cudatest.Require(t)
	tensors := map[string][]float32{"absent": {1, 1, 1, 1}, "present": {1, 1, 1, 1}}
	pack, err := NewTensorPack(tensors, absentGeometry)
	if err != nil {
		t.Fatal(err)
	}
	stepper, err := pack.NewStepper(Config{BaseLearningRate: 0.1, Momentum: 0.9, Schedule: ScheduleConstant})
	if err != nil {
		t.Fatal(err)
	}
	defer stepper.Close()
	stepAbsentGradient(t, tensors, pack, func() {
		if err := stepper.Step(); err != nil {
			t.Fatal(err)
		}
	})
}
