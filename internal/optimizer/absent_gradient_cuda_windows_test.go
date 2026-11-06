//go:build windows

package optimizer

import (
	"testing"

	cudatest "overgo/internal/cuda/testutil"
	"overgo/internal/testcheck"
)

// TestAbsentGradientKeepsItsTensorOnTheDevice holds a pack's device stepper to the gathers it reads.
func TestAbsentGradientKeepsItsTensorOnTheDevice(t *testing.T) {
	cudatest.Require(t)
	tensors := map[string][]float32{"absent": {1, 1, 1, 1}, "present": {1, 1, 1, 1}}
	pack := testcheck.Must(NewTensorPack(tensors, absentGeometry))(t)
	stepper := testcheck.Must(pack.NewStepper(Config{BaseLearningRate: 0.1, Momentum: 0.9, Schedule: ScheduleConstant}))(t)
	defer stepper.Close()
	stepAbsentGradient(t, tensors, pack, func() {
		testcheck.NoError(t, stepper.Step())
	})
}
