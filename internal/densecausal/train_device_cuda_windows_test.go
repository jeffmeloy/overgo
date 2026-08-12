//go:build windows

package densecausal

import (
	"math"
	"testing"

	"overgo/internal/cuda/device"
	cudatest "overgo/internal/cuda/testutil"
)

// TestTrainDeviceMatchesHostTrajectory runs the seeded tiny model through Train
// (host optimizer) and TrainDevice (device Muon step) from identical inits and
// checks the loss trajectories track within tolerance -- the device-accelerated
// optimizer reproduces the host learning curve, and training reduces the loss.
func TestTrainDeviceMatchesHostTrajectory(t *testing.T) {
	cudatest.Require(t)
	worker, err := device.New(0)
	if err != nil {
		t.Fatal(err)
	}
	defer worker.Close()

	tokens := []int{1, 5, 9, 3, 7, 2, 11, 4}
	const steps = 12

	hostModel := tinyMuonModel(t)
	hostTraj, err := hostModel.Train(tokens, steps, 0, 0.9)
	if err != nil {
		t.Fatal(err)
	}

	devModel := tinyMuonModel(t) // same seed => identical initialization
	devTraj, err := devModel.TrainDevice(worker, tokens, steps, 0, 0.9)
	if err != nil {
		t.Fatal(err)
	}

	var maxDiff float64
	for i := range hostTraj {
		if d := math.Abs(hostTraj[i] - devTraj[i]); d > maxDiff {
			maxDiff = d
		}
	}
	t.Logf("host %.5f->%.5f | device %.5f->%.5f | max traj diff %.3e",
		hostTraj[0], hostTraj[len(hostTraj)-1], devTraj[0], devTraj[len(devTraj)-1], maxDiff)

	if !(devTraj[len(devTraj)-1] < devTraj[0]) {
		t.Fatalf("device training did not reduce loss: %.5f -> %.5f", devTraj[0], devTraj[len(devTraj)-1])
	}
	const tolerance = 1e-3
	if maxDiff > tolerance {
		t.Fatalf("device trajectory diverges from host: max diff %.3e > %.1e", maxDiff, tolerance)
	}
}
