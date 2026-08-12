//go:build windows

package densecausal

import (
	"math"
	"testing"

	"overgo/internal/cuda/device"
	cudatest "overgo/internal/cuda/testutil"
)

// TestTrainDeviceFullMatchesHost checks that TrainDeviceFull (device backward +
// device Muon step) reproduces host Train's loss trajectory within fp32
// tolerance on identical seeded models.
func TestTrainDeviceFullMatchesHost(t *testing.T) {
	cudatest.Require(t)
	worker, err := device.New(0)
	if err != nil {
		t.Fatal(err)
	}
	defer worker.Close()

	mHost := tinyMuonModel(t)
	mDev := tinyMuonModel(t) // same seed -> identical initial weights
	tokens := []int{1, 5, 9, 3, 7, 2, 11, 4}
	const steps = 6

	trajHost, err := mHost.Train(tokens, steps, 0, 0.9)
	if err != nil {
		t.Fatal(err)
	}
	trajDev, err := mDev.TrainDeviceFull(worker, tokens, steps, 0, 0.9)
	if err != nil {
		t.Fatal(err)
	}

	if len(trajHost) != len(trajDev) {
		t.Fatalf("trajectory lengths differ: host %d device %d", len(trajHost), len(trajDev))
	}
	var worst float64
	for i := range trajHost {
		if d := math.Abs(trajHost[i] - trajDev[i]); d > worst {
			worst = d
		}
	}
	t.Logf("trajectory host[0]=%.6f device[0]=%.6f ... host[last]=%.6f device[last]=%.6f; worst |d|=%.3e",
		trajHost[0], trajDev[0], trajHost[len(trajHost)-1], trajDev[len(trajDev)-1], worst)
	if worst > 1e-3 {
		t.Fatalf("device-full trajectory diverges from host: worst |d|=%.3e > 1e-3", worst)
	}
}
