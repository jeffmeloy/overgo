//go:build windows

package densecausal

import (
	"math"
	"slices"
	"testing"

	"overgo/internal/cuda/device"
	cudatest "overgo/internal/cuda/testutil"
)

// TestTrainDeviceResidentMatchesHost checks resident loss parity.
func TestTrainDeviceResidentMatchesHost(t *testing.T) {
	cudatest.Require(t)
	worker, err := device.New(0)
	if err != nil {
		t.Fatal(err)
	}
	defer worker.Close()

	mHost := tinyMuonModel(t)
	mDev := tinyMuonModel(t) // same seed -> identical initial weights
	tokens := []int{1, 5, 9, 3, 7, 2, 11, 4}
	const steps = 12

	batches := slices.Repeat([][]int{tokens}, steps)
	trajHost, err := mHost.TrainBatches(batches, 0, 0.9)
	if err != nil {
		t.Fatal(err)
	}
	deviceResult, err := mDev.TrainDeviceResident(worker, batches, 0, 0.9, DeviceTrainingOptions{})
	if err != nil {
		t.Fatal(err)
	}
	trajDev := deviceResult.Losses

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
	if !(trajDev[len(trajDev)-1] < trajDev[0]) {
		t.Fatalf("resident device training did not reduce loss: %.5f -> %.5f", trajDev[0], trajDev[len(trajDev)-1])
	}
	if worst > 1.2e-3 {
		t.Fatalf("resident device trajectory diverges from host: worst |d|=%.3e > 1.2e-3", worst)
	}

}

func TestTrainDeviceResidentQwen2BiasMatchesHost(t *testing.T) {
	cudatest.Require(t)
	worker, err := device.New(0)
	if err != nil {
		t.Fatal(err)
	}
	defer worker.Close()
	golden := readQwen2Golden(t)
	host := modelFromGolden(t, golden)
	resident := modelFromGolden(t, golden)
	batches := slices.Repeat([][]int{golden.Tokens}, 2)
	wantTrajectory, wantState, err := host.TrainBatchesResume(batches, 0, 0.9, nil)
	if err != nil {
		t.Fatal(err)
	}
	deviceResult, err := resident.TrainDeviceResident(worker, batches, 0, 0.9, DeviceTrainingOptions{})
	if err != nil {
		t.Fatal(err)
	}
	gotTrajectory, gotState := deviceResult.Losses, deviceResult.State
	if delta := maxF64Delta(wantTrajectory, gotTrajectory); delta > 1e-3 {
		t.Fatalf("Qwen2 trajectory delta %.3e", delta)
	}
	for name, want := range host.Weights {
		if delta := maxSliceDelta(want, resident.Weights[name]); delta > 6e-3 {
			t.Fatalf("Qwen2 updated weight %q delta %.3e", name, delta)
		}
	}
	if delta := maxF64Delta(wantState.Momentum, gotState.Momentum); delta > 6e-3 {
		t.Fatalf("Qwen2 momentum delta %.3e", delta)
	}
}

func maxF64Delta(left, right []float64) float64 {
	if len(left) != len(right) {
		return math.Inf(1)
	}
	var worst float64
	for index := range left {
		worst = max(worst, math.Abs(left[index]-right[index]))
	}
	return worst
}

func TestTrainDeviceResidentFrozenLexicalMatchesInitialLoss(t *testing.T) {
	cudatest.Require(t)
	worker, err := device.New(0)
	if err != nil {
		t.Fatal(err)
	}
	defer worker.Close()

	model := tinyMuonModel(t)
	tokens := []int{1, 5, 9, 3, 7, 2, 11, 4}
	want, _, err := model.Loss(tokens)
	if err != nil {
		t.Fatal(err)
	}
	embedBefore := slices.Clone(model.tensors.embedding.values)
	deviceResult, err := model.TrainDeviceResident(worker, slices.Repeat([][]int{tokens}, 8), 0, 0.9, DeviceTrainingOptions{FrozenLexical: true})
	if err != nil {
		t.Fatal(err)
	}
	trajectory := deviceResult.Losses
	if delta := math.Abs(trajectory[0] - want); delta > 2e-4 {
		t.Fatalf("initial loss delta %.3e exceeds device floor", delta)
	}
	if !(trajectory[len(trajectory)-1] < trajectory[0]) {
		t.Fatalf("frozen resident loss %.6f -> %.6f", trajectory[0], trajectory[len(trajectory)-1])
	}
	if !slices.Equal(embedBefore, model.tensors.embedding.values) {
		t.Fatal("frozen lexical table changed")
	}
}

func TestTrainDeviceResidentResumeExact(t *testing.T) {
	cudatest.Require(t)
	worker, err := device.New(0)
	if err != nil {
		t.Fatal(err)
	}
	defer worker.Close()
	batches := [][]int{{1, 5, 9, 3, 7, 2, 11, 4}, {4, 11, 2, 7, 3, 9, 5, 1}}
	uninterrupted := tinyMuonModel(t)
	want, err := uninterrupted.TrainDeviceResident(worker, batches, 0, 0.9, DeviceTrainingOptions{})
	if err != nil {
		t.Fatal(err)
	}
	resumed := tinyMuonModel(t)
	first, err := resumed.TrainDeviceResident(worker, batches[:1], 0, 0.9, DeviceTrainingOptions{})
	if err != nil {
		t.Fatal(err)
	}
	got, err := resumed.TrainDeviceResident(worker, batches[1:], 0, 0.9, DeviceTrainingOptions{Resume: &first.State})
	if err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(want.State.Momentum, got.State.Momentum) {
		t.Fatal("resumed device Muon momentum differs")
	}
	for name, want := range uninterrupted.Weights {
		if !slices.Equal(want, resumed.Weights[name]) {
			t.Fatalf("resumed device weight %q differs", name)
		}
	}
}
