//go:build windows

package densecausal

import (
	"math"
	"slices"
	"testing"

	"overgo/internal/cuda/device"
	cudatest "overgo/internal/cuda/testutil"
)

// TestTrainDeviceResidentBatchesMatchesHost checks that resident training keeps
// the layer matrices and their Muon momentum resident on the device across all steps
// (uploaded once, downloaded only at checkpoint, no per-step weight scatter/gather)
// -- reproduces host Train's loss trajectory within fp32 tolerance on identical
// seeded models, and that training reduces the loss.
func TestTrainDeviceResidentBatchesMatchesHost(t *testing.T) {
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
	trajDev, _, err := mDev.TrainDeviceResidentBatches(worker, batches, 0, 0.9, nil)
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
	if !(trajDev[len(trajDev)-1] < trajDev[0]) {
		t.Fatalf("resident device training did not reduce loss: %.5f -> %.5f", trajDev[0], trajDev[len(trajDev)-1])
	}
	if worst > 1e-3 {
		t.Fatalf("resident device trajectory diverges from host: worst |d|=%.3e > 1e-3", worst)
	}

	// The final checkpointed weights must match the host-trained weights: the resident
	// matrices came back from the device, the host-owned tensors from the host optimizer.
	var worstW float64
	for name, hw := range mHost.Weights {
		dw := mDev.Weights[name]
		for i := range hw {
			if d := math.Abs(float64(hw[i] - dw[i])); d > worstW {
				worstW = d
			}
		}
	}
	t.Logf("final weight worst |d|=%.3e", worstW)
	if worstW > 1e-3 {
		t.Fatalf("resident device final weights diverge from host: worst |d|=%.3e > 1e-3", worstW)
	}
}

func TestTrainDeviceResidentFrozenLexicalBatchesMatchesInitialLoss(t *testing.T) {
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
	embedBefore := slices.Clone(model.Weights["model.embed_tokens.weight"])
	trajectory, _, err := model.TrainDeviceResidentFrozenLexicalBatches(worker, slices.Repeat([][]int{tokens}, 8), 0, 0.9, nil)
	if err != nil {
		t.Fatal(err)
	}
	if delta := math.Abs(trajectory[0] - want); delta > 2e-4 {
		t.Fatalf("initial loss delta %.3e exceeds device floor", delta)
	}
	if !(trajectory[len(trajectory)-1] < trajectory[0]) {
		t.Fatalf("frozen resident loss %.6f -> %.6f", trajectory[0], trajectory[len(trajectory)-1])
	}
	if !slices.Equal(embedBefore, model.Weights["model.embed_tokens.weight"]) {
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
	_, wantState, err := uninterrupted.TrainDeviceResidentBatches(worker, batches, 0, 0.9, nil)
	if err != nil {
		t.Fatal(err)
	}
	resumed := tinyMuonModel(t)
	_, firstState, err := resumed.TrainDeviceResidentBatches(worker, batches[:1], 0, 0.9, nil)
	if err != nil {
		t.Fatal(err)
	}
	_, gotState, err := resumed.TrainDeviceResidentBatches(worker, batches[1:], 0, 0.9, &firstState)
	if err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(wantState.Momentum, gotState.Momentum) {
		t.Fatal("resumed device Muon momentum differs")
	}
	for name, want := range uninterrupted.Weights {
		if !slices.Equal(want, resumed.Weights[name]) {
			t.Fatalf("resumed device weight %q differs", name)
		}
	}
}
