//go:build windows

package densecausaltrain

import (
	"math"
	"slices"
	"testing"

	"overgo/internal/cuda/device"
	cudatest "overgo/internal/cuda/testutil"
	"overgo/internal/testutil"
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
	trajHost, _, err := mHost.Train(batches, derivedTestLearningRate(t, mHost), 0.9, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	deviceResult, err := TrainDeviceResident(mDev, worker, batches, derivedTestLearningRate(t, mDev), 0.9, Options{})
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
	wantTrajectory, wantState, err := host.Train(batches, derivedTestLearningRate(t, host), 0.9, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	deviceResult, err := TrainDeviceResident(resident, worker, batches, derivedTestLearningRate(t, resident), 0.9, Options{})
	if err != nil {
		t.Fatal(err)
	}
	gotTrajectory, gotState := deviceResult.Losses, deviceResult.State
	if delta := testutil.MaxAbsDiff(wantTrajectory, gotTrajectory); delta > 1e-3 {
		t.Fatalf("Qwen2 trajectory delta %.3e", delta)
	}
	for name, want := range host.Weights {
		if delta := testutil.MaxAbsDiff(want, resident.Weights[name]); delta > 6e-3 {
			t.Fatalf("Qwen2 updated weight %q delta %.3e", name, delta)
		}
	}
	if delta := testutil.MaxAbsDiff(wantState.Momentum, gotState.Momentum); delta > 6e-3 {
		t.Fatalf("Qwen2 momentum delta %.3e", delta)
	}
}

// TestTrainDeviceResidentUntiedHeadMatchesHost holds a trained untied head
// and embedding, whose gradients the device writes apart, to the host
// trainer over a window that repeats a token. One step holds every weight
// tightly. A second step holds the trajectory, the momentum and the lexical
// tensors, whose update the second forward reads; it does not hold the layer
// matrices, where Muon's orthogonalization of a rank-limited momentum
// amplifies f32 rounding into differences of several 1e-3, tied or untied.
func TestTrainDeviceResidentUntiedHeadMatchesHost(t *testing.T) {
	cudatest.Require(t)
	worker, err := device.New(0)
	if err != nil {
		t.Fatal(err)
	}
	defer worker.Close()
	const (
		momentum        = 0.9
		trajectoryLimit = 1e-5
		stepLimit       = 1e-4
	)
	tokens := []int{1, 5, 9, 3, 7, 2, 11, 5}
	for _, steps := range []int{1, 2} {
		host, resident := untiedMuonModel(t), untiedMuonModel(t)
		embeddingName, normName, headName := resident.LexicalTensorNames()
		if embeddingName == headName {
			t.Fatal("fixture head is tied")
		}
		before := map[string][]float32{embeddingName: slices.Clone(resident.Weights[embeddingName]), headName: slices.Clone(resident.Weights[headName])}
		batches := slices.Repeat([][]int{tokens}, steps)
		wantTrajectory, wantState, err := host.Train(batches, derivedTestLearningRate(t, host), momentum, nil, nil)
		if err != nil {
			t.Fatal(err)
		}
		deviceResult, err := TrainDeviceResident(resident, worker, batches, derivedTestLearningRate(t, resident), momentum, Options{})
		if err != nil {
			t.Fatal(err)
		}
		if delta := testutil.MaxAbsDiff(wantTrajectory, deviceResult.Losses); delta > trajectoryLimit {
			t.Fatalf("%d step(s): untied trajectory delta %.3e", steps, delta)
		}
		if delta := testutil.MaxAbsDiff(wantState.Momentum, deviceResult.State.Momentum); delta > stepLimit {
			t.Fatalf("%d step(s): untied momentum delta %.3e", steps, delta)
		}
		held := []string{embeddingName, normName, headName}
		if steps == 1 {
			held = held[:0]
			for name := range host.Weights {
				held = append(held, name)
			}
		}
		for _, name := range held {
			if delta := testutil.MaxAbsDiff(host.Weights[name], resident.Weights[name]); delta > stepLimit {
				t.Fatalf("%d step(s): untied updated weight %q delta %.3e", steps, name, delta)
			}
		}
		for name, initial := range before {
			if slices.Equal(initial, resident.Weights[name]) {
				t.Fatalf("%d step(s): device left trained lexical tensor %q unchanged", steps, name)
			}
		}
	}
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
	embedding, _, _ := model.LexicalTensorNames()
	embedBefore := slices.Clone(model.Weights[embedding])
	deviceResult, err := TrainDeviceResident(model, worker, slices.Repeat([][]int{tokens}, 8), derivedTestLearningRate(t, model), 0.9, Options{FrozenLexical: true})
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
	if !slices.Equal(embedBefore, model.Weights[embedding]) {
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
	want, err := TrainDeviceResident(uninterrupted, worker, batches, derivedTestLearningRate(t, uninterrupted), 0.9, Options{})
	if err != nil {
		t.Fatal(err)
	}
	resumed := tinyMuonModel(t)
	first, err := TrainDeviceResident(resumed, worker, batches[:1], derivedTestLearningRate(t, resumed), 0.9, Options{})
	if err != nil {
		t.Fatal(err)
	}
	got, err := TrainDeviceResident(resumed, worker, batches[1:], derivedTestLearningRate(t, resumed), 0.9, Options{Resume: &first.State})
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
