package densecausal

import (
	"math"
	"slices"
	"testing"

	"overgo/internal/testutil"
)

func tinyMuonModel(t *testing.T) *Model {
	t.Helper()
	spec := testutil.DenseCausalSpec{
		Vocab: 32, Hidden: 16, Heads: 2, HeadDim: 8,
		KVHeads: 2, Intermediate: 32, Layers: 2, Seed: 1,
	}
	weights, shapes := testutil.DenseCausalWeights(t, spec)
	m, err := NewModel(weights, shapes, spec.Heads, spec.HeadDim, 10000, 1e-6)
	if err != nil {
		t.Fatal(err)
	}
	return m
}

func TestTensorGeometryRejectsNonPositiveDimensions(t *testing.T) {
	tests := map[string]struct {
		shape  []int
		length int
	}{
		"negative matrix": {shape: []int{-1, -1}, length: 1},
		"empty vector":    {shape: []int{0}, length: 0},
	}
	for name, test := range tests {
		t.Run(name, func(t *testing.T) {
			if _, _, err := tensorGeometry(test.shape, test.length); err == nil {
				t.Fatal("invalid tensor geometry accepted")
			}
		})
	}
}

// TestMuonTrainingDecreasesLossTiny: multi-step Muon training over a fixed batch
// reduces the loss on the seeded in-memory model, with the base learning rate
// DERIVED from the parameter count (baseLR=0), not hand-tuned. Loss finite
// throughout.
func TestMuonTrainingDecreasesLossTiny(t *testing.T) {
	m := tinyMuonModel(t)
	tokens := []int{1, 5, 9, 3, 7, 2, 11, 4}
	const steps = 20
	trajectory, err := m.TrainBatches(slices.Repeat([][]int{tokens}, steps), 0, 0.9)
	if err != nil {
		t.Fatal(err)
	}
	if len(trajectory) != steps {
		t.Fatalf("trajectory length %d, want %d", len(trajectory), steps)
	}
	for i, loss := range trajectory {
		if math.IsNaN(loss) || math.IsInf(loss, 0) {
			t.Fatalf("trajectory[%d] = %g is not finite", i, loss)
		}
	}
	after, _, err := m.Loss(tokens)
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("loss %.6f -> %.6f over %d Muon steps", trajectory[0], after, steps)
	if !(after < trajectory[0]) {
		t.Fatalf("Muon training did not reduce loss: before %.6f after %.6f", trajectory[0], after)
	}
}

func TestTrainBatchesResumeMatchesUninterrupted(t *testing.T) {
	batches := [][]int{
		{1, 5, 9, 3, 7, 2, 11, 4},
		{4, 11, 2, 7, 3, 9, 5, 1},
		{1, 2, 3, 4, 5, 6, 7, 8},
	}
	uninterrupted := tinyMuonModel(t)
	if _, err := uninterrupted.TrainBatches(batches, 0, 0.9); err != nil {
		t.Fatal(err)
	}
	resumed := tinyMuonModel(t)
	_, state, err := resumed.TrainBatchesResume(batches[:2], 0, 0.9, nil)
	if err != nil {
		t.Fatal(err)
	}
	if _, _, err := resumed.TrainBatchesResume(batches[2:], 0, 0.9, &state); err != nil {
		t.Fatal(err)
	}
	for name, want := range uninterrupted.Weights {
		if got := resumed.Weights[name]; !slices.Equal(got, want) {
			t.Fatalf("resumed batch weights differ for %q", name)
		}
	}
}

// Real-artifact Muon training is deliberately NOT gated here: one host
// Newton-Schulz step over a 500M model's large matrices exceeds go test's 600s
// timeout. Real-model Muon verification belongs to the device-NS rung (rung 2)
// or a research lane with its own timeout; the existing single-step SGD
// TestRealArtifactTrainingStepDecreasesLoss already exercises the real backward.
