package densecausal

import (
	"math"
	"testing"

	"overgo/internal/testutil"
)

func graftFixtureModels(t *testing.T) (*Model, *Model) {
	t.Helper()
	targetWeights, targetShapes := testutil.DenseCausalWeights(t, testutil.DenseCausalSpec{
		Vocab: 32, Hidden: 16, Heads: 2, HeadDim: 8,
		KVHeads: 2, Intermediate: 32, Layers: 2, Seed: 1,
	})
	target, err := NewModel(targetWeights, targetShapes, 2, 8, 10000, 1e-6)
	if err != nil {
		t.Fatal(err)
	}
	donorWeights, donorShapes := testutil.DenseCausalWeights(t, testutil.DenseCausalSpec{
		Vocab: 32, Hidden: 12, Heads: 2, HeadDim: 6,
		KVHeads: 2, Intermediate: 24, Layers: 1, Seed: 2,
	})
	donor, err := NewModel(donorWeights, donorShapes, 2, 6, 10000, 1e-6)
	if err != nil {
		t.Fatal(err)
	}
	return target, donor
}

func fixtureGraft(t *testing.T, target, donor *Model, seed int64) *Graft {
	t.Helper()
	graft, err := NewGraft(target, 0,
		donor.Weights["model.layers.0.mlp.gate_proj.weight"],
		donor.Weights["model.layers.0.mlp.up_proj.weight"],
		donor.Weights["model.layers.0.mlp.down_proj.weight"],
		donor.Dims.Hidden, donor.Dims.Intermediate, seed,
	)
	if err != nil {
		t.Fatal(err)
	}
	return graft
}

// TestGraftZeroUpIsExactNoOp pins the baseline identity: with the zero-
// initialized Up bridge, the grafted model's loss is bitwise the unmodified
// target's loss, so any held-out delta after training is attributable to the
// bridge alone.
func TestGraftZeroUpIsExactNoOp(t *testing.T) {
	target, donor := graftFixtureModels(t)
	graft := fixtureGraft(t, target, donor, 7)
	tokens := []int{3, 1, 4, 1, 5, 9, 2, 6}
	base, _, _, err := target.LossAndGrads(tokens)
	if err != nil {
		t.Fatal(err)
	}
	grafted, err := target.GraftLoss(graft, tokens)
	if err != nil {
		t.Fatal(err)
	}
	if base != grafted {
		t.Fatalf("zero-Up graft loss %v differs from target loss %v", grafted, base)
	}
}

// TestGraftBridgeGradientsMatchFiniteDifference verifies the bridge VJP
// against central finite differences on both bridge matrices. Tolerance
// rationale: f32 forward with f64 loss accumulation; 1e-2 relative on
// gradients of order 1e-2 with h=1e-3 keeps truncation and rounding bounded.
func TestGraftBridgeGradientsMatchFiniteDifference(t *testing.T) {
	target, donor := graftFixtureModels(t)
	graft := fixtureGraft(t, target, donor, 11)
	// A non-zero Up makes the Down gradient path observable.
	for i := range graft.Up {
		graft.Up[i] = float32((i%7)-3) * 0.02
	}
	tokens := []int{3, 1, 4, 1, 5, 9, 2, 6}
	bridge := Grads{
		BridgeDownName: make([]float32, len(graft.Down)),
		BridgeUpName:   make([]float32, len(graft.Up)),
	}
	_, err := target.GraftLossAndBridgeGrads(graft, tokens, bridge)
	if err != nil {
		t.Fatal(err)
	}
	const h = 1e-3
	probe := func(matrix []float32, index int) float64 {
		original := matrix[index]
		matrix[index] = original + h
		plus, err := target.GraftLoss(graft, tokens)
		if err != nil {
			t.Fatal(err)
		}
		matrix[index] = original - h
		minus, err := target.GraftLoss(graft, tokens)
		if err != nil {
			t.Fatal(err)
		}
		matrix[index] = original
		return (plus - minus) / (2 * h)
	}
	checks := []struct {
		name     string
		matrix   []float32
		gradient []float32
		index    int
	}{
		{"down", graft.Down, bridge[BridgeDownName], 5},
		{"down2", graft.Down, bridge[BridgeDownName], 33},
		{"up", graft.Up, bridge[BridgeUpName], 9},
		{"up2", graft.Up, bridge[BridgeUpName], 41},
	}
	for _, check := range checks {
		numeric := probe(check.matrix, check.index)
		analytic := float64(check.gradient[check.index])
		if diff := math.Abs(numeric - analytic); diff > 1e-2*max(1, math.Abs(numeric)) {
			t.Errorf("%s[%d]: |%g - %g| = %g exceeds tolerance", check.name, check.index, numeric, analytic, diff)
		}
	}
}

// TestTrainBridgeReducesTrainingLoss proves bridge-only Muon moves the
// composed model: training loss falls from step one, the bridge matrices
// change, and donor plus target weights are byte-identical afterward.
func TestTrainBridgeReducesTrainingLoss(t *testing.T) {
	target, donor := graftFixtureModels(t)
	graft := fixtureGraft(t, target, donor, 13)
	frozenProbe := append([]float32(nil), target.Weights["model.layers.0.mlp.gate_proj.weight"]...)
	donorProbe := append([]float32(nil), graft.DonorGate...)
	batches := [][]int{{3, 1, 4, 1, 5, 9, 2, 6}, {2, 7, 1, 8, 2, 8, 1, 8}}
	losses, err := target.TrainBridge(graft, batches, 0.1, 0.9, 12)
	if err != nil {
		t.Fatal(err)
	}
	// Batches alternate, so progress is same-batch: step 10 vs step 0.
	if len(losses) != 12 || !(losses[10] < losses[0]) {
		t.Fatalf("bridge training did not reduce loss: %v", losses)
	}
	for i, value := range target.Weights["model.layers.0.mlp.gate_proj.weight"] {
		if value != frozenProbe[i] {
			t.Fatal("target weights changed during bridge-only training")
		}
	}
	for i, value := range graft.DonorGate {
		if value != donorProbe[i] {
			t.Fatal("donor weights changed during bridge-only training")
		}
	}
}
