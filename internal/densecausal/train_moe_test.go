package densecausal

import (
	"fmt"
	"math"
	"slices"
	"testing"

	"overgo/internal/testutil"
)

// tinyMixtureModel converts the tiny dense fixture's second layer into a
// routed mixture (router + four experts + shared), exercising the mixed
// dense/MoE schedule the DeepSeek-V2 family declares.
func tinyMixtureModel(t *testing.T) *Model {
	t.Helper()
	spec := testutil.DenseCausalSpec{
		Vocab: 32, Hidden: 16, Heads: 2, HeadDim: 8,
		KVHeads: 2, Intermediate: 32, Layers: 2, Seed: 1,
	}
	weights, shapes := testutil.DenseCausalWeights(t, spec)
	const experts, expertInter, sharedInter = 4, 8, 8
	prefix := "model.layers.1.mlp."
	for _, name := range []string{prefix + "gate_proj.weight", prefix + "up_proj.weight", prefix + "down_proj.weight"} {
		delete(weights, name)
		delete(shapes, name)
	}
	seed := uint32(19650218)
	fill := func(name string, rows, cols int) {
		values := make([]float32, rows*cols)
		for i := range values {
			seed ^= seed << 13
			seed ^= seed >> 17
			seed ^= seed << 5
			values[i] = (float32(seed%2000)/1000 - 1) * 0.2
		}
		weights[name] = values
		shapes[name] = []int{rows, cols}
	}
	fill(prefix+"gate.weight", experts, spec.Hidden)
	for e := 0; e < experts; e++ {
		fill(fmt.Sprintf("%sexperts.%d.gate_proj.weight", prefix, e), expertInter, spec.Hidden)
		fill(fmt.Sprintf("%sexperts.%d.up_proj.weight", prefix, e), expertInter, spec.Hidden)
		fill(fmt.Sprintf("%sexperts.%d.down_proj.weight", prefix, e), spec.Hidden, expertInter)
	}
	fill(prefix+"shared_experts.gate_proj.weight", sharedInter, spec.Hidden)
	fill(prefix+"shared_experts.up_proj.weight", sharedInter, spec.Hidden)
	fill(prefix+"shared_experts.down_proj.weight", spec.Hidden, sharedInter)
	policy := MoERouterPolicy{TopK: 2, Scoring: MoEScoringSoftmax, NormalizeTopKProb: true, RoutedScaling: 1, ExpertInter: expertInter}
	m, err := NewMixtureModel(weights, shapes, spec.Heads, spec.HeadDim, 10000, 1e-6, policy, 0)
	if err != nil {
		t.Fatal(err)
	}
	return m
}

// TestMixtureMuonTrainingDecreasesLossTiny: the mixed dense/routed schedule
// trains end to end through the derived-hyperparameter Muon lane, and the
// mixture tensors demonstrably move.
func TestMixtureMuonTrainingDecreasesLossTiny(t *testing.T) {
	m := tinyMixtureModel(t)
	routerBefore := append([]float32(nil), m.Weights["model.layers.1.mlp.gate.weight"]...)
	expertBefore := append([]float32(nil), m.Weights["model.layers.1.mlp.experts.0.down_proj.weight"]...)
	tokens := []int{1, 5, 9, 3, 7, 2, 11, 4}
	const steps = 20
	trajectory, _, err := m.Train(slices.Repeat([][]int{tokens}, steps), derivedTestLearningRate(t, m), 0.9, nil, nil)
	if err != nil {
		t.Fatal(err)
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
	t.Logf("mixture loss %.6f -> %.6f over %d Muon steps", trajectory[0], after, steps)
	if !(after < trajectory[0]) {
		t.Fatalf("mixture Muon training did not reduce loss: before %.6f after %.6f", trajectory[0], after)
	}
	changed := func(label string, before, now []float32) {
		t.Helper()
		for i := range before {
			if before[i] != now[i] {
				return
			}
		}
		t.Fatalf("%s did not move during training", label)
	}
	changed("router", routerBefore, m.Weights["model.layers.1.mlp.gate.weight"])
	changed("expert 0 down", expertBefore, m.Weights["model.layers.1.mlp.experts.0.down_proj.weight"])
}
