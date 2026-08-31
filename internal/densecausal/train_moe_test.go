package densecausal

import (
	"math"
	"reflect"
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
		MoELayer: 1, MoEExperts: 4, MoEIntermediate: 8, MoEShared: 8,
	}
	weights, shapes := testutil.DenseCausalWeights(t, spec)
	const expertInter = 8
	policy := MoERouterPolicy{TopK: 2, Scoring: MoEScoringSoftmax, NormalizeTopKProb: true, RoutedScaling: 1, ExpertInter: expertInter}
	m, err := NewMixtureModel(weights, shapes, spec.Heads, spec.HeadDim, 10000, 1e-6, policy, 0)
	if err != nil {
		t.Fatal(err)
	}
	return m
}

func TestMoETrainingPublishesRouterObservationsWithoutChangingNumerics(t *testing.T) {
	baseline, observed := tinyMixtureModel(t), tinyMixtureModel(t)
	batches := [][]int{{1, 5, 9, 3}, {7, 2, 11, 4}}
	learningRate := derivedTestLearningRate(t, baseline)
	wantLosses, wantState, err := baseline.Train(batches, learningRate, 0.9, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	var routes []MoERouterObservation
	gotLosses, gotState, err := observed.TrainWithRouterObservations(
		batches, learningRate, 0.9, nil, nil,
		func(_ int, observation MoERouterObservation) error {
			routes = append(routes, observation)
			return nil
		},
	)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(gotLosses, wantLosses) || !reflect.DeepEqual(gotState, wantState) ||
		!reflect.DeepEqual(observed.Weights, baseline.Weights) {
		t.Fatal("router observation changed host training numerics")
	}
	if len(routes) != len(batches) {
		t.Fatalf("router observations=%d want=%d", len(routes), len(batches))
	}
	for _, route := range routes {
		if route.Layer != 1 || route.Rows != len(batches[0]) || route.Experts != 4 || route.TopK != 2 ||
			len(route.Selections) != route.Rows*route.TopK || len(route.CombineWeights) != len(route.Selections) ||
			len(route.Accepted) != len(route.Selections) || len(route.Margins) != route.Rows {
			t.Fatalf("incomplete router observation: %+v", route)
		}
		for row, margin := range route.Margins {
			if !margin.Observed || margin.Value < 0 {
				t.Fatalf("margin[%d]=%+v", row, margin)
			}
		}
	}
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
