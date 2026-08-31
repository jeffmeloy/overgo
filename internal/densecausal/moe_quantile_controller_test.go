package densecausal

import (
	"math"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/testutil"
)

func TestQuantileBiasChangesSelectionButNotUnbiasedCombineWeights(t *testing.T) {
	stratum := testutil.ArtifactID(t, artifact.KindDatasetShard, "controller-stratum")
	controller := moeQuantileBiasController{policy: moeQuantileBiasPolicy{
		Stratum: stratum, Authority: testutil.ArtifactID(t, artifact.KindRecipe, "controller-policy"),
		Target: moeQuantileSpec{Method: moeQuantileExact, Numerator: 1, Denominator: 2},
	}}
	policy := MoERouterPolicy{TopK: 1, Scoring: MoEScoringSigmoid, RoutedScaling: 1, ExpertInter: 1}
	x, router := []float32{1, 2}, []float32{2, -2}

	baseline, baselineScores, err := routeTopKWithQuantileController(x, router, 2, 1, 2, policy, stratum, &controller)
	if err != nil || baseline.indices[0] != 0 || baseline.indices[1] != 0 || len(controller.nextBias) != 2 {
		t.Fatalf("baseline route = (%+v, %v), bias=%v", baseline, err, controller.nextBias)
	}
	if math.Abs(float64(controller.nextBias[0])) > 0.5 || math.Abs(float64(controller.nextBias[1])) > 0.5 {
		t.Fatalf("derived bias is not bounded by half the activated-score range: %v", controller.nextBias)
	}

	biased, unbiasedScores, err := routeTopKWithQuantileController(x, router, 2, 1, 2, policy, stratum, &controller)
	if err != nil {
		t.Fatal(err)
	}
	if biased.indices[0] != 1 || biased.indices[1] != 0 {
		t.Fatalf("next-step bias did not change only the expected selection: %v", biased.indices)
	}
	for row, selected := range biased.indices {
		want := unbiasedScores[row*2+selected]
		if biased.weights[row] != want {
			t.Fatalf("row %d combine weight=%v want unbiased score=%v", row, biased.weights[row], want)
		}
	}
	if baselineScores[0] != unbiasedScores[0] || baselineScores[1] != unbiasedScores[1] {
		t.Fatal("selection bias changed activated router scores")
	}

	approximate := controller
	approximate.policy.Target = moeQuantileSpec{Method: moeQuantileHistogram, Numerator: 1, Denominator: 2, Bins: 2}
	if _, _, err := routeTopKWithQuantileController(x, router, 2, 1, 2, policy, stratum, &approximate); err == nil {
		t.Fatal("approximate quantile evidence acquired controller authority")
	}
	other := testutil.ArtifactID(t, artifact.KindDatasetShard, "other-stratum")
	if _, _, err := routeTopKWithQuantileController(x, router, 2, 1, 2, policy, other, &controller); err == nil {
		t.Fatal("controller crossed its declared stratum")
	}
}
