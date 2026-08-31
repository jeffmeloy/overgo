package composition

import (
	"strings"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/runrecord"
	"overgo/internal/testutil"
)

func fitnessScore(
	t *testing.T, name string, quality, latency, bytes [2]float64,
) CompositeScore {
	t.Helper()
	return CompositeScore{
		Composite:  testutil.ArtifactID(t, artifact.KindModel, name),
		Evaluation: testutil.ArtifactID(t, artifact.KindEvidence, name+"-held-out"),
		Dimensions: map[string]CompositeDimension{
			"quality":      {Baseline: quality[0], Candidate: quality[1], Direction: runrecord.DirectionMaximize},
			"latency":      {Baseline: latency[0], Candidate: latency[1], Direction: runrecord.DirectionMinimize},
			"device-bytes": {Baseline: bytes[0], Candidate: bytes[1], Direction: runrecord.DirectionMinimize},
		},
	}
}

// TestFitnessScoredCompositeSelection pins the unscalarized selection
// contract: each dimension compares in its own direction, a composite is
// fit only when no dimension regressed and at least one strictly
// improved, an average-only win — a large quality gain paid for with a
// latency regression — cannot pass, verdicts name exactly which
// dimensions moved, ordering is deterministic with fit composites first,
// and non-finite or directionless measurements refuse.
func TestFitnessScoredCompositeSelection(t *testing.T) {
	pareto := fitnessScore(t, "fitness-pareto", [2]float64{0.70, 0.78}, [2]float64{100, 95}, [2]float64{1000, 1000})
	tradeoff := fitnessScore(t, "fitness-tradeoff", [2]float64{0.70, 0.95}, [2]float64{100, 140}, [2]float64{1000, 900})
	flat := fitnessScore(t, "fitness-flat", [2]float64{0.70, 0.70}, [2]float64{100, 100}, [2]float64{1000, 1000})

	verdicts, err := ScoreCompositeSelection([]CompositeScore{tradeoff, flat, pareto})
	if err != nil {
		t.Fatal(err)
	}
	if len(verdicts) != 3 || verdicts[0].Composite != pareto.Composite || !verdicts[0].Fit {
		t.Fatalf("selection order = %+v", verdicts)
	}
	if len(verdicts[0].Improved) != 2 || len(verdicts[0].Regressed) != 0 {
		t.Fatalf("pareto verdict = %+v", verdicts[0])
	}
	for _, verdict := range verdicts[1:] {
		if verdict.Fit {
			t.Fatalf("unfit composite passed: %+v", verdict)
		}
	}
	byComposite := map[artifact.ID]CompositeFitnessVerdict{}
	for _, verdict := range verdicts {
		byComposite[verdict.Composite] = verdict
	}
	blended := byComposite[tradeoff.Composite]
	if len(blended.Improved) != 2 || len(blended.Regressed) != 1 || blended.Regressed[0] != "latency" {
		t.Fatalf("average-only win was not held to its regression: %+v", blended)
	}
	if unfit := byComposite[flat.Composite]; len(unfit.Improved) != 0 || unfit.Fit {
		t.Fatalf("flat composite verdict = %+v", unfit)
	}

	broken := pareto
	broken.Dimensions = map[string]CompositeDimension{
		"quality": {Baseline: 0.5, Candidate: 0.6, Direction: runrecord.DirectionNeutral},
	}
	if _, err := ScoreCompositeSelection([]CompositeScore{broken}); err == nil ||
		!strings.Contains(err.Error(), "comparison direction") {
		t.Fatalf("directionless dimension scored: %v", err)
	}
	if _, err := ScoreCompositeSelection(nil); err == nil {
		t.Fatal("empty selection scored")
	}
}

// TestSelectionEmitsAblationGatedPromotion pins the emission door: only a
// composite the multidimensional fitness judged fit may enter the
// existing ablation-armed promotion gate, the emitted evidence carries
// the dropped-source and shuffled-source arms and validates entirely in
// the existing promoter, and an unfit composite refuses with its
// regressions named.
func TestSelectionEmitsAblationGatedPromotion(t *testing.T) {
	policy, err := NewRepresentationBridgePromotionPolicy(RepresentationBridgePromotionPolicy{
		Direction:                 runrecord.DirectionMaximize,
		MinimumSeeds:              3,
		MinimumHeldOutGain:        0.05,
		MinimumSourceDependence:   0.15,
		MaximumRegression:         0.02,
		MaximumSeedSpread:         0.05,
		MaximumLatencyIncrease:    0.2,
		MaximumDeviceByteIncrease: 64,
	})
	if err != nil {
		t.Fatal(err)
	}
	evidence := RepresentationBridgePromotion{
		Bridge:         testutil.ArtifactID(t, artifact.KindAdapter, "selection bridge"),
		SourceModel:    testutil.ArtifactID(t, artifact.KindModel, "selection donor"),
		TargetModel:    testutil.ArtifactID(t, artifact.KindModel, "selection target"),
		SourceContract: testutil.ArtifactID(t, artifact.KindProfile, "selection source contract"),
		TargetContract: testutil.ArtifactID(t, artifact.KindProfile, "selection target contract"),
		HeldOutSplit:   testutil.ArtifactID(t, artifact.KindDatasetShard, "selection held-out split"),
		RegressionSet:  testutil.ArtifactID(t, artifact.KindDatasetShard, "selection regression suite"),
		Evaluator:      testutil.ArtifactID(t, artifact.KindEvidence, "selection evaluator"),
		Trials: []RepresentationBridgePromotionTrial{
			{Seed: 31, BridgeScore: 0.83, CheapBaselineScore: 0.75, DroppedSourceScore: 0.60, ShuffledSourceScore: 0.62, RegressionBaselineScore: 0.90, RegressionCandidateScore: 0.89, BaselineLatencyNS: 100, ComposedLatencyNS: 110, BaselinePeakDeviceBytes: 1000, ComposedPeakDeviceBytes: 1030},
			{Seed: 11, BridgeScore: 0.81, CheapBaselineScore: 0.74, DroppedSourceScore: 0.61, ShuffledSourceScore: 0.59, RegressionBaselineScore: 0.88, RegressionCandidateScore: 0.88, BaselineLatencyNS: 100, ComposedLatencyNS: 112, BaselinePeakDeviceBytes: 1000, ComposedPeakDeviceBytes: 1020},
			{Seed: 23, BridgeScore: 0.82, CheapBaselineScore: 0.76, DroppedSourceScore: 0.63, ShuffledSourceScore: 0.65, RegressionBaselineScore: 0.91, RegressionCandidateScore: 0.90, BaselineLatencyNS: 100, ComposedLatencyNS: 108, BaselinePeakDeviceBytes: 1000, ComposedPeakDeviceBytes: 1024},
		},
	}
	fit := CompositeFitnessVerdict{
		Composite:  testutil.ArtifactID(t, artifact.KindModel, "selection composite"),
		Evaluation: testutil.ArtifactID(t, artifact.KindEvidence, "selection evaluation"),
		Improved:   []string{"quality"}, Fit: true,
	}
	promotion, err := EmitAblationGatedPromotion(fit, policy, evidence)
	if err != nil {
		t.Fatal(err)
	}
	if promotion.ID.Kind() != artifact.KindEvidence || promotion.PolicyID != policy.ID ||
		promotion.WorstSourceDependence < policy.MinimumSourceDependence {
		t.Fatalf("emitted promotion = %+v", promotion)
	}

	unfit := fit
	unfit.Fit = false
	unfit.Regressed = []string{"latency"}
	if _, err := EmitAblationGatedPromotion(unfit, policy, evidence); err == nil ||
		!strings.Contains(err.Error(), "did not clear the multidimensional fitness") {
		t.Fatalf("unfit composite emitted: %v", err)
	}

	unarmed := evidence
	unarmed.Trials = append([]RepresentationBridgePromotionTrial(nil), evidence.Trials...)
	unarmed.Trials[0].ShuffledSourceScore = unarmed.Trials[0].BridgeScore
	if _, err := EmitAblationGatedPromotion(fit, policy, unarmed); err == nil {
		t.Fatal("promotion without source dependence emitted")
	}
}
