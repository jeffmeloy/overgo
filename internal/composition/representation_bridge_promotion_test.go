package composition

import (
	"math"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/runrecord"
	"overgo/internal/testutil"
)

func TestRepresentationBridgePromotion(t *testing.T) {
	promoter := RepresentationBridgePromoter{}
	request := RepresentationBridgePromotion{
		Bridge:         testutil.ArtifactID(t, artifact.KindAdapter, "trained representation bridge"),
		SourceModel:    testutil.ArtifactID(t, artifact.KindModel, "audio source model"),
		TargetModel:    testutil.ArtifactID(t, artifact.KindModel, "text target model"),
		SourceContract: testutil.ArtifactID(t, artifact.KindProfile, "audio hidden-state contract"),
		TargetContract: testutil.ArtifactID(t, artifact.KindProfile, "text embedding contract"),
		HeldOutSplit:   testutil.ArtifactID(t, artifact.KindDatasetShard, "held-out cross-modal split"),
		RegressionSet:  testutil.ArtifactID(t, artifact.KindDatasetShard, "target-only regression suite"),
		Evaluator:      testutil.ArtifactID(t, artifact.KindEvidence, "bridge promotion evaluator"),
		Policy: RepresentationBridgePromotionPolicy{
			Direction:               runrecord.DirectionMaximize,
			MinimumSeeds:            3,
			MinimumHeldOutGain:      0.05,
			MinimumSourceDependence: 0.15,
			MaximumRegression:       0.02,
		},
		Trials: []RepresentationBridgePromotionTrial{
			{Seed: 31, BridgeScore: 0.83, CheapBaselineScore: 0.75, DroppedSourceScore: 0.60, ShuffledSourceScore: 0.62, RegressionBaselineScore: 0.90, RegressionCandidateScore: 0.89},
			{Seed: 11, BridgeScore: 0.81, CheapBaselineScore: 0.74, DroppedSourceScore: 0.61, ShuffledSourceScore: 0.59, RegressionBaselineScore: 0.88, RegressionCandidateScore: 0.88},
			{Seed: 23, BridgeScore: 0.82, CheapBaselineScore: 0.76, DroppedSourceScore: 0.63, ShuffledSourceScore: 0.65, RegressionBaselineScore: 0.91, RegressionCandidateScore: 0.90},
		},
	}
	promotion, err := promoter.Evaluate(request)
	if err != nil {
		t.Fatal(err)
	}
	if promotion.ID.Kind() != artifact.KindEvidence || promotion.Trials[0].Seed != 11 ||
		promotion.WorstHeldOutGain < request.Policy.MinimumHeldOutGain ||
		promotion.WorstSourceDependence < request.Policy.MinimumSourceDependence ||
		promotion.WorstRegression > request.Policy.MaximumRegression {
		t.Fatalf("promotion evidence = %+v", promotion)
	}
	content, err := promotion.Content()
	if err != nil {
		t.Fatal(err)
	}
	parsed, err := promoter.Parse(content.Data)
	if err != nil || parsed.ID != promotion.ID || len(parsed.Lineage()) != len(promotion.Lineage()) {
		t.Fatalf("promotion round trip: parsed=%+v err=%v", parsed, err)
	}

	tests := []struct {
		name   string
		mutate func(*RepresentationBridgePromotion)
	}{
		{"single seed", func(value *RepresentationBridgePromotion) {
			value.Policy.MinimumSeeds = 2
			value.Trials = value.Trials[:1]
		}},
		{"no held-out gain", func(value *RepresentationBridgePromotion) {
			value.Trials[0].CheapBaselineScore = value.Trials[0].BridgeScore
		}},
		{"no source dependence", func(value *RepresentationBridgePromotion) {
			value.Trials[0].ShuffledSourceScore = value.Trials[0].BridgeScore
		}},
		{"target regression", func(value *RepresentationBridgePromotion) {
			value.Trials[0].RegressionCandidateScore = value.Trials[0].RegressionBaselineScore - 0.03
		}},
		{"non-finite score", func(value *RepresentationBridgePromotion) {
			value.Trials[0].BridgeScore = math.NaN()
		}},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			candidate := request
			candidate.Trials = append([]RepresentationBridgePromotionTrial(nil), request.Trials...)
			test.mutate(&candidate)
			if _, err := promoter.Evaluate(candidate); err == nil {
				t.Fatal("invalid promotion evidence accepted")
			}
		})
	}
}
