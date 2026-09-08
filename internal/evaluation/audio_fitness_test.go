package evaluation

import (
	"slices"
	"strings"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/runrecord"
	"overgo/internal/testutil"
)

// These are policy counterexamples, not measured speech/model results. The
// real audio acceptance separately binds repeated source, model and process
// evidence. Keep quality and resources in the existing vector-policy owner.
func TestAudioFitnessPolicyAcceptance(t *testing.T) {
	contract := CrossDomainEvaluationContract{CandidateDimensions: []CandidateDeltaDimension{CandidateDeltaSystem}, Requirements: []CandidateFitnessRequirement{
		{Name: "cer", Direction: runrecord.DirectionMinimize, MinimumImprovement: 0.001},
		{Name: "peak-bytes", Direction: runrecord.DirectionMinimize, MinimumImprovement: 1},
		{Name: "wall-ns", Direction: runrecord.DirectionMinimize, MinimumImprovement: 1},
		{Name: "wer", Direction: runrecord.DirectionMinimize, MinimumImprovement: 0.001},
	}}
	baseline := CandidateEvaluationArm{Fitness: []runrecord.Metric{
		{Name: "cer", Value: 0.01, Unit: "ratio", Direction: runrecord.DirectionMinimize},
		{Name: "peak-bytes", Value: 100, Unit: "bytes", Direction: runrecord.DirectionMinimize},
		{Name: "wall-ns", Value: 100, Unit: "ns", Direction: runrecord.DirectionMinimize},
		{Name: "wer", Value: 0.02, Unit: "ratio", Direction: runrecord.DirectionMinimize},
	}}
	for _, test := range []struct {
		name, reason string
		mutate       func(*CandidateEvaluationArm)
	}{
		{"no-gain", "no predeclared measurable improvement", func(*CandidateEvaluationArm) {}},
		{"faster-but-less-accurate", "regresses metric wer", func(v *CandidateEvaluationArm) { v.Fitness[2].Value = 50; v.Fitness[3].Value = 0.03 }},
		{"lower-wer-higher-cer", "regresses metric cer", func(v *CandidateEvaluationArm) { v.Fitness[3].Value = 0.01; v.Fitness[0].Value = 0.02 }},
		{"missing-memory", "fitness vector differs", func(v *CandidateEvaluationArm) { v.Fitness = slices.Delete(v.Fitness, 1, 2) }},
		{"changed-unit", "fitness vector differs", func(v *CandidateEvaluationArm) { v.Fitness[2].Unit = "seconds" }},
	} {
		t.Run(test.name, func(t *testing.T) {
			candidate := CandidateEvaluationArm{Dimensions: slices.Clone(contract.CandidateDimensions), Fitness: slices.Clone(baseline.Fitness)}
			test.mutate(&candidate)
			reasons, accepted := evaluateCandidateFitness(contract, baseline, candidate)
			if !slices.ContainsFunc(reasons, func(reason string) bool { return strings.Contains(reason, test.reason) }) || len(accepted) != 0 {
				t.Fatalf("policy failed: %v, %v", reasons, accepted)
			}
		})
	}
	// Only a predeclared authority can permit a quality/resource tradeoff.
	authority := testutil.ArtifactID(t, artifact.KindEvidence, "audio-policy-fixture-explicit-tradeoff")
	contract.AcceptedTradeoffs = []CandidateAcceptedTradeoff{{RegressedMetric: "peak-bytes", MaximumRegression: 10, ImprovedMetric: "wer", MinimumImprovement: 0.005, Authority: authority}}
	candidate := CandidateEvaluationArm{Dimensions: slices.Clone(contract.CandidateDimensions), Fitness: slices.Clone(baseline.Fitness)}
	candidate.Fitness[1].Value = 105
	candidate.Fitness[3].Value = 0.01
	reasons, accepted := evaluateCandidateFitness(contract, baseline, candidate)
	if len(reasons) != 0 || !slices.Equal(accepted, []artifact.ID{authority}) {
		t.Fatalf("explicit tradeoff refused: %v, %v", reasons, accepted)
	}
	candidate.Fitness[1].Value = 111
	if reasons, _ := evaluateCandidateFitness(contract, baseline, candidate); len(reasons) == 0 {
		t.Fatal("tradeoff ceiling ignored")
	}
	t.Log("shared policy refuses no gain, WER/CER regression, missing memory and changed units; explicit bounded tradeoff authority required; synthetic policy cases, not measured model evidence")
}
