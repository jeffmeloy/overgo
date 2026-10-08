package scorecard

import (
	"math"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/evaluation"
	"overgo/internal/recipe"
	"overgo/internal/runrecord"
	"overgo/internal/testutil"
)

// TestHeldoutReadingStatesItsCasesAndNoise holds what a held-out reading says
// of how much stands behind it. A share of a suite's cases answered right
// states those cases and the standard error of the share. A reading whose
// record counted its own cases and noise, as a held-out verdict's does, states
// those, beside its base model's value. An error rate over pooled words states
// its cases and no noise, and a reading no record counts states neither and
// reads as it did. An evaluated set's cases are stated and do not rank it:
// the set named first still heads the row.
func TestHeldoutReadingStatesItsCasesAndNoise(t *testing.T) {
	t.Parallel()
	summary := func(set, metric string, value float64) evaluation.EvalSummary {
		return evaluation.EvalSummary{
			Suite: set, Prompting: "protocol", Record: testutil.ArtifactID(t, artifact.KindEvaluation, set+metric),
			Metrics: map[string]float64{metric: value}, Directions: map[string]runrecord.Direction{metric: runrecord.DirectionMaximize},
		}
	}
	read := func(descriptors map[string]evaluation.SuiteDescriptor, summaries ...evaluation.EvalSummary) Cell {
		cell, _ := qualityCell(recipe.TaskInference, summaries, descriptors)
		return cell
	}
	suite := map[string]evaluation.SuiteDescriptor{"suite": {Kind: "multiple-choice", Source: "suite", Cases: 64}}
	share := read(suite, summary("suite", "accuracy", 0.25))
	if want := math.Sqrt(0.25 * 0.75 / 64); share.Cases != 64 || math.Abs(share.Noise-want) > 1e-15 || share.QualityText() != "0.25 ±0.054 on 64 cases (accuracy, suite (protocol))" {
		t.Fatalf("a share of 64 cases reads %q with noise %g, want its cases and a standard error of %g", share.QualityText(), share.Noise, want)
	}
	judged := summary("view", "token-accuracy", 0.5082)
	judged.Cases, judged.Noise = 24, 0.0075
	verdict := read(nil, judged)
	base := 0.5001
	verdict.Base = &base
	if verdict.Cases != 24 || verdict.Noise != 0.0075 || verdict.QualityText() != "0.5082 ±0.0075 vs base 0.5001 on 24 cases (token-accuracy, view (protocol))" {
		t.Fatalf("a reading whose record counted its cases reads %q", verdict.QualityText())
	}
	pooled := summary("speech", "wer", 0.03209)
	pooled.Cases = 32
	if cell := read(nil, pooled); cell.Noise != 0 || cell.QualityText() != "0.03209 on 32 cases (wer, speech (protocol))" {
		t.Fatalf("an error rate over pooled words reads %q with noise %g, want its cases and no noise", cell.QualityText(), cell.Noise)
	}
	if cell := read(nil, summary("unknown", "accuracy", 0.5)); cell.Cases != 0 || cell.Noise != 0 || cell.QualityText() != "0.5 (accuracy, unknown (protocol))" {
		t.Fatalf("a reading no record counts reads %q", cell.QualityText())
	}
	// An evaluated set has no kind: its cases are stated and it is not ranked by them.
	evaluated := map[string]evaluation.SuiteDescriptor{"set-a": {Source: "frames", Cases: 16}, "set-b": {Source: "clips", Cases: 128}}
	if cell := read(evaluated, summary("set-b", "accuracy", 0.5), summary("set-a", "accuracy", 0.75)); cell.Contract.Suite != "set-a" || cell.Cases != 16 {
		t.Fatalf("of two evaluated sets the row is headed by %s on %d cases, want the set named first with its own 16 cases", cell.Contract.Suite, cell.Cases)
	}
}
