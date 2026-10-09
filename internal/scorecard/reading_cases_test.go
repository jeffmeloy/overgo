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

// TestHeldoutReadingSaysHowManyAnswersNamedNoOption holds a reading whose
// answers were generated to saying how many of them named no option: each was
// counted wrong, and a reader who is not told takes a model that began to
// reason for one that chose badly. A reading whose every answer named one
// reads as it did.
func TestHeldoutReadingSaysHowManyAnswersNamedNoOption(t *testing.T) {
	t.Parallel()
	summary := evaluation.EvalSummary{
		Suite: "suite", Prompting: "chat-template", Record: testutil.ArtifactID(t, artifact.KindEvaluation, "unread"),
		Metrics: map[string]float64{"accuracy": 0.25}, Directions: map[string]runrecord.Direction{"accuracy": runrecord.DirectionMaximize},
	}
	suite := map[string]evaluation.SuiteDescriptor{"suite": {Kind: "multiple-choice", Source: "suite", Cases: 64, Chance: 0.1}}
	named, _ := qualityCell(recipe.TaskInference, []evaluation.EvalSummary{summary}, suite)
	if named.Unread != 0 || named.QualityText() != "0.25 ±0.054 on 64 cases, a constant answer scores 0.1 (accuracy, suite (chat-template))" {
		t.Fatalf("a reading whose every answer named an option reads %q", named.QualityText())
	}
	summary.Unread = 9
	unread, _ := qualityCell(recipe.TaskInference, []evaluation.EvalSummary{summary}, suite)
	if unread.Unread != 9 || unread.QualityText() != "0.25 ±0.054 on 64 cases, 9 of them answered with no option named and counted wrong, a constant answer scores 0.1 (accuracy, suite (chat-template))" {
		t.Fatalf("a reading nine of whose answers named no option reads %q", unread.QualityText())
	}
}

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

// TestHeldoutShareAtAnEndSaysHowNearItMustBe holds a share that is every case
// or none to saying so. Its standard error is zero there, so the reading
// would stand as if exact; a suite answered wholly right is at its ceiling,
// and all it tells is how near the ceiling the true share must be. A share
// between the ends, a reading with no cases counted, and a value of one on a
// measure that is no share say nothing more than they did.
func TestHeldoutShareAtAnEndSaysHowNearItMustBe(t *testing.T) {
	t.Parallel()
	read := func(metric string, value float64, cases uint64) string {
		cell, _ := qualityCell(recipe.TaskImageGen, []evaluation.EvalSummary{{
			Suite: "set", Prompting: "protocol", Record: testutil.ArtifactID(t, artifact.KindEvaluation, metric), Cases: cases,
			Metrics: map[string]float64{metric: value}, Directions: map[string]runrecord.Direction{metric: runrecord.DirectionMaximize},
		}}, nil)
		return cell.QualityText()
	}
	for want, got := range map[string]string{
		"1 on 100 cases, every one right, which at 95 in 100 puts the true share at 0.963 or more (accuracy, set (protocol))": read("accuracy", 1, 100),
		"0 on 100 cases, none right, which at 95 in 100 puts the true share at 0.037 or less (accuracy, set (protocol))":      read("accuracy", 0, 100),
		"1 on 4 cases, every one right, which at 95 in 100 puts the true share at 0.51 or more (accuracy, set (protocol))":    read("accuracy", 1, 4),
		"0.99 ±0.0099 on 100 cases (accuracy, set (protocol))":                                                                read("accuracy", 0.99, 100),
		"1 (accuracy, set (protocol))":                     read("accuracy", 1, 0),
		"1 on 100 cases (signal-to-noise, set (protocol))": read("signal-to-noise", 1, 100),
	} {
		if got != want {
			t.Errorf("the reading says %q, want %q", got, want)
		}
	}
}

// TestHeldoutReadingNearAConstantAnswerSaysSo holds a reading set beside what
// a constant answer scores to saying which side of its own noise it stands
// on. A share below the constant answer's is no better than it; a share above
// it by less than two of its own standard errors is within its noise of it,
// and tells nothing a constant answer does not; only a share further above
// states the constant answer's score as a floor it has cleared.
func TestHeldoutReadingNearAConstantAnswerSaysSo(t *testing.T) {
	t.Parallel()
	suite := map[string]evaluation.SuiteDescriptor{"suite": {Kind: "multiple-choice", Source: "suite", Cases: 12032, Chance: 0.1109}}
	read := func(value float64) string {
		cell, _ := qualityCell(recipe.TaskInference, []evaluation.EvalSummary{{
			Suite: "suite", Prompting: "protocol", Record: testutil.ArtifactID(t, artifact.KindEvaluation, "near"),
			Metrics: map[string]float64{"accuracy": value}, Directions: map[string]runrecord.Direction{"accuracy": runrecord.DirectionMaximize},
		}}, suite)
		return cell.QualityText()
	}
	for want, got := range map[string]string{
		"0.0786 ±0.0025 on 12032 cases, no better than the 0.1109 a constant answer scores (accuracy, suite (protocol))":      read(0.0786),
		"0.1119 ±0.0029 on 12032 cases, within its noise of the 0.1109 a constant answer scores (accuracy, suite (protocol))": read(0.1119),
		"0.119 ±0.003 on 12032 cases, a constant answer scores 0.1109 (accuracy, suite (protocol))":                           read(0.119),
	} {
		if got != want {
			t.Errorf("the reading says %q, want %q", got, want)
		}
	}
}
