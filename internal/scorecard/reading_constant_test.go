package scorecard

import (
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/evaluation"
	"overgo/internal/recipe"
	"overgo/internal/runrecord"
	"overgo/internal/testutil"
)

// TestHeldoutReadingStatesWhatAConstantAnswerScores holds what a held-out
// reading says of the answer that needs no model. A share of choices answered
// right states chance, from its suite's descriptor; a reading whose record
// names its own constant answer, as a held-out verdict on label accuracy does,
// states that one. A reading that beats it says what it scores; one that does
// not says in words that it is no better. A measure with no such answer, a
// judged generation or an error rate, states none and reads as it did.
func TestHeldoutReadingStatesWhatAConstantAnswerScores(t *testing.T) {
	t.Parallel()
	summary := func(set, metric string, value float64, direction runrecord.Direction) evaluation.EvalSummary {
		return evaluation.EvalSummary{
			Suite: set, Prompting: "protocol", Record: testutil.ArtifactID(t, artifact.KindEvaluation, set+metric),
			Metrics: map[string]float64{metric: value}, Directions: map[string]runrecord.Direction{metric: direction},
		}
	}
	read := func(descriptors map[string]evaluation.SuiteDescriptor, summary evaluation.EvalSummary) string {
		cell, _ := qualityCell(recipe.TaskInference, []evaluation.EvalSummary{summary}, descriptors)
		return cell.QualityText()
	}
	choices := map[string]evaluation.SuiteDescriptor{"choices": {Kind: "multiple-choice", Source: "choices", Cases: 64, Chance: 0.25}}
	for value, want := range map[float64]string{
		0.5:  "0.5 ±0.062 on 64 cases, a constant answer scores 0.25 (accuracy, choices (protocol))",
		0.25: "0.25 ±0.054 on 64 cases, no better than the 0.25 a constant answer scores (accuracy, choices (protocol))",
		0.2:  "0.2 ±0.05 on 64 cases, no better than the 0.25 a constant answer scores (accuracy, choices (protocol))",
	} {
		if got := read(choices, summary("choices", "accuracy", value, runrecord.DirectionMaximize)); got != want {
			t.Errorf("a share of choices at %g reads %q, want %q", value, got, want)
		}
	}
	majority := 0.7
	for value, want := range map[float64]string{
		0.6611: "0.6611 on 6 cases, no better than the 0.7 a constant answer scores (table-accuracy, view (protocol))",
		0.75:   "0.75 on 6 cases, a constant answer scores 0.7 (table-accuracy, view (protocol))",
	} {
		judged := summary("view", "table-accuracy", value, runrecord.DirectionMaximize)
		judged.Cases, judged.Constant = 6, &majority
		if got := read(nil, judged); got != want {
			t.Errorf("a label accuracy of %g reads %q, want %q", value, got, want)
		}
	}
	// A judged generation's suite is answered in words, so its descriptor has no chance.
	judged := map[string]evaluation.SuiteDescriptor{"clips": {Source: "clips", Cases: 64}}
	if got := read(judged, summary("clips", "accuracy", 0.25, runrecord.DirectionMaximize)); got != "0.25 ±0.054 on 64 cases (accuracy, clips (protocol))" {
		t.Errorf("a judged generation reads %q, want no constant answer", got)
	}
	// Chance belongs to a share of cases, not to another metric of the same suite.
	if got := read(choices, summary("choices", "perplexity", 12, runrecord.DirectionMinimize)); got != "12 on 64 cases (perplexity, choices (protocol))" {
		t.Errorf("a perplexity over a suite of choices reads %q, want no constant answer", got)
	}
}
