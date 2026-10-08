package main

import (
	"strings"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/recipe"
	"overgo/internal/scorecard"
	"overgo/internal/scorecard/metric"
	"overgo/internal/testutil"
)

// TestScorecardMarksAReadingOutsideItsModelsUse holds what the printed
// scorecard says of a model whose held-out reading is not a measure of what
// the model is for. Such a model is listed with its use and where its
// documentation states it, what is measured in its place, and what a measure
// of its use needs; a model with a recorded use and no reading is listed
// apart; a model with no recorded use, a model whose reading is on another
// measure than its recorded use speaks of, and a use recorded for a model the
// scorecard does not hold, are not listed. The reading's own cell says it is
// not a measure of what the model is for, and a reading that is says nothing.
//
// The uses the repository holds are read too: each is whole, and each names
// where its model's documentation states the use or that none ships.
func TestScorecardMarksAReadingOutsideItsModelsUse(t *testing.T) {
	t.Parallel()
	row := func(name string, task recipe.Task, measured bool, outside string) scorecard.Row {
		cell := scorecard.Cell{Reason: "nothing measures it", NotApplicable: true}
		if measured {
			cell = scorecard.Cell{Value: 910, Label: "perplexity, a corpus", Source: testutil.ArtifactID(t, artifact.KindEvaluation, name), OutsideUse: outside}
		}
		return scorecard.Row{Name: name, Task: task, Cells: map[metric.Name]scorecard.Cell{metric.Quality: cell}}
	}
	uses := []scorecard.ModelUse{
		{Model: "a drafter", Task: recipe.TaskInference, Measure: "a corpus", Use: "to draft for a larger model", Source: "its card", Measured: "its perplexity alone", Needs: "the larger model"},
		{Model: "a model nothing documents", Task: recipe.TaskImageGen, Use: "not stated", Source: "no documentation", Measured: "nothing", Needs: "the owner to say"},
		{Model: "a model not on the scorecard", Task: recipe.TaskInference, Measure: "a corpus", Use: "something", Source: "its card", Measured: "something else", Needs: "data"},
		{Model: "an editor read on its use", Task: recipe.TaskVideoGen, Measure: "whole clips", Use: "to change one thing", Source: "its card", Measured: "changes of the whole clip", Needs: "nothing"},
	}
	text := useStanding([]scorecard.Row{
		row("a drafter", recipe.TaskInference, true, "its perplexity alone"),
		row("a drafter", recipe.TaskTraining, true, ""),
		row("an assistant", recipe.TaskInference, true, ""),
		row("an editor read on its use", recipe.TaskVideoGen, true, ""),
		row("a model nothing documents", recipe.TaskImageGen, false, ""),
	}, uses)
	want := "\nHeld-out readings that measure something other than what their model is for, by the model's own documentation:\n" +
		"- a drafter inference: it is for: to draft for a larger model (its card). Measured: its perplexity alone. A measure of its use needs: the larger model.\n" +
		"\nModels with no held-out reading, and what naming a measure of their use waits on:\n" +
		"- a model nothing documents image-gen: it is for: not stated (no documentation). Measured: nothing. A measure of its use needs: the owner to say.\n"
	if text != want {
		t.Fatalf("the uses read:\n%s\nwant:\n%s", text, want)
	}
	if text := useStanding([]scorecard.Row{row("an assistant", recipe.TaskInference, true, "")}, uses); text != "" {
		t.Fatalf("a scorecard whose models have no recorded use says %q of uses", text)
	}

	held, err := scorecard.ModelUses()
	if err != nil {
		t.Fatal(err)
	}
	if len(held) == 0 {
		t.Fatal("the repository records no model's use")
	}
	for _, entry := range held {
		if !strings.Contains(entry.Source, "model card") && !strings.Contains(entry.Source, "no documentation") {
			t.Errorf("the use of %s %s does not say where its documentation states it, or that none ships: %q", entry.Model, entry.Task, entry.Source)
		}
	}
}
