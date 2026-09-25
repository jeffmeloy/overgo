package main

import (
	"slices"
	"strings"
	"testing"
)

// TestTrainObjectiveModeRefusesRawDatasetInputs holds the -objective mode to
// the objective's registered split: every set flag that names a raw dataset,
// a recipe or recipe-session input, a preview or a checkpoint output is named
// back, sorted; an unset one and the flags the mode uses are not.
func TestTrainObjectiveModeRefusesRawDatasetInputs(t *testing.T) {
	t.Parallel()
	set := map[string]bool{
		"dataset": true, "recipe": true, "resume": false, "reference": true, "audio-manifest": false,
		"bootstrap-recipe": true, "out": true, "objective-scale": true, "model": true, "steps": true,
	}
	want := []string{"-bootstrap-recipe", "-dataset", "-objective-scale", "-out", "-recipe", "-reference"}
	if got := objectiveModeConflicts(set); !slices.Equal(got, want) {
		t.Fatalf("conflicts = %v, want %v", got, want)
	}
	if got := objectiveModeConflicts(map[string]bool{"model": true, "steps": true, "host": true}); len(got) != 0 {
		t.Fatalf("a flag the mode uses conflicted: %v", got)
	}
	err := runObjectiveMode(t.Context(), nil, objectiveRun{Alias: "objective.registered.texts", Model: "m", Steps: 1}, set, nil)
	if err == nil || !strings.Contains(err.Error(), "-dataset") {
		t.Fatalf("a conflicting run = %v, want the conflicting flags named", err)
	}
}
