package main

import (
	"slices"
	"strings"
	"testing"

	"overgo/internal/trainingprogram"
	"overgo/internal/trainingworkflow"
)

// TestTrainObjectiveModeRefusesRawDatasetInputs holds the -objective mode to
// the objective's registered split: every set flag that names a raw dataset,
// a recipe or recipe-session input or a preview is named back, sorted; an
// unset one and the flags the mode uses, the trained model's output among
// them, are not.
func TestTrainObjectiveModeRefusesRawDatasetInputs(t *testing.T) {
	t.Parallel()
	set := map[string]bool{
		"dataset": true, "recipe": true, "resume": false, "reference": true, "audio-manifest": false,
		"bootstrap-recipe": true, "out": true, "objective-scale": true, "model": true, "steps": true,
	}
	want := []string{"-bootstrap-recipe", "-dataset", "-objective-scale", "-recipe", "-reference"}
	if got := objectiveModeConflicts(set); !slices.Equal(got, want) {
		t.Fatalf("conflicts = %v, want %v", got, want)
	}
	if got := objectiveModeConflicts(map[string]bool{"model": true, "out": true, "steps": true, "host": true}); len(got) != 0 {
		t.Fatalf("a flag the mode uses conflicted: %v", got)
	}
	err := runObjectiveMode(t.Context(), nil, objectiveRun{Alias: "objective.registered.texts", Model: "m", Steps: 1}, set, nil)
	if err == nil || !strings.Contains(err.Error(), "-dataset") {
		t.Fatalf("a conflicting run = %v, want the conflicting flags named", err)
	}
}

// TestPreferenceRunsSayWhenUnjudged: a run of another objective publishes
// no preference verdict, and a resumed DPO run, which names no base policy,
// says it is not judged rather than passing silently.
func TestPreferenceRunsSayWhenUnjudged(t *testing.T) {
	t.Parallel()
	var output strings.Builder
	token := trainingworkflow.Result{Objective: trainingprogram.ObjectiveTokenPrediction}
	if err := judgePreference(t.Context(), nil, trainingworkflow.Request{}, token, &output); err != nil || output.Len() != 0 {
		t.Fatalf("a token run = (%q, %v), want nothing judged or said", output.String(), err)
	}
	resumed := trainingworkflow.Request{ResumeDirectory: "resume"}
	if err := judgePreference(t.Context(), nil, resumed, trainingworkflow.Result{Objective: trainingprogram.ObjectiveDPO}, &output); err != nil ||
		!strings.Contains(output.String(), "not judged") {
		t.Fatalf("a resumed DPO run = (%q, %v), want it said to be not judged", output.String(), err)
	}
}
