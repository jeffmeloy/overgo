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
// a raw session input or a preview is named back, sorted; an
// unset one and the flags the mode uses, the trained model's output among
// them and the selected recipe, are not.
func TestTrainObjectiveModeRefusesRawDatasetInputs(t *testing.T) {
	t.Parallel()
	set := map[string]bool{
		"dataset": true, "recipe": true, "resume": false, "reference": true, "audio-manifest": false,
		"bootstrap-recipe": true, "out": true, "objective-scale": true, "model": true, "steps": true,
	}
	want := []string{"-bootstrap-recipe", "-dataset", "-objective-scale", "-reference"}
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

// TestTrainObjectiveModeFreezesLexicalOnTheDevice: -freeze-lexical is an
// -objective run's input, not a conflicting flag, and with -host the run is
// refused before it reads the store, since only the resident device
// trainer freezes the embedding and head.
func TestTrainObjectiveModeFreezesLexicalOnTheDevice(t *testing.T) {
	t.Parallel()
	set := map[string]bool{"objective": true, "model": true, "out": true, "steps": true, "freeze-lexical": true}
	if got := objectiveModeConflicts(set); len(got) != 0 {
		t.Fatalf("conflicts = %v, want -freeze-lexical accepted", got)
	}
	set["host"] = true
	run := objectiveRun{Alias: "objective.registered.texts", Model: "m", Out: "o", Steps: 1, Host: true, FreezeLexical: true}
	if err := runObjectiveMode(t.Context(), nil, run, set, nil); err == nil || !strings.Contains(err.Error(), "host and frozen lexical") {
		t.Fatalf("a -host -freeze-lexical run = %v, want it refused", err)
	}
}

// TestLoRAFlagsMakeTheRunTheyAskFor: no -lora flag asks for no adapter; a rank
// asks for a new one whose alpha defaults to twice the rank and whose targets
// split from a comma list, blanks dropped; an adapter to continue brings its
// own rank, alpha and targets, so the others are not carried beside it.
func TestLoRAFlagsMakeTheRunTheyAskFor(t *testing.T) {
	t.Parallel()
	if lora := loraRequest(0, 0, "", 0, ""); lora != nil {
		t.Fatalf("no flags asked for %+v", lora)
	}
	lora := loraRequest(8, 0, "q_proj, v_proj,,", 7, "")
	if lora == nil || lora.Rank != 8 || lora.Alpha != 16 || !slices.Equal(lora.Targets, []string{"q_proj", "v_proj"}) || lora.Seed != 7 || lora.From != "" {
		t.Fatalf("rank 8 asked for %+v, want alpha 16 and the two targets", lora)
	}
	if lora := loraRequest(8, 32, "", 0, ""); lora == nil || lora.Alpha != 32 || len(lora.Targets) != 0 {
		t.Fatalf("an explicit alpha asked for %+v", lora)
	}
	if lora := loraRequest(8, 32, "q_proj", 7, "adapter"); lora == nil || lora.From != "adapter" || lora.Rank != 0 || lora.Alpha != 0 || len(lora.Targets) != 0 || lora.Seed != 0 {
		t.Fatalf("continuing an adapter asked for %+v, want only its directory", lora)
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
