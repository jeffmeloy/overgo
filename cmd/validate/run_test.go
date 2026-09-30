package main

import (
	"bytes"
	"slices"
	"strings"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/recipe"
	"overgo/internal/testutil"
)

// TestReacquireCommand proves the bound guard command is fully determined by the
// cell and the selected store, and that an unbound kind reports rather than
// fabricates a command. The guard publishes into the store the command read,
// never the default its own working directory would resolve.
func TestReacquireCommand(t *testing.T) {
	guard := ModelValidation{Kind: recipe.TaskInference, Validation: "guard", Location: "C:/models/qwen.gguf"}
	name, args, err := reacquireCommand(guard, "C:/stores/selected")
	if err != nil || name != "go" {
		t.Fatalf("guard command = (%q, %v)", name, err)
	}
	if !slices.Contains(args, "-guard") || !slices.Contains(args, "-publish") || args[len(args)-1] != "C:/models/qwen.gguf" {
		t.Fatalf("guard args = %v", args)
	}
	if at := slices.Index(args, "-repo"); at < 0 || args[at+1] != "C:/stores/selected" {
		t.Fatalf("guard args = %v, want -repo C:/stores/selected", args)
	}

	if _, _, err := reacquireCommand(ModelValidation{Kind: recipe.TaskInference, Validation: "guard"}, "C:/stores/selected"); err == nil {
		t.Fatal("guard without a location bound a command")
	}
	if _, _, err := reacquireCommand(ModelValidation{Kind: recipe.TaskImageGen, Validation: "image-proof"}, "C:/stores/selected"); err == nil {
		t.Fatal("an image proof without a location bound a command")
	}
	// A DNA model's guard cell is owned by its smoke run: the text guard is
	// the wrong producer, and the smoke lane cannot publish into the
	// selected store, so nothing is bound.
	dna := ModelValidation{Kind: recipe.TaskInference, Modality: dnaModality, Validation: "guard", Location: "C:/models/carbon.gguf"}
	if name, args, err := reacquireCommand(dna, "C:/stores/selected"); err == nil {
		t.Fatalf("a DNA model's guard cell bound %s %v", name, args)
	}
}

// TestRunBindsMediaMeasurement holds each media and specialized validation
// to one measurement of the model through its active capability, into the
// selected store, so the loop's empty-plan run fills those models' scorecard
// rows in the store it read.
func TestRunBindsMediaMeasurement(t *testing.T) {
	t.Parallel()
	for _, cell := range []ModelValidation{
		{Kind: recipe.TaskImageGen, Validation: "image-proof", Location: "C:/models/image"},
		{Kind: recipe.TaskSpeech, Validation: "speech-proof", Location: "C:/models/speech"},
		{Kind: recipe.TaskTabular, Validation: "specialized-proof", Location: "C:/models/tabular"},
	} {
		name, args, err := reacquireCommand(cell, "C:/stores/selected")
		if err != nil || name != "go" {
			t.Fatalf("%s command = (%q, %v)", cell.Validation, name, err)
		}
		want := []string{"run", "./cmd/recipe", "measure", "-repo", "C:/stores/selected", "-task", string(cell.Kind), cell.Location}
		if !slices.Equal(args, want) {
			t.Fatalf("%s args = %v, want %v", cell.Validation, args, want)
		}
	}
}

// TestValidateRunEmpty proves an empty run set is the clean no-op the dry run
// predicts for a commit that moves no surface: no command executes.
func TestValidateRunEmpty(t *testing.T) {
	var out bytes.Buffer
	plan := Plan{Reuse: []Selected{{}, {}}}
	if err := executeRun(t.Context(), &out, ".", "C:/stores/selected", plan); err != nil {
		t.Fatalf("empty run set errored: %v", err)
	}
	if !strings.Contains(out.String(), "nothing to re-acquire") {
		t.Fatalf("empty run output = %q", out.String())
	}
}

// TestValidateRunUnbound proves an unbound cell is reported as incomplete
// coverage without executing anything, never silently skipped.
func TestValidateRunUnbound(t *testing.T) {
	var out bytes.Buffer
	model := testutil.ArtifactID(t, artifact.KindModel, "wan")
	plan := Plan{Run: []Selected{{ModelValidation: ModelValidation{Model: model, ModelName: "Wan", Kind: recipe.TaskVideoGen, Validation: "video-proof"}}}}
	err := executeRun(t.Context(), &out, ".", "C:/stores/selected", plan)
	if err == nil || !strings.Contains(err.Error(), "no bound command") {
		t.Fatalf("unbound run = %v", err)
	}
}
