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
// cell and that an unbound kind reports rather than fabricates a command.
func TestReacquireCommand(t *testing.T) {
	guard := ModelValidation{Kind: recipe.TaskInference, Validation: "guard", Location: "C:/models/qwen.gguf"}
	name, args, err := reacquireCommand(guard)
	if err != nil || name != "go" {
		t.Fatalf("guard command = (%q, %v)", name, err)
	}
	if !slices.Contains(args, "-guard") || !slices.Contains(args, "-publish") || args[len(args)-1] != "C:/models/qwen.gguf" {
		t.Fatalf("guard args = %v", args)
	}

	if _, _, err := reacquireCommand(ModelValidation{Kind: recipe.TaskInference, Validation: "guard"}); err == nil {
		t.Fatal("guard without a location bound a command")
	}
	if _, _, err := reacquireCommand(ModelValidation{Kind: recipe.TaskImageGen, Validation: "image-proof"}); err == nil {
		t.Fatal("an image proof without a location bound a command")
	}
}

// TestRunBindsMediaMeasurement holds each media and specialized validation
// to one measurement of the model through its active capability, so the
// loop's empty-plan run fills those models' scorecard rows.
func TestRunBindsMediaMeasurement(t *testing.T) {
	t.Parallel()
	for _, cell := range []ModelValidation{
		{Kind: recipe.TaskImageGen, Validation: "image-proof", Location: "C:/models/image"},
		{Kind: recipe.TaskSpeech, Validation: "speech-proof", Location: "C:/models/speech"},
		{Kind: recipe.TaskTabular, Validation: "specialized-proof", Location: "C:/models/tabular"},
	} {
		name, args, err := reacquireCommand(cell)
		if err != nil || name != "go" {
			t.Fatalf("%s command = (%q, %v)", cell.Validation, name, err)
		}
		want := []string{"run", "./cmd/recipe", "measure", "-task", string(cell.Kind), cell.Location}
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
	if err := executeRun(t.Context(), &out, ".", plan); err != nil {
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
	err := executeRun(t.Context(), &out, ".", plan)
	if err == nil || !strings.Contains(err.Error(), "no bound command") {
		t.Fatalf("unbound run = %v", err)
	}
}
