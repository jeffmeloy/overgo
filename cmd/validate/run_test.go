package main

import (
	"bytes"
	"slices"
	"strings"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/recipe"
	"overgo/internal/runrecord"
	"overgo/internal/testutil"
)

// TestReacquireCommand proves the bound guard command is fully determined by the
// cell and the selected store, and that an unbound kind reports rather than
// fabricates a command. The guard publishes into the store the command read,
// never the default its own working directory would resolve.
func TestReacquireCommand(t *testing.T) {
	t.Parallel()
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
}

// TestDNAGuardBindsTheSmokeLane holds a DNA model's stale guard cell to its
// owner. Its smoke run predates an inference surface move, so the
// current-evidence decision refuses it and the cell runs; -run then binds
// validate -smoke on that model into the selected store, the very command the
// refusal names, never the text guard and never no command at all.
func TestDNAGuardBindsTheSmokeLane(t *testing.T) {
	t.Parallel()
	root, beforeSurface, _, _ := surfaceHistory(t)
	const store = "C:/stores/selected"
	carbon := testutil.ArtifactID(t, artifact.KindModel, "carbon")
	cell := ModelValidation{
		Model: carbon, ModelName: "Carbon", Location: "C:/models/carbon.gguf",
		Kind: recipe.TaskInference, Modality: dnaModality, Validation: guardValidation, Surface: "inference",
		Recipe: testutil.ArtifactID(t, artifact.KindRecipe, "carbon-inference"),
	}
	stale := runrecord.Run{Outcome: runrecord.OutcomeSucceeded, CodeCommit: beforeSurface}
	read := acceptedRecords{smoke: map[artifact.ID]error{carbon: smokeCurrent(t.Context(), root, store, carbon, stale, true)}}
	plan := SelectValidation(Inputs{AffectedSurfaces: map[SurfaceID]AffectRecord{"inference": {}}, Registered: []ModelValidation{cell}, Current: read.current})
	if len(plan.Run) != 1 || plan.Run[0].Model != carbon || !strings.Contains(plan.Run[0].Reason, "inference surface moved") {
		t.Fatalf("a stale DNA smoke run was not selected to run: run=%+v reuse=%+v", plan.Run, plan.Reuse)
	}

	name, args, err := reacquireCommand(plan.Run[0].ModelValidation, store)
	want := []string{"run", "./cmd/validate", "-smoke", "-budget", "15m", "-model", carbon.String(), "-repo", store}
	if err != nil || name != "go" || !slices.Equal(args, want) {
		t.Fatalf("a stale DNA cell bound (%q, %v, %v), want go %v", name, args, err, want)
	}
	if bound := "run: go " + strings.Join(want, " "); !strings.HasSuffix(plan.Run[0].Reason, bound) {
		t.Fatalf("the refusal %q does not name the bound command %q", plan.Run[0].Reason, bound)
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
	t.Parallel()
	var out bytes.Buffer
	plan := Plan{Reuse: []Selected{{}, {}}}
	if err := executeRun(t.Context(), &out, ".", "C:/stores/selected", plan, false); err != nil {
		t.Fatalf("empty run set errored: %v", err)
	}
	if !strings.Contains(out.String(), "nothing to re-acquire") {
		t.Fatalf("empty run output = %q", out.String())
	}
}

// TestValidateRunCycleLanes runs each selected cycle lane's go command with
// its environment and only those: a lane that passes is counted, one that
// fails fails the run by name, so the loop withholds promotion, and a reused
// lane never starts. A command other than go test answers by its exit; a go
// test lane is acquired only by the outcomes it requires, read from its own
// run's events: each package it names completes with a passing test its
// selector matches, a declared exclusion beside them visible but uncredited.
// A failed package, a zero-exit skip that a same-named test passing in
// another package does not excuse, and a selector matching nothing each
// withhold by name; a lane whose selected tests only declared themselves
// inapplicable acquires nothing and is reported apart from the failures.
func TestValidateRunCycleLanes(t *testing.T) {
	t.Parallel()
	const fixtures = "./internal/testevidence/testdata/"
	var out bytes.Buffer
	plan := Plan{Suites: []SelectedSuite{
		{Lane: "passes", Run: true, Arguments: []string{"env", "GOFLAGS"}, Environment: []string{"GOFLAGS=-tags=lane-environment"}},
		{Lane: "fails", Run: true, Arguments: []string{"vet", "./no-such-package"}},
		{Lane: "reused", Arguments: []string{"env", "GOFLAGS"}},
		{Lane: "tested", Run: true, Arguments: []string{"test", "-tags", "", fixtures + "probefixture"}},
		{Lane: "failed", Run: true, Arguments: []string{"test", "./no-such-package"}},
		{Lane: "skipped", Run: true, Arguments: []string{"test", fixtures + "probefixture", fixtures + "skipfixture", "-run", "^TestUnavailableCapability$"}},
		{Lane: "absent", Run: true, Arguments: []string{"test", fixtures + "probefixture", "-run", "^TestAbsent$"}},
		{Lane: "inapplicable", Run: true, Arguments: []string{"test", fixtures + "probefixture", "-run", "^TestElsewhere$"}},
	}}
	err := executeRun(t.Context(), &out, testutil.RepoRoot(t), "C:/stores/selected", plan, false)
	if err == nil {
		t.Fatalf("cycle lanes run passed; output:\n%s", out.String())
	}
	failed, inapplicable, _ := strings.Cut(err.Error(), "declared inapplicable here")
	for lane, want := range map[string]string{
		"fails":   "exit=",
		"failed":  "./no-such-package failed",
		"skipped": "testdata/skipfixture: TestUnavailableCapability skipped",
		"absent":  "testdata/probefixture is absent",
	} {
		if !strings.Contains(failed, "cycle lane "+lane+": ") || !strings.Contains(failed, want) {
			t.Errorf("cycle lane %s is not withheld as a failure saying %q: %v", lane, want, err)
		}
	}
	if !strings.Contains(inapplicable, "cycle lane inapplicable: ") || !strings.Contains(inapplicable, "TestElsewhere") || strings.Contains(failed, "cycle lane inapplicable") {
		t.Errorf("a lane whose required test declared itself inapplicable is not reported apart from the failures: %v", err)
	}
	for _, lane := range []string{"passes", "tested", "reused"} {
		if strings.Contains(err.Error(), "cycle lane "+lane+":") {
			t.Errorf("cycle lane %s was withheld: %v", lane, err)
		}
	}
	if !strings.Contains(err.Error(), "re-acquired 2 cell(s)") {
		t.Errorf("cycle lanes run = %v, want the passing and the tested lanes acquired", err)
	}
	if !strings.Contains(out.String(), "lane-environment") || strings.Contains(out.String(), "cycle lane reused") {
		t.Fatalf("cycle lane output = %q, want the passing lane run with its environment and the reused one not started", out.String())
	}
}

// TestValidateRunUnbound proves an unbound cell is reported as incomplete
// coverage without executing anything, never silently skipped.
func TestValidateRunUnbound(t *testing.T) {
	t.Parallel()
	var out bytes.Buffer
	model := testutil.ArtifactID(t, artifact.KindModel, "wan")
	plan := Plan{Run: []Selected{{ModelValidation: ModelValidation{Model: model, ModelName: "Wan", Kind: recipe.TaskVideoGen, Validation: "video-proof"}}}}
	err := executeRun(t.Context(), &out, ".", "C:/stores/selected", plan, false)
	if err == nil || !strings.Contains(err.Error(), "no bound command") {
		t.Fatalf("unbound run = %v", err)
	}
}
