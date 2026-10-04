package main

import (
	"fmt"
	"maps"
	"os"
	"os/exec"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/evaluation"
	"overgo/internal/modelrecipe"
	"overgo/internal/overgodb"
	"overgo/internal/recipe"
	"overgo/internal/runrecord"
	"overgo/internal/testutil"
)

// surfaceHistory writes a fixture repository whose history moves the
// inference surface, and so the evaluation surface that holds it, and then
// only documentation, and returns its root with the commit before the surface
// move, the one before the documentation change, and HEAD.
func surfaceHistory(t *testing.T) (root, beforeSurface, beforeDocs, head string) {
	t.Helper()
	root = t.TempDir()
	write := func(name, content string) {
		t.Helper()
		path := filepath.Join(root, filepath.FromSlash(name))
		if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(path, []byte(content), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	commit := func(message string) string {
		t.Helper()
		for _, arguments := range [][]string{{"add", "-A"}, {"commit", "-q", "-m", message}, {"rev-parse", "HEAD"}} {
			command := exec.Command("git", append([]string{"-c", "user.name=fixture", "-c", "user.email=fixture@example.com"}, arguments...)...)
			command.Dir = root
			output, err := command.CombinedOutput()
			if err != nil {
				t.Fatalf("git %v: %v: %s", arguments, err, output)
			}
			if arguments[0] == "rev-parse" {
				return strings.TrimSpace(string(output))
			}
		}
		return ""
	}
	write("go.mod", "module fixture\n\ngo 1.26\n")
	for _, surface := range []string{"internal/inference", "internal/cuda/executor", "internal/modelrecipe", "internal/evaluation"} {
		write(surface+"/"+filepath.Base(surface)+".go", "package "+filepath.Base(surface)+"\n")
	}
	write("docs/notes.md", "fixture\n")
	if output, err := exec.Command("git", "init", "-q", root).CombinedOutput(); err != nil {
		t.Fatalf("git init: %v: %s", err, output)
	}
	beforeSurface = commit("fixture")
	write("internal/inference/inference.go", "package inference\n\n// Rate moves the surface.\nconst Rate = 1\n")
	beforeDocs = commit("move the surface")
	write("docs/notes.md", "fixture, revised\n")
	head = commit("revise the notes")
	return root, beforeSurface, beforeDocs, head
}

// TestBenchmarkDoesNotReuseUnknownExecution makes missing execution authority
// select a fresh measurement, while a complete real campaign remains reusable.
func TestBenchmarkDoesNotReuseUnknownExecution(t *testing.T) {
	t.Parallel()
	store, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	authorities := evaluation.ListingAuthorities()
	data, _ := replayFixtureSuite(t, 1)
	suite, err := evaluation.CompileSuite(data, authorities)
	if err != nil {
		t.Fatal(err)
	}
	model := testutil.ArtifactID(t, artifact.KindModel, "unknown execution model")
	output := testutil.ArtifactID(t, artifact.KindEvaluation, "legacy report")
	if _, err := store.Commit(t.Context(), artifact.Batch{Key: "benchmark/authorities", Artifacts: []artifact.Descriptor{
		{ID: model}, {ID: authorities.ModelDefinition}, {ID: authorities.RuntimeRecipe}, {ID: suite.Descriptor().Dataset}, {ID: output},
	}}); err != nil {
		t.Fatal(err)
	}
	legacy, err := runrecord.NewRun(authorities.RuntimeRecipe, runrecord.OutcomeSucceeded, nil, []artifact.ID{output}, "")
	if err != nil {
		t.Fatal(err)
	}
	content, err := legacy.Content()
	if err != nil {
		t.Fatal(err)
	}
	batch, err := artifact.NewDocumentBatch("benchmark/legacy-run", []artifact.Content{content}, legacy.Lineage(), nil)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := store.Commit(t.Context(), batch); err != nil {
		t.Fatal(err)
	}
	record, err := runrecord.NewEvaluation(authorities.RuntimeRecipe, legacy.ID, suite.Descriptor().Dataset, []runrecord.Metric{{Name: "accuracy", Value: 0.5, Direction: runrecord.DirectionMaximize}})
	if err != nil {
		t.Fatal(err)
	}
	batch, err = record.Batch("benchmark/legacy-evaluation")
	if err != nil {
		t.Fatal(err)
	}
	if _, err := store.Commit(t.Context(), batch); err != nil {
		t.Fatal(err)
	}
	cell := ModelValidation{Model: model, Recipe: authorities.RuntimeRecipe, Kind: recipe.TaskInference, Validation: "mmlu", Surface: "evaluation"}
	read := func() boundRecord {
		t.Helper()
		benchmarks := benchmarkEvidence{suites: []evaluation.CompiledSuite{suite}, evidence: evaluation.LatestEvidence(t.Context(), store, map[artifact.ID]string{suite.Descriptor().Dataset: "store/mmlu"}, 0)}
		value, err := benchmarks.record(t.Context(), store, cell)
		if err != nil {
			t.Fatal(err)
		}
		return value
	}
	if value := read(); value.found || value.replay.Valid() || value.protocol != evaluation.PromptingRawCompletion.Label() {
		t.Errorf("unknown execution was reused: %+v, want a fresh raw measurement without replay", value)
	}
	platform, err := runrecord.NewEnvironment(runrecord.Environment{Host: "fixture", OS: "fixture", Arch: "fixture", Device: "fixture", Backend: "host", Driver: "fixture"})
	if err != nil {
		t.Fatal(err)
	}
	scored := 0
	campaign, err := evaluation.NewIsolatedCampaign(store, replayFixtureRuntime{scored: &scored}, modelrecipe.ProgramIdentity{
		Model: model, Definition: authorities.ModelDefinition, Recipe: authorities.RuntimeRecipe,
	}, platform, strings.Repeat("a", 40))
	if err != nil {
		t.Fatal(err)
	}
	suite, err = evaluation.CompileSuite(data, campaign.Authorities())
	if err != nil {
		t.Fatal(err)
	}
	fresh, err := campaign.Evaluate(t.Context(), suite)
	if err != nil {
		t.Fatal(err)
	}
	if value := read(); !value.found || value.run.ID != fresh.Run || value.replay != fresh.Evaluation || value.protocol != evaluation.PromptingRawCompletion.Label() || scored != 1 {
		t.Fatalf("known real execution was not retained: %+v, scored=%d", value, scored)
	}
}

// TestBenchmarkCurrencyFollowsItsSurface holds a benchmark cell to the commit
// its record ran at, in a fixture repository whose history moves the
// evaluation surface and then only documentation. A cell whose record ran
// before the move runs, and so does one no record measures yet; one whose
// record ran after it, with only documentation since, is reused; and one its
// model's declared domains exclude is skipped. Each cell that runs binds one
// evaluate worker on its model and family into the selected store, under the
// protocol of the record it replaces, and the inventory takes the same
// decision.
func TestBenchmarkCurrencyFollowsItsSurface(t *testing.T) {
	t.Parallel()
	root, beforeSurface, beforeDocs, _ := surfaceHistory(t)
	const store = "C:/stores/selected"
	directories, err := surfaceDirectories(t.Context(), root, modalitySurfaces["evaluation"].roots)
	if err != nil {
		t.Fatal(err)
	}
	chat, raw := evaluation.PromptingChatTemplate.Label(), evaluation.PromptingRawCompletion.Label()
	benchmark := func(name, protocol string) ModelValidation {
		return ModelValidation{
			Model: testutil.ArtifactID(t, artifact.KindModel, name), ModelName: name, Location: "C:/models/" + name + "/m.gguf",
			Kind: recipe.TaskInference, Validation: "mmlu-pro", Surface: "evaluation",
			Recipe: testutil.ArtifactID(t, artifact.KindRecipe, name), Protocol: protocol,
		}
	}
	older, current, unmeasured, excluded := benchmark("Older", chat), benchmark("Current", chat), benchmark("Unmeasured", raw), benchmark("Excluded", raw)
	ran := func(commit string) boundRecord {
		return boundRecord{run: runrecord.Run{Outcome: runrecord.OutcomeSucceeded, CodeCommit: commit}, found: true}
	}
	records := map[string]boundRecord{
		older.ModelName: ran(beforeSurface), current.ModelName: ran(beforeDocs), unmeasured.ModelName: {},
		excluded.ModelName: {refusal: fmt.Errorf("%w: it declares dna", errCellInapplicable)},
	}
	cells := []ModelValidation{older, current, unmeasured, excluded}
	read := acceptedRecords{bound: map[boundKey]error{}}
	for _, cell := range cells {
		read.bound[boundKey{cell.Recipe, cell.Validation, cell.Protocol}] = recordCurrent(t.Context(), root, directories, cell, records[cell.ModelName])
	}
	plan := SelectValidation(Inputs{Registered: cells, Current: read.current})

	// No baseline scorecard compares any of them, so those that run are
	// background.
	ranCells := map[string]Selected{}
	for _, selected := range plan.Background {
		ranCells[selected.ModelName] = selected
	}
	if len(plan.Run) != 0 || len(plan.Background) != 2 || !strings.Contains(ranCells[older.ModelName].Reason, "evaluation surface moved since") ||
		!strings.Contains(ranCells[unmeasured.ModelName].Reason, "no mmlu-pro record") {
		t.Fatalf("the stale and the unmeasured benchmark did not run in the background: run=%+v background=%+v", plan.Run, plan.Background)
	}
	if len(plan.Reuse) != 1 || plan.Reuse[0].Model != current.Model {
		t.Fatalf("a benchmark bound to the current surface was not reused: reuse=%+v run=%+v", plan.Reuse, plan.Run)
	}
	if len(plan.SkipInappropriate) != 1 || plan.SkipInappropriate[0].Model != excluded.Model || len(plan.Unknown) != 0 {
		t.Fatalf("an excluded benchmark was not skipped: skip=%+v unknown=%+v", plan.SkipInappropriate, plan.Unknown)
	}

	for _, cell := range []ModelValidation{older, unmeasured} {
		name, args, err := reacquireCommand(ranCells[cell.ModelName].ModelValidation, store)
		want := []string{
			"run", "./cmd/evaluate", "-all", "-worker", "-repo", store, "-model-path", cell.Location,
			"-family", "mmlu-pro", "-budget", benchmarkBudget.String(),
		}
		if cell.Protocol == chat {
			want = append(want, "-chat-protocol")
		}
		if err != nil || name != "go" || !slices.Equal(args, want) {
			t.Fatalf("%s bound (%q, %v, %v), want go %v", cell.ModelName, name, args, err, want)
		}
	}

	var models []modelDirectory
	for _, cell := range cells {
		models = append(models, modelDirectory{Path: "C:/models/" + cell.ModelName, Weights: true})
	}
	standings := map[string]string{}
	for _, entry := range buildInventory(models, cells, nil, read.current, store) {
		standings[strings.TrimPrefix(entry.Directory, "C:/models/")] = entry.Standing
	}
	want := map[string]string{
		older.ModelName: standingActivated, current.ModelName: standingValidated,
		unmeasured.ModelName: standingActivated, excluded.ModelName: standingActivated,
	}
	if !maps.Equal(standings, want) {
		t.Fatalf("the inventory disagrees with the plan: %v, want %v", standings, want)
	}
}
