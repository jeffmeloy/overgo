package main

import (
	"fmt"
	"maps"
	"slices"
	"strings"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/evaluation"
	"overgo/internal/recipe"
	"overgo/internal/runrecord"
	"overgo/internal/testutil"
)

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
		read.bound[boundKey{cell.Recipe, cell.Validation}] = recordCurrent(t.Context(), root, directories, cell, records[cell.ModelName])
	}
	plan := SelectValidation(Inputs{AffectedSurfaces: map[SurfaceID]AffectRecord{"evaluation": {}}, Registered: cells, Current: read.current})

	ranCells := map[string]Selected{}
	for _, selected := range plan.Run {
		ranCells[selected.ModelName] = selected
	}
	if len(plan.Run) != 2 || !strings.Contains(ranCells[older.ModelName].Reason, "evaluation surface moved since") ||
		!strings.Contains(ranCells[unmeasured.ModelName].Reason, "no mmlu-pro record") {
		t.Fatalf("the stale and the unmeasured benchmark did not run: run=%+v", plan.Run)
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
