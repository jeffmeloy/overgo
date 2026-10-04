package main

import (
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

// TestBenchmarkDoesNotReuseUnknownExecution makes missing execution authority
// bind a fresh measurement, while a complete real campaign is the record its
// re-run replays: the cell takes that record's protocol, cost and identity,
// and its command names the identity after -replay. A benchmark its model's
// declared domains exclude is inapplicable and binds no record.
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
	cell := ModelValidation{Model: model, Location: "C:/models/m.gguf", Recipe: authorities.RuntimeRecipe, Kind: recipe.TaskInference, Validation: "mmlu"}
	read := func(cell ModelValidation) ModelValidation {
		t.Helper()
		benchmarks := benchmarkEvidence{suites: []evaluation.CompiledSuite{suite}, evidence: evaluation.LatestEvidence(t.Context(), store, map[artifact.ID]string{suite.Descriptor().Dataset: "store/mmlu"}, 0)}
		value, err := benchmarks.bind(t.Context(), store, cell)
		if err != nil {
			t.Fatal(err)
		}
		return value
	}
	raw := evaluation.PromptingRawCompletion.Label()
	if value := read(cell); value.Replay.Valid() || value.Cost != int64(benchmarkBudget) || value.Protocol != raw {
		t.Errorf("unknown execution was replayed: %+v, want a fresh raw measurement without replay", value)
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
	run, err := runrecord.RequireRun(t.Context(), store, fresh.Run)
	if err != nil {
		t.Fatal(err)
	}
	bound := read(cell)
	if bound.Replay != fresh.Evaluation || bound.Cost != int64(run.MeasuredNS) || bound.Protocol != raw || scored != 1 {
		t.Fatalf("known real execution was not retained: %+v, scored=%d", bound, scored)
	}
	_, args, err := reacquireCommand(bound, "C:/stores/selected")
	if at := slices.Index(args, "-replay"); err != nil || at < 0 || args[at+1] != fresh.Evaluation.String() {
		t.Fatalf("a benchmark with a succeeded record bound %v, %v, want -replay %s", args, err, fresh.Evaluation)
	}

	// A DNA model's declaration admits no text suite: its cell is inapplicable
	// whatever its recipe recorded, and binds no replay.
	cell.Modality = dnaModality
	if value := read(cell); value.Inapplicable == "" || value.Replay.Valid() {
		t.Errorf("a benchmark the model's declaration excludes was bound: %+v", value)
	}
}
