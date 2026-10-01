package scorecard

import (
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/evaluation"
	"overgo/internal/overgodb"
	"overgo/internal/recipe"
	"overgo/internal/runrecord"
	"overgo/internal/testutil"
)

// TestServingCountsWhatUsersHit holds the serving summary to the attempts
// users made: a failed attempt counts against its recipe, and one a gate
// result depends on -- a validation run's deliberate interrupt -- does not.
// The cells read the summary as a user met it: one failure in two attempts,
// a whole-output capability's time to its output, and the device memory
// the success peaked at. Each keeps its origin and scope, the serving
// attempts of its recipe, so it compares only with the same population.
func TestServingCountsWhatUsersHit(t *testing.T) {
	t.Parallel()
	ctx := t.Context()
	store, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	model := testutil.ArtifactID(t, artifact.KindModel, "served-model")
	recipeID := testutil.ArtifactID(t, artifact.KindRecipe, "served-recipe")
	environment := testutil.ArtifactID(t, artifact.KindEvidence, "served-environment")
	if _, err := store.Commit(ctx, artifact.Batch{Key: "serving/authorities", Artifacts: []artifact.Descriptor{{ID: model}, {ID: recipeID}, {ID: environment}}}); err != nil {
		t.Fatal(err)
	}
	observe := func(label string, attempt runrecord.ServingObservation) runrecord.ServingObservation {
		t.Helper()
		attempt.Model, attempt.Recipe, attempt.Environment, attempt.Task = model, recipeID, environment, recipe.TaskImageGen
		observation, err := runrecord.NewServingObservation(attempt)
		if err != nil {
			t.Fatal(err)
		}
		batch, err := observation.Batch("serving/" + label)
		if err != nil {
			t.Fatal(err)
		}
		if _, err := store.Commit(ctx, batch); err != nil {
			t.Fatal(err)
		}
		return observation
	}
	const peak, wholeOutputNS = 3 << 20, 2_000_000
	observe("user", runrecord.ServingObservation{Outcome: runrecord.OutcomeFailed, Failure: "execution_failed", StartedUnixNS: 1, MeasuredNS: 1})
	observe("success", runrecord.ServingObservation{Outcome: runrecord.OutcomeSucceeded, StartedUnixNS: 2, MeasuredNS: wholeOutputNS,
		Resources: runrecord.ServingResources{PeakDeviceBytes: peak}})
	interrupt := observe("interrupt", runrecord.ServingObservation{Outcome: runrecord.OutcomeFailed, Failure: "execution_failed", StartedUnixNS: 3, MeasuredNS: 1})
	gate, err := runrecord.NewGateRecord(recipeID, environment, "0123456789abcdef0123456789abcdef01234567", runrecord.OutcomeSucceeded, "", 1,
		[]runrecord.GateStep{{Name: "interrupts", Phase: runrecord.PhaseTest, Outcome: runrecord.StepSucceeded, DurationNS: 1}})
	if err != nil {
		t.Fatal(err)
	}
	batch, err := gate.Batch("validation/interrupts")
	if err != nil {
		t.Fatal(err)
	}
	batch.Lineage = append(batch.Lineage, artifact.DependencyLineage(gate.Result.ID, interrupt.ID)...)
	if _, err := store.Commit(ctx, batch); err != nil {
		t.Fatal(err)
	}
	summaries, err := servingByRecipe(ctx, store)
	if err != nil {
		t.Fatal(err)
	}
	summary := summaries[recipeID]
	if summary == nil || summary.attempts != 2 || summary.failures != 1 {
		t.Fatalf("serving summary = %+v, want the user's two attempts and not the validation run's", summary)
	}
	var none evaluation.BenchmarkSummary
	if cell := failureCell(summary); cell.Value != 0.5 || cell.Contract.Origin != originServing || cell.Contract.Scope != recipeID {
		t.Fatalf("failure cell = %+v, want one failure in two attempts, scoped to the recipe's serving attempts", cell)
	}
	if cell := firstOutputCell(recipe.TaskImageGen, none, false, summary); cell.Value != milliseconds(wholeOutputNS) || cell.Unit != "ms" {
		t.Fatalf("first-output cell = %+v, want the whole output's time", cell)
	}
	if cell := peakMemoryCell(recipe.TaskImageGen, none, false, summary); cell.Value != peak {
		t.Fatalf("peak-memory cell = %+v, want %d bytes", cell, peak)
	}
}
