package scorecard

import (
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/overgodb"
	"overgo/internal/recipe"
	"overgo/internal/runrecord"
	"overgo/internal/testutil"
)

// TestServingCountsWhatUsersHit holds the serving summary to the attempts
// users made: a failed attempt counts against its recipe, and one a gate
// result depends on -- a validation run's deliberate interrupt -- does not.
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
	observe := func(label string, started int64) runrecord.ServingObservation {
		t.Helper()
		observation, err := runrecord.NewServingObservation(runrecord.ServingObservation{
			Model: model, Recipe: recipeID, Environment: environment, Task: recipe.TaskInference,
			Outcome: runrecord.OutcomeFailed, Failure: "execution_failed", StartedUnixNS: started, MeasuredNS: 1,
		})
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
	observe("user", 1)
	interrupt := observe("interrupt", 2)
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
	if summary := summaries[recipeID]; summary == nil || summary.attempts != 1 || summary.failures != 1 {
		t.Fatalf("serving summary = %+v, want the user's one failed attempt and not the validation run's", summary)
	}
}
