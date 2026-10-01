package main

import (
	"slices"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/overgodb"
	"overgo/internal/recipe"
	"overgo/internal/runrecord"
	"overgo/internal/testutil"
)

// TestServingCellRunsUntilAnAttemptTimesTheLoad holds an inference model's
// serving cell to the attempt the scorecard's serving cells read. A page's
// attempt runs on a model already loaded and times no load, so the cell still
// runs, and -run binds recipe measure on the model into the selected store;
// once a measured attempt timed the load beside its prefill and decode, the
// cell is reused.
func TestServingCellRunsUntilAnAttemptTimesTheLoad(t *testing.T) {
	t.Parallel()
	ctx := t.Context()
	store, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	model := testutil.ArtifactID(t, artifact.KindModel, "served")
	served := testutil.ArtifactID(t, artifact.KindRecipe, "served-inference")
	environment := testutil.ArtifactID(t, artifact.KindEvidence, "served-environment")
	for _, id := range []artifact.ID{model, served, environment} {
		testutil.PublishArtifact(t, store, id)
	}
	const selected = "C:/stores/selected"
	cell := ModelValidation{
		Model: model, Location: "C:/models/served.gguf", Kind: recipe.TaskInference,
		Validation: servingValidation, Surface: "inference", Recipe: served,
	}
	page := []runrecord.PhaseMetric{{Phase: runrecord.PhasePrefill, DurationNS: 1}, {Phase: runrecord.PhaseDecode, DurationNS: 1}}
	measurement := append(slices.Clone(page), runrecord.PhaseMetric{Phase: runrecord.PhaseLoad, DurationNS: 1})
	for index, phases := range [][]runrecord.PhaseMetric{page, measurement} {
		if _, err := runrecord.PublishServingObservation(ctx, store, runrecord.ServingObservation{
			Version: artifact.InitialDocumentVersion, Model: model, Recipe: served, Environment: environment, Task: recipe.TaskInference,
			Outcome: runrecord.OutcomeSucceeded, StartedUnixNS: int64(index + 1), MeasuredNS: uint64(len(phases)), Phases: phases,
		}); err != nil {
			t.Fatal(err)
		}
		measured, err := measuredRecipes(ctx, store)
		if err != nil {
			t.Fatal(err)
		}
		read := acceptedRecords{serving: measured}
		plan := SelectValidation(Inputs{AffectedSurfaces: map[SurfaceID]AffectRecord{"inference": {}}, Registered: []ModelValidation{cell}, Current: read.current})
		if index == 0 {
			if len(plan.Run) != 1 {
				t.Fatalf("a recipe only a page served was not selected to measure: run=%+v reuse=%+v", plan.Run, plan.Reuse)
			}
			name, args, err := reacquireCommand(plan.Run[0].ModelValidation, selected)
			want := []string{"run", "./cmd/recipe", "measure", "-repo", selected, "-task", string(recipe.TaskInference), cell.Location}
			if err != nil || name != "go" || !slices.Equal(args, want) {
				t.Fatalf("an unmeasured serving cell bound (%q, %v, %v), want go %v", name, args, err, want)
			}
			continue
		}
		if len(plan.Reuse) != 1 {
			t.Fatalf("a measured recipe was not reused: run=%+v reuse=%+v", plan.Run, plan.Reuse)
		}
	}
}
