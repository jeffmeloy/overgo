package workflowruntime

import (
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/artifact/repositorytest"
	"overgo/internal/overgodbtest"
	"overgo/internal/recipe"
	"overgo/internal/testcheck"
	"overgo/internal/testutil"
	"overgo/internal/workflowrecipe"
)

// TestSemanticTransitionUsesOneCommit proves execution-authority
// publication -- the recipe content plus every dependency identity --
// lands as exactly one atomic batch, and replays idempotently instead
// of depending on interleaved store state.
func TestSemanticTransitionUsesOneCommit(t *testing.T) {
	store := overgodbtest.Open(t, t.TempDir())
	batch := recipe.Node{ID: "batch", Module: workflowrecipe.ModuleBatchDataset, Placement: recipe.PlacementHost}
	definition, err := recipe.NewDefinitionWithDependencies(
		recipe.TaskTraining,
		[]recipe.Dependency{
			{Role: recipe.DependencyModel, Artifact: testutil.ArtifactID(t, artifact.KindModel, "authority-model")},
			{Role: recipe.DependencyDataset, Artifact: testutil.ArtifactID(t, artifact.KindDataset, "authority-dataset")},
		},
		[]recipe.Node{batch}, nil, nil,
		[]recipe.Output{{Name: "batch", Data: recipe.DataBatch, Source: recipe.Endpoint{Node: batch.ID, Port: "batch"}}},
	)
	testcheck.NoError(t, err)
	counting := &repositorytest.CountingRepository{Repository: store}
	runtime := &Runtime{store: counting}
	operation := testutil.ArtifactID(t, artifact.KindEvidence, "authority-operation")
	testcheck.NoError(t, runtime.publishExecutionAuthority(t.Context(), definition, operation))
	if counting.Commits != 1 {
		t.Fatalf("authority publication used %d commits, want 1", counting.Commits)
	}
	if err := runtime.publishExecutionAuthority(t.Context(), definition, operation); err != nil {
		t.Fatalf("authority republication must replay idempotently: %v", err)
	}
	if counting.Commits != 2 {
		t.Fatalf("republication used %d total commits, want 2", counting.Commits)
	}
}
