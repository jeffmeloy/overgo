package modelrecipe

import (
	"context"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/artifact/repositorytest"
	"overgo/internal/overgodb"
	"overgo/internal/recipe"
	"overgo/internal/testutil"
)

// TestSemanticTransitionUsesOneCommit proves candidate publication --
// definition, lifecycle event, lineage, and status alias together --
// lands as exactly one atomic batch.
func TestSemanticTransitionUsesOneCommit(t *testing.T) {
	ctx := context.Background()
	store, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = store.Close() })
	weights := []byte("semantic-batch-weights")
	weightsID := testutil.ArtifactBytesID(t, artifact.KindTensorSet, weights)
	model, err := artifact.NewManifest(artifact.KindModel, []artifact.Component{{
		Role: artifact.ComponentWeights, Name: "weights", Artifact: weightsID,
	}})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := store.Commit(ctx, artifact.Batch{
		Key:       "semantic-batch/artifacts",
		Artifacts: []artifact.Descriptor{{ID: weightsID, Size: uint64(len(weights))}},
		Manifests: []artifact.Manifest{model},
	}); err != nil {
		t.Fatal(err)
	}
	definition, err := CapabilityDefinition(recipe.TaskGeneration, model.ID)
	if err != nil {
		t.Fatal(err)
	}
	counting := &repositorytest.CountingRepository{Repository: store}
	if _, _, err := PublishCandidate(ctx, counting, "semantic-batch/candidate", definition); err != nil {
		t.Fatal(err)
	}
	if counting.Commits != 1 {
		t.Fatalf("candidate publication used %d commits, want 1", counting.Commits)
	}
}
