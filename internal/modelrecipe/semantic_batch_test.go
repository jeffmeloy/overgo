package modelrecipe

import (
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/artifact/repositorytest"
	"overgo/internal/overgodb"
	"overgo/internal/recipe"
	"overgo/internal/testcheck"
	"overgo/internal/testutil"
)

// TestSemanticTransitionUsesOneCommit proves candidate publication --
// definition, lifecycle event, lineage, and status alias together --
// lands as exactly one atomic batch.
func TestSemanticTransitionUsesOneCommit(t *testing.T) {
	ctx := t.Context()
	store, err := overgodb.Open(t.TempDir())
	testcheck.NoError(t, err)
	t.Cleanup(func() { _ = store.Close() })
	weights := []byte("semantic-batch-weights")
	weightsID := testutil.ArtifactBytesID(t, artifact.KindTensorSet, weights)
	model, err := artifact.NewManifest(artifact.KindModel, []artifact.Component{{
		Role: artifact.ComponentWeights, Name: "weights", Artifact: weightsID,
	}})
	testcheck.NoError(t, err)
	if _, err := store.Commit(ctx, artifact.Batch{
		Key:       "semantic-batch/artifacts",
		Artifacts: []artifact.Descriptor{{ID: weightsID, Size: uint64(len(weights))}},
		Manifests: []artifact.Manifest{model},
	}); err != nil {
		t.Fatal(err)
	}
	definition, err := CapabilityDefinition(recipe.TaskGeneration, model.ID)
	testcheck.NoError(t, err)
	counting := &repositorytest.CountingRepository{Repository: store}
	if _, _, err := PublishCandidate(ctx, counting, "semantic-batch/candidate", definition); err != nil {
		t.Fatal(err)
	}
	if counting.Commits != 1 {
		t.Fatalf("candidate publication used %d commits, want 1", counting.Commits)
	}
}
