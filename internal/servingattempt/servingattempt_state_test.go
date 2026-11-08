package servingattempt

import (
	"slices"
	"testing"
	"time"

	"overgo/internal/artifact"
	"overgo/internal/modelrecipe"
	"overgo/internal/modelrecipetest"
	"overgo/internal/overgodbtest"
	"overgo/internal/recipe"
	"overgo/internal/runrecord"
	"overgo/internal/testcheck"
	"overgo/internal/testutil"
)

// TestAServingAttemptCarriesTheStateOfTheRecipeThatRan holds an attempt to the state of the recipe that served it
// (SQA-283): Record read the model's state for the attempt's task and ignored which recipe that state rests on, so an
// attempt by a candidate recipe was stamped with the active recipe's verification. An attempt by the active recipe is
// recorded verified, and one by a candidate recipe of the same task beside it not verified.
func TestAServingAttemptCarriesTheStateOfTheRecipeThatRan(t *testing.T) {
	t.Parallel()
	ctx := t.Context()
	store := overgodbtest.Open(t, t.TempDir())
	weights := []byte("serving-state-weights")
	weightsID := testutil.ArtifactBytesID(t, artifact.KindTensorSet, weights)
	manifest := testcheck.Must(artifact.NewManifest(artifact.KindModel, []artifact.Component{{
		Role: artifact.ComponentWeights, Name: "weights", Artifact: weightsID,
	}}))(t)
	testcheck.Must(store.Commit(ctx, artifact.Batch{Key: "state/model", Artifacts: []artifact.Descriptor{{ID: weightsID, Size: uint64(len(weights))}},
		Manifests: []artifact.Manifest{manifest}}))(t)
	active := testcheck.Must(modelrecipe.CapabilityDefinition(recipe.TaskGeneration, manifest.ID))(t)
	if _, _, err := modelrecipe.PublishCandidate(ctx, store, "state/candidate/active", active); err != nil {
		t.Fatal(err)
	}
	verification := testcheck.Must(modelrecipetest.PublishVerification(ctx, store, "state/verification", active.ID))(t)
	testcheck.NoError(t, modelrecipe.ActivateCapability(ctx, store, active, verification, recipe.EvidenceParity, "serving state fixture"))
	// A candidate of the same task beside the active recipe: the active one with an evaluation profile more.
	evaluation := testutil.ArtifactID(t, artifact.KindProfile, "serving-state-evaluation")
	testutil.PublishArtifact(t, store, evaluation)
	candidate := testcheck.Must(recipe.NewDefinitionWithDependencies(active.Task,
		append(slices.Clone(active.Dependencies), recipe.Dependency{Role: recipe.DependencyEvaluation, Artifact: evaluation}),
		active.Nodes, active.Edges, active.Inputs, active.Outputs))(t)
	if _, _, err := modelrecipe.PublishCandidate(ctx, store, "state/candidate/other", candidate); err != nil {
		t.Fatal(err)
	}
	environment := testcheck.Must(runrecord.CurrentEnvironment("process", "go"))(t)
	testcheck.Must(artifact.CommitBatch(ctx, store, testcheck.Must(environment.Batch("state/environment"))(t)))(t)

	attempt := func(served recipe.Definition) string {
		operation := testutil.ArtifactID(t, artifact.KindEvidence, "serving-state-operation-"+served.ID.String())
		testutil.PublishArtifact(t, store, operation)
		t.Helper()
		recorded := testcheck.Must(Record(ctx, store, runrecord.ServingObservation{
			Version: artifact.InitialDocumentVersion, Model: manifest.ID, Recipe: served.ID, Environment: environment.ID,
			Operation: operation, Task: served.Task, Outcome: runrecord.OutcomeSucceeded,
			StartedUnixNS: time.Now().UnixNano(), MeasuredNS: uint64(time.Millisecond),
		}))(t)
		return recorded.Verification
	}
	if state := attempt(active); state != string(modelrecipe.StateVerified) {
		t.Errorf("an attempt by the active recipe is recorded %q, want verified", state)
	}
	if state := attempt(candidate); state != string(modelrecipe.StateNotVerified) {
		t.Errorf("an attempt by a candidate recipe beside the active one is recorded %q, want not verified", state)
	}
}
