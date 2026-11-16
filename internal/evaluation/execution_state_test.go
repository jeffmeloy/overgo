package evaluation

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/inference"
	"overgo/internal/modelrecipe"
	"overgo/internal/modelrecipetest"
	"overgo/internal/overgodb"
	"overgo/internal/overgodbtest"
	"overgo/internal/recipe"
	"overgo/internal/runrecord"
	"overgo/internal/testcheck"
	"overgo/internal/testutil"
	"overgo/internal/tokenizer"
)

// activatingRuntime is a runtime that, as it answers its first case, runs activate: an activation of the recipe the
// campaign evaluates, made while its run executes.
type activatingRuntime struct {
	observedRuntime
	activate func()
}

func (r *activatingRuntime) Generate(ctx context.Context, prompt string, options inference.GenerateOptions) ([]tokenizer.TokenID, string, error) {
	if r.activate != nil {
		r.activate()
		r.activate = nil
	}
	return r.observedRuntime.Generate(ctx, prompt, options)
}

// unreadRecipeStore is a store that cannot say what its recipe is: every read of it fails.
type unreadRecipeStore struct {
	*overgodb.Store
	recipe artifact.ID
}

func (s unreadRecipeStore) OpenContent(ctx context.Context, id artifact.ID) (artifact.Descriptor, io.Reader, bool, error) {
	if id == s.recipe {
		return artifact.Descriptor{}, nil, false, errors.New("the store cannot read the recipe")
	}
	return s.Store.OpenContent(ctx, id)
}

// TestAPublishedEvaluationHoldsTheStateItsExecutionRead holds a held-out reading to the recipe state its run executed
// under (execution-state/do, SQA-283): the campaign read the state at publication and dropped a failed read, so a
// candidate activated while its run executed published that run verified, and a state the store could not say left
// the reading with none, which the scorecard took for a reading from before states were kept. A candidate whose
// verification and activation come as its first case is answered publishes its reading not verified, and a candidate
// whose state the store cannot read publishes runrecord.StateUnread, which earns no verified row.
func TestAPublishedEvaluationHoldsTheStateItsExecutionRead(t *testing.T) {
	t.Parallel()
	ctx := t.Context()
	store := overgodbtest.Open(t, t.TempDir())
	weights := []byte("execution-state-weights")
	weightsID := testutil.ArtifactBytesID(t, artifact.KindTensorSet, weights)
	manifest := testcheck.Must(artifact.NewManifest(artifact.KindModel, []artifact.Component{{
		Role: artifact.ComponentWeights, Name: "weights", Artifact: weightsID,
	}}))(t)
	definition := planID(t, artifact.KindModelDefinition, "execution-state definition")
	testcheck.Must(store.Commit(ctx, artifact.Batch{Key: "state/model", Artifacts: []artifact.Descriptor{{ID: weightsID, Size: uint64(len(weights))}, {ID: definition}},
		Manifests: []artifact.Manifest{manifest}}))(t)
	candidate := testcheck.Must(modelrecipe.CapabilityDefinition(recipe.TaskGeneration, manifest.ID))(t)
	if _, _, err := modelrecipe.PublishCandidate(ctx, store, "state/candidate", candidate); err != nil {
		t.Fatal(err)
	}
	environment := testcheck.Must(runrecord.NewEnvironment(runrecord.Environment{Host: "fixture", OS: "fixture", Arch: "fixture", Device: "fixture", Backend: "fixture", Driver: "fixture"}))(t)
	identity := modelrecipe.ProgramIdentity{Model: manifest.ID, Definition: definition, Recipe: candidate.ID}
	published := func(repository artifact.Repository, runtime Runtime) string {
		t.Helper()
		campaign := testcheck.Must(newCampaign(repository, runtime, identity, environment, runtimeLedgerCommit, LifecycleIsolated))(t)
		suite := testcheck.Must(CompileSuite(testcheck.Must(json.Marshal(exactFixture()))(t), campaign.Authorities()))(t)
		result := testcheck.Must(campaign.Evaluate(ctx, suite))(t)
		return testcheck.Must(runrecord.RequireEvaluation(ctx, store, result.Evaluation))(t).Verification
	}

	unread := published(unreadRecipeStore{Store: store, recipe: candidate.ID}, &observedRuntime{})
	if unread != runrecord.StateUnread {
		t.Errorf("a reading whose recipe's state the store could not read was published %q; want %q", unread, runrecord.StateUnread)
	}
	activating := &activatingRuntime{activate: func() {
		verification := testcheck.Must(modelrecipetest.PublishVerification(ctx, store, "state/verification", candidate.ID))(t)
		testcheck.NoError(t, modelrecipe.ActivateCapability(ctx, store, candidate, verification, recipe.EvidenceParity, "activated while the run executed"))
	}}
	if state := published(store, activating); state != string(modelrecipe.StateNotVerified) {
		t.Errorf("a run executed before its recipe was activated was published %q; want %q, the state it ran under", state, modelrecipe.StateNotVerified)
	}
}
