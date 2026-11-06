package modelrecipe

import (
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/overgodb"
	"overgo/internal/overgodbtest"
	"overgo/internal/recipe"
	"overgo/internal/testcheck"
	"overgo/internal/testutil"
)

func TestCapabilityBundleRecipeSelection(t *testing.T) {
	ctx := t.Context()
	store, modelID, definition, bundle := capabilityBundleFixture(t)
	verification := publishVerification(t, store, definition.ID, "bundle/verification")
	testcheck.NoError(t, ActivateCapability(ctx, store, definition, verification, recipe.EvidenceParity, "bundle fixture"))
	selection := testcheck.Must(ResolveActiveExecution(ctx, store, modelID, recipe.TaskGeneration, SessionWarm))(t)
	if len(selection.Bundles) != 1 || selection.Bundles[0].Manifest.ID != bundle.ID ||
		selection.Bundles[0].Evidence != selection.Activation.Event.ID {
		t.Fatalf("bundle selection = %+v", selection.Bundles)
	}
}

func TestUnpromotedBundleRefusal(t *testing.T) {
	ctx := t.Context()
	store, modelID, _, _ := capabilityBundleFixture(t)
	if _, err := ResolveActiveExecution(ctx, store, modelID, recipe.TaskGeneration, SessionWarm); err == nil {
		t.Fatal("unpromoted bundle recipe accepted")
	}
}

func capabilityBundleFixture(t *testing.T) (*overgodb.Store, artifact.ID, recipe.Definition, artifact.Manifest) {
	t.Helper()
	ctx := t.Context()
	store := overgodbtest.Open(t, t.TempDir())
	weights := []byte("bundle-weights")
	weightsID := testutil.ArtifactBytesID(t, artifact.KindTensorSet, weights)
	model := testcheck.Must(artifact.NewManifest(artifact.KindModel, []artifact.Component{{
		Role: artifact.ComponentWeights, Name: "weights", Artifact: weightsID,
	}}))(t)
	instruction := testcheck.Must((artifact.DocumentContract{
		Kind: artifact.KindFile, MediaType: "text/plain", Schema: "overgo/capability-instruction/v1",
	}).ContentBytes([]byte("Use exact artifact facts.")))(t)
	bundle := testcheck.Must(artifact.NewManifest(artifact.KindProfile, []artifact.Component{{
		Role: artifact.ComponentInstruction, Name: "system", Artifact: instruction.Descriptor.ID,
	}}))(t)
	testcheck.NoError(t, artifact.ValidateCapabilityBundle(bundle))
	if _, err := store.Commit(ctx, artifact.Batch{
		Key:       "bundle/artifacts",
		Artifacts: []artifact.Descriptor{{ID: weightsID, Size: uint64(len(weights))}},
		Contents:  []artifact.Content{instruction},
		Manifests: []artifact.Manifest{model, bundle},
	}); err != nil {
		t.Fatal(err)
	}
	base := testcheck.Must(CapabilityDefinition(recipe.TaskGeneration, model.ID))(t)
	dependencies := append(base.Dependencies, recipe.Dependency{
		Role: recipe.DependencyCapabilityBundle, Artifact: bundle.ID,
	})
	definition := testcheck.Must(recipe.NewDefinitionWithDependencies(
		base.Task, dependencies, base.Nodes, base.Edges, base.Inputs, base.Outputs,
	))(t)
	if _, _, err := PublishCandidate(ctx, store, "bundle/candidate", definition); err != nil {
		t.Fatal(err)
	}
	return store, model.ID, definition, bundle
}
