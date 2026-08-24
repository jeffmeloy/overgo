package modelrecipe

import (
	"context"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/recipe"
	"overgo/internal/repodb"
	"overgo/internal/testutil"
)

func TestCapabilityBundleRecipeSelection(t *testing.T) {
	ctx := context.Background()
	store, modelID, definition, bundle := capabilityBundleFixture(t)
	verification := publishVerification(t, store, definition.ID, "bundle/verification")
	if err := ActivateCapability(ctx, store, definition, verification, recipe.EvidenceParity, "bundle fixture"); err != nil {
		t.Fatal(err)
	}
	selection, err := ResolveActiveExecution(ctx, store, modelID, recipe.TaskGeneration, SessionWarm)
	if err != nil {
		t.Fatal(err)
	}
	if len(selection.Bundles) != 1 || selection.Bundles[0].Manifest.ID != bundle.ID ||
		selection.Bundles[0].Evidence != selection.Activation.Event.ID {
		t.Fatalf("bundle selection = %+v", selection.Bundles)
	}
}

func TestUnpromotedBundleRefusal(t *testing.T) {
	ctx := context.Background()
	store, modelID, _, _ := capabilityBundleFixture(t)
	if _, err := ResolveActiveExecution(ctx, store, modelID, recipe.TaskGeneration, SessionWarm); err == nil {
		t.Fatal("unpromoted bundle recipe accepted")
	}
}

func capabilityBundleFixture(t *testing.T) (*repodb.Store, artifact.ID, recipe.Definition, artifact.Manifest) {
	t.Helper()
	ctx := context.Background()
	store, err := repodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = store.Close() })
	weights := []byte("bundle-weights")
	weightsID := testutil.ArtifactBytesID(t, artifact.KindTensorSet, weights)
	model, err := artifact.NewManifest(artifact.KindModel, []artifact.Component{{
		Role: artifact.ComponentWeights, Name: "weights", Artifact: weightsID,
	}})
	if err != nil {
		t.Fatal(err)
	}
	instruction, err := (artifact.DocumentContract{
		Kind: artifact.KindFile, MediaType: "text/plain", Schema: "overgo/capability-instruction/v1",
	}).ContentBytes([]byte("Use exact artifact facts."))
	if err != nil {
		t.Fatal(err)
	}
	bundle, err := artifact.NewManifest(artifact.KindProfile, []artifact.Component{{
		Role: artifact.ComponentInstruction, Name: "system", Artifact: instruction.Descriptor.ID,
	}})
	if err != nil {
		t.Fatal(err)
	}
	if err := artifact.ValidateCapabilityBundle(bundle); err != nil {
		t.Fatal(err)
	}
	if _, err := store.Commit(ctx, artifact.Batch{
		Key:       "bundle/artifacts",
		Artifacts: []artifact.Descriptor{{ID: weightsID, Size: uint64(len(weights))}},
		Contents:  []artifact.Content{instruction},
		Manifests: []artifact.Manifest{model, bundle},
	}); err != nil {
		t.Fatal(err)
	}
	base, err := CapabilityDefinition(recipe.TaskGeneration, model.ID)
	if err != nil {
		t.Fatal(err)
	}
	dependencies := append(base.Dependencies, recipe.Dependency{
		Role: recipe.DependencyCapabilityBundle, Artifact: bundle.ID,
	})
	definition, err := recipe.NewDefinitionWithDependencies(
		base.Task, dependencies, base.Nodes, base.Edges, base.Inputs, base.Outputs,
	)
	if err != nil {
		t.Fatal(err)
	}
	if _, _, err := PublishCandidate(ctx, store, "bundle/candidate", definition); err != nil {
		t.Fatal(err)
	}
	return store, model.ID, definition, bundle
}
