//go:build modeltest

package inference

import (
	"testing"

	"overgo/internal/artifact"
	cudatest "overgo/internal/cuda/testutil"
	"overgo/internal/gguf"
	"overgo/internal/model"
	"overgo/internal/modelartifact"
	"overgo/internal/modelrecipe"
	"overgo/internal/modelrecipetest"
	"overgo/internal/modeltest"
	"overgo/internal/overgodb"
	"overgo/internal/recipe"
	"overgo/internal/testcheck"
)

func TestActiveRecipeQwen35Open(t *testing.T) {
	requireIntegration(t)
	cudatest.Require(t)
	path := modeltest.GGUF(t, "qwen35")
	store, err := overgodb.Open(t.TempDir())
	testcheck.NoError(t, err)
	defer store.Close()
	definition := publishActiveGGUFRecipe(t, store, path)
	loaded, err := modelrecipe.ResolveActiveGGUF(t.Context(), store, path)
	testcheck.NoError(t, err)
	runner, err := OpenWithProgram(t.Context(), &loaded, OpenOptions{})
	if err != nil {
		_ = loaded.Close()
		t.Fatal(err)
	}
	if runner.program.Identity.Recipe != definition.ID || runner.spec.Architecture != "qwen35" {
		_ = runner.Close()
		t.Fatalf("resolved Qwen3.5 program = %+v", runner.program.Identity)
	}
	dense, recurrent := false, false
	for index := range runner.program.Model.LayerCount() {
		layer, layerErr := runner.program.Model.Layer(uint32(index))
		if layerErr != nil {
			t.Fatal(layerErr)
		}
		dense = dense || !layer.Recurrent
		recurrent = recurrent || layer.Recurrent
	}
	if !dense || !recurrent {
		_ = runner.Close()
		t.Fatal("Qwen3.5 layer program lacks dense/recurrent coverage")
	}
	if state := runner.ModelState(); state.Verification == modelrecipe.StateNotVerified || state.Verification == modelrecipe.StateBroken {
		_ = runner.Close()
		t.Fatalf("Qwen3.5 state = %+v", state)
	}
	testcheck.NoError(t, runner.Close())
}

func publishActiveGGUFRecipe(
	t *testing.T,
	store artifact.Repository,
	path string,
) recipe.Definition {
	t.Helper()
	ctx := t.Context()
	file, err := gguf.Open(path)
	testcheck.NoError(t, err)
	defer file.Close()
	inventory, err := modelartifact.FromGGUF(file, artifact.KindModel)
	testcheck.NoError(t, err)
	spec, err := model.ReadSpec(file)
	testcheck.NoError(t, err)
	profile, ok := model.LookupArchitecture(spec.Architecture)
	if !ok {
		t.Fatalf("profile %q is unavailable", spec.Architecture)
	}
	profileDocument, err := modelrecipe.NewProfileDocument(profile)
	testcheck.NoError(t, err)
	document, err := modelrecipe.NewModelDefinitionDocument(profileDocument, inventory.TensorInventory, spec)
	testcheck.NoError(t, err)
	resolved, err := document.Resolve(profileDocument, inventory.TensorInventory)
	testcheck.NoError(t, err)
	testcheck.NoError(t, modelrecipe.PublishResolvedModelDefinition(ctx, store, inventory, resolved))
	definition, err := modelrecipe.InferenceWithModelDefinition(
		inventory.Manifest.ID, profileDocument.ID, document.ID, recipe.PlacementHybrid,
		modelrecipe.DecodeSessionCapacity, recipe.ResidencyHybridNative,
	)
	testcheck.NoError(t, err)
	if _, _, err := modelrecipe.PublishCandidate(ctx, store, "integration/qwen35/candidate", definition); err != nil {
		t.Fatal(err)
	}
	if _, _, err := modelrecipe.Transition(ctx, store, "integration/qwen35/validated", definition, recipe.StatusValidated, nil, nil); err != nil {
		t.Fatal(err)
	}
	verification, err := modelrecipetest.PublishVerification(
		ctx, store, "integration/qwen35/verification", definition.ID,
	)
	testcheck.NoError(t, err)
	if _, _, err := modelrecipe.ActivateVerified(
		ctx, store, "integration/qwen35/active", definition, verification,
		recipe.EvidenceVerified, "inference integration activation", nil, nil,
	); err != nil {
		t.Fatal(err)
	}
	return definition
}
