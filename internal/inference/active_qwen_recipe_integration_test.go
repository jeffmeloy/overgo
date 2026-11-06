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
	"overgo/internal/overgodbtest"
	"overgo/internal/recipe"
	"overgo/internal/testcheck"
)

func TestActiveRecipeQwen35Open(t *testing.T) {
	requireIntegration(t)
	cudatest.Require(t)
	path := modeltest.GGUF(t, "qwen35")
	store := overgodbtest.Open(t, t.TempDir())
	definition := publishActiveGGUFRecipe(t, store, path)
	loaded := testcheck.Must(modelrecipe.ResolveActiveGGUF(t.Context(), store, path))(t)
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
	file := testcheck.Must(gguf.Open(path))(t)
	defer file.Close()
	inventory := testcheck.Must(modelartifact.FromGGUF(file, artifact.KindModel))(t)
	spec := testcheck.Must(model.ReadSpec(file))(t)
	profile, ok := model.LookupArchitecture(spec.Architecture)
	if !ok {
		t.Fatalf("profile %q is unavailable", spec.Architecture)
	}
	profileDocument := testcheck.Must(modelrecipe.NewProfileDocument(profile))(t)
	document := testcheck.Must(modelrecipe.NewModelDefinitionDocument(profileDocument, inventory.TensorInventory, spec))(t)
	resolved := testcheck.Must(document.Resolve(profileDocument, inventory.TensorInventory))(t)
	testcheck.NoError(t, modelrecipe.PublishResolvedModelDefinition(ctx, store, inventory, resolved))
	definition := testcheck.Must(modelrecipe.InferenceWithModelDefinition(
		inventory.Manifest.ID, profileDocument.ID, document.ID, recipe.PlacementHybrid,
		modelrecipe.DecodeSessionCapacity, recipe.ResidencyHybridNative,
	))(t)
	if _, _, err := modelrecipe.PublishCandidate(ctx, store, "integration/qwen35/candidate", definition); err != nil {
		t.Fatal(err)
	}
	if _, _, err := modelrecipe.Transition(ctx, store, "integration/qwen35/validated", definition, recipe.StatusValidated, nil, nil); err != nil {
		t.Fatal(err)
	}
	verification := testcheck.Must(modelrecipetest.PublishVerification(
		ctx, store, "integration/qwen35/verification", definition.ID,
	))(t)
	if _, _, err := modelrecipe.ActivateVerified(
		ctx, store, "integration/qwen35/active", definition, verification,
		recipe.EvidenceVerified, "inference integration activation", nil, nil,
	); err != nil {
		t.Fatal(err)
	}
	return definition
}
