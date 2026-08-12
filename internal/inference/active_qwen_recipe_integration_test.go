package inference

import (
	"context"
	"os"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/gguf"
	"overgo/internal/model"
	"overgo/internal/modelartifact"
	"overgo/internal/modelrecipe"
	"overgo/internal/modelrecipetest"
	"overgo/internal/recipe"
	"overgo/internal/repodb"
)

func TestActiveRecipeQwen35Open(t *testing.T) {
	path := os.Getenv("OVERGO_QWEN35_MODEL")
	if path == "" {
		t.Skip("OVERGO_QWEN35_MODEL is not set")
	}
	store, err := repodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	definition := publishActiveGGUFRecipe(t, store, path)
	loaded, err := modelrecipe.ResolveActiveGGUF(context.Background(), store, path)
	if err != nil {
		t.Fatal(err)
	}
	identity, identityOK := loaded.Identity()
	architecture, architectureOK := loaded.Architecture()
	if !identityOK || !architectureOK || identity.Recipe != definition.ID || architecture != "qwen35" {
		_ = loaded.Close()
		t.Fatalf("resolved Qwen3.5 program = %+v", identity)
	}
	dense, recurrent := false, false
	layers, ok := loaded.LayerPlans()
	if !ok {
		_ = loaded.Close()
		t.Fatal("resolved Qwen3.5 model plan is unavailable")
	}
	for _, layer := range layers {
		dense = dense || !layer.Recurrent
		recurrent = recurrent || layer.Recurrent
	}
	if !dense || !recurrent {
		_ = loaded.Close()
		t.Fatalf("Qwen3.5 layer program lacks dense/recurrent coverage")
	}
	runner, err := OpenWithProgram(&loaded, OpenOptions{})
	if err != nil {
		_ = loaded.Close()
		t.Fatal(err)
	}
	if runner.EvidenceTier() != recipe.EvidenceExperimental {
		_ = runner.Close()
		t.Fatalf("Qwen3.5 evidence tier = %q", runner.EvidenceTier())
	}
	if err := runner.Close(); err != nil {
		t.Fatal(err)
	}
}

func publishActiveGGUFRecipe(
	t *testing.T,
	store artifact.Repository,
	path string,
) recipe.Definition {
	t.Helper()
	ctx := context.Background()
	file, err := gguf.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer file.Close()
	inventory, err := modelartifact.FromGGUF(file)
	if err != nil {
		t.Fatal(err)
	}
	spec, err := model.ReadSpec(file)
	if err != nil {
		t.Fatal(err)
	}
	profile, ok := model.LookupArchitecture(spec.Architecture)
	if !ok {
		t.Fatalf("profile %q is unavailable", spec.Architecture)
	}
	profileDocument, err := modelrecipe.NewProfileDocument(profile)
	if err != nil {
		t.Fatal(err)
	}
	document, err := modelrecipe.NewModelDefinitionDocument(profileDocument, inventory.TensorInventory, spec)
	if err != nil {
		t.Fatal(err)
	}
	resolved, err := document.Resolve(profileDocument, inventory.TensorInventory)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := modelrecipe.PublishResolvedModelDefinition(ctx, store, "integration/qwen35/facts", inventory, resolved); err != nil {
		t.Fatal(err)
	}
	definition, err := modelrecipe.InferenceWithModelDefinition(
		inventory.Manifest.ID, profileDocument.ID, document.ID, recipe.PlacementHybrid,
		modelrecipe.DecodeSessionCapacity,
	)
	if err != nil {
		t.Fatal(err)
	}
	if _, _, err := modelrecipe.PublishCandidate(ctx, store, "integration/qwen35/candidate", definition); err != nil {
		t.Fatal(err)
	}
	if _, _, err := modelrecipe.Transition(ctx, store, "integration/qwen35/validated", definition, recipe.StatusValidated, nil, nil); err != nil {
		t.Fatal(err)
	}
	verification, err := modelrecipetest.PublishVerification(
		ctx, store, "integration/qwen35/verification", definition.ID,
	)
	if err != nil {
		t.Fatal(err)
	}
	if _, _, err := modelrecipe.ActivateVerified(
		ctx, store, "integration/qwen35/active", definition, verification, nil, nil,
	); err != nil {
		t.Fatal(err)
	}
	return definition
}
