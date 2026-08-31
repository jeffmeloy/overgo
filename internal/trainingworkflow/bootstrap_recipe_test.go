package trainingworkflow

import (
	"os"
	"path/filepath"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/overgodb"
	"overgo/internal/recipe"
	"overgo/internal/trainingprogram"
)

// TestBootstrapRecordsModelLocation pins the discoverability contract:
// bootstrapping a recipe from a weights file records that file's location
// for the identified model, so every bootstrapped activation resolves to
// its bytes without a separate intake; the rerun keeps the recorded fact.
func TestBootstrapRecordsModelLocation(t *testing.T) {
	root := t.TempDir()
	modelPath := filepath.Join(root, "model.gguf")
	datasetPath := filepath.Join(root, "dataset.txt")
	if err := os.WriteFile(modelPath, []byte("weights"), 0o600); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(datasetPath, []byte("training data"), 0o600); err != nil {
		t.Fatal(err)
	}
	store, err := overgodb.Open(filepath.Join(root, "repodb"))
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()

	ctx := t.Context()
	if _, err := BootstrapTokenRecipe(ctx, store, modelPath, datasetPath); err != nil {
		t.Fatal(err)
	}
	modelID, err := artifact.IdentifyBytes(artifact.KindModel, []byte("weights"))
	if err != nil {
		t.Fatal(err)
	}
	recorded, err := artifact.AvailablePath(ctx, store, modelID, artifact.LocationFile)
	if err != nil {
		t.Fatalf("bootstrapped model has no recorded weights location: %v", err)
	}
	data, err := os.ReadFile(recorded)
	if err != nil || string(data) != "weights" {
		t.Fatalf("recorded location %s = (%q, %v), want the bootstrapped weights", recorded, data, err)
	}
	if _, err := BootstrapTokenRecipe(ctx, store, modelPath, datasetPath); err != nil {
		t.Fatalf("repeat bootstrap with a recorded location: %v", err)
	}
}

func TestBootstrapTokenRecipeBindsStoredOptimizerPolicy(t *testing.T) {
	root := t.TempDir()
	modelPath := filepath.Join(root, "model.gguf")
	datasetPath := filepath.Join(root, "dataset.txt")
	if err := os.WriteFile(modelPath, []byte("model"), 0o600); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(datasetPath, []byte("training data"), 0o600); err != nil {
		t.Fatal(err)
	}
	store, err := overgodb.Open(filepath.Join(root, "repodb"))
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()

	ctx := t.Context()
	definitionID, err := BootstrapTokenRecipe(ctx, store, modelPath, datasetPath)
	if err != nil {
		t.Fatal(err)
	}
	content, ok, err := artifact.ReadContent(ctx, store, definitionID)
	if err != nil {
		t.Fatal(err)
	}
	if !ok {
		t.Fatalf("recipe definition %s was not stored", definitionID)
	}
	definition, err := recipe.ParseDefinition(content.Data)
	if err != nil {
		t.Fatal(err)
	}
	optimizerID, ok := definition.PrimaryDependency(recipe.DependencyOptimizer)
	if !ok {
		t.Fatal("bootstrap recipe has no optimizer policy dependency")
	}
	if optimizerID != trainingprogram.BuiltinOptimizerPolicy().ID {
		t.Fatalf("optimizer policy = %s, want %s", optimizerID, trainingprogram.BuiltinOptimizerPolicy().ID)
	}
	if _, err := trainingprogram.RequireOptimizerPolicy(ctx, store, optimizerID); err != nil {
		t.Fatalf("load optimizer policy: %v", err)
	}

	repeatedID, err := BootstrapTokenRecipe(ctx, store, modelPath, datasetPath)
	if err != nil {
		t.Fatalf("repeat bootstrap: %v", err)
	}
	if repeatedID != definitionID {
		t.Fatalf("repeat recipe = %s, want %s", repeatedID, definitionID)
	}
}
