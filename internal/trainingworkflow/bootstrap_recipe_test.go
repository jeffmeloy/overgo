package trainingworkflow

import (
	"context"
	"os"
	"path/filepath"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/recipe"
	"overgo/internal/repodb"
	"overgo/internal/trainingprogram"
)

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
	store, err := repodb.Open(filepath.Join(root, "repodb"))
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()

	ctx := context.Background()
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
