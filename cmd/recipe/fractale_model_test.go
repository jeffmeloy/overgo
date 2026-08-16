//go:build modeltest

package main

import (
	"context"
	"os"
	"testing"
	"time"

	"overgo/internal/modelrecipe"
	"overgo/internal/recipe"
	"overgo/internal/repodb"
	"overgo/internal/thoughtbank"
)

const fractaleModelDirectory = `C:\Users\jeffm\adaptive_new\models\Fractale-350M-base`

func TestFractaleRecipeActivation(t *testing.T) {
	modelPath := os.Getenv("OVERGO_THOUGHTBANK_MODEL")
	if modelPath == "" {
		modelPath = fractaleModelDirectory
	}
	if _, err := os.Stat(modelPath); err != nil {
		t.Skipf("Fractale checkpoint unavailable: %v", err)
	}
	ctx := context.Background()
	store, err := repodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	capability := capabilities[recipe.TaskGeneration]
	modelID, definition, err := prepareCapability(ctx, store, modelPath, recipe.TaskGeneration, capability)
	if err != nil {
		t.Fatal(err)
	}
	if _, _, err := modelrecipe.PublishCandidate(ctx, store, "fixture/fractale/candidate", definition); err != nil {
		t.Fatal(err)
	}
	program, err := modelrecipe.CompileCapability(definition)
	if err != nil {
		t.Fatal(err)
	}
	output, err := capability.execute(
		ctx, store, modelPath, modelID, program,
		`{"text":"def fibonacci(n):","max_tokens":8}`,
	)
	if err != nil {
		t.Fatal(err)
	}
	generation, ok := output.(thoughtbank.Generation)
	if !ok {
		t.Fatalf("generation type = %T", output)
	}
	const reference = "\n    \"\"\"\n    Returns the number of iterations"
	if generation.Text != reference || len(generation.Tokens) != 8 {
		t.Fatalf("generation = %q/%v", generation.Text, generation.Tokens)
	}
	verification, err := publishCapabilityVerification(
		ctx, store, definition, "0123456789abcdef0123456789abcdef01234567", time.Millisecond,
		"host", "go", "real Fractale generation matched adaptive_new reference",
	)
	if err != nil {
		t.Fatal(err)
	}
	if err := modelrecipe.ActivateCapability(
		ctx, store, definition, verification, recipe.EvidenceParity, "real checkpoint completion matched",
	); err != nil {
		t.Fatal(err)
	}
	activation, activeProgram, err := modelrecipe.ResolveActiveCapability(ctx, store, modelID, recipe.TaskGeneration)
	if err != nil {
		t.Fatal(err)
	}
	if activation.Definition.ID != definition.ID || activeProgram.Definition().ID != definition.ID {
		t.Fatalf("active recipe = %s/%s, want %s", activation.Definition.ID, activeProgram.Definition().ID, definition.ID)
	}
}
