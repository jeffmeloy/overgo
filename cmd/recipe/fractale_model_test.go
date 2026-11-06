//go:build modeltest

package main

import (
	"context"
	"testing"
	"time"

	"overgo/internal/capabilityruntime"
	"overgo/internal/mediacapability"
	"overgo/internal/modelintake"
	"overgo/internal/modelrecipe"
	"overgo/internal/modeltest"
	"overgo/internal/overgodbtest"
	"overgo/internal/recipe"
	"overgo/internal/testcheck"
	"overgo/internal/thoughtbank"
)

func TestFractaleRecipeActivation(t *testing.T) {
	modelPath := modeltest.Directory(t, "Fractale-350M-base")
	ctx := context.Background()
	store := overgodbtest.Open(t, t.TempDir())
	capability := mediacapability.Catalog[recipe.TaskGeneration]
	modelID, definition, err := mediacapability.Prepare(ctx, store, modelPath, recipe.TaskGeneration, capability)
	testcheck.NoError(t, err)
	if _, _, err := modelrecipe.PublishCandidate(ctx, store, "fixture/fractale/candidate", definition); err != nil {
		t.Fatal(err)
	}
	program, err := modelrecipe.CompileCapability(definition)
	testcheck.NoError(t, err)
	execution, err := modelrecipe.CompileCandidateExecution(ctx, store, program)
	testcheck.NoError(t, err)
	output, err := capability.Execute(
		ctx, store, modelPath, execution,
		`{"text":"def fibonacci(n):","max_tokens":8}`,
	)
	testcheck.NoError(t, err)
	generation, ok := capabilityruntime.Unwrap(output).(thoughtbank.Generation)
	if !ok {
		t.Fatalf("generation type = %T", output)
	}
	const reference = "\n    \"\"\"\n    Returns the number of iterations"
	if generation.Text != reference || len(generation.Tokens) != 8 {
		t.Fatalf("generation = %q/%v", generation.Text, generation.Tokens)
	}
	verification, err := modelintake.PublishVerification(
		ctx, store, definition, "0123456789abcdef0123456789abcdef01234567", time.Millisecond,
		"host", "go", "real Fractale generation matched adaptive_new reference",
	)
	testcheck.NoError(t, err)
	if err := modelrecipe.ActivateCapability(
		ctx, store, definition, verification, recipe.EvidenceParity, "real checkpoint completion matched",
	); err != nil {
		t.Fatal(err)
	}
	activation, activeProgram, err := modelrecipe.ResolveActiveCapability(ctx, store, modelID, recipe.TaskGeneration)
	testcheck.NoError(t, err)
	if activation.Definition.ID != definition.ID || activeProgram.Definition().ID != definition.ID {
		t.Fatalf("active recipe = %s/%s, want %s", activation.Definition.ID, activeProgram.Definition().ID, definition.ID)
	}
}
