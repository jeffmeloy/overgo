//go:build modeltest

package main

import (
	"testing"

	"overgo/internal/dataroot"
	"overgo/internal/mediacapability"
	"overgo/internal/modelrecipe"
	"overgo/internal/modeltest"
	"overgo/internal/overgodbtest"
	"overgo/internal/recipe"
	"overgo/internal/testcheck"
	"overgo/internal/testutil"
)

// TestEveryReadGeneratorHasAnActiveRecipe holds the generators whose guidance presets are read (guidance-ops) to a
// recipe active under the identity their inventory gives them now. A change to how a model's inventory is read moves
// its identity and leaves its activation under the old one, and a render then refuses the model: 22960ddad came to
// count a sharded model's shard index, tokenizer config and scheduler config on 2026-10-18, and the NAG readings'
// campaign found Krea-2-Turbo with no active image-gen recipe on 2026-11-16 (guidance-ops/krea-recipe).
func TestEveryReadGeneratorHasAnActiveRecipe(t *testing.T) {
	t.Parallel()
	ctx := t.Context()
	roots := testcheck.Must(dataroot.Resolve(testutil.RepoRoot(t)))(t)
	store := overgodbtest.OpenReadOnly(t, retainedReferenceStore(roots.Store))
	for name, task := range map[string]recipe.Task{
		"Krea-2-Turbo":                       recipe.TaskImageGen,
		"MOSS-Image-SFT-797fa8e678b4f":       recipe.TaskImageGen,
		"SenseNova-U1-8B-MoT-Infographic-V3": recipe.TaskImageGen,
		"Wan2.1-T2V-1.3B":                    recipe.TaskVideoGen,
		"Self-Forcing":                       recipe.TaskVideoGen,
	} {
		model := testcheck.Must(mediacapability.Catalog[task].Resolve(ctx, modeltest.Directory(t, name)))(t).Inventory.Manifest.ID
		if _, err := modelrecipe.ResolveActiveExecution(ctx, store, model, task, modelrecipe.SessionWarm); err != nil {
			t.Errorf("%s has no %s recipe active under %s, the identity its inventory gives it: %v", name, task, model, err)
		}
	}
}
