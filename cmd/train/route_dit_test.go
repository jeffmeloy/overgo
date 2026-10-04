package main

import (
	"context"
	"errors"
	"os"
	"path/filepath"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/modelrecipe"
	"overgo/internal/overgodb"
	"overgo/internal/recipecontract"
	"overgo/internal/trainingprogram"
	"overgo/internal/trainingworkflow"
)

// TestDiTRouteRecordsItsProgram holds the dit route to the program its
// trainer steps, flow matching from text to video. Over a stand-in video
// stack whose weights load no model, the route's session publishes the
// flow-matching text-to-video recipe for those weights over its stimulus,
// not the token recipe, and a run whose context has ended stops with its
// cancellation and publishes nothing.
func TestDiTRouteRecordsItsProgram(t *testing.T) {
	t.Parallel()
	root := t.TempDir()
	stack, fixtures := filepath.Join(root, "stack"), filepath.Join(root, "fixtures")
	weights, stimulus := filepath.Join(stack, "diffusion_pytorch_model.safetensors"), filepath.Join(root, "stimulus.txt")
	// The run stops at the stand-in stack before it uses the fixture's
	// request, so the fixture requests nothing.
	for path, data := range map[string]string{
		weights: "weights", stimulus: routeFixtureDataset, filepath.Join(fixtures, "g3_denoise.json"): "{}",
	} {
		if err := os.MkdirAll(filepath.Dir(path), 0o700); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(path, []byte(data), 0o600); err != nil {
			t.Fatal(err)
		}
	}
	run := func(ctx context.Context) (store, out string, err error) {
		scratch := t.TempDir()
		store, out = filepath.Join(scratch, "store"), filepath.Join(scratch, "checkpoint")
		return store, out, runRoute(ctx, []string{"dit", "-model-dir", stack, "-fixtures", fixtures, "-stimulus", stimulus,
			"-schedule-index", "0", "-steps", "1", "-store", store, "-out", out})
	}
	t.Run("flow matching, text to video", func(t *testing.T) {
		identities, err := overgodb.Open(filepath.Join(t.TempDir(), "identities"))
		if err != nil {
			t.Fatal(err)
		}
		defer identities.Close()
		recipeOf := func(contract trainingworkflow.RecipeContract) artifact.ID {
			id, err := trainingworkflow.BootstrapRecipe(t.Context(), identities, weights, []byte(routeFixtureDataset), contract)
			if err != nil {
				t.Fatal(err)
			}
			return id
		}
		flow := recipeOf(generationContract(trainingprogram.ObjectiveFlowMatching, recipecontract.ModalityText, recipecontract.ModalityVideo, trainingprogram.MetricVideoPSNRUnit))
		token := recipeOf(trainingworkflow.TokenContract)
		storePath, _, ran := run(t.Context())
		if ran == nil {
			t.Fatal("the dit route trained a stand-in stack that loads no model")
		}
		store, err := overgodb.OpenReadOnly(storePath)
		if err != nil {
			t.Fatal(err)
		}
		defer store.Close()
		published := map[artifact.ID]bool{}
		for _, id := range []artifact.ID{flow, token} {
			if _, published[id], err = modelrecipe.Status(t.Context(), store, id); err != nil {
				t.Fatal(err)
			}
		}
		if !published[flow] || published[token] {
			t.Fatalf("the dit route published the flow-matching text-to-video recipe %s: %t, the token recipe %s: %t; want only the flow-matching one (the run stopped at %v)",
				flow, published[flow], token, published[token], ran)
		}
	})
	t.Run("cancelled", func(t *testing.T) {
		ctx, cancel := context.WithCancel(t.Context())
		cancel()
		_, out, err := run(ctx)
		if !errors.Is(err, context.Canceled) {
			t.Fatalf("a dit run whose context ended = %v, want its cancellation", err)
		}
		if _, err := os.Stat(out); !os.IsNotExist(err) {
			t.Fatalf("a cancelled dit run published at %s (%v)", out, err)
		}
	})
}
