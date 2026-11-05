package trainingworkflow

import (
	"bytes"
	"os"
	"path/filepath"
	"slices"
	"testing"

	"overgo/internal/discovery"
	"overgo/internal/modelrecipe"
	"overgo/internal/overgodb"
	"overgo/internal/recipe"
	"overgo/internal/testcheck"
	"overgo/internal/testutil"
)

// TestATrainedModelIsListedWhenItsTrainingEnds holds a run to what it owes the model it trained, in either format: with no
// hand activation the catalog a person picks from lists it, trained and not yet measured, and its one measurement, once
// recorded, is read from the catalog every run of every model selects from, a retried run recording nothing new.
func TestATrainedModelIsListedWhenItsTrainingEnds(t *testing.T) {
	t.Parallel()
	for name, base := range map[string]func(t *testing.T) string{
		"gguf": writeTrainingGGUF,
		"directory": func(t *testing.T) string {
			directory := filepath.Join(t.TempDir(), "model")
			weights, shapes := testutil.DenseCausalWeights(t, testutil.DenseCausalSpec{
				Vocab: 8, Hidden: 8, Heads: 2, HeadDim: 4, KVHeads: 1, Intermediate: 16, Layers: 1, Seed: 5,
			})
			writeModelDocument(t, directory, weights, shapes, `{"model_type":"llama","hidden_size":8,"intermediate_size":16,"max_position_embeddings":16,"num_attention_heads":2,"num_hidden_layers":1,"num_key_value_heads":1,"head_dim":4,"rms_norm_eps":1e-6,"rope_theta":10000.0,"vocab_size":8,"tie_word_embeddings":true}`)
			return directory
		},
	} {
		t.Run(name, func(t *testing.T) {
			t.Parallel()
			ctx := t.Context()
			root := t.TempDir()
			model := base(t)
			dataset := filepath.Join(root, "dataset.txt")
			testcheck.NoError(t, os.WriteFile(dataset, bytes.Repeat([]byte("abcd"), 64), 0o600))
			store, err := overgodb.Create(filepath.Join(root, "store"))
			testcheck.NoError(t, err)
			defer store.Close()
			if _, err := modelrecipe.PublishArchitectureProfileCatalog(ctx, store); err != nil {
				t.Fatal(err)
			}
			weights, err := IdentifyModelWeights(model)
			testcheck.NoError(t, err)
			recipeID, err := BootstrapTokenRecipe(ctx, store, weights, dataset)
			testcheck.NoError(t, err)
			result, err := Execute(ctx, Request{
				Repository: store, Recipe: recipeID, Observations: store, ModelDirectory: model, DatasetPath: dataset,
				OutputDirectory: filepath.Join(root, "out"), Steps: 1, MaximumSequence: 8, Host: true,
			})
			testcheck.NoError(t, err)
			definition, err := recipe.RequireDefinition(ctx, store, result.Candidate)
			if err != nil {
				t.Fatalf("the run published no recipe for the model it trained: %v", err)
			}
			listed := func() discovery.CatalogEntry {
				t.Helper()
				entries, truncated, err := discovery.ModelCatalog(ctx, store, nil)
				if err != nil || truncated {
					t.Fatalf("the catalog reads %v (truncated %t)", err, truncated)
				}
				at := slices.IndexFunc(entries, func(entry discovery.CatalogEntry) bool { return entry.Model == definition.Model })
				if at < 0 {
					t.Fatalf("the trained model %s is not in the catalog a person picks from", definition.Model)
				}
				return entries[at]
			}
			if entry := listed(); !entry.Trained || entry.Measured || !entry.Present {
				t.Fatalf("a model just trained reads trained %t, measured %t, present %t", entry.Trained, entry.Measured, entry.Present)
			}
			measure := func() {
				t.Helper()
				testcheck.NoError(t, modelrecipe.MarkMeasured(ctx, store, definition.Model, definition.Task, definition.ID))
			}
			measure()
			measure()
			if entry := listed(); !entry.Trained || !entry.Measured {
				t.Fatalf("a measured trained model reads trained %t, measured %t", entry.Trained, entry.Measured)
			}
		})
	}
}
