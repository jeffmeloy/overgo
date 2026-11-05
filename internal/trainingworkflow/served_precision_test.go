package trainingworkflow

import (
	"os"
	"path/filepath"
	"testing"

	"overgo/internal/safetensors"
	"overgo/internal/tensor/dtype"
	"overgo/internal/testutil"
	"overgo/internal/trainingprogram"
)

// TestATrainedCheckpointServesAtItsBasesPrecision holds a model trained from a BF16 base to its base's precision: its
// checkpoint stores every tensor in BF16, each value its F32 weight's nearest, and keeps the F32 weights beside them,
// which a run that resumes from the checkpoint reads, so it trains on to the uninterrupted run's weights.
func TestATrainedCheckpointServesAtItsBasesPrecision(t *testing.T) {
	t.Parallel()
	weights, shapes := testutil.DenseCausalWeights(t, testutil.DenseCausalSpec{
		Vocab: 8, Hidden: 8, Heads: 2, HeadDim: 4, KVHeads: 1, Intermediate: 16, Layers: 1, Seed: 5,
	})
	root := t.TempDir()
	model := filepath.Join(root, "model")
	writeModel(t, model, weights, shapes)
	types := map[string]string{}
	for name := range weights {
		types[name] = servedBF16
	}
	if err := safetensors.SaveTyped(filepath.Join(model, trainingprogram.CheckpointWeights), weights, shapes, types, nil); err != nil {
		t.Fatal(err)
	}
	dataset := filepath.Join(root, "dataset.txt")
	if err := os.WriteFile(dataset, []byte("abcd"), 0o600); err != nil {
		t.Fatal(err)
	}
	store, recipeID := trainingAuthority(t, model, "", dataset, trainingprogram.ObjectiveTokenPrediction)
	verifyWorkflowResume(t, root, Request{
		Repository: store, Recipe: recipeID, ModelDirectory: model, DatasetPath: dataset,
		Host: true, Observations: store,
	})
	checkpoint := filepath.Join(root, "uninterrupted")
	served, err := safetensors.OpenFile(filepath.Join(checkpoint, trainingprogram.CheckpointWeights))
	if err != nil {
		t.Fatal(err)
	}
	defer served.Close()
	resumed, err := safetensors.OpenFile(filepath.Join(checkpoint, resumeDirectory, resumeWeightsFile))
	if err != nil {
		t.Fatal(err)
	}
	defer resumed.Close()
	for name := range weights {
		tensor, servedHeld := served.Tensors[name]
		kept, keptHeld := resumed.Tensors[name]
		if !servedHeld || !keptHeld || tensor.DType != servedBF16 || kept.DType != "F32" {
			t.Fatalf("%s is served as %s and kept as %s, want BF16 served and F32 kept", name, tensor.DType, kept.DType)
		}
		values, err := safetensors.ReadF32(tensor)
		if err != nil {
			t.Fatal(err)
		}
		exact, err := safetensors.ReadF32(kept)
		if err != nil {
			t.Fatal(err)
		}
		for index, value := range exact {
			if values[index] != dtype.RoundBF16(value) {
				t.Fatalf("%s[%d] serves %g for the weight %g", name, index, values[index], value)
			}
		}
	}
}
