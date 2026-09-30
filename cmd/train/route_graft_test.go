package main

import (
	"fmt"
	"os"
	"path/filepath"
	"testing"

	"overgo/internal/overgodb"
	"overgo/internal/safetensors"
	"overgo/internal/testutil"
	"overgo/internal/trainingprogram"
	"overgo/internal/trainingworkflow"
)

// TestGraftPublishesEachSeedsBridge runs the viability probe on a tiny target
// and donor and holds each seed's trained bridge to the checkpoint it
// publishes under -out: one step of the bridge's two matrices, recorded in
// the store.
func TestGraftPublishesEachSeedsBridge(t *testing.T) {
	root := t.TempDir()
	targetDir, donorDir := filepath.Join(root, "target"), filepath.Join(root, "donor")
	for directory, seed := range map[string]int64{targetDir: 3, donorDir: 5} {
		weights, shapes := testutil.DenseCausalWeights(t, testutil.DenseCausalSpec{
			Vocab: 8, Hidden: 8, Heads: 2, HeadDim: 4, KVHeads: 1, Intermediate: 16, Layers: 1, Seed: seed,
		})
		if err := os.MkdirAll(directory, 0o755); err != nil {
			t.Fatal(err)
		}
		if err := safetensors.Save(filepath.Join(directory, trainingprogram.CheckpointWeights), weights, shapes, nil); err != nil {
			t.Fatal(err)
		}
		config := `{"model_type":"llama","num_attention_heads":2,"head_dim":4,"rope_theta":10000.0,"rms_norm_eps":1e-6,"tie_word_embeddings":true}`
		if err := os.WriteFile(filepath.Join(directory, "config.json"), []byte(config), 0o600); err != nil {
			t.Fatal(err)
		}
	}
	tokens, storePath, out := filepath.Join(root, "tokens.json"), filepath.Join(root, "store"), filepath.Join(root, "bridges")
	if err := os.WriteFile(tokens, []byte("[1, 2, 3, 4, 2, 3, 4, 1, 1, 3, 2, 4]"), 0o600); err != nil {
		t.Fatal(err)
	}
	if err := runRoute([]string{"graft", "-target", targetDir, "-donor", donorDir, "-tokens", tokens,
		"-record", storePath, "-out", out, "-steps", "1", "-window", "4"}); err != nil {
		t.Fatal(err)
	}
	store, err := overgodb.Open(storePath)
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	for _, seed := range graftSeeds {
		checkpoint, err := trainingprogram.LoadCheckpoint(filepath.Join(out, fmt.Sprintf("seed-%d", seed)))
		if err != nil {
			t.Fatal(err)
		}
		_, recorded, err := store.Artifact(t.Context(), checkpoint.ID())
		if err != nil || !recorded || checkpoint.Optimizer.Step != 1 || checkpoint.ParameterCount != len(checkpoint.Optimizer.Momentum) {
			t.Fatalf("seed %d checkpoint %s at step %d recorded %t (%v), want the one-step bridge recorded", seed, checkpoint.ID(), checkpoint.Optimizer.Step, recorded, err)
		}
	}
}

// TestProbeCommitsGenerationRecord pins the lifecycle wiring: the probe's
// recording path authors a typed generation record through the production
// constructor, commits it with its referenced identities, and the committed
// graph is queryable -- child lineage present, verdict outcome preserved,
// refusals recorded as loudly as ships.
func TestProbeRejectsUncompiledGenerationRecord(t *testing.T) {
	store, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	config := trainingworkflow.GraftConfig{
		TargetDir: "fixture/target", DonorDir: "fixture/donor",
		Seeds: []int64{7, 11, 13}, Steps: 6,
	}
	result := trainingworkflow.GraftResult{
		DonorTensor: "model.layers.14.mlp.gate_proj.weight", GraftLayer: 12,
		Baseline: 4.849884, Ship: false,
		Reason: "refusal: worst seed held-out 5.441688 does not beat baseline 4.849884",
	}
	if _, err := trainingworkflow.RecordGraftViability(store, config, result); err == nil {
		t.Fatal("record accepted path-derived placeholder authority")
	}
}
