package main

import (
	"math"
	"os"
	"path/filepath"
	"testing"

	"overgo/internal/densecausal"
	"overgo/internal/safetensors"
	"overgo/internal/testutil"
)

func writeArtifactDir(t *testing.T, dir string, weights map[string][]float32, shapes map[string][]int) {
	t.Helper()
	if err := os.MkdirAll(dir, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := safetensors.Save(filepath.Join(dir, "model.safetensors"), weights, shapes, nil); err != nil {
		t.Fatalf("Save fixture: %v", err)
	}
	config := `{"model_type":"llama","num_attention_heads":2,"head_dim":4,"rope_theta":10000.0,"rms_norm_eps":1e-6,"tie_word_embeddings":true}`
	if err := os.WriteFile(filepath.Join(dir, "config.json"), []byte(config), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "tokenizer.json"), []byte(`{"model":{"type":"BPE"}}`), 0o644); err != nil {
		t.Fatal(err)
	}
}

// TestTrainGlueLoadStepSaveReload exercises cmd/train's real path end to end on
// the host: a synthetic on-disk model loads, trains one step, and the checkpoint
// saveCheckpoint writes reloads through densecausal.Load with matching geometry.
// This proves the writer produces loader-compatible artifacts and that the
// production caller is not inert.
func TestTrainGlueLoadStepSaveReload(t *testing.T) {
	weights, shapes := testutil.DenseCausalWeights(t, testutil.DenseCausalSpec{
		Vocab: 8, Hidden: 8, Heads: 2, HeadDim: 4,
		KVHeads: 1, Intermediate: 16, Layers: 1, Seed: 1,
	})
	src := filepath.Join(t.TempDir(), "src")
	writeArtifactDir(t, src, weights, shapes)

	model, err := densecausal.Load(src)
	if err != nil {
		t.Fatalf("Load fixture (writer not loader-compatible?): %v", err)
	}

	before := append([]float32(nil), model.Weights["model.embed_tokens.weight"]...)
	tokens := []int{1, 2, 3, 4, 5}
	traj, backend, err := runTraining(model, tokens, 1, 0, 0.9, false) // host path
	if err != nil {
		t.Fatalf("runTraining: %v", err)
	}
	if backend != "host" {
		t.Fatalf("backend = %q, want host", backend)
	}
	if len(traj) != 1 || math.IsNaN(traj[0]) || math.IsInf(traj[0], 0) {
		t.Fatalf("trajectory = %v, want one finite loss", traj)
	}
	changed := false
	for i, v := range model.Weights["model.embed_tokens.weight"] {
		if v != before[i] {
			changed = true
			break
		}
	}
	if !changed {
		t.Fatalf("training left weights unchanged")
	}

	out := filepath.Join(t.TempDir(), "out")
	if err := saveCheckpoint(src, out, model); err != nil {
		t.Fatalf("saveCheckpoint: %v", err)
	}
	reloaded, err := densecausal.Load(out)
	if err != nil {
		t.Fatalf("reload checkpoint: %v", err)
	}
	if reloaded.Dims != model.Dims {
		t.Fatalf("reloaded dims %+v != %+v", reloaded.Dims, model.Dims)
	}
	got := reloaded.Weights["model.embed_tokens.weight"]
	for i, v := range model.Weights["model.embed_tokens.weight"] {
		if got[i] != v {
			t.Fatalf("reloaded weight[%d] = %v, want %v (trained value not persisted)", i, got[i], v)
		}
	}
	for _, name := range []string{"config.json", "tokenizer.json", "model.safetensors"} {
		if _, err := os.Stat(filepath.Join(out, name)); err != nil {
			t.Fatalf("checkpoint missing %s: %v", name, err)
		}
	}
}
