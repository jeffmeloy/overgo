package main

import (
	"fmt"
	"math"
	"os"
	"path/filepath"
	"testing"

	"overgo/internal/densecausal"
	"overgo/internal/safetensors"
)

// tinyLlama returns the weights+shapes of a minimal valid tied-embedding llama
// (no attention bias) that densecausal.Load/NewModel accept and Train can step.
// Values are small deterministic magnitudes so the training step stays finite.
func tinyLlama() (map[string][]float32, map[string][]int) {
	const vocab, hidden, heads, headDim, kvHeads, inter, layers = 8, 8, 2, 4, 1, 16, 1
	qOut := heads * headDim
	kvOut := kvHeads * headDim
	weights := map[string][]float32{}
	shapes := map[string][]int{}
	seed := 0
	add := func(name string, dims ...int) {
		n := 1
		for _, d := range dims {
			n *= d
		}
		values := make([]float32, n)
		for i := range values {
			seed++
			values[i] = float32(math.Sin(float64(seed))) * 0.1
		}
		weights[name] = values
		shapes[name] = dims
	}
	add("model.embed_tokens.weight", vocab, hidden)
	for l := 0; l < layers; l++ {
		p := fmt.Sprintf("model.layers.%d.", l)
		add(p+"self_attn.q_proj.weight", qOut, hidden)
		add(p+"self_attn.k_proj.weight", kvOut, hidden)
		add(p+"self_attn.v_proj.weight", kvOut, hidden)
		add(p+"self_attn.o_proj.weight", hidden, qOut)
		add(p+"mlp.gate_proj.weight", inter, hidden)
		add(p+"mlp.up_proj.weight", inter, hidden)
		add(p+"mlp.down_proj.weight", hidden, inter)
		add(p+"input_layernorm.weight", hidden)
		add(p+"post_attention_layernorm.weight", hidden)
	}
	add("model.norm.weight", hidden)
	return weights, shapes
}

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
	weights, shapes := tinyLlama()
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
