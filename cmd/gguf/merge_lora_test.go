package main

import (
	"math"
	"os"
	"path/filepath"
	"testing"

	"overgo/internal/peft"
	"overgo/internal/safetensors"
)

const loRAModule = "model.layers.0.self_attn.q_proj"

// TestMergeLoRAWritesTheMergedCheckpoint: a PEFT adapter merged into a
// checkpoint adds exactly its update to the adapted weight, leaves the other
// tensors as they were, and carries the config and tokenizer files over; a
// merge into an existing directory, or with a flag missing, is refused.
func TestMergeLoRAWritesTheMergedCheckpoint(t *testing.T) {
	root := t.TempDir()
	base, adapterDirectory, output := filepath.Join(root, "base"), filepath.Join(root, "adapter"), filepath.Join(root, "merged")
	if err := os.MkdirAll(base, 0o755); err != nil {
		t.Fatal(err)
	}
	weights := map[string][]float32{loRAModule + ".weight": {1, 2, 3, 4, 5, 6}, "model.norm.weight": {1, 1, 1}}
	shapes := map[string][]int{loRAModule + ".weight": {2, 3}, "model.norm.weight": {3}}
	if err := safetensors.Save(filepath.Join(base, "model.safetensors"), weights, shapes, nil); err != nil {
		t.Fatal(err)
	}
	for name, contents := range map[string]string{"config.json": `{"model_type":"llama"}`, "tokenizer.json": `{"model":{}}`} {
		if err := os.WriteFile(filepath.Join(base, name), []byte(contents), 0o600); err != nil {
			t.Fatal(err)
		}
	}
	// Alpha 6 over rank 2 scales the product by three.
	adapter := peft.Adapter{
		Config:  peft.Config{Rank: 2, Alpha: 6},
		Modules: map[string]peft.Module{loRAModule: {Rank: 2, In: 3, Out: 2, Alpha: 6, Down: []float32{1, 0, 2, 0, 1, -1}, Up: []float32{1, 2, 0, 1}}},
	}
	if err := peft.Save(adapterDirectory, adapter); err != nil {
		t.Fatal(err)
	}
	// Up·Down by hand: up rows (1, 2) and (0, 1) over down rows (1, 0, 2) and (0, 1, -1).
	update := []float64{1, 2, 0, 0, 1, -1}
	if err := runSubcommand([]string{"merge-lora", "-adapter", adapterDirectory, "-base", base, "-output", output}); err != nil {
		t.Fatal(err)
	}
	source, err := safetensors.OpenSource(output)
	if err != nil {
		t.Fatal(err)
	}
	defer source.Close()
	got, err := safetensors.ReadF32(source.Tensors[loRAModule+".weight"])
	if err != nil {
		t.Fatal(err)
	}
	for index, value := range got {
		want := float64(weights[loRAModule+".weight"][index]) + 3*update[index]
		if math.Abs(float64(value)-want) > 1e-5 {
			t.Fatalf("merged[%d] = %v, want %v", index, value, want)
		}
	}
	if norm, err := safetensors.ReadF32(source.Tensors["model.norm.weight"]); err != nil || norm[0] != 1 || norm[2] != 1 {
		t.Fatalf("an unadapted tensor changed: %v (%v)", norm, err)
	}
	for _, name := range []string{"config.json", "tokenizer.json"} {
		if _, err := os.Stat(filepath.Join(output, name)); err != nil {
			t.Errorf("%s was not carried over: %v", name, err)
		}
	}
	if err := runSubcommand([]string{"merge-lora", "-adapter", adapterDirectory, "-base", base, "-output", output}); err == nil {
		t.Fatal("a merge into an existing directory was accepted")
	}
	if err := runSubcommand([]string{"merge-lora", "-adapter", adapterDirectory, "-base", base}); err == nil {
		t.Fatal("a merge with no output was accepted")
	}
}
