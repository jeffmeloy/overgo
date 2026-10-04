package main

import (
	"encoding/json"
	"os"
	"path/filepath"
	"slices"
	"testing"

	"overgo/internal/gguf"
	"overgo/internal/peft"
	"overgo/internal/safetensors"
	"overgo/internal/testutil"
)

// TestConvertLoRAWritesTheGGUFAdapter: a PEFT adapter converts to a GGUF adapter
// file of its own kind and architecture holding its pair under the GGUF name of
// the weight it adapts; a conversion with a flag missing, or onto an existing
// file, is refused.
func TestConvertLoRAWritesTheGGUFAdapter(t *testing.T) {
	root := t.TempDir()
	base := filepath.Join(root, "base")
	if err := os.MkdirAll(base, 0o755); err != nil {
		t.Fatal(err)
	}
	spec := testutil.DenseCausalSpec{Vocab: 3, Hidden: 8, Heads: 2, HeadDim: 4, KVHeads: 1, Intermediate: 16, Layers: 1, Seed: 3}
	weights, shapes := testutil.DenseCausalWeights(t, spec)
	if err := safetensors.Save(filepath.Join(base, "model.safetensors"), weights, shapes, nil); err != nil {
		t.Fatal(err)
	}
	config, err := json.Marshal(map[string]any{
		"model_type": "llama", "hidden_size": spec.Hidden, "intermediate_size": spec.Intermediate, "num_attention_heads": spec.Heads,
		"num_key_value_heads": spec.KVHeads, "head_dim": spec.HeadDim, "num_hidden_layers": spec.Layers, "vocab_size": spec.Vocab,
		"max_position_embeddings": 16, "rms_norm_eps": 1e-5, "rope_theta": 10000,
	})
	if err != nil {
		t.Fatal(err)
	}
	testutil.WriteTextFile(t, base, "config.json", string(config))
	adapterDirectory := filepath.Join(root, "adapter")
	adapter := peft.Adapter{
		Config:  peft.Config{Rank: 2, Alpha: 4},
		Modules: map[string]peft.Module{"model.layers.0.self_attn.v_proj": {Rank: 2, In: 8, Out: 4, Alpha: 4, Down: make([]float32, 16), Up: make([]float32, 8)}},
	}
	if err := peft.Save(adapterDirectory, adapter); err != nil {
		t.Fatal(err)
	}
	output := filepath.Join(root, "adapter.gguf")
	if err := runSubcommand([]string{"convert-lora", "-adapter", adapterDirectory, "-base", base, "-output", output}); err != nil {
		t.Fatal(err)
	}
	file, err := gguf.Open(output)
	if err != nil {
		t.Fatal(err)
	}
	defer file.Close()
	var names []string
	for _, info := range file.Tensors {
		names = append(names, info.Name)
	}
	slices.Sort(names)
	if !slices.Equal(names, []string{"blk.0.attn_v.weight.lora_a", "blk.0.attn_v.weight.lora_b"}) {
		t.Fatalf("tensors = %v, want the value pair under the GGUF name of its weight", names)
	}
	for key, want := range map[string]string{"general.type": "adapter", "general.architecture": "llama", "adapter.type": "lora"} {
		if value, found := file.MetadataValue(key); !found || value.Data != want {
			t.Errorf("%s = %v (%t), want %q", key, value.Data, found, want)
		}
	}
	if err := runSubcommand([]string{"convert-lora", "-adapter", adapterDirectory, "-base", base, "-output", output}); err == nil {
		t.Fatal("a conversion onto an existing file was accepted")
	}
	if err := runSubcommand([]string{"convert-lora", "-adapter", adapterDirectory, "-base", base}); err == nil {
		t.Fatal("a conversion with no output was accepted")
	}
}
