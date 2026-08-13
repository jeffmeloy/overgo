//go:build windows

package main

import (
	"encoding/binary"
	"encoding/json"
	"os"
	"path/filepath"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/modelrecipe"
	"overgo/internal/recipe"
	"overgo/internal/testutil"
)

func TestLatentImageInventoryAcceptsDiffusersShards(t *testing.T) {
	root := t.TempDir()
	for _, name := range []string{"text_encoder", "transformer", "vae", "tokenizer", "scheduler"} {
		if err := os.MkdirAll(filepath.Join(root, name), 0o755); err != nil {
			t.Fatal(err)
		}
	}
	for _, name := range []string{
		"model_index.json", "scheduler/scheduler_config.json", "text_encoder/config.json",
		"tokenizer/tokenizer.json", "tokenizer/tokenizer_config.json", "transformer/config.json", "vae/config.json",
	} {
		if err := os.WriteFile(filepath.Join(root, name), []byte("{}"), 0o600); err != nil {
			t.Fatal(err)
		}
	}
	writeInventorySafetensor(t, filepath.Join(root, "text_encoder", "model.safetensors"), "embed")
	writeInventorySafetensor(t, filepath.Join(root, "vae", "diffusion_pytorch_model.safetensors"), "decode")
	shards := []string{
		"diffusion_pytorch_model-00001-of-00002.safetensors",
		"diffusion_pytorch_model-00002-of-00002.safetensors",
	}
	writeInventorySafetensor(t, filepath.Join(root, "transformer", shards[0]), "first")
	writeInventorySafetensor(t, filepath.Join(root, "transformer", shards[1]), "second")
	index, err := json.Marshal(map[string]any{
		"weight_map": map[string]string{"first": shards[0], "second": shards[1]},
	})
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(
		filepath.Join(root, "transformer", "diffusion_pytorch_model.safetensors.index.json"), index, 0o600,
	); err != nil {
		t.Fatal(err)
	}

	inventory, err := latentImageInventory(root)
	if err != nil {
		t.Fatal(err)
	}
	if got := len(inventory.TensorInventory.Tensors); got != 4 {
		t.Fatalf("tensor facts = %d, want 4", got)
	}
}

func writeInventorySafetensor(t *testing.T, path, tensor string) {
	t.Helper()
	header, err := json.Marshal(map[string]any{
		tensor: map[string]any{"dtype": "U8", "shape": []uint64{1}, "data_offsets": []uint64{0, 1}},
	})
	if err != nil {
		t.Fatal(err)
	}
	data := make([]byte, 8, 9+len(header))
	binary.LittleEndian.PutUint64(data, uint64(len(header)))
	data = append(data, header...)
	data = append(data, 1)
	if err := os.WriteFile(path, data, 0o600); err != nil {
		t.Fatal(err)
	}
}

func TestTypedImageRecipeSelectsRuntimeWithoutPlacement(t *testing.T) {
	modelID := testutil.ArtifactID(t, artifact.KindModel, "typed-image-runtime")
	tests := []struct {
		define func(artifact.ID) (recipe.Definition, error)
		want   recipe.ModuleID
	}{
		{modelrecipe.LatentImageDefinition, modelrecipe.ModuleLatentImagePrepare},
		{modelrecipe.OscillatorImageDefinition, modelrecipe.ModuleOscillatorImagePrepare},
	}
	for _, test := range tests {
		definition, err := test.define(modelID)
		if err != nil {
			t.Fatal(err)
		}
		program, err := modelrecipe.CompileCapability(definition)
		if err != nil {
			t.Fatal(err)
		}
		if got := imageProgramModule(program); got != test.want {
			t.Fatalf("operator module=%q, want %q", got, test.want)
		}
	}
}

func TestImagePolicyComesFromRecipeProfile(t *testing.T) {
	profile, ok, err := modelrecipe.ImageProfileForPipeline("Krea2Pipeline")
	if err != nil {
		t.Fatal(err)
	}
	if !ok || profile.Recognition.Scheduler != "FlowMatchEulerDiscreteScheduler" ||
		profile.Conditioning.PadToken != "<|endoftext|>" || profile.Sampling.Steps != 8 {
		t.Fatalf("image recipe profile=%+v present=%v", profile, ok)
	}
}
