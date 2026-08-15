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
	writeLatentImageInventoryFixture(t, root)

	inventory, err := latentImageInventory(root)
	if err != nil {
		t.Fatal(err)
	}
	if got := len(inventory.TensorInventory.Tensors); got != 4 {
		t.Fatalf("tensor facts = %d, want 4", got)
	}
}

func writeLatentImageInventoryFixture(t testing.TB, root string) {
	t.Helper()
	for _, name := range []string{"text_encoder", "transformer", "vae", "tokenizer", "scheduler"} {
		if err := os.MkdirAll(filepath.Join(root, name), 0o755); err != nil {
			t.Fatal(err)
		}
	}
	for _, name := range []string{
		"model_index.json", "scheduler/scheduler_config.json", "text_encoder/config.json",
		"tokenizer/tokenizer.json", "tokenizer/tokenizer_config.json", "transformer/config.json", "vae/config.json",
	} {
		path := filepath.Join(root, name)
		if _, err := os.Stat(path); err == nil {
			continue
		}
		if err := os.WriteFile(path, []byte("{}"), 0o600); err != nil {
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
}

func writeInventorySafetensor(t testing.TB, path, tensor string) {
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
	profileID := testutil.ArtifactID(t, artifact.KindProfile, "typed-image-profile")
	tests := []struct {
		define func(artifact.ID) (recipe.Definition, error)
		want   recipe.ModuleID
	}{
		{func(model artifact.ID) (recipe.Definition, error) {
			return modelrecipe.LatentImageDefinition(model, profileID)
		}, modelrecipe.ModuleLatentImagePrepare},
		{modelrecipe.OscillatorImageDefinition, modelrecipe.ModuleOscillatorImagePrepare},
		{modelrecipe.RoutedImageDefinition, modelrecipe.ModuleRoutedImagePrepare},
		{modelrecipe.DiffusionImageDefinition, modelrecipe.ModuleDiffusionImagePrepare},
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
		if got := program.Stages()[0].Module.ID; got != test.want {
			t.Fatalf("operator module=%q, want %q", got, test.want)
		}
	}
}

func TestImagePolicyComesFromRecipeProfile(t *testing.T) {
	root := t.TempDir()
	testutil.WriteImageProfileFixture(t, root)
	writeLatentImageInventoryFixture(t, root)
	modelID := testutil.ArtifactID(t, artifact.KindModel, "profile-bound-image")
	source, err := imageCapability().resolve(root)
	if err != nil {
		t.Fatal(err)
	}
	definition, facts, err := source.define(modelID)
	if err != nil {
		t.Fatal(err)
	}
	profileID, ok := definition.Dependency(recipe.DependencyProfile, 0)
	if !ok || len(facts) != 1 || facts[0].Descriptor.ID != profileID {
		t.Fatalf("profile binding = (%s, %v), facts=%+v", profileID, ok, facts)
	}
}

func TestImageCapabilityBindsSenseNovaByArchitecture(t *testing.T) {
	root := t.TempDir()
	if err := os.WriteFile(
		filepath.Join(root, "config.json"),
		[]byte(`{"architectures":["NEOChatModel"],"model_type":"neo_chat"}`),
		0o600,
	); err != nil {
		t.Fatal(err)
	}
	writeInventorySafetensor(t, filepath.Join(root, "model.safetensors"), "weight")
	modelID := testutil.ArtifactID(t, artifact.KindModel, "sensenova-image")
	source, err := imageCapability().resolve(root)
	if err != nil {
		t.Fatal(err)
	}
	definition, facts, err := source.define(modelID)
	if err != nil {
		t.Fatal(err)
	}
	if len(facts) != 0 || definition.Model != modelID {
		t.Fatalf("SenseNova binding model=%s facts=%d", definition.Model, len(facts))
	}
	program, err := modelrecipe.CompileCapability(definition)
	if err != nil {
		t.Fatal(err)
	}
	if got := program.Stages()[0].Module.ID; got != modelrecipe.ModuleRoutedImagePrepare {
		t.Fatalf("SenseNova operator=%q", got)
	}
}

func TestImagePublicationStreamsEncodedArtifact(t *testing.T) {
	modelID := testutil.ArtifactID(t, artifact.KindModel, "encoded-image-runtime")
	profileID := testutil.ArtifactID(t, artifact.KindProfile, "encoded-image-profile")
	definition, err := modelrecipe.LatentImageDefinition(modelID, profileID)
	if err != nil {
		t.Fatal(err)
	}
	program, err := modelrecipe.CompileCapability(definition)
	if err != nil {
		t.Fatal(err)
	}
	if program.Stages()[0].Module.ID != modelrecipe.ModuleLatentImagePrepare ||
		len(definition.Outputs) != 1 || definition.Outputs[0].Data != recipe.DataImage {
		t.Fatalf("latent image program=%+v", definition)
	}
	if capability := imageCapability(); capability.execute == nil {
		t.Fatal("typed image runtime has no executor")
	}
}
