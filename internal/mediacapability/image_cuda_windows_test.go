//go:build windows

package mediacapability

import (
	"encoding/binary"
	"encoding/json"
	"os"
	"path/filepath"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/modelartifact"
	"overgo/internal/modelrecipe"
	"overgo/internal/recipe"
	"overgo/internal/testutil"
)

func TestLatentImageInventoryAcceptsDiffusersShards(t *testing.T) {
	root := t.TempDir()
	testutil.WriteImageInventoryFixture(t, root)

	inventory, err := modelartifact.FromHFPath(root)
	if err != nil {
		t.Fatal(err)
	}
	if got := len(inventory.TensorInventory.Tensors); got != 4 {
		t.Fatalf("tensor facts = %d, want 4", got)
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
		prepare recipe.ModuleID
		profile artifact.ID
	}{
		{modelrecipe.ModuleLatentImagePrepare, profileID},
		{modelrecipe.ModuleOscillatorImagePrepare, artifact.ID{}},
		{modelrecipe.ModuleRoutedImagePrepare, profileID},
		{modelrecipe.ModuleDiffusionImagePrepare, artifact.ID{}},
	}
	for _, test := range tests {
		definition, err := modelrecipe.GenerationDefinition(test.prepare, modelID, test.profile)
		if err != nil {
			t.Fatal(err)
		}
		program, err := modelrecipe.CompileCapability(definition)
		if err != nil {
			t.Fatal(err)
		}
		if got := program.Stages()[0].Module.ID; got != test.prepare {
			t.Fatalf("operator module=%q, want %q", got, test.prepare)
		}
	}
}

func TestImagePolicyComesFromRecipeProfile(t *testing.T) {
	root := t.TempDir()
	testutil.WriteImageProfileFixture(t, root)
	testutil.WriteImageInventoryFixture(t, root)
	modelID := testutil.ArtifactID(t, artifact.KindModel, "profile-bound-image")
	source, err := imageCapability().Resolve(root)
	if err != nil {
		t.Fatal(err)
	}
	definition, facts, err := source.Define(modelID)
	if err != nil {
		t.Fatal(err)
	}
	profileID, ok := definition.PrimaryDependency(recipe.DependencyProfile)
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
	source, err := imageCapability().Resolve(root)
	if err != nil {
		t.Fatal(err)
	}
	definition, facts, err := source.Define(modelID)
	if err != nil {
		t.Fatal(err)
	}
	profileID, bound := definition.PrimaryDependency(recipe.DependencyFlowProfile)
	if len(facts) != 1 || !bound || facts[0].Descriptor.ID != profileID || definition.Model != modelID {
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
	definition, err := modelrecipe.GenerationDefinition(modelrecipe.ModuleLatentImagePrepare, modelID, profileID)
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
	if capability := imageCapability(); capability.Execute == nil {
		t.Fatal("typed image runtime has no executor")
	}
}
