package modelartifact

import (
	"bytes"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"slices"
	"testing"

	"llamacpp2go/internal/artifact"
	"llamacpp2go/internal/gguf"
	"llamacpp2go/internal/hfrepo"
	"llamacpp2go/internal/repodb"
)

const (
	fixtureTensorName = "weight"
	fixtureConfig     = `{"model_type":"fixture"}`
	fixtureTokenizer  = `{"version":"1.0"}`
)

func TestGGUFInventoryUsesPhysicalContent(t *testing.T) {
	path := filepath.Join(t.TempDir(), "fixture.gguf")
	file, err := os.Create(path)
	if err != nil {
		t.Fatal(err)
	}
	err = gguf.Write(file, nil, []gguf.TensorData{{
		Name: fixtureTensorName, Shape: []uint64{1}, Type: gguf.DTypeF32,
		Data: bytes.NewReader(make([]byte, 4)),
	}}, gguf.WriteOptions{})
	closeErr := file.Close()
	if err != nil {
		t.Fatal(err)
	}
	if closeErr != nil {
		t.Fatal(closeErr)
	}
	opened, err := gguf.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer opened.Close()
	inventory, err := FromGGUF(opened)
	if err != nil {
		t.Fatal(err)
	}
	if len(inventory.Components) != 1 || len(inventory.Manifest.Components) != 1 ||
		inventory.Manifest.Components[0].Role != artifact.ComponentWeights {
		t.Fatalf("inventory = %+v", inventory)
	}
	if inventory.Locations[0].Value != path {
		t.Fatalf("component location = %q, want %q", inventory.Locations[0].Value, path)
	}
	tensor, ok := inventory.TensorInventory.Tensor(fixtureTensorName)
	if !ok || tensor.Storage != "f32" || tensor.Bytes != 4 || !slices.Equal(tensor.Shape, []uint64{1}) {
		t.Fatalf("tensor fact = (%+v, %v)", tensor, ok)
	}
}

func TestHFInventoryIsRelocationStable(t *testing.T) {
	first := writeHFRepository(t)
	second := writeHFRepository(t)
	firstRepository, err := hfrepo.Open(first)
	if err != nil {
		t.Fatal(err)
	}
	defer firstRepository.Close()
	secondRepository, err := hfrepo.Open(second)
	if err != nil {
		t.Fatal(err)
	}
	defer secondRepository.Close()
	firstInventory, err := FromHFRepository(firstRepository)
	if err != nil {
		t.Fatal(err)
	}
	secondInventory, err := FromHFRepository(secondRepository)
	if err != nil {
		t.Fatal(err)
	}
	if firstInventory.Manifest.ID != secondInventory.Manifest.ID {
		t.Fatalf("relocated identities differ: %s != %s", firstInventory.Manifest.ID, secondInventory.Manifest.ID)
	}
	if firstInventory.TensorInventory.ID != secondInventory.TensorInventory.ID {
		t.Fatalf("relocated tensor inventories differ: %s != %s", firstInventory.TensorInventory.ID, secondInventory.TensorInventory.ID)
	}
	if len(firstInventory.Components) != 3 || len(firstInventory.Locations) != 4 {
		t.Fatalf("components/locations = %d/%d", len(firstInventory.Components), len(firstInventory.Locations))
	}
	batch, err := firstInventory.Batch("fixture/hf/import")
	if err != nil {
		t.Fatal(err)
	}
	if len(batch.Manifests) != 1 || len(batch.Contents) != 1 || len(batch.Lineage) != 1 ||
		len(batch.Locations) != len(firstInventory.Locations) {
		t.Fatalf("batch = %+v", batch)
	}
	store, err := repodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	if _, err := store.Commit(t.Context(), batch); err != nil {
		t.Fatal(err)
	}
	loaded, ok, err := LoadTensorInventory(t.Context(), store, firstInventory.Manifest.ID)
	if err != nil || !ok || loaded.ID != firstInventory.TensorInventory.ID {
		t.Fatalf("stored tensor inventory = (%s, %v, %v)", loaded.ID, ok, err)
	}
}

func TestHFShardedInventoryUsesOrderedLogicalSlots(t *testing.T) {
	directory := t.TempDir()
	if err := os.WriteFile(filepath.Join(directory, "config.json"), []byte(fixtureConfig), 0o600); err != nil {
		t.Fatal(err)
	}
	shards := []string{"model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"}
	writeSafetensor(t, filepath.Join(directory, shards[0]), "first")
	writeSafetensor(t, filepath.Join(directory, shards[1]), "second")
	index, err := json.Marshal(map[string]any{
		"weight_map": map[string]string{"first": shards[0], "second": shards[1]},
	})
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(directory, "model.safetensors.index.json"), index, 0o600); err != nil {
		t.Fatal(err)
	}
	repository, err := hfrepo.Open(directory)
	if err != nil {
		t.Fatal(err)
	}
	defer repository.Close()
	inventory, err := FromHFRepository(repository)
	if err != nil {
		t.Fatal(err)
	}
	var weights []artifact.Component
	for _, component := range inventory.Manifest.Components {
		if component.Role == artifact.ComponentWeightsShard {
			weights = append(weights, component)
		}
	}
	if len(weights) != len(shards) || weights[0].Name != "weights/00001" || weights[1].Name != "weights/00002" {
		t.Fatalf("weight components = %+v", weights)
	}
}

func writeHFRepository(t *testing.T) string {
	t.Helper()
	directory := t.TempDir()
	if err := os.WriteFile(filepath.Join(directory, "config.json"), []byte(fixtureConfig), 0o600); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(directory, "tokenizer.json"), []byte(fixtureTokenizer), 0o600); err != nil {
		t.Fatal(err)
	}
	writeSafetensor(t, filepath.Join(directory, "model.safetensors"), fixtureTensorName)
	return directory
}

func writeSafetensor(t *testing.T, path, name string) {
	t.Helper()
	header := fmt.Sprintf(`{"%s":{"dtype":"U8","shape":[1],"data_offsets":[0,1]}}`, name)
	data := make([]byte, 8, 8+len(header)+1)
	binary.LittleEndian.PutUint64(data, uint64(len(header)))
	data = append(data, header...)
	data = append(data, 1)
	if err := os.WriteFile(path, data, 0o600); err != nil {
		t.Fatal(err)
	}
}
