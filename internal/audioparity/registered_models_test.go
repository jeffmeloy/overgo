package audioparity

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"slices"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/modelartifact"
	"overgo/internal/overgodb"
)

func TestCanonicalAudioModelRegistration(t *testing.T) {
	t.Parallel()
	data, err := os.ReadFile("testdata/registered_models.json")
	if err != nil {
		t.Fatal(err)
	}
	// The reference store and the models beside it come from the declared
	// roots, the same reference the CPU ASR acceptance reads.
	reference := resolveReferenceRoots(t)
	storeRoot, root := reference.store, filepath.Dir(reference.store)
	empty, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	if err := verifyRegisteredAudioModels(t.Context(), empty, root, data); err == nil {
		t.Fatal("accepted an empty catalog")
	}
	if err := empty.Close(); err != nil {
		t.Fatal(err)
	}
	store, err := overgodb.OpenReadOnly(storeRoot)
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	if err := verifyRegisteredAudioModels(t.Context(), store, root, data); err != nil {
		t.Fatal(err)
	}
	t.Log("registration accepted: 4/4 exact models, 1287/1287 tensor declarations; physical components and source/license lineage verified; native Nemotron Safetensors selected; Q8 and AED excluded; no runtime activation, task-quality, training or streaming claim")
}

func verifyRegisteredAudioModels(ctx context.Context, reader artifact.Reader, root string, data []byte) error {
	var entries []struct {
		Directory  string                   `json:"directory"`
		Model      artifact.ID              `json:"model"`
		Tensors    artifact.ID              `json:"tensor_inventory"`
		Components []modelartifact.FileSpec `json:"components"`
		License    struct {
			Artifact artifact.ID `json:"artifact"`
			Path     string      `json:"path"`
		} `json:"license"`
	}
	if err := json.Unmarshal(data, &entries); err != nil {
		return err
	}
	want := map[string]int{
		"1712f360981e18b30efad1e17611d16066549b820da88c0eb7c7b6f6ad39c098": 550,
		"e61f34926cd36a73ee0814b0a01ee4e515d983c204676df0e9bc0652a571869d": 655,
		"f8b932cdc72a739cb65d0d99e81c6f9c761372d7df26ca970b6a084e4e55b19d": 45,
		"04e117c24c7749e8a71a8d5ee92492265f6c38cb069f42c3a727bc50fc06fb61": 37,
	}
	if len(entries) != len(want) {
		return fmt.Errorf("registration denominator %d differs from %d", len(entries), len(want))
	}
	specID, err := artifact.IdentifyBytes(artifact.KindFile, data)
	if err != nil {
		return err
	}
	stored, found, err := artifact.ReadContent(ctx, reader, specID)
	if err != nil || !found || !bytes.Equal(stored.Data, data) {
		return fmt.Errorf("exact source declaration absent or changed: %v", err)
	}
	for _, entry := range entries {
		count, wanted := want[entry.Model.DigestHex()]
		if !wanted {
			return fmt.Errorf("unexpected or duplicate registered model %s", entry.Model)
		}
		delete(want, entry.Model.DigestHex())
		manifest, found, err := reader.Manifest(ctx, entry.Model)
		if err != nil || !found || manifest.ID != entry.Model {
			return fmt.Errorf("missing manifest %s: %v", entry.Model, err)
		}
		tensors, found, err := modelartifact.ReadTensorInventoryDocument(ctx, reader, entry.Tensors)
		if err != nil || !found || tensors.Owner != entry.Model || len(tensors.Tensors) != count {
			return fmt.Errorf("tensor coverage differs for %s: %v", entry.Model, err)
		}
		directory := filepath.Join(root, "models", entry.Directory)
		available, err := artifact.AvailablePath(ctx, reader, entry.Model, artifact.LocationDirectory)
		if err != nil {
			return err
		}
		actual, err := os.Stat(available)
		if err != nil {
			return err
		}
		expected, err := os.Stat(directory)
		if err != nil || !os.SameFile(actual, expected) {
			return fmt.Errorf("canonical model location differs: %s", entry.Model)
		}
		var inventory modelartifact.Inventory
		if len(entry.Components) == 0 {
			inventory, err = modelartifact.FromHFPath(directory)
		} else {
			for index := range entry.Components {
				entry.Components[index].Path = filepath.Join(directory, entry.Components[index].Path)
			}
			inventory, err = modelartifact.FromFiles(directory, entry.Components)
		}
		if err != nil || inventory.Manifest.ID != entry.Model || inventory.TensorInventory.ID != entry.Tensors {
			return fmt.Errorf("physical inventory differs: %s: %v", entry.Model, err)
		}
		license, found, err := artifact.ReadContent(ctx, reader, entry.License.Artifact)
		if err != nil || !found {
			return fmt.Errorf("license is missing: %v", err)
		}
		licenseBytes, err := os.ReadFile(filepath.Join(directory, entry.License.Path))
		if err != nil || !bytes.Equal(license.Data, licenseBytes) {
			return fmt.Errorf("license content differs: %s", entry.Model)
		}
		parents, err := reader.Parents(ctx, entry.Model)
		if err != nil {
			return err
		}
		for _, parent := range []artifact.ID{specID, entry.License.Artifact} {
			if !slices.ContainsFunc(parents, func(edge artifact.Lineage) bool {
				return edge.Parent == parent && edge.Relation == artifact.RelationDependsOn
			}) {
				return fmt.Errorf("source/license lineage missing for %s", entry.Model)
			}
		}
	}
	return nil
}
