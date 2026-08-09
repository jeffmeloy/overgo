package modelartifact

import (
	"os"
	"path/filepath"
	"testing"

	"overgo/internal/artifact"
)

// TestFromFilesNamespacesDualHeadTensors: two sub-model heads reuse the same
// tensor name; the explicit-file inventory must keep them distinct.
func TestFromFilesNamespacesDualHeadTensors(t *testing.T) {
	directory := t.TempDir()
	var specs []FileSpec
	for _, head := range []string{"classification", "regression"} {
		headDir := filepath.Join(directory, head)
		if err := os.MkdirAll(headDir, 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(filepath.Join(headDir, "config.json"), []byte(fixtureConfig), 0o600); err != nil {
			t.Fatal(err)
		}
		writeSafetensor(t, filepath.Join(headDir, "model.safetensors"), fixtureTensorName)
		specs = append(specs,
			FileSpec{Path: filepath.Join(headDir, "config.json"), Name: head + "/config", Role: artifact.ComponentConfig},
			FileSpec{Path: filepath.Join(headDir, "model.safetensors"), Name: head + "/weights", Role: artifact.ComponentWeights},
		)
	}
	inventory, err := FromFiles(directory, specs)
	if err != nil {
		t.Fatal(err)
	}
	if len(inventory.Components) != 4 || len(inventory.Manifest.Components) != 4 || len(inventory.Locations) != 5 {
		t.Fatalf("inventory shape = %d/%d/%d", len(inventory.Components), len(inventory.Manifest.Components), len(inventory.Locations))
	}
	roles := map[artifact.ComponentRole]int{}
	for _, component := range inventory.Manifest.Components {
		roles[component.Role]++
	}
	if roles[artifact.ComponentConfig] != 2 || roles[artifact.ComponentWeights] != 2 {
		t.Fatalf("roles = %v", roles)
	}
	for _, name := range []string{
		"classification/weights/" + fixtureTensorName,
		"regression/weights/" + fixtureTensorName,
	} {
		if _, ok := inventory.TensorInventory.Tensor(name); !ok {
			t.Fatalf("tensor fact %q absent", name)
		}
	}
	if _, ok := inventory.TensorInventory.Tensor(fixtureTensorName); ok {
		t.Fatal("un-namespaced tensor fact leaked")
	}
	// Relocation-stable identity: a copy elsewhere yields the same manifest.
	other := t.TempDir()
	var otherSpecs []FileSpec
	for _, head := range []string{"classification", "regression"} {
		headDir := filepath.Join(other, head)
		if err := os.MkdirAll(headDir, 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(filepath.Join(headDir, "config.json"), []byte(fixtureConfig), 0o600); err != nil {
			t.Fatal(err)
		}
		writeSafetensor(t, filepath.Join(headDir, "model.safetensors"), fixtureTensorName)
		otherSpecs = append(otherSpecs,
			FileSpec{Path: filepath.Join(headDir, "config.json"), Name: head + "/config", Role: artifact.ComponentConfig},
			FileSpec{Path: filepath.Join(headDir, "model.safetensors"), Name: head + "/weights", Role: artifact.ComponentWeights},
		)
	}
	relocated, err := FromFiles(other, otherSpecs)
	if err != nil {
		t.Fatal(err)
	}
	if relocated.Manifest.ID != inventory.Manifest.ID || relocated.TensorInventory.ID != inventory.TensorInventory.ID {
		t.Fatalf("relocated identities differ: %s != %s", relocated.Manifest.ID, inventory.Manifest.ID)
	}
}

// TestFromFilesRejectsSharedWeightsDirectory: a weights file must stand alone
// in its directory; sibling safetensors would smuggle facts.
func TestFromFilesRejectsSharedWeightsDirectory(t *testing.T) {
	directory := t.TempDir()
	writeSafetensor(t, filepath.Join(directory, "model.safetensors"), "first")
	writeSafetensor(t, filepath.Join(directory, "extra.safetensors"), "second")
	_, err := FromFiles(directory, []FileSpec{
		{Path: filepath.Join(directory, "model.safetensors"), Name: "weights", Role: artifact.ComponentWeights},
	})
	if err == nil {
		t.Fatal("accepted a weights file with safetensors siblings")
	}
}
