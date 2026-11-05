package modelartifact

import (
	"archive/zip"
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/testcheck"
)

func TestFromFilesPyTorchCompoundSuffix(t *testing.T) {
	// A protocol-2 torch state_dict with one float32 weight, shape [1].
	metadata := "\x80\x02}X\x06\x00\x00\x00weightctorch._utils\n_rebuild_tensor_v2\n((X\x07\x00\x00\x00storagectorch\nFloatStorage\nX\x01\x00\x00\x000X\x03\x00\x00\x00cpuK\x01tQK\x00K\x01\x85K\x01\x85\x89)tRs."
	var encoded bytes.Buffer
	archive := zip.NewWriter(&encoded)
	for name, body := range map[string]string{"archive/data.pkl": metadata, "archive/data/0": "\x00\x00\x80\x3f"} {
		entry, err := archive.CreateHeader(&zip.FileHeader{Name: name, Method: zip.Store})
		testcheck.NoError(t, err)
		if _, err := entry.Write([]byte(body)); err != nil {
			t.Fatal(err)
		}
	}
	testcheck.NoError(t, archive.Close())
	var expected artifact.ID
	for _, suffix := range []string{".pth", ".pt", ".pth.tar", ".PT.TAR"} {
		t.Run(suffix, func(t *testing.T) {
			root := t.TempDir()
			path := filepath.Join(root, "model"+suffix)
			testcheck.NoError(t, os.WriteFile(path, encoded.Bytes(), 0o600))
			inventory, err := FromFiles(root, []FileSpec{{Path: path, Name: "weights", Role: artifact.ComponentWeights}})
			testcheck.NoError(t, err)
			fact, found := inventory.TensorInventory.Tensor("weights/weight")
			if !found || fact.Storage != "f32" || fact.Bytes != 4 || len(fact.Shape) != 1 || fact.Shape[0] != 1 {
				t.Fatalf("tensor facts: %+v", inventory.TensorInventory)
			}
			if expected.Valid() && inventory.Manifest.ID != expected {
				t.Fatal("filename suffix changed the same checkpoint's identity")
			}
			expected = inventory.Manifest.ID
			testcheck.NoError(t, os.WriteFile(path, []byte("not a ZIP checkpoint"), 0o600))
			if _, err := FromFiles(root, []FileSpec{{Path: path, Name: "weights", Role: artifact.ComponentWeights}}); err == nil {
				t.Fatal("accepted invalid PyTorch container")
			}
		})
	}
}

func TestFromFilesInventoriesIndexedShardsOnce(t *testing.T) {
	directory := t.TempDir()
	shards := []string{"model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"}
	writeSafetensor(t, filepath.Join(directory, shards[0]), "first")
	writeSafetensor(t, filepath.Join(directory, shards[1]), "second")
	index, err := json.Marshal(map[string]any{
		"weight_map": map[string]string{"first": shards[0], "second": shards[1]},
	})
	testcheck.NoError(t, err)
	indexPath := filepath.Join(directory, "model.safetensors.index.json")
	testcheck.NoError(t, os.WriteFile(indexPath, index, 0o600))
	inventory, err := FromFiles(directory, []FileSpec{
		{Path: indexPath, Name: "shard-index", Role: artifact.ComponentShardIndex},
		{Path: filepath.Join(directory, shards[0]), Name: "weights-0", Role: artifact.ComponentWeightsShard},
		{Path: filepath.Join(directory, shards[1]), Name: "weights-1", Role: artifact.ComponentWeightsShard},
	})
	testcheck.NoError(t, err)
	if len(inventory.TensorInventory.Tensors) != len(shards) {
		t.Fatalf("tensor facts = %d, want %d", len(inventory.TensorInventory.Tensors), len(shards))
	}
	if inventory.TensorInventory.Tensors[0].Name != "weights-0/first" ||
		inventory.TensorInventory.Tensors[1].Name != "weights-1/second" {
		t.Fatalf("tensor facts = %+v", inventory.TensorInventory.Tensors)
	}
}

func TestFromFilesInventoriesNonstandardIndexedShards(t *testing.T) {
	directory := t.TempDir()
	shards := []string{
		"diffusion_pytorch_model-00001-of-00002.safetensors",
		"diffusion_pytorch_model-00002-of-00002.safetensors",
	}
	writeSafetensor(t, filepath.Join(directory, shards[0]), "first")
	writeSafetensor(t, filepath.Join(directory, shards[1]), "second")
	index, err := json.Marshal(map[string]any{
		"weight_map": map[string]string{"first": shards[0], "second": shards[1]},
	})
	testcheck.NoError(t, err)
	indexPath := filepath.Join(directory, "diffusion_pytorch_model.safetensors.index.json")
	testcheck.NoError(t, os.WriteFile(indexPath, index, 0o600))
	inventory, err := FromFiles(directory, []FileSpec{
		{Path: indexPath, Name: "shard-index", Role: artifact.ComponentShardIndex},
		{Path: filepath.Join(directory, shards[0]), Name: "weights-0", Role: artifact.ComponentWeightsShard},
		{Path: filepath.Join(directory, shards[1]), Name: "weights-1", Role: artifact.ComponentWeightsShard},
	})
	testcheck.NoError(t, err)
	if len(inventory.TensorInventory.Tensors) != 2 {
		t.Fatalf("tensor facts = %d, want 2", len(inventory.TensorInventory.Tensors))
	}
}

// TestFromFilesNamespacesDualHeadTensors: two sub-model heads reuse the same
// tensor name; the explicit-file inventory must keep them distinct.
func TestFromFilesNamespacesDualHeadTensors(t *testing.T) {
	directory := t.TempDir()
	var specs []FileSpec
	for _, head := range []string{"classification", "regression"} {
		headDir := filepath.Join(directory, head)
		testcheck.NoError(t, os.MkdirAll(headDir, 0o755))
		testcheck.NoError(t, os.WriteFile(filepath.Join(headDir, "config.json"), []byte(fixtureConfig), 0o600))
		writeSafetensor(t, filepath.Join(headDir, "model.safetensors"), fixtureTensorName)
		specs = append(specs,
			FileSpec{Path: filepath.Join(headDir, "config.json"), Name: head + "/config", Role: artifact.ComponentConfig},
			FileSpec{Path: filepath.Join(headDir, "model.safetensors"), Name: head + "/weights", Role: artifact.ComponentWeights},
		)
	}
	inventory, err := FromFiles(directory, specs)
	testcheck.NoError(t, err)
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
		testcheck.NoError(t, os.MkdirAll(headDir, 0o755))
		testcheck.NoError(t, os.WriteFile(filepath.Join(headDir, "config.json"), []byte(fixtureConfig), 0o600))
		writeSafetensor(t, filepath.Join(headDir, "model.safetensors"), fixtureTensorName)
		otherSpecs = append(otherSpecs,
			FileSpec{Path: filepath.Join(headDir, "config.json"), Name: head + "/config", Role: artifact.ComponentConfig},
			FileSpec{Path: filepath.Join(headDir, "model.safetensors"), Name: head + "/weights", Role: artifact.ComponentWeights},
		)
	}
	relocated, err := FromFiles(other, otherSpecs)
	testcheck.NoError(t, err)
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
