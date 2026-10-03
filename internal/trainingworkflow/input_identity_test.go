package trainingworkflow

import (
	"fmt"
	"os"
	"path/filepath"
	"slices"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/gguf"
	"overgo/internal/modelartifact"
	"overgo/internal/overgodb"
	"overgo/internal/safetensors"
	"overgo/internal/trainingprogram"
)

func identityFixtureFile(t *testing.T, directory, name, content string) artifact.ID {
	t.Helper()
	path := filepath.Join(directory, name)
	if err := os.WriteFile(path, []byte(content), 0o600); err != nil {
		t.Fatal(err)
	}
	id, err := artifact.IdentifyBytes(artifact.KindModel, []byte(content))
	if err != nil {
		t.Fatal(err)
	}
	return id
}

// TestTrainingIdentifiesRecordedInputs pins the input-identity contract:
// the workflow identifies every input the store records — a GGUF file by
// its complete manifest, model.safetensors directly, and a single
// safetensors file under any recorded name. Two unordered
// safetensors files are ambiguous and refuse, and an activation whose
// recorded bytes exist never fails at identification.
func TestTrainingIdentifiesRecordedInputs(t *testing.T) {
	canonical := t.TempDir()
	expected := identityFixtureFile(t, canonical, "model.safetensors", "canonical-weights")
	id, err := identifyModel(canonical)
	if err != nil || id != expected {
		t.Fatalf("canonical identity = (%s, %v), want %s", id, err, expected)
	}

	alternate := t.TempDir()
	expected = identityFixtureFile(t, alternate, "diffusion_pytorch_model.safetensors", "diffusion-weights")
	id, err = identifyModel(alternate)
	if err != nil || id != expected {
		t.Fatalf("alternate-name identity = (%s, %v), want %s", id, err, expected)
	}

	ggufPath := writeTrainingGGUF(t)
	ggufFile, err := gguf.Open(ggufPath)
	if err != nil {
		t.Fatal(err)
	}
	inventory, err := modelartifact.FromGGUF(ggufFile, artifact.KindModel)
	if closeErr := ggufFile.Close(); err != nil || closeErr != nil {
		t.Fatal(err, closeErr)
	}
	expected = inventory.Manifest.ID
	id, err = identifyModel(ggufPath)
	if err != nil || id != expected {
		t.Errorf("gguf identity = (%s, %v), want %s", id, err, expected)
	}

	ambiguous := t.TempDir()
	identityFixtureFile(t, ambiguous, "first.safetensors", "first-weights")
	identityFixtureFile(t, ambiguous, "second.safetensors", "second-weights")
	if _, err := identifyModel(ambiguous); err == nil {
		t.Fatal("ambiguous directory identified")
	}
	if _, err := identifyModel(t.TempDir()); err == nil {
		t.Fatal("empty directory identified")
	}
	t.Run("indexed-safetensors", func(t *testing.T) {
		directory := t.TempDir()
		for index, name := range []string{"first", "second", "third"} {
			file := fmt.Sprintf("model-%05d-of-00003.safetensors", index+1)
			if err := safetensors.Save(filepath.Join(directory, file), map[string][]float32{name: {float32(index)}}, map[string][]int{name: {1}}, nil); err != nil {
				t.Fatal(err)
			}
		}
		if err := os.WriteFile(filepath.Join(directory, "model.safetensors.index.json"), []byte(`{"weight_map":{"first":"model-00001-of-00003.safetensors","second":"model-00002-of-00003.safetensors","third":"model-00003-of-00003.safetensors"}}`), 0o644); err != nil {
			t.Fatal(err)
		}
		inventory, err := modelartifact.FromSafetensorsPath(directory)
		if err != nil {
			t.Fatal(err)
		}
		id, err := identifyModel(directory)
		if err != nil || id != inventory.Manifest.ID {
			t.Fatalf("indexed identity = (%s, %v), want %s", id, err, inventory.Manifest.ID)
		}
		sibling := filepath.Join(directory, "model-00002-of-00003.safetensors")
		if err := safetensors.Save(sibling, map[string][]float32{"second": {5}}, map[string][]int{"second": {1}}, nil); err != nil {
			t.Fatal(err)
		}
		changed, err := identifyModel(directory)
		if err != nil || changed == id {
			t.Fatalf("changed Safetensors sibling = (%s, %v), old %s", changed, err, id)
		}
		if err := os.Remove(sibling); err != nil {
			t.Fatal(err)
		}
		if _, err := identifyModel(directory); err == nil {
			t.Fatal("missing indexed sibling identified")
		}
	})
	t.Run("split-gguf", func(t *testing.T) {
		root := t.TempDir()
		store, err := overgodb.Open(filepath.Join(root, "store"))
		if err != nil {
			t.Fatal(err)
		}
		defer store.Close()
		base := writeTrainedFixtureBase(t, root)
		directory := filepath.Join(root, "split")
		if err := os.Mkdir(directory, 0o755); err != nil {
			t.Fatal(err)
		}
		if err := WriteTrainedModel(t.Context(), store, base, directory, 1, lookupTrained(map[string][]float32{"changed.weight": {5, 6}})); err != nil {
			t.Fatal(err)
		}
		path := filepath.Join(directory, trainingprogram.CheckpointModel)
		id, err := identifyModel(path)
		if err != nil {
			t.Fatal(err)
		}
		recorded, err := identifyAndRecordWeights(t.Context(), store, path)
		if err != nil || recorded != id {
			t.Fatalf("bootstrap identity = (%s, %v), want %s", recorded, err, id)
		}
		frozen := filepath.Join(directory, trainingprogram.CheckpointFrozenShard)
		original, err := os.ReadFile(frozen)
		if err != nil {
			t.Fatal(err)
		}
		changed := slices.Clone(original)
		changed[len(changed)-1] ^= 1
		if err := os.WriteFile(frozen, changed, 0o644); err != nil {
			t.Fatal(err)
		}
		changedID, err := identifyModel(path)
		if err != nil || changedID == id {
			t.Fatalf("changed sibling identity = (%s, %v), old %s", changedID, err, id)
		}
		recorded, err = identifyAndRecordWeights(t.Context(), store, path)
		if err != nil || recorded != changedID {
			t.Fatalf("changed bootstrap identity = (%s, %v), want %s", recorded, err, changedID)
		}
		if err := os.Remove(frozen); err != nil {
			t.Fatal(err)
		}
		if _, err := identifyModel(path); err == nil {
			t.Error("missing sibling identified")
		}
		if _, err := identifyAndRecordWeights(t.Context(), store, path); err == nil {
			t.Error("missing sibling bootstrapped")
		}
	})
}
