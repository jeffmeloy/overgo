package trainingworkflow

import (
	"os"
	"path/filepath"
	"testing"

	"overgo/internal/artifact"
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
// its own hash, model.safetensors directly, a single safetensors file
// under any recorded name, and a sharded checkpoint by its first shard in
// order, the file intake recorded as the model location. Two unordered
// safetensors files are ambiguous and refuse, and an activation whose
// recorded bytes exist never fails at identification.
func TestTrainingIdentifiesRecordedInputs(t *testing.T) {
	canonical := t.TempDir()
	expected := identityFixtureFile(t, canonical, "model.safetensors", "canonical-weights")
	identityFixtureFile(t, canonical, "extra.safetensors", "extra-weights")
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

	sharded := t.TempDir()
	expected = identityFixtureFile(t, sharded, "model-00001-of-00003.safetensors", "shard-one")
	identityFixtureFile(t, sharded, "model-00002-of-00003.safetensors", "shard-two")
	identityFixtureFile(t, sharded, "model-00003-of-00003.safetensors", "shard-three")
	id, err = identifyModel(sharded)
	if err != nil || id != expected {
		t.Fatalf("sharded identity = (%s, %v), want the first shard %s", id, err, expected)
	}

	ggufPath := writeTrainingGGUF(t)
	ggufFile, err := os.Open(ggufPath)
	if err != nil {
		t.Fatal(err)
	}
	expected, _, err = artifact.Identify(artifact.KindModel, ggufFile)
	if closeErr := ggufFile.Close(); err != nil || closeErr != nil {
		t.Fatal(err, closeErr)
	}
	id, err = identifyModel(ggufPath)
	if err != nil || id != expected {
		t.Fatalf("gguf identity = (%s, %v), want %s", id, err, expected)
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
}
