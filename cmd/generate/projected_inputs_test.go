package main

import (
	"os"
	"overgo/internal/testskip"
	"path/filepath"
	"testing"
)

func requireIntegration(t *testing.T) {
	t.Helper()
	if testing.Short() {
		t.Skip(testskip.ShortIntegration)
	}
}

func TestReadProjectedInputs(t *testing.T) {
	path := filepath.Join(t.TempDir(), "projected.json")
	data := []byte(`{
  "embedding_overrides": [{"token_index": 1, "embedding": [1, 2]}],
  "multi_axis_positions": [[0, 1], [0, 1], [0, 1], [0, 1]],
  "deepstack_embeddings": [{"shape": [2, 2], "data": [1, 2, 3, 4]}]
}`)
	if err := os.WriteFile(path, data, 0o600); err != nil {
		t.Fatal(err)
	}
	inputs, err := readProjectedInputs(path)
	if err != nil {
		t.Fatal(err)
	}
	if len(inputs.EmbeddingOverrides) != 1 ||
		inputs.EmbeddingOverrides[0].TokenIndex != 1 ||
		inputs.MultiAxisPositions == nil ||
		len(inputs.MultiAxisPositions[3]) != 2 ||
		len(inputs.DeepstackEmbeddings) != 1 ||
		inputs.DeepstackEmbeddings[0].Shape.Dims[1] != 2 {
		t.Fatalf("unexpected projected inputs: %+v", inputs)
	}
}

func TestReadProjectedInputsRejectsInvalidTensor(t *testing.T) {
	for _, data := range []string{
		`{"unknown": true}`,
		`{"deepstack_embeddings":[{"shape":[2,2],"data":[1]}]}`,
	} {
		path := filepath.Join(t.TempDir(), "projected.json")
		if err := os.WriteFile(path, []byte(data), 0o600); err != nil {
			t.Fatal(err)
		}
		if _, err := readProjectedInputs(path); err == nil {
			t.Fatalf("invalid projected inputs accepted: %s", data)
		}
	}
}
