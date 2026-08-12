package modelartifact

import (
	"bytes"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/testutil"
)

func TestTensorInventoryDocumentRoundTrip(t *testing.T) {
	modelID := testutil.ArtifactID(t, artifact.KindModel, "tensor-inventory-model")
	document, err := NewTensorInventoryDocument(modelID, TensorFormatSafetensors, []TensorFact{
		{Name: "matrix", Shape: []uint64{2, 3}, Storage: "bf16", Bytes: 12},
		{Name: "scalar", Shape: []uint64{}, Storage: "f32", Bytes: 4},
	})
	if err != nil {
		t.Fatal(err)
	}
	content, err := document.Content()
	if err != nil {
		t.Fatal(err)
	}
	parsed, err := ParseTensorInventoryDocument(content.Data)
	if err != nil {
		t.Fatal(err)
	}
	parsedContent, err := parsed.Content()
	if err != nil {
		t.Fatal(err)
	}
	if parsed.ID != document.ID || !bytes.Equal(parsedContent.Data, content.Data) {
		t.Fatal("tensor inventory round trip drifted")
	}
}

func TestTensorInventoryDocumentRejectsInvalidFacts(t *testing.T) {
	modelID := testutil.ArtifactID(t, artifact.KindModel, "invalid-tensor-inventory-model")
	tests := map[string][]TensorFact{
		"empty":     {},
		"unordered": {{Name: "z", Shape: []uint64{}, Storage: "f32"}, {Name: "a", Shape: []uint64{}, Storage: "f32"}},
		"duplicate": {{Name: "a", Shape: []uint64{}, Storage: "f32"}, {Name: "a", Shape: []uint64{}, Storage: "f32"}},
		"storage":   {{Name: "a", Shape: []uint64{}, Storage: "F32"}},
		"nil shape": {{Name: "a", Storage: "f32"}},
	}
	for name, tensors := range tests {
		t.Run(name, func(t *testing.T) {
			if _, err := NewTensorInventoryDocument(modelID, TensorFormatGGUF, tensors); err == nil {
				t.Fatal("invalid tensor inventory accepted")
			}
		})
	}
}
