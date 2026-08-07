package modelartifact

import (
	"bytes"
	"testing"

	"overgo/internal/artifact"
)

func TestTensorInventoryDocumentRoundTrip(t *testing.T) {
	modelID, err := artifact.IdentifyBytes(artifact.KindModel, []byte("tensor-inventory-model"))
	if err != nil {
		t.Fatal(err)
	}
	document, err := NewTensorInventoryDocument(modelID, TensorFormatSafetensors, []TensorFact{
		{Name: "matrix", Shape: []uint64{2, 3}, Storage: "bf16", Bytes: 12},
		{Name: "scalar", Shape: []uint64{}, Storage: "f32", Bytes: 4},
	})
	if err != nil {
		t.Fatal(err)
	}
	content, err := document.ContentBytes()
	if err != nil {
		t.Fatal(err)
	}
	parsed, err := ParseTensorInventoryDocument(content)
	if err != nil {
		t.Fatal(err)
	}
	parsedContent, err := parsed.ContentBytes()
	if err != nil {
		t.Fatal(err)
	}
	if parsed.ID != document.ID || !bytes.Equal(parsedContent, content) {
		t.Fatal("tensor inventory round trip drifted")
	}
}

func TestTensorInventoryDocumentRejectsInvalidFacts(t *testing.T) {
	modelID, err := artifact.IdentifyBytes(artifact.KindModel, []byte("invalid-tensor-inventory-model"))
	if err != nil {
		t.Fatal(err)
	}
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
