package modelartifact

import (
	"context"
	"errors"

	"bytes"
	"math"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/overgodb"
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
	parsed, err := tensorInventoryCodec.Parse(content.Data)
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
		"empty":          {},
		"unordered":      {{Name: "z", Shape: []uint64{}, Storage: "f32"}, {Name: "a", Shape: []uint64{}, Storage: "f32"}},
		"duplicate":      {{Name: "a", Shape: []uint64{}, Storage: "f32"}, {Name: "a", Shape: []uint64{}, Storage: "f32"}},
		"storage":        {{Name: "a", Shape: []uint64{}, Storage: "F32"}},
		"nil shape":      {{Name: "a", Storage: "f32"}},
		"zero bytes":     {{Name: "a", Shape: []uint64{}, Storage: "f32"}},
		"zero dimension": {{Name: "a", Shape: []uint64{0}, Storage: "f32", Bytes: 4}},
		"shape overflow": {{Name: "a", Shape: []uint64{math.MaxUint64, 2}, Storage: "f32", Bytes: 4}},
	}
	for name, tensors := range tests {
		t.Run(name, func(t *testing.T) {
			if _, err := NewTensorInventoryDocument(modelID, TensorFormatGGUF, tensors); err == nil {
				t.Fatal("invalid tensor inventory accepted")
			}
		})
	}
}

func TestPyTorchTensorInventoryDocument(t *testing.T) {
	modelID := testutil.ArtifactID(t, artifact.KindModel, "pytorch-inventory-model")
	document, err := NewTensorInventoryDocument(modelID, TensorFormatPyTorch, []TensorFact{
		{Name: "weights/embed", Shape: []uint64{16, 8}, Storage: "bf16", Bytes: 256},
	})
	if err != nil {
		t.Fatal(err)
	}
	if document.Format != TensorFormatPyTorch {
		t.Fatalf("format = %s", document.Format)
	}
}

func TestTensorInventoryAcceptsProjectorOwner(t *testing.T) {
	owner := testutil.ArtifactID(t, artifact.KindProjector, "tensor-inventory-projector")
	document, err := NewTensorInventoryDocument(owner, TensorFormatGGUF, []TensorFact{{
		Name: "projection.weight", Shape: []uint64{2, 2}, Storage: "f32", Bytes: 16,
	}})
	if err != nil {
		t.Fatal(err)
	}
	if document.Owner != owner || document.Version != TensorInventoryVersion {
		t.Fatalf("inventory owner/version = %s/%d", document.Owner, document.Version)
	}
	content, err := document.Content()
	if err != nil {
		t.Fatal(err)
	}
	store, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	_, err = store.Commit(t.Context(), artifact.Batch{
		Key: "test/projector-inventory", Artifacts: []artifact.Descriptor{{ID: owner}},
		Contents: []artifact.Content{content}, Lineage: []artifact.Lineage{{
			Child: document.ID, Parent: owner, Relation: artifact.RelationDerivedFrom,
		}},
	})
	if err != nil {
		t.Fatal(err)
	}
	loaded, found, err := LoadTensorInventory(t.Context(), store, owner)
	if err != nil || !found || loaded.ID != document.ID {
		t.Fatalf("loaded projector inventory = (%s, %t, %v)", loaded.ID, found, err)
	}
}

// LoadTensorInventory resolves the unique inventory derived from its owner.
func LoadTensorInventory(
	ctx context.Context,
	store artifact.Reader,
	owner artifact.ID,
) (TensorInventoryDocument, bool, error) {
	if store == nil || !tensorBearingKind(owner.Kind()) {
		return TensorInventoryDocument{}, false, errors.New("tensor inventory: invalid owner lookup")
	}
	edges, err := store.Children(ctx, owner)
	if err != nil {
		return TensorInventoryDocument{}, false, err
	}
	var found TensorInventoryDocument
	matched := false
	for _, edge := range edges {
		if edge.Relation != artifact.RelationDerivedFrom || edge.Child.Kind() != artifact.KindTensorInventory {
			continue
		}
		if matched {
			return TensorInventoryDocument{}, false, errors.New("tensor inventory: multiple owner inventories")
		}
		document, ok, contentErr := tensorInventoryCodec.Read(ctx, store, edge.Child)
		if contentErr != nil {
			return TensorInventoryDocument{}, false, contentErr
		}
		if !ok {
			return TensorInventoryDocument{}, false, errors.New("tensor inventory: content is absent or incompatible")
		}
		found = document
		if found.Owner != owner || found.ID != edge.Child {
			return TensorInventoryDocument{}, false, errors.New("tensor inventory: owner lineage mismatch")
		}
		matched = true
	}
	return found, matched, nil
}
