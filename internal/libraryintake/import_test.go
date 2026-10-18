package libraryintake

import (
	"context"
	"errors"
	"os"
	"path/filepath"
	"testing"

	"overgo/internal/overgodb"
	"overgo/internal/safetensors"
)

func TestComponentImportInspectionRoutes(t *testing.T) {
	path := t.TempDir()
	if err := os.WriteFile(filepath.Join(path, "config.json"), []byte(`{"model_type":"unregistered","architectures":["RemoteModel"]}`), 0600); err != nil {
		t.Fatal(err)
	}
	weights := filepath.Join(path, "model.safetensors")
	if err := safetensors.Save(weights, map[string][]float32{"weight": {1}}, map[string][]int{"weight": {1}}, nil); err != nil {
		t.Fatal(err)
	}
	store, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	model, projector, err := ModelFiles(path, "")
	if err != nil {
		t.Fatal(err)
	}
	initial, _ := store.Head()
	ctx, cancel := context.WithCancel(t.Context())
	cancel()
	if _, err := Register(ctx, store, model, projector); !errors.Is(err, context.Canceled) {
		t.Fatalf("cancelled library import = %v", err)
	}
	after, _ := store.Head()
	if after != initial {
		t.Fatal("cancelled import published facts")
	}
	receipt, err := Register(t.Context(), store, model, projector)
	if err != nil {
		t.Fatal(err)
	}
	if receipt["inspection"] == nil || receipt["recipe"] != nil || receipt["qualification"] != "structural-only" {
		t.Fatalf("source registration receipt = %+v", receipt)
	}
	beforeCorruption, _ := store.Head()
	if err := os.WriteFile(weights, []byte("corrupt"), 0600); err != nil {
		t.Fatal(err)
	}
	if _, err := Register(t.Context(), store, model, projector); err == nil {
		t.Fatal("corrupt source was registered")
	}
	after, _ = store.Head()
	if after != beforeCorruption {
		t.Fatal("corrupt import changed prior registration")
	}
}
