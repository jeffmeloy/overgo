package libraryintake

import (
	"context"
	"errors"
	"os"
	"path/filepath"
	"testing"

	"overgo/internal/modelrecipe"
	"overgo/internal/overgodb"
	"overgo/internal/safetensors"
	"overgo/internal/testutil"
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

// TestARegisteredDirectoryRunsUnderItsRegisteredIdentity: a directory the view serves is the one model whether it is
// registered or run, so the catalog lists it once.
func TestARegisteredDirectoryRunsUnderItsRegisteredIdentity(t *testing.T) {
	t.Parallel()
	store, err := overgodb.Create(filepath.Join(t.TempDir(), "store"))
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	if _, err := modelrecipe.PublishArchitectureProfileCatalog(t.Context(), store); err != nil {
		t.Fatal(err)
	}
	directory := testutil.Qwen35Directory(t)
	receipt, err := Register(t.Context(), store, directory, "")
	if err != nil {
		t.Fatal(err)
	}
	loaded, err := modelrecipe.ResolveActiveGGUF(t.Context(), store, directory)
	if err != nil {
		t.Fatal(err)
	}
	defer loaded.Close()
	identity, err := loaded.Identity()
	if err != nil {
		t.Fatal(err)
	}
	if identity.Model != receipt["model"] {
		t.Fatalf("ran as %v, registered as %v", identity.Model, receipt["model"])
	}
}

// TestAViewedDirectoryValidatesAsItRegisters: a directory the view serves has no generation source, and validates under
// the recipe it registers by, as a trained model's directory must to have its measurement.
func TestAViewedDirectoryValidatesAsItRegisters(t *testing.T) {
	t.Parallel()
	store, err := overgodb.Create(filepath.Join(t.TempDir(), "store"))
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	if _, err := modelrecipe.PublishArchitectureProfileCatalog(t.Context(), store); err != nil {
		t.Fatal(err)
	}
	directory := testutil.Qwen35Directory(t)
	request, execute, err := validationAt(t.Context(), store, directory, "", "a revision", validationPolicy{})
	if err != nil || execute == nil {
		t.Fatalf("the directory does not validate: %v", err)
	}
	receipt, err := Register(t.Context(), store, directory, "")
	if err != nil {
		t.Fatal(err)
	}
	if request.Recipe != receipt["recipe"] {
		t.Fatalf("validates under %v, registers under %v", request.Recipe, receipt["recipe"])
	}
}
