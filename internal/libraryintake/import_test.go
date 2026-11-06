package libraryintake

import (
	"context"
	"errors"
	"os"
	"path/filepath"
	"testing"

	"overgo/internal/modelrecipe"
	"overgo/internal/overgodbtest"
	"overgo/internal/safetensors"
	"overgo/internal/testcheck"
	"overgo/internal/testutil"
)

func TestComponentImportInspectionRoutes(t *testing.T) {
	path := t.TempDir()
	testcheck.NoError(t, os.WriteFile(filepath.Join(path, "config.json"), []byte(`{"model_type":"unregistered","architectures":["RemoteModel"]}`), 0600))
	weights := filepath.Join(path, "model.safetensors")
	testcheck.NoError(t, safetensors.Save(weights, map[string][]float32{"weight": {1}}, map[string][]int{"weight": {1}}, nil))
	store := overgodbtest.Open(t, t.TempDir())
	model, projector, err := ModelFiles(path, "")
	testcheck.NoError(t, err)
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
	receipt := testcheck.Must(Register(t.Context(), store, model, projector))(t)
	if receipt["inspection"] == nil || receipt["recipe"] != nil || receipt["qualification"] != "structural-only" {
		t.Fatalf("source registration receipt = %+v", receipt)
	}
	beforeCorruption, _ := store.Head()
	testcheck.NoError(t, os.WriteFile(weights, []byte("corrupt"), 0600))
	if _, err := Register(t.Context(), store, model, projector); err == nil {
		t.Fatal("corrupt source was registered")
	}
	after, _ = store.Head()
	if after != beforeCorruption {
		t.Fatal("corrupt import changed prior registration")
	}
}

// TestARegisteredDirectoryRunsUnderItsRegisteredIdentity: a directory the view serves is the one model whether it is
// registered, validated or run, so the catalog lists it once. It has no generation source, and validates under the recipe
// it registers by, as a trained model's directory must to have its measurement.
func TestARegisteredDirectoryRunsUnderItsRegisteredIdentity(t *testing.T) {
	t.Parallel()
	store := overgodbtest.Create(t, filepath.Join(t.TempDir(), "store"))
	if _, err := modelrecipe.PublishArchitectureProfileCatalog(t.Context(), store); err != nil {
		t.Fatal(err)
	}
	directory := testutil.Qwen35Directory(t)
	request, execute, err := validationAt(t.Context(), store, directory, "", "a revision", validationPolicy{})
	if err != nil || execute == nil {
		t.Fatalf("the directory does not validate: %v", err)
	}
	receipt := testcheck.Must(Register(t.Context(), store, directory, ""))(t)
	if request.Recipe != receipt["recipe"] {
		t.Fatalf("validates under %v, registers under %v", request.Recipe, receipt["recipe"])
	}
	loaded := testcheck.Must(modelrecipe.ResolveActiveGGUF(t.Context(), store, directory))(t)
	defer loaded.Close()
	identity := testcheck.Must(loaded.Identity())(t)
	if identity.Model != receipt["model"] {
		t.Fatalf("ran as %v, registered as %v", identity.Model, receipt["model"])
	}
}
