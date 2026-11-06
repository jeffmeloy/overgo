package modelartifact

import (
	"context"
	"errors"
	"os"
	"path/filepath"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/overgodbtest"
	"overgo/internal/testcheck"
)

func TestReinspectStoredFilesRefusesPhysicalDrift(t *testing.T) {
	root := t.TempDir()
	path := filepath.Join(root, "model.safetensors")
	writeSafetensor(t, path, "first")
	inventory := testcheck.Must(FromFiles(root, []FileSpec{{Path: path, Name: "weights", Role: artifact.ComponentWeights}}))(t)
	store := overgodbtest.Open(t, t.TempDir())
	batch := testcheck.Must(inventory.Batch("test/model"))(t)
	if _, err := artifact.CommitBatch(t.Context(), store, batch); err != nil {
		t.Fatal(err)
	}
	actual, err := ReinspectFiles(t.Context(), store, inventory.Manifest.ID)
	if err != nil || actual.TensorInventory.ID != inventory.TensorInventory.ID {
		t.Fatalf("exact reinspection: %v", err)
	}
	ctx, cancel := context.WithCancelCause(t.Context())
	cancel(context.Canceled)
	if _, err := ReinspectFiles(ctx, store, inventory.Manifest.ID); !errors.Is(err, context.Canceled) {
		t.Fatalf("cancellation: %v", err)
	}
	writeSafetensor(t, path, "changed")
	if _, err := ReinspectFiles(t.Context(), store, inventory.Manifest.ID); err == nil {
		t.Fatal("modified checkpoint accepted")
	}
	testcheck.NoError(t, os.Remove(path))
	if _, err := ReinspectFiles(t.Context(), store, inventory.Manifest.ID); err == nil {
		t.Fatal("missing checkpoint accepted")
	}
	if _, err := ReinspectFiles(nil, store, inventory.Manifest.ID); err == nil {
		t.Fatal("nil context accepted")
	}
}
