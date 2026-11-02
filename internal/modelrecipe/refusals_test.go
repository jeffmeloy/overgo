package modelrecipe

import (
	"bytes"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/gguf"
	"overgo/internal/modelartifact"
	"overgo/internal/testutil"
)

// TestOnlyMissingPartsRefuseARun: a load is refused only for what it cannot run without, each refusal naming what is
// missing, and a path only remembered runs as what it holds now.
func TestOnlyMissingPartsRefuseARun(t *testing.T) {
	t.Parallel()
	open := func(t *testing.T) artifact.Repository {
		t.Helper()
		store := openProgramStore(t)
		t.Cleanup(func() { store.Close() })
		return store
	}
	t.Run("weights that are not there", func(t *testing.T) {
		t.Parallel()
		path := filepath.Join(t.TempDir(), "gone.gguf")
		if _, err := ResolveActiveGGUF(t.Context(), open(t), path); err == nil || !strings.Contains(err.Error(), "gone.gguf") {
			t.Fatalf("missing weights: %v", err)
		}
	})
	t.Run("an architecture with no prototype", func(t *testing.T) {
		t.Parallel()
		path := filepath.Join(t.TempDir(), "model.gguf")
		testutil.WriteGGUF(t, path, []gguf.Metadata{definitionMetadataValue("general.architecture", gguf.ValueTypeString, "nonesuch")}, []gguf.TensorData{
			{Name: "weight", Shape: []uint64{1}, Type: gguf.DTypeF32, Data: bytes.NewReader(make([]byte, definitionF32Bytes))},
		})
		if _, err := ResolveActiveGGUF(t.Context(), open(t), path); err == nil || !strings.Contains(err.Error(), "nonesuch") {
			t.Fatalf("an architecture nothing compiles: %v", err)
		}
	})
	t.Run("an activated path that holds other bytes", func(t *testing.T) {
		t.Parallel()
		directory := t.TempDir()
		path, other := filepath.Join(directory, "model.gguf"), filepath.Join(directory, "other.gguf")
		writeServingDefinitionGGUF(t, path, false)
		writeServingDefinitionGGUF(t, other, true)
		store := open(t)
		inventory, resolved := publishProgramFacts(t, store, path)
		activateProgram(t, store, definitionRecipe(t, inventory, resolved))
		if err := os.Rename(other, path); err != nil {
			t.Fatal(err)
		}
		replaced, err := modelartifact.FromGGUFPath(path)
		if err != nil {
			t.Fatal(err)
		}
		_, err = ResolveActiveGGUF(t.Context(), store, path)
		if err == nil || !strings.Contains(err.Error(), inventory.Manifest.ID.String()) || !strings.Contains(err.Error(), replaced.Manifest.ID.String()) {
			t.Fatalf("replaced bytes: %v, want a refusal naming both identities", err)
		}
	})
	t.Run("a path nothing activated", func(t *testing.T) {
		t.Parallel()
		directory := t.TempDir()
		path, other := filepath.Join(directory, "model.gguf"), filepath.Join(directory, "other.gguf")
		writeServingDefinitionGGUF(t, path, false)
		writeServingDefinitionGGUF(t, other, true)
		store := open(t)
		publishProgramFacts(t, store, path)
		if err := os.Rename(other, path); err != nil {
			t.Fatal(err)
		}
		loaded, err := ResolveActiveGGUF(t.Context(), store, path)
		if err != nil {
			t.Fatal(err)
		}
		defer loaded.Close()
		if loaded.state.State.Verification != StateNotVerified {
			t.Fatalf("state = %+v", loaded.state.State)
		}
	})
}
