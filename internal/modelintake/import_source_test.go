package modelintake

import (
	"context"
	"errors"
	"os"
	"path/filepath"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/overgodb"
	"overgo/internal/testcheck"
)

func TestComponentImportSourceBinding(t *testing.T) {
	for _, problem := range []string{"same-size-corruption", "missing-file", "missing-location", "cancelled", "publication-failure"} {
		t.Run(problem, func(t *testing.T) {
			root := importFixture(t)
			inspection, err := InspectImport(t.Context(), root, "")
			testcheck.NoError(t, err)
			path := filepath.Join(root, "model.safetensors")
			original, err := os.ReadFile(path)
			testcheck.NoError(t, err)
			info, err := os.Stat(path)
			testcheck.NoError(t, err)
			store, err := overgodb.Open(t.TempDir())
			testcheck.NoError(t, err)
			defer store.Close()
			before, sequence := store.Head()
			ctx := t.Context()
			var repository artifact.Repository = store
			locations := inspection.Inventory.Locations
			switch problem {
			case "same-size-corruption":
				changed := append([]byte(nil), original...)
				changed[len(changed)-1] ^= 1
				testcheck.NoError(t, os.WriteFile(path, changed, 0600))
				testcheck.NoError(t, os.Chtimes(path, info.ModTime(), info.ModTime()))
			case "missing-file":
				testcheck.NoError(t, os.Remove(path))
			case "missing-location":
				inspection.Inventory.Locations = nil
			case "cancelled":
				var cancel context.CancelFunc
				ctx, cancel = context.WithCancel(ctx)
				cancel()
			case "publication-failure":
				repository = refusedImportPublication{Repository: store}
			}
			if err := RegisterImport(ctx, repository, inspection); err == nil {
				t.Fatal("published invalid or incomplete import")
			}
			after, afterSequence := store.Head()
			if before != after || sequence != afterSequence {
				t.Fatal("failed import changed the catalog")
			}
			testcheck.NoError(t, os.WriteFile(path, original, 0600))
			inspection.Inventory.Locations = locations
			if err := RegisterImport(t.Context(), store, inspection); err != nil {
				t.Fatalf("valid retry: %v", err)
			}
			before, sequence = store.Head()
			testcheck.NoError(t, RegisterImport(t.Context(), store, inspection))
			after, afterSequence = store.Head()
			if before != after || sequence != afterSequence {
				t.Fatal("successful retry changed published facts")
			}
		})
	}
	t.Run("existing-source-provenance", func(t *testing.T) {
		root := t.TempDir()
		testcheck.NoError(t, os.CopyFS(filepath.Join(root, "encoder"), os.DirFS(importFixture(t))))
		testcheck.NoError(t, os.WriteFile(filepath.Join(root, "model_index.json"), []byte(`{"_class_name":"Fixture","encoder":["upstream","LlamaForCausalLM"]}`), 0600))
		code := []byte("# retained upstream source\n")
		testcheck.NoError(t, os.WriteFile(filepath.Join(root, "upstream.py"), code, 0600))
		store, err := overgodb.Open(t.TempDir())
		testcheck.NoError(t, err)
		defer store.Close()
		id, err := artifact.IdentifyBytes(artifact.KindFile, code)
		testcheck.NoError(t, err)
		original := artifact.Descriptor{ID: id, Size: uint64(len(code)), MediaType: "text/plain"}
		if _, err := artifact.CommitBatch(t.Context(), store, artifact.Batch{Key: "upstream-source", Artifacts: []artifact.Descriptor{original}}); err != nil {
			t.Fatal(err)
		}
		inspection, err := InspectImport(t.Context(), root, "")
		testcheck.NoError(t, err)
		testcheck.NoError(t, RegisterImport(t.Context(), store, inspection))
		retained, found, err := store.Artifact(t.Context(), id)
		if err != nil || !found || retained != original {
			t.Fatalf("source provenance changed: %+v %v", retained, err)
		}
	})
}

type refusedImportPublication struct{ artifact.Repository }

func (refusedImportPublication) Commit(context.Context, artifact.Batch) (artifact.CommitID, error) {
	return artifact.CommitID{}, errors.New("injected publication write failure")
}
