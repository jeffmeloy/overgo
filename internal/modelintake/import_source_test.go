package modelintake

import (
	"context"
	"errors"
	"os"
	"path/filepath"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/overgodb"
)

func TestComponentImportSourceBinding(t *testing.T) {
	for _, problem := range []string{"same-size-corruption", "missing-file", "missing-location", "cancelled", "publication-failure"} {
		t.Run(problem, func(t *testing.T) {
			root := importFixture(t)
			inspection, err := InspectImport(t.Context(), root, "")
			if err != nil {
				t.Fatal(err)
			}
			path := filepath.Join(root, "model.safetensors")
			original, err := os.ReadFile(path)
			if err != nil {
				t.Fatal(err)
			}
			info, err := os.Stat(path)
			if err != nil {
				t.Fatal(err)
			}
			store, err := overgodb.Open(t.TempDir())
			if err != nil {
				t.Fatal(err)
			}
			defer store.Close()
			before, sequence := store.Head()
			ctx := t.Context()
			var repository artifact.Repository = store
			locations := inspection.Inventory.Locations
			switch problem {
			case "same-size-corruption":
				changed := append([]byte(nil), original...)
				changed[len(changed)-1] ^= 1
				if err := os.WriteFile(path, changed, 0600); err != nil {
					t.Fatal(err)
				}
				if err := os.Chtimes(path, info.ModTime(), info.ModTime()); err != nil {
					t.Fatal(err)
				}
			case "missing-file":
				if err := os.Remove(path); err != nil {
					t.Fatal(err)
				}
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
			if err := os.WriteFile(path, original, 0600); err != nil {
				t.Fatal(err)
			}
			inspection.Inventory.Locations = locations
			if err := RegisterImport(t.Context(), store, inspection); err != nil {
				t.Fatalf("valid retry: %v", err)
			}
			before, sequence = store.Head()
			if err := RegisterImport(t.Context(), store, inspection); err != nil {
				t.Fatal(err)
			}
			after, afterSequence = store.Head()
			if before != after || sequence != afterSequence {
				t.Fatal("successful retry changed published facts")
			}
		})
	}
	t.Run("existing-source-provenance", func(t *testing.T) {
		root := t.TempDir()
		if err := os.CopyFS(filepath.Join(root, "encoder"), os.DirFS(importFixture(t))); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(filepath.Join(root, "model_index.json"), []byte(`{"_class_name":"Fixture","encoder":["upstream","LlamaForCausalLM"]}`), 0600); err != nil {
			t.Fatal(err)
		}
		code := []byte("# retained upstream source\n")
		if err := os.WriteFile(filepath.Join(root, "upstream.py"), code, 0600); err != nil {
			t.Fatal(err)
		}
		store, err := overgodb.Open(t.TempDir())
		if err != nil {
			t.Fatal(err)
		}
		defer store.Close()
		id, err := artifact.IdentifyBytes(artifact.KindFile, code)
		if err != nil {
			t.Fatal(err)
		}
		original := artifact.Descriptor{ID: id, Size: uint64(len(code)), MediaType: "text/plain"}
		if _, err := artifact.CommitBatch(t.Context(), store, artifact.Batch{Key: "upstream-source", Artifacts: []artifact.Descriptor{original}}); err != nil {
			t.Fatal(err)
		}
		inspection, err := InspectImport(t.Context(), root, "")
		if err != nil {
			t.Fatal(err)
		}
		if err := RegisterImport(t.Context(), store, inspection); err != nil {
			t.Fatal(err)
		}
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
