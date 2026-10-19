package libraryintake

import (
	"bytes"
	"context"
	"errors"
	"os"
	"path/filepath"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/latentimage"
	"overgo/internal/mediacapability"
	"overgo/internal/modelintake"
	"overgo/internal/modelrecipe"
	"overgo/internal/overgodb"
	"overgo/internal/recipe"
	"overgo/internal/testutil"
)

func TestDeterministicImportAcrossFamilies(t *testing.T) {
	for _, family := range []string{"moss", "krea", "gguf"} {
		t.Run(family, func(t *testing.T) {
			var root string
			switch family {
			case "moss":
				root = testutil.ExtractTarGzip(t, testutil.FixturePath(t, "moss", "generator-compact.tar.gz"), "150c4263057d72476ef4399a442c2b7dbadcb955670232fc2ea869c71f07c5db")
			case "krea":
				root = t.TempDir()
				testutil.WriteImageInventoryFixture(t, root)
			case "gguf":
				root = testutil.HermeticLlamaGGUF(t, 16)
			}
			inspection, err := modelintake.InspectImport(t.Context(), root, "")
			if err != nil {
				t.Fatal(err)
			}
			storeRoot := t.TempDir()
			store, err := overgodb.Open(storeRoot)
			if err != nil {
				t.Fatal(err)
			}
			defer func() {
				if store != nil {
					store.Close()
				}
			}()
			if _, err := modelrecipe.PublishArchitectureProfileCatalog(t.Context(), store); err != nil {
				t.Fatal(err)
			}
			if family == "krea" {
				data, err := os.ReadFile(filepath.Join(root, "tokenizer", "tokenizer_config.json"))
				if err != nil {
					t.Fatal(err)
				}
				id, err := artifact.IdentifyBytes(artifact.KindFile, data)
				if err != nil {
					t.Fatal(err)
				}
				prior := artifact.Content{Descriptor: artifact.Descriptor{ID: id, Size: uint64(len(data)), MediaType: artifact.JSONMediaType}, Data: data}
				if _, err := store.Commit(t.Context(), artifact.Batch{Key: "legacy-krea-config", Contents: []artifact.Content{prior}}); err != nil {
					t.Fatal(err)
				}
			}
			before, sequence := store.Head()
			ctx, cancel := context.WithCancel(t.Context())
			cancel()
			if _, err := Register(ctx, store, root, ""); !errors.Is(err, context.Canceled) {
				t.Fatal("cancelled import", err)
			}
			if after, next := store.Head(); after != before || next != sequence {
				t.Fatal("cancelled import published")
			}
			if err := modelintake.RegisterImport(t.Context(), store, inspection); err != nil {
				t.Fatal(err)
			}
			receipt, err := Register(t.Context(), store, root, "")
			if err != nil {
				t.Fatal(err)
			}
			if receipt["model"] != inspection.Inventory.Manifest.ID {
				t.Fatalf("Library and inspection disagree: %v != %s", receipt["model"], inspection.Inventory.Manifest.ID)
			}
			id, ok := receipt["recipe"].(artifact.ID)
			if !ok {
				t.Fatal("missing recipe", receipt)
			}
			status, found, err := modelrecipe.Status(t.Context(), store, id)
			if err != nil || !found || status != recipe.StatusCandidate {
				t.Fatal("import granted activation or omitted candidate", status, err)
			}
			if family != "gguf" {
				profile, err := latentimage.ResolveProfile(root)
				if err != nil {
					t.Fatal(err)
				}
				if (profile.Rotary != nil) != (family == "moss") {
					t.Fatal("image architectures conflated")
				}
				model, definition, err := mediacapability.Prepare(t.Context(), store, root, recipe.TaskImageGen, mediacapability.Catalog[recipe.TaskImageGen])
				if err != nil || model != receipt["model"] || definition.ID != id {
					t.Fatal("shared preparation disagrees", err)
				}
				program, err := modelrecipe.CompileCapability(definition)
				if err != nil {
					t.Fatal(err)
				}
				if _, err := modelrecipe.CompileCandidateExecution(t.Context(), store, program); err != nil {
					t.Fatal("candidate cannot bind execution", err)
				}
			}
			before, sequence = store.Head()
			if err := store.Close(); err != nil {
				t.Fatal(err)
			}
			store = nil
			store, err = overgodb.Open(storeRoot)
			if err != nil {
				t.Fatal(err)
			}
			repeated, err := Register(t.Context(), store, root, "")
			if err != nil || repeated["model"] != receipt["model"] || repeated["recipe"] != id {
				t.Fatal("restart changed identity", repeated, err)
			}
			if after, next := store.Head(); after != before || next != sequence {
				t.Fatal("restart duplicated registration")
			}
			relocated := filepath.Join(t.TempDir(), "source")
			if family == "gguf" {
				data, err := os.ReadFile(root)
				if err != nil {
					t.Fatal(err)
				}
				relocated += ".gguf"
				if err := os.WriteFile(relocated, data, 0600); err != nil {
					t.Fatal(err)
				}
			} else if err := os.CopyFS(relocated, os.DirFS(root)); err != nil {
				t.Fatal(err)
			}
			moved, err := modelintake.InspectImport(t.Context(), relocated, "")
			if err != nil {
				t.Fatal(err)
			}
			if !bytes.Equal(moved.Content.Data, inspection.Content.Data) {
				t.Fatal("relocation changed canonical description")
			}
			repeated, err = Register(t.Context(), store, relocated, "")
			if err != nil || repeated["model"] != receipt["model"] || repeated["recipe"] != id {
				t.Fatal("relocation changed candidate", repeated, err)
			}
			before, sequence = store.Head()
			if _, err := Register(t.Context(), store, relocated, ""); err != nil {
				t.Fatal(err)
			}
			if after, next := store.Head(); after != before || next != sequence {
				t.Fatal("relocated retry changed journal")
			}
			weight := root
			if family != "gguf" {
				weight = filepath.Join(root, "vae", "diffusion_pytorch_model.safetensors")
			}
			original, err := os.ReadFile(weight)
			if err != nil {
				t.Fatal(err)
			}
			if err := os.WriteFile(weight, []byte("corrupt"), 0600); err != nil {
				t.Fatal(err)
			}
			if _, err := Register(t.Context(), store, root, ""); err == nil {
				t.Fatal("corrupt source registered")
			}
			if after, next := store.Head(); after != before || next != sequence {
				t.Fatal("corruption changed prior candidate")
			}
			if err := os.WriteFile(weight, original, 0600); err != nil {
				t.Fatal(err)
			}
			if _, err := Register(t.Context(), store, root, ""); err != nil {
				t.Fatal("restored source cannot resume", err)
			}
			if after, next := store.Head(); after != before || next != sequence {
				t.Fatal("restored retry changed journal")
			}
		})
	}
}
