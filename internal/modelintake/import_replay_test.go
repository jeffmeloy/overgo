package modelintake

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/hfgguf"
	"overgo/internal/overgodb"
)

func TestFrozenComponentImportReplay(t *testing.T) {
	root := importFixture(t)
	for _, purpose := range []hfgguf.LanguageOutput{"", hfgguf.LanguageHidden, hfgguf.LanguageTokens} {
		t.Run(string(purpose), func(t *testing.T) {
			original, err := InspectImport(t.Context(), root, purpose)
			if err != nil {
				t.Fatal(err)
			}
			var formatted bytes.Buffer
			if err := json.Indent(&formatted, original.Content.Data, "", "  "); err != nil {
				t.Fatal(err)
			}
			relocated := filepath.Join(t.TempDir(), "relocated")
			if err := os.CopyFS(relocated, os.DirFS(root)); err != nil {
				t.Fatal(err)
			}
			replay, err := ReplayImport(t.Context(), relocated, formatted.Bytes())
			if err != nil {
				t.Fatal(err)
			}
			if replay.Content.Descriptor.ID != original.Content.Descriptor.ID || !bytes.Equal(replay.Content.Data, original.Content.Data) {
				t.Fatal("replay changed canonical description")
			}
			storePath := t.TempDir()
			publish := func() artifact.CommitID {
				t.Helper()
				store, err := overgodb.Open(storePath)
				if err != nil {
					t.Fatal(err)
				}
				defer store.Close()
				replay, err := ReplayImport(t.Context(), relocated, formatted.Bytes())
				if err != nil {
					t.Fatal(err)
				}
				if err := RegisterImport(t.Context(), store, replay); err != nil {
					t.Fatal(err)
				}
				head, _ := store.Head()
				return head
			}
			if publish() != publish() {
				t.Fatal("restarted publication changed catalog")
			}
		})
	}
	original, err := InspectImport(t.Context(), root, hfgguf.LanguageHidden)
	if err != nil {
		t.Fatal(err)
	}
	for name, change := range map[string]func(*importDescription){
		"version":           func(d *importDescription) { d.Version++ },
		"qualification":     func(d *importDescription) { d.Qualification = "verified" },
		"missing-component": func(d *importDescription) { d.Components = nil },
		"declaration":       func(d *importDescription) { d.Components[0].Declaration.Class = "OtherClass" },
		"operator":          func(d *importDescription) { d.Components[0].Operations[0] = "unavailable" },
		"target-binding":    func(d *importDescription) { d.Components[0].Bindings[0].Target = "other.weight" },
		"ordered-transform": func(d *importDescription) {
			d.Components[0].Bindings[0].Transforms = []hfgguf.TensorTransform{{Operation: "add-one-f32"}}
		},
		"tensor-bytes": func(d *importDescription) { d.StoredTensorBytes++ },
		"profile":      func(d *importDescription) { d.Components[0].Profile = nil },
	} {
		t.Run(name, func(t *testing.T) {
			var description importDescription
			if err := json.Unmarshal(original.Content.Data, &description); err != nil {
				t.Fatal(err)
			}
			change(&description)
			data, err := json.Marshal(description)
			if err != nil {
				t.Fatal(err)
			}
			if result, err := ReplayImport(t.Context(), root, data); err == nil || result.Content.Descriptor.ID != (artifact.ID{}) {
				t.Fatalf("changed contract accepted: %v", err)
			}
		})
	}
	for _, data := range [][]byte{nil, []byte("null"), append(append([]byte{}, original.Content.Data...), []byte(" {}")...), []byte(`{"version":1,"version":1}`), []byte(`{"unknown":true}`)} {
		if _, err := ReplayImport(t.Context(), root, data); err == nil {
			t.Fatal("malformed saved description accepted")
		}
	}
	if _, err := ReplayImport(nil, root, original.Content.Data); err == nil {
		t.Fatal("nil context accepted")
	}
	ctx, cancel := context.WithCancel(t.Context())
	cancel()
	if _, err := ReplayImport(ctx, root, original.Content.Data); !errors.Is(err, context.Canceled) {
		t.Fatalf("cancel = %v", err)
	}
	weightPath := filepath.Join(root, "model.safetensors")
	data, err := os.ReadFile(weightPath)
	if err != nil {
		t.Fatal(err)
	}
	data[len(data)-1] ^= 1
	if err := os.WriteFile(weightPath, data, 0600); err != nil {
		t.Fatal(err)
	}
	if _, err := ReplayImport(t.Context(), root, original.Content.Data); err == nil {
		t.Fatal("changed weight payload accepted")
	}
}
