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
	"overgo/internal/testcheck"
)

func TestFrozenComponentImportReplay(t *testing.T) {
	root := importFixture(t)
	for _, purpose := range []hfgguf.LanguageOutput{"", hfgguf.LanguageHidden, hfgguf.LanguageTokens} {
		t.Run(string(purpose), func(t *testing.T) {
			original, err := InspectImport(t.Context(), root, purpose)
			testcheck.NoError(t, err)
			var formatted bytes.Buffer
			testcheck.NoError(t, json.Indent(&formatted, original.Content.Data, "", "  "))
			relocated := filepath.Join(t.TempDir(), "relocated")
			testcheck.NoError(t, os.CopyFS(relocated, os.DirFS(root)))
			replay, err := ReplayImport(t.Context(), relocated, formatted.Bytes())
			testcheck.NoError(t, err)
			if replay.Content.Descriptor.ID != original.Content.Descriptor.ID || !bytes.Equal(replay.Content.Data, original.Content.Data) {
				t.Fatal("replay changed canonical description")
			}
			storePath := t.TempDir()
			publish := func() artifact.CommitID {
				t.Helper()
				store, err := overgodb.Create(storePath)
				testcheck.NoError(t, err)
				defer store.Close()
				replay, err := ReplayImport(t.Context(), relocated, formatted.Bytes())
				testcheck.NoError(t, err)
				testcheck.NoError(t, RegisterImport(t.Context(), store, replay))
				head, _ := store.Head()
				return head
			}
			if publish() != publish() {
				t.Fatal("restarted publication changed catalog")
			}
		})
	}
	original, err := InspectImport(t.Context(), root, hfgguf.LanguageHidden)
	testcheck.NoError(t, err)
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
			testcheck.NoError(t, json.Unmarshal(original.Content.Data, &description))
			change(&description)
			data, err := json.Marshal(description)
			testcheck.NoError(t, err)
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
	testcheck.NoError(t, err)
	data[len(data)-1] ^= 1
	testcheck.NoError(t, os.WriteFile(weightPath, data, 0600))
	if _, err := ReplayImport(t.Context(), root, original.Content.Data); err == nil {
		t.Fatal("changed weight payload accepted")
	}
}
