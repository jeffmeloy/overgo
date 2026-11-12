package main

import (
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"overgo/internal/dataroot"
	"overgo/internal/modelartifact"
	"overgo/internal/overgodb"
	"overgo/internal/testcheck"
	"overgo/internal/testutil"
)

// TestAnImportOfChangedBytesIsRefused holds recipe import-local to the bytes its declaration pins
// (moss-ultra-import/refusal-fixture), on two small files in a temporary install and a data root of the default
// layout: the import hard-links both into the models folder beside the store and publishes the model; an install one
// of whose files holds other bytes is refused, the links it made removed and the store's head unmoved. It held the
// refusal only through the library, in a model test that needed the six installed files.
func TestAnImportOfChangedBytesIsRefused(t *testing.T) {
	data, repository := t.TempDir(), t.TempDir()
	t.Setenv(dataroot.Env, data)
	header := `{"w":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}`
	weights := append(binary.LittleEndian.AppendUint64(nil, uint64(len(header))), header+"\x00\x00\x00\x00"...)
	pinned := map[string][]byte{"weights.safetensors": weights, "adapter.safetensors": []byte("adapter bytes")}
	install := func(files map[string][]byte) string {
		folder := t.TempDir()
		for name, bytes := range files {
			testcheck.NoError(t, os.WriteFile(filepath.Join(folder, name), bytes, 0o600))
		}
		return folder
	}
	declare := func(name string) string {
		declaration := modelartifact.LocalImport{Name: name}
		for _, file := range []struct{ source, role, folder string }{
			{"weights.safetensors", "weights", "transformer"}, {"adapter.safetensors", "adapter", "adapters"},
		} {
			sum := sha256.Sum256(pinned[file.source])
			declaration.Files = append(declaration.Files, modelartifact.LocalImportFile{
				Path: file.folder + "/" + file.source, Role: file.role, Source: file.source, Origin: "https://example.com/" + file.source,
				License: "Apache-2.0", SHA256: hex.EncodeToString(sum[:]), Bytes: int64(len(pinned[file.source])),
			})
		}
		path := filepath.Join(t.TempDir(), name+".json")
		testutil.WriteJSON(t, path, declaration)
		return path
	}
	head := func() string {
		t.Helper()
		store := testcheck.Must(overgodb.OpenReadOnly(repository))(t)
		defer store.Close()
		at, _ := store.Head()
		return at.String()
	}

	kept := install(pinned)
	testcheck.NoError(t, importLocal([]string{"-repo", repository, "-declaration", declare("fixture"), "-from", kept}))
	for source, path := range map[string]string{"weights.safetensors": "transformer", "adapter.safetensors": "adapters"} {
		linked, err := os.Stat(filepath.Join(data, "models", "fixture", path, source))
		original := testcheck.Must(os.Stat(filepath.Join(kept, source)))(t)
		if err != nil || !os.SameFile(linked, original) {
			t.Fatalf("the import left %s at %v (%v); want it hard-linked into the models folder beside the store", source, linked, err)
		}
	}
	published := head()

	changed := install(map[string][]byte{"weights.safetensors": weights, "adapter.safetensors": []byte("other bytes!!")})
	err := importLocal([]string{"-repo", repository, "-declaration", declare("changed"), "-from", changed})
	if err == nil || !strings.Contains(err.Error(), "not the declared") {
		t.Fatalf("an import whose adapter holds other bytes than declared gave %v; want it refused naming the file", err)
	}
	for _, path := range []string{"transformer/weights.safetensors", "adapters/adapter.safetensors"} {
		if _, err := os.Stat(filepath.Join(data, "models", "changed", filepath.FromSlash(path))); !os.IsNotExist(err) {
			t.Errorf("the refused import left %s linked (%v)", path, err)
		}
	}
	if now := head(); now != published {
		t.Errorf("the refused import moved the store's head from %s to %s", published, now)
	}
}
