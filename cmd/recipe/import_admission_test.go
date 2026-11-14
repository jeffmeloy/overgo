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

// TestALocalImportRefusesItsDeclarationBeforeHashing holds recipe import-local to admitting its declaration and its
// destination before it links or reads a file (SQA-387): a declaration was checked against the files' hashes after
// every file was linked and read, and the store opened after that. A malformed digest is refused from an install that
// holds no file at all, a declared size the install's file does not have is refused by its size, and a store that
// cannot be opened is refused, each with no file linked into the models folder.
func TestALocalImportRefusesItsDeclarationBeforeHashing(t *testing.T) {
	data := t.TempDir()
	t.Setenv(dataroot.Env, data)
	repository := t.TempDir()
	testcheck.NoError(t, testcheck.Must(overgodb.Create(repository))(t).Close())
	header := `{"w":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}`
	content := append(binary.LittleEndian.AppendUint64(nil, uint64(len(header))), append([]byte(header), 0, 0, 0, 0)...)
	sum := sha256.Sum256(content)
	install := t.TempDir()
	testcheck.NoError(t, os.WriteFile(filepath.Join(install, "weights.safetensors"), content, 0o600))
	declare := func(name, digest string, size int64) string {
		t.Helper()
		path := filepath.Join(t.TempDir(), name+".json")
		testutil.WriteJSON(t, path, modelartifact.LocalImport{Name: name, Files: []modelartifact.LocalImportFile{{
			Path: "transformer/weights.safetensors", Role: "weights", Source: "weights.safetensors",
			Origin: "https://example.com/weights.safetensors", License: "Apache-2.0", SHA256: digest, Bytes: size,
		}}})
		return path
	}
	unlinked := func(name string) {
		t.Helper()
		if _, err := os.Stat(filepath.Join(data, "models", name, "transformer", "weights.safetensors")); !os.IsNotExist(err) {
			t.Errorf("the refused import %s left its file linked (%v)", name, err)
		}
	}
	err := importLocal([]string{"-repo", repository, "-declaration", declare("malformed", "not-a-digest", int64(len(content))), "-from", t.TempDir()})
	if err == nil || !strings.Contains(err.Error(), `declares "not-a-digest"`) {
		t.Errorf("a malformed digest, from an install holding nothing, gave %v; want it refused by its declaration", err)
	}
	unlinked("malformed")
	err = importLocal([]string{"-repo", repository, "-declaration", declare("larger", hex.EncodeToString(sum[:]), int64(len(content))+1), "-from", install})
	if err == nil || !strings.Contains(err.Error(), "bytes in the install, not the declared") {
		t.Errorf("a declared size the file does not have gave %v; want it refused by the file's size", err)
	}
	unlinked("larger")
	unopenable := filepath.Join(t.TempDir(), "a file, not a store")
	testcheck.NoError(t, os.WriteFile(unopenable, content, 0o600))
	err = importLocal([]string{"-repo", unopenable, "-declaration", declare("storeless", hex.EncodeToString(sum[:]), int64(len(content))), "-from", install})
	if err == nil {
		t.Error("an import into a store that cannot be opened was taken")
	}
	unlinked("storeless")
}
