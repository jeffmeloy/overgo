package discovery

import (
	"os"
	"path/filepath"
	"testing"
	"time"

	"overgo/internal/artifact"
	"overgo/internal/gguf"
	"overgo/internal/modelartifact"
	"overgo/internal/testutil"
)

// TestADirectoryModelIsItsViewsIdentity holds the catalog to a model registered from its own files: its directory is
// present as the bytes gguf.Open serves from it, the identity its inventory names, remembered under the signature of
// the files the directory holds, so the next listing and a load's claim answer without reading them again; a change to
// any of its files ends the claim, and a directory gguf does not serve names nothing.
func TestADirectoryModelIsItsViewsIdentity(t *testing.T) {
	t.Parallel()
	directory := testutil.Qwen35Directory(t)
	file, err := gguf.Open(directory)
	if err != nil {
		t.Fatal(err)
	}
	inventory, _, err := modelartifact.FromGGUFClaimed(file, artifact.KindModel, nil)
	if closeErr := file.Close(); err != nil || closeErr != nil {
		t.Fatal(err, closeErr)
	}
	want := inventory.Components[0]
	memo := NewMemo()
	identity := identifyLocation(directory, artifact.KindTensorSet, map[string]fileIdentity{}, memo)
	if !identity.present || identity.id != want.ID || identity.size != want.Size {
		t.Fatalf("the directory's identity = %+v, want present as %s of %d bytes", identity, want.ID, want.Size)
	}
	if id, size, claimed := memo.Claim(directory); !claimed || id != want.ID || size != want.Size {
		t.Fatalf("the directory's claim = %s, %d, %t; want the identity it was found as", id, size, claimed)
	}
	later := time.Now().Add(time.Hour)
	if err := os.Chtimes(filepath.Join(directory, "config.json"), later, later); err != nil {
		t.Fatal(err)
	}
	if _, _, claimed := memo.Claim(directory); claimed {
		t.Error("the claim outlived a change to a file the directory holds")
	}
	if empty := identifyLocation(t.TempDir(), artifact.KindTensorSet, map[string]fileIdentity{}, memo); empty.present {
		t.Errorf("a directory gguf does not serve = %+v, want nothing", empty)
	}
}
