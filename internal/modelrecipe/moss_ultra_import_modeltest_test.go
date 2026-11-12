//go:build modeltest

package modelrecipe

import (
	"crypto/sha256"
	"encoding/hex"
	"path/filepath"
	"slices"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/dataroot"
	"overgo/internal/modelartifact"
	"overgo/internal/overgodbtest"
	"overgo/internal/testcheck"
	"overgo/internal/testutil"
)

// TestTheUltraRecipeListsItsSixFilesByHash holds the MOSS Ultra workflow's six files to their import
// (moss-ultra-import/import): the store locates each by the SHA-256 its declaration pins, hard-linked into the data
// root's model folder under moss-ultra, the transformer, encoder and VAE as weights and the three LoRAs as adapters.
// An import of changed bytes is refused through the command, on small files (cmd/recipe
// TestAnImportOfChangedBytesIsRefused).
func TestTheUltraRecipeListsItsSixFilesByHash(t *testing.T) {
	declared := testcheck.Must(modelartifact.ReadLocalImport("moss_ultra_import.json"))(t)
	if len(declared.Files) != 6 {
		t.Fatalf("the declaration names %d files; want the workflow's six", len(declared.Files))
	}
	roots := testcheck.Must(dataroot.Resolve(testutil.RepoRoot(t)))(t)
	store := overgodbtest.OpenReadOnly(t, roots.Store)
	for _, file := range declared.Files {
		kind := artifact.KindTensorSet
		if file.Role == "adapter" {
			kind = artifact.KindFile
		}
		var digest [sha256.Size]byte
		testcheck.Must(hex.Decode(digest[:], []byte(file.SHA256)))(t)
		id := testcheck.Must(artifact.NewID(kind, digest))(t)
		located := testcheck.Must(store.Locations(t.Context(), id))(t)
		want := filepath.Join(roots.StoreModels, declared.Name, filepath.FromSlash(file.Path))
		if !slices.ContainsFunc(located, func(location artifact.Location) bool { return filepath.Clean(location.Value) == want }) {
			t.Errorf("the store locates %s (%s) at %v; want it at %s", file.Path, id, located, want)
		}
	}
}
