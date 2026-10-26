//go:build modeltest

package modelcli

import (
	"os"
	"path/filepath"
	"testing"

	"overgo/internal/dataroot"
	"overgo/internal/discovery"
	"overgo/internal/modelrecipe"
	"overgo/internal/overgodb"
	"overgo/internal/testutil"
)

// TestFlashNextServesPromptTensorCores holds Flash-Next's active inference definition to the profile that admits the
// tensor-core prompt attention for its architecture. The program a served request compiles reads the stored profile
// document its definition names, so the catalog's admission reaches Flash-Next's requests once the definition is
// activated over a profile derived with it.
func TestFlashNextServesPromptTensorCores(t *testing.T) {
	roots, err := dataroot.Resolve(testutil.RepoRoot(t))
	if err != nil {
		t.Fatal(err)
	}
	path := roots.ResolveModelPath(filepath.Join("Qwen3.8-Flash-Next", "unsloth-UD-IQ4_XS", "Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf"))
	if _, err := os.Stat(path); err != nil {
		t.Skipf("UNAVAILABLE: Flash-Next's file %s cannot be read: %v", path, err)
	}
	store, err := overgodb.OpenReadOnly(roots.Store)
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	loaded, err := modelrecipe.ResolveClaimedGGUF(t.Context(), store, path, discovery.LoadMemo(t.Context(), store).Claim)
	if err != nil {
		t.Fatal(err)
	}
	file, _, spec, _, _, _, _, err := loaded.Take()
	if err != nil {
		t.Fatal(err)
	}
	if err := file.Close(); err != nil {
		t.Fatal(err)
	}
	if !spec.Profile().Runtime.PromptTensorCores {
		t.Fatal("Flash-Next's active definition names a profile that reads its prompts on the F32 paths: activate it over the current catalog's profile")
	}
}
