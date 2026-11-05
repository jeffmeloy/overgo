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
	"overgo/internal/testcheck"
	"overgo/internal/testutil"
)

// TestDenseModelsServePromptTensorCores holds the dense models whose long-prompt rows the tensor-core prompt attention
// restores to active definitions over profiles that admit it, each as it is served: Qwen2.5-0.5B, Carbon-500M and
// MiniCPM5-1B from their own directories, gemma-4-E4B-it and Unlimited-OCR from their conversions.
// The program a served request compiles reads the profile its definition stored, so the catalog's admission reaches
// their requests once each is activated over a profile derived with it, as Flash-Next's was
// (TestFlashNextServesPromptTensorCores).
func TestDenseModelsServePromptTensorCores(t *testing.T) {
	t.Parallel()
	roots, err := dataroot.Resolve(testutil.RepoRoot(t))
	testcheck.NoError(t, err)
	store, err := overgodb.OpenReadOnly(roots.Store)
	testcheck.NoError(t, err)
	defer store.Close()
	claim := discovery.LoadMemo(t.Context(), store).Claim
	for _, served := range []string{
		roots.ResolveModelPath("Qwen2.5-0.5B"),
		roots.ResolveModelPath("Carbon-500M"),
		roots.ResolveModelPath("MiniCPM5-1B"),
		filepath.Join(roots.Checkpoints, "overgo-hfconvert", "gemma-4-E4B-it-bf16.gguf"),
		filepath.Join(roots.Checkpoints, "overgo-hfconvert", "Unlimited-OCR-bf16.gguf"),
	} {
		if _, err := os.Stat(served); err != nil {
			t.Errorf("%s cannot be read: %v", served, err)
			continue
		}
		loaded, err := modelrecipe.ResolveClaimedGGUF(t.Context(), store, served, claim)
		if err != nil {
			t.Errorf("%s: %v", served, err)
			continue
		}
		file, _, spec, _, _, _, err := loaded.Take()
		if err != nil {
			t.Errorf("%s: %v", served, err)
			continue
		}
		testcheck.NoError(t, file.Close())
		if !spec.Profile().Runtime.PromptTensorCores {
			t.Errorf("%s's active definition names a profile that reads its prompts on the F32 paths: activate it over the current catalog's profile", filepath.Base(served))
		}
	}
}
