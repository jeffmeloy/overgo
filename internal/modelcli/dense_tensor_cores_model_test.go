//go:build modeltest

package modelcli

import (
	"os"
	"path/filepath"
	"testing"

	"overgo/internal/dataroot"
	"overgo/internal/discovery"
	"overgo/internal/modelrecipe"
	"overgo/internal/overgodbtest"
	"overgo/internal/testcheck"
	"overgo/internal/testutil"
)

// TestDenseModelsServePromptTensorCores holds the models whose long-prompt rows the tensor-core prompt attention
// restores to active definitions over profiles that admit it, each as it is served: Qwen2.5-0.5B, Carbon-500M and
// MiniCPM5-1B from their own directories, gemma-4-E4B-it and Unlimited-OCR from their conversions, and Flash-Next from
// the first of its three files, when the data root holds it.
// The program a served request compiles reads the profile its definition stored, so the catalog's admission reaches
// their requests once each is activated over a profile derived with it.
func TestDenseModelsServePromptTensorCores(t *testing.T) {
	t.Parallel()
	roots, err := dataroot.Resolve(testutil.RepoRoot(t))
	testcheck.NoError(t, err)
	store := overgodbtest.OpenReadOnly(t, roots.Store)
	claim := discovery.LoadMemo(t.Context(), store).Claim
	for _, served := range []struct {
		path     string
		optional bool
	}{
		{roots.ResolveModelPath("Qwen2.5-0.5B"), false},
		{roots.ResolveModelPath("Carbon-500M"), false},
		{roots.ResolveModelPath("MiniCPM5-1B"), false},
		{filepath.Join(roots.Checkpoints, "overgo-hfconvert", "gemma-4-E4B-it-bf16.gguf"), false},
		{filepath.Join(roots.Checkpoints, "overgo-hfconvert", "Unlimited-OCR-bf16.gguf"), false},
		{roots.ResolveModelPath(filepath.Join("Qwen3.8-Flash-Next", "unsloth-UD-IQ4_XS", "Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf")), true},
	} {
		t.Run(filepath.Base(served.path), func(t *testing.T) {
			if _, err := os.Stat(served.path); err != nil {
				if served.optional {
					t.Skipf("UNAVAILABLE: %s cannot be read: %v", served.path, err)
				}
				t.Fatalf("%s cannot be read: %v", served.path, err)
			}
			loaded, err := modelrecipe.ResolveClaimedGGUF(t.Context(), store, served.path, claim)
			testcheck.NoError(t, err)
			file, _, spec, _, _, _, err := loaded.Take()
			testcheck.NoError(t, err)
			testcheck.NoError(t, file.Close())
			if !spec.Profile().Runtime.PromptTensorCores {
				t.Fatal("its active definition names a profile that reads its prompts on the F32 paths: activate it over the current catalog's profile")
			}
		})
	}
}
