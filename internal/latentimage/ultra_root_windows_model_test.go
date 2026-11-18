//go:build modeltest && windows

package latentimage

import (
	"encoding/json"
	"maps"
	"os"
	"path/filepath"
	"slices"
	"testing"

	"overgo/internal/testcheck"
	"overgo/internal/testutil"
)

// ultraFiles are the MOSS Ultra workflow's model files a MOSS root reads, by the folder of the root each goes in: the
// Base transformer and the abliterated encoder in int8 ConvRot, and flux2-vae_small, whose decoder is MOSS-Image-SFT's
// VAE's and whose encoder and latent statistics are its own.
var ultraFiles = map[string]string{
	"transformer":  "transformer/moss_image_base_int8_convrot.safetensors",
	"text_encoder": "text_encoder/Qwen3.5-9B-abliterated_int8_convrot.safetensors",
	"vae":          "vae/flux2-vae_small.safetensors",
}

// ultraAdapters are the workflow's adapters, which a MOSS root holds under adapters/ by name (modelAdapter).
var ultraAdapters = []string{"adapters/8-steps-dmd2-lora.safetensors", "adapters/moss_image_sft.safetensors", "adapters/moss-control.safetensors"}

// ultraModelRoot is a MOSS model root of the Ultra workflow's files, hard-linked from the import (moss_ultra_import.json
// pins each, as the test confirms), its adapters under adapters/, beside MOSS-Image-SFT's pipeline index, scheduler,
// configs and tokenizer, which the Ultra files share: what overgo reads as the workflow's MOSS.
func ultraModelRoot(t *testing.T) string {
	t.Helper()
	repo := testutil.RepoRoot(t)
	var recipe struct {
		Files []struct {
			Path string `json:"path"`
		} `json:"files"`
	}
	testcheck.NoError(t, json.Unmarshal(testcheck.Must(os.ReadFile(filepath.Join(repo, "internal", "modelrecipe", "moss_ultra_import.json")))(t), &recipe))
	sft, ultra, root := testutil.ModelArtifactDir(t, "MOSS-Image-SFT-797fa8e678b4f"), testutil.ModelArtifactDir(t, "moss-ultra"), t.TempDir()
	for _, folder := range []string{"scheduler", "transformer", "text_encoder", "tokenizer", "vae", "adapters"} {
		testcheck.NoError(t, os.Mkdir(filepath.Join(root, folder), 0o700))
	}
	for _, name := range []string{"model_index.json", "scheduler/scheduler_config.json", "transformer/config.json", "text_encoder/config.json",
		"tokenizer/tokenizer.json", "tokenizer/tokenizer_config.json", "tokenizer/chat_template.jinja", "vae/config.json"} {
		testcheck.NoError(t, os.WriteFile(filepath.Join(root, filepath.FromSlash(name)), testcheck.Must(os.ReadFile(filepath.Join(sft, filepath.FromSlash(name))))(t), 0o600))
	}
	pinned := map[string]bool{}
	for _, file := range recipe.Files {
		pinned[file.Path] = true
	}
	for _, path := range append(slices.Collect(maps.Values(ultraFiles)), ultraAdapters...) {
		if !pinned[path] {
			t.Fatalf("the import recipe pins no %s", path)
		}
		testcheck.NoError(t, os.Link(filepath.Join(ultra, filepath.FromSlash(path)), filepath.Join(root, filepath.FromSlash(path))))
	}
	return root
}

// ultraDepth gives a root its depth folder (depthDirectory), the Depth Anything V2 Small checkpoint its depth sections
// estimate a picture's depth with, hard-linked from the model store.
func ultraDepth(t *testing.T, root string) {
	t.Helper()
	testcheck.NoError(t, os.MkdirAll(filepath.Join(root, depthDirectory), 0o755))
	testcheck.NoError(t, os.Link(filepath.Join(testutil.ModelArtifactDir(t, "depth-anything-v2-small"), depthCheckpoint), filepath.Join(root, depthDirectory, depthCheckpoint)))
}

// ultraEnhancer gives an Ultra root its enhancer folder (enhancerDirectory), the encoder file served as a chat model:
// Qwen3.5-9B's own config for generation (fixtures/comfy/moss-ultra/enhancer/config.json, Qwen/Qwen3.5-9B's
// config.json), the root's tokenizer and chat template, and the encoder file hard-linked; it returns the folder.
func ultraEnhancer(t *testing.T, root string) string {
	t.Helper()
	directory := filepath.Join(root, enhancerDirectory)
	testcheck.NoError(t, os.Mkdir(directory, 0o700))
	config := filepath.Join(testutil.RepoRoot(t), "fixtures", "comfy", "moss-ultra", "enhancer", "config.json")
	testcheck.NoError(t, os.WriteFile(filepath.Join(directory, "config.json"), testcheck.Must(os.ReadFile(config))(t), 0o600))
	for _, name := range []string{"tokenizer.json", "tokenizer_config.json", "chat_template.jinja"} {
		testcheck.NoError(t, os.WriteFile(filepath.Join(directory, name), testcheck.Must(os.ReadFile(filepath.Join(root, "tokenizer", name)))(t), 0o600))
	}
	encoder := filepath.FromSlash(ultraFiles["text_encoder"])
	testcheck.NoError(t, os.Link(filepath.Join(root, encoder), filepath.Join(directory, filepath.Base(encoder))))
	return directory
}
