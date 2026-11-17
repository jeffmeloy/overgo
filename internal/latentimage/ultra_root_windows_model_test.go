//go:build modeltest && windows

package latentimage

import (
	"encoding/json"
	"os"
	"path/filepath"
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

// ultraModelRoot is a MOSS model root of the Ultra workflow's files, hard-linked from the import (moss_ultra_import.json
// pins each, as the test confirms), beside MOSS-Image-SFT's pipeline index, scheduler, configs and tokenizer, which the
// Ultra files share: what overgo reads as the workflow's MOSS.
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
	for _, folder := range []string{"scheduler", "transformer", "text_encoder", "tokenizer", "vae"} {
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
	for _, path := range ultraFiles {
		if !pinned[path] {
			t.Fatalf("the import recipe pins no %s", path)
		}
		testcheck.NoError(t, os.Link(filepath.Join(ultra, filepath.FromSlash(path)), filepath.Join(root, filepath.FromSlash(path))))
	}
	return root
}
