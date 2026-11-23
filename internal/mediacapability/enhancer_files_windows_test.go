//go:build windows

package mediacapability

import (
	"errors"
	"io/fs"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"overgo/internal/capabilityruntime"
	"overgo/internal/modelrecipe"
	"overgo/internal/testcheck"
)

// TestAnEnhancedAnswerIsKeptUnderTheFilesItsEnhancerOpens holds enhancer-files/do (SQA-469): a prompt mode 2 request
// is named (capabilityruntime.Contents, pictureFiles) by the files its enhancer's chat opens in the enhancer folder, as
// overgo reads a Hugging Face folder: an unrelated file added to the folder leaves the naming as it was, a config
// renamed away refuses the request naming the config, and the config back under its own name names the request as
// before. Every file of the folder had been named by its content alone, so a renamed config gave its stored answer again
// and an unrelated file drew it afresh.
func TestAnEnhancedAnswerIsKeptUnderTheFilesItsEnhancerOpens(t *testing.T) {
	t.Parallel()
	root := t.TempDir()
	folder := filepath.Join(root, "enhancer")
	testcheck.NoError(t, os.Mkdir(folder, 0o700))
	for name, content := range map[string]string{"config.json": `{"model_type":"qwen3_5"}`, "tokenizer.json": `{"version":"1.0"}`,
		"tokenizer_config.json": `{}`, "chat_template.jinja": "{{ messages }}", "model.safetensors": "weights"} {
		testcheck.NoError(t, os.WriteFile(filepath.Join(folder, name), []byte(content), 0o600))
	}
	naming := capabilityruntime.Contents(nil, pictureFiles)
	const enhanced = `{"prompt_mode":"enhanced","prompt":"a fox","width":256,"height":144,"seed":3}`
	name := func() (string, error) {
		named, release, err := naming(t.Context(), nil, root, modelrecipe.CapabilityEvidenceSelection{}, enhanced)
		if release != nil {
			testcheck.NoError(t, release())
		}
		return named, err
	}
	first := testcheck.Must(name())(t)
	if first == "" {
		t.Fatal("an enhanced request named no file of its enhancer")
	}
	testcheck.NoError(t, os.WriteFile(filepath.Join(folder, "notes.txt"), []byte("unrelated"), 0o600))
	if again := testcheck.Must(name())(t); again != first {
		t.Error("an unrelated file in the enhancer folder changed the request's naming")
	}
	config := filepath.Join(folder, "config.json")
	testcheck.NoError(t, os.Rename(config, config+".bak"))
	if _, err := name(); !errors.Is(err, fs.ErrNotExist) || !strings.Contains(err.Error(), "config.json") {
		t.Errorf("with its config renamed away the request was named, %v; want it refused naming the config", err)
	}
	testcheck.NoError(t, os.Rename(config+".bak", config))
	if back := testcheck.Must(name())(t); back != first {
		t.Error("with its config back under its own name the request was named otherwise")
	}
}
