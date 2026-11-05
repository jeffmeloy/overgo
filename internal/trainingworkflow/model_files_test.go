package trainingworkflow

import (
	"os"
	"path/filepath"
	"slices"
	"testing"

	"overgo/internal/testcheck"
	"overgo/internal/testutil"
)

// TestACheckpointKeepsWhatServesItsModel: a trained directory serves as its base did, so it keeps the chat template and
// generation settings beside the config and tokenizer, and no weights come with them. A base with no tokenizer is refused.
func TestACheckpointKeepsWhatServesItsModel(t *testing.T) {
	t.Parallel()
	source, stage := t.TempDir(), t.TempDir()
	for _, name := range []string{"config.json", "generation_config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json"} {
		testutil.WriteTextFile(t, source, name, name)
	}
	testcheck.NoError(t, CopyModelFiles(source, stage))
	entries, err := os.ReadDir(stage)
	testcheck.NoError(t, err)
	var kept []string
	for _, entry := range entries {
		kept = append(kept, entry.Name())
	}
	if want := []string{"config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json"}; !slices.Equal(kept, want) {
		t.Fatalf("the checkpoint keeps %v, want %v", kept, want)
	}
	bare := t.TempDir()
	testutil.WriteTextFile(t, bare, "config.json", "{}")
	if err := CopyModelFiles(bare, filepath.Join(stage, "another")); err == nil {
		t.Fatal("a model with no tokenizer was copied")
	}
}
