package modelartifact

import (
	"os"
	"path/filepath"
	"testing"
)

// TestARecordedConfigNamesItsWeights holds a model's recorded config to the
// weights it was built from: each weight file at the top of the source
// directory is named with its digest beside the config files, so the same
// files read from two directories are the same weights (WeightsKey), and a
// trained checkpoint, which keeps its base's config files, is other weights.
// A directory's other files are no weights, and a record without weights has
// no key.
func TestARecordedConfigNamesItsWeights(t *testing.T) {
	t.Parallel()
	write := func(directory, name, content string) {
		t.Helper()
		if err := os.WriteFile(filepath.Join(directory, name), []byte(content), 0o600); err != nil {
			t.Fatal(err)
		}
	}
	base, copied, trained := t.TempDir(), t.TempDir(), t.TempDir()
	for _, directory := range []string{base, copied, trained} {
		write(directory, "config.json", `{"model_type": "qwen3_5"}`)
		write(directory, "README.md", "not weights")
	}
	write(base, "model.safetensors", "the base's weights")
	write(copied, "model.safetensors", "the base's weights")
	write(trained, "model.safetensors", "a trained checkpoint's weights")
	key := func(directory string) string {
		t.Helper()
		read, err := weightSources(directory)
		if err != nil || len(read) != 1 || read[0].Name != "weights:model.safetensors" {
			t.Fatalf("weight sources of %s = %v (%v), want the one weight file", directory, read, err)
		}
		return WeightsKey(append([]ConfigSource{{Name: "config.json", SHA256: "c0"}}, read...))
	}
	if key(base) == "" || key(base) != key(copied) || key(base) == key(trained) {
		t.Fatal("the same weights read from two directories differ, or a trained checkpoint's weights are its base's")
	}
	if WeightsKey([]ConfigSource{{Name: "config.json", SHA256: "c0"}}) != "" {
		t.Fatal("a record without weights names weights")
	}
}
