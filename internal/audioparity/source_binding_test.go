package audioparity

import (
	"os"
	"path/filepath"
	"slices"
	"testing"

	"overgo/internal/gosource"
	"overgo/internal/testutil"
)

func TestAudioProducerSourceBinding(t *testing.T) {
	t.Parallel()
	t.Run("live producer closure", func(t *testing.T) {
		source := audioSources(t, testutil.RepoRoot(t))
		for path, want := range map[string]bool{
			"internal/audioparity/cpu_baseline_test.go":      true,
			"internal/evaluation/transcription_resources.go": true,
			"cmd/evaluate/main.go":                           true,
			"cmd/recipe/main.go":                             true,
			"internal/gate/verification.go":                  false,
			"internal/evaluation/qwen_retained_text_test.go": false,
		} {
			found := slices.ContainsFunc(source.Files, func(file gosource.File) bool { return file.Path == path })
			if found != want {
				t.Fatalf("live source %s present=%t, want %t", path, found, want)
			}
		}
		t.Logf("producer sources=%d identity=%s; no model acquisition or selector authority", len(source.Files), source.Identity())
	})
	root := t.TempDir()
	sources := map[string]string{
		"go.mod":                             "module example\n\ngo 1.26\n",
		"internal/audioparity/audio.go":      "package audioparity\n",
		"internal/audioparity/audio_test.go": "package audioparity\nimport (\"testing\"; \"example/internal/shared\")\nfunc TestAudio(t *testing.T) { shared.Value() }\n",
		"internal/shared/shared.go":          "package shared\nfunc Value() int { return 1 }\n",
		"internal/shared/shared_test.go":     "package shared\nimport \"testing\"\nfunc TestShared(t *testing.T) {}\n",
		"internal/foreign/foreign.go":        "package foreign\nfunc Value() int { return 1 }\n",
		"cmd/evaluate/main.go":               "package main\nimport \"example/internal/shared\"\nfunc main() { shared.Value() }\n",
		"cmd/evaluate/main_test.go":          "package main\nimport \"testing\"\nfunc TestCLI(t *testing.T) {}\n",
		"cmd/recipe/main.go":                 "package main\nfunc main() {}\n",
	}
	for name, source := range sources {
		path := filepath.Join(root, name)
		if err := os.MkdirAll(filepath.Dir(path), 0700); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(path, []byte(source), 0600); err != nil {
			t.Fatal(err)
		}
	}
	baseline := audioSources(t, root).Identity()
	for _, mutation := range []struct {
		path        string
		invalidates bool
	}{
		{"internal/foreign/foreign.go", false},
		{"internal/shared/shared_test.go", false},
		{"cmd/evaluate/main_test.go", false},
		{"internal/shared/shared.go", true},
		{"internal/audioparity/audio_test.go", true},
		{"cmd/evaluate/main.go", true},
		{"cmd/recipe/main.go", true},
	} {
		t.Run(mutation.path, func(t *testing.T) {
			path := filepath.Join(root, mutation.path)
			if err := os.WriteFile(path, []byte(sources[mutation.path]+"\n// source binding mutation\n"), 0600); err != nil {
				t.Fatal(err)
			}
			changed := audioSources(t, root).Identity() != baseline
			if err := os.WriteFile(path, []byte(sources[mutation.path]), 0600); err != nil {
				t.Fatal(err)
			}
			if changed != mutation.invalidates {
				t.Fatalf("source invalidation=%t, want %t", changed, mutation.invalidates)
			}
		})
	}
}
