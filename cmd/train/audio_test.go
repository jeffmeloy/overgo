package main

import (
	"flag"
	"io"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"overgo/internal/artifact"
)

func TestAudioTrainingContract(t *testing.T) {
	id, err := artifact.IdentifyBytes(artifact.KindRecipe, []byte("audio training admission fixture"))
	if err != nil {
		t.Fatal(err)
	}
	for _, tc := range []struct {
		name, manifest, refusal string
		arguments               []string
	}{
		{name: "preview conflict", manifest: `{}`, arguments: []string{"-preview-dataset", "unused"}, refusal: "cannot be combined"},
		{name: "bootstrap conflict", manifest: `{}`, arguments: []string{"-bootstrap-recipe", "unused"}, refusal: "cannot be combined"},
		{name: "unknown field", manifest: `{"unknown":true}`, refusal: "unknown field"},
		{name: "missing explicit processing", manifest: `{}`, refusal: "explicit base, target case"},
		{name: "dense input conflict", manifest: `{}`, arguments: []string{"-dataset", "unused"}, refusal: "dense-only inputs refuse"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			root := t.TempDir()
			manifest := filepath.Join(root, "audio.json")
			if err := os.WriteFile(manifest, []byte(tc.manifest), 0600); err != nil {
				t.Fatal(err)
			}
			previousArgs, previousFlags := os.Args, flag.CommandLine
			t.Cleanup(func() { os.Args, flag.CommandLine = previousArgs, previousFlags })
			flag.CommandLine = flag.NewFlagSet("train", flag.ContinueOnError)
			flag.CommandLine.SetOutput(io.Discard)
			output := filepath.Join(root, "checkpoint")
			os.Args = append([]string{"train", "-store", filepath.Join(root, "store"), "-recipe", id.String(), "-audio-manifest", manifest, "-out", output}, tc.arguments...)
			if err := run(); err == nil || !strings.Contains(err.Error(), tc.refusal) {
				t.Fatalf("refusal=%v; want %q", err, tc.refusal)
			}
			if _, err := os.Stat(output); !os.IsNotExist(err) {
				t.Fatalf("refused command created output: %v", err)
			}
		})
	}
}
