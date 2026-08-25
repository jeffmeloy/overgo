package main

import (
	"os"
	"path/filepath"
	"slices"
	"testing"
)

func TestSanitizerCommandFailsOnHazard(t *testing.T) {
	for _, tool := range deviceRaceTools {
		command := sanitizerCmd(tool, "fixture.test")
		flag := slices.Index(command, "--error-exitcode")
		if flag < 0 || flag+1 >= len(command) || command[flag+1] != "1" {
			t.Errorf("%s command does not fail on a detected hazard: %v", tool, command)
		}
	}
}

func TestCompilerUsesWorktreeToolchainWhenConfiguredCompilerIsAbsent(t *testing.T) {
	root := t.TempDir()
	compiler := filepath.Join(root, filepath.FromSlash(localRaceCompiler))
	if err := os.MkdirAll(filepath.Dir(compiler), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(compiler, nil, 0o755); err != nil {
		t.Fatal(err)
	}
	got, err := selectCompiler("missing-compiler", root, func(candidate string) (string, error) {
		if candidate == compiler {
			return candidate, nil
		}
		return "", os.ErrNotExist
	})
	if err != nil || got != compiler {
		t.Fatalf("compiler=%q err=%v want %q", got, err, compiler)
	}
}
