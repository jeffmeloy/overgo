package main

import (
	"os"
	"os/exec"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"overgo/internal/artifact"
)

// TestTrailerRootsScanCommitMessages collects every identity the commit
// messages of a checkout name, once each and sorted, across commits.
func TestTrailerRootsScanCommitMessages(t *testing.T) {
	checkout := t.TempDir()
	git := func(args ...string) {
		t.Helper()
		command := exec.Command("git", append([]string{"-C", checkout}, args...)...)
		command.Env = append(os.Environ(),
			"GIT_AUTHOR_NAME=test", "GIT_AUTHOR_EMAIL=test@example.invalid",
			"GIT_COMMITTER_NAME=test", "GIT_COMMITTER_EMAIL=test@example.invalid",
		)
		if output, err := command.CombinedOutput(); err != nil {
			t.Fatalf("git %v: %v\n%s", args, err, output)
		}
	}
	git("init", "-q")
	preparation := "evidence:sha256:" + strings.Repeat("a", 64)
	recipe := "recipe:sha256:" + strings.Repeat("b", 64)
	if err := os.WriteFile(filepath.Join(checkout, "file"), []byte("one"), 0o644); err != nil {
		t.Fatal(err)
	}
	git("add", "file")
	git("commit", "-q", "-m", "first\n\nOvergo-Gate-Preparation: "+preparation+"\nOvergo-Manifest-Plan: "+recipe)
	if err := os.WriteFile(filepath.Join(checkout, "file"), []byte("two"), 0o644); err != nil {
		t.Fatal(err)
	}
	git("add", "file")
	git("commit", "-q", "-m", "second\n\nOvergo-Gate-Preparation: "+preparation+"\nnot:an:identity sha256:short")
	roots, err := trailerRoots(checkout)
	if err != nil {
		t.Fatal(err)
	}
	want := []artifact.ID{}
	for _, text := range []string{preparation, recipe} {
		id, err := artifact.ParseID(text)
		if err != nil {
			t.Fatal(err)
		}
		want = append(want, id)
	}
	slices.SortFunc(want, artifact.CompareID)
	if !slices.Equal(roots, want) {
		t.Fatalf("trailer roots = %v, want %v", roots, want)
	}
	if merged := mergeRoots(roots, want[:1]); !slices.Equal(merged, want) {
		t.Fatalf("merged roots = %v, want %v", merged, want)
	}
}
