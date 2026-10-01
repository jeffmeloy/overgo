package main

import (
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/runrecord"
	"overgo/internal/testutil"
)

// surfaceHistory writes a fixture repository whose history moves the
// inference surface, and so the evaluation surface that holds it, and then
// only documentation, and returns its root with the commit before the surface
// move, the one before the documentation change, and HEAD.
func surfaceHistory(t *testing.T) (root, beforeSurface, beforeDocs, head string) {
	t.Helper()
	root = t.TempDir()
	write := func(name, content string) {
		t.Helper()
		path := filepath.Join(root, filepath.FromSlash(name))
		if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(path, []byte(content), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	commit := func(message string) string {
		t.Helper()
		for _, arguments := range [][]string{{"add", "-A"}, {"commit", "-q", "-m", message}, {"rev-parse", "HEAD"}} {
			command := exec.Command("git", append([]string{"-c", "user.name=fixture", "-c", "user.email=fixture@example.com"}, arguments...)...)
			command.Dir = root
			output, err := command.CombinedOutput()
			if err != nil {
				t.Fatalf("git %v: %v: %s", arguments, err, output)
			}
			if arguments[0] == "rev-parse" {
				return strings.TrimSpace(string(output))
			}
		}
		return ""
	}
	write("go.mod", "module fixture\n\ngo 1.26\n")
	for _, surface := range []string{"internal/inference", "internal/cuda/executor", "internal/modelrecipe", "internal/evaluation"} {
		write(surface+"/"+filepath.Base(surface)+".go", "package "+filepath.Base(surface)+"\n")
	}
	write("docs/notes.md", "fixture\n")
	if output, err := exec.Command("git", "init", "-q", root).CombinedOutput(); err != nil {
		t.Fatalf("git init: %v: %s", err, output)
	}
	beforeSurface = commit("fixture")
	write("internal/inference/inference.go", "package inference\n\n// Rate moves the surface.\nconst Rate = 1\n")
	beforeDocs = commit("move the surface")
	write("docs/notes.md", "fixture, revised\n")
	head = commit("revise the notes")
	return root, beforeSurface, beforeDocs, head
}

// TestSmokeCurrentFollowsTheInferenceSurface holds a DNA model's currency to
// its smoke run and the inference surface, in a fixture repository whose
// history the test writes: a passing run at the current commit is current,
// one at a commit the surface has moved past is not, one at a commit only
// documentation has moved past still is, a failed run is not, and no run
// says how to acquire one into the selected store.
func TestSmokeCurrentFollowsTheInferenceSurface(t *testing.T) {
	t.Parallel()
	root, beforeSurface, beforeDocs, head := surfaceHistory(t)
	const store = "C:/stores/selected"
	model := testutil.ArtifactID(t, artifact.KindModel, "carbon")
	passed := func(commit string) runrecord.Run {
		return runrecord.Run{Outcome: runrecord.OutcomeSucceeded, CodeCommit: commit}
	}
	for _, current := range []string{head, beforeDocs} {
		if err := smokeCurrent(t.Context(), root, store, model, passed(current), true); err != nil {
			t.Errorf("a passing run at %.12s, with only documentation since, is not current: %v", current, err)
		}
	}
	err := smokeCurrent(t.Context(), root, store, model, passed(beforeSurface), true)
	if err == nil || !strings.Contains(err.Error(), "inference surface moved") {
		t.Errorf("a run before a surface move was judged current: %v", err)
	}
	err = smokeCurrent(t.Context(), root, store, model, runrecord.Run{Outcome: runrecord.OutcomeFailed, Failure: "smoke_failed", CodeCommit: head}, true)
	if err == nil || !strings.Contains(err.Error(), "failed") {
		t.Errorf("a failed run was judged current: %v", err)
	}
	err = smokeCurrent(t.Context(), root, store, model, runrecord.Run{}, false)
	if err == nil || !strings.Contains(err.Error(), "validate -smoke") || !strings.Contains(err.Error(), "-repo "+store) {
		t.Errorf("no run did not name validate -smoke on the selected store: %v", err)
	}
}
