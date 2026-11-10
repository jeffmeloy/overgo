package main

import (
	"os"
	"os/exec"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"overgo/internal/testcheck"
)

// TestAMeasurementIsNoTaskForTheWorker holds which landed steps a replay leaves out as measurements
// (loop-proof/measurement-skip): a step whose landing wrote a record under docs/verification, which a model run on the
// card makes and no worker's tools can, as loop-proof/local-demonstrated's vision-bridge steps failed every try of runs
// 2 to 4; a step whose landing changed code alone is a task.
func TestAMeasurementIsNoTaskForTheWorker(t *testing.T) {
	t.Parallel()
	root := t.TempDir()
	git := func(arguments ...string) string {
		t.Helper()
		output, err := exec.Command("git", append([]string{"-C", root, "-c", "user.name=fixture", "-c", "user.email=fixture@example.com"}, arguments...)...).CombinedOutput()
		if err != nil {
			t.Fatalf("git %v: %v: %s", arguments, err, output)
		}
		return strings.TrimSpace(string(output))
	}
	commit := func(name string) string {
		t.Helper()
		path := filepath.Join(root, filepath.FromSlash(name))
		testcheck.NoError(t, os.MkdirAll(filepath.Dir(path), 0o755))
		testcheck.NoError(t, os.WriteFile(path, []byte(name), 0o644))
		git("add", "-A")
		git("commit", "-q", "-m", name)
		return git("rev-parse", "HEAD")
	}
	git("init", "-q")
	parent := commit("internal/a/a.go")
	coded := commit("internal/a/b.go")
	measured := commit("docs/verification/vision-bridge-9b-envelope.json")
	for _, step := range []struct {
		parent, landed string
		want           []string
	}{
		{parent, coded, nil},
		{coded, measured, []string{"docs/verification/vision-bridge-9b-envelope.json"}},
	} {
		got := testcheck.Must(replayed{parent: step.parent, landed: step.landed}.measurement(t.Context(), root))(t)
		if !slices.Equal(got, step.want) {
			t.Errorf("the landing %.8s names the records %q, want %q", step.landed, got, step.want)
		}
	}
}
