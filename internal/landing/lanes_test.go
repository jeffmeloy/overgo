package landing

import (
	"io"
	"os"
	"path/filepath"
	"slices"
	"testing"

	"overgo/internal/testskip"
)

// TestLaneReachSkipsOtherPackagesTests reaches a lane from production code
// in its closure and from its own packages' tests, never from another
// package's tests: a test file compiles only into its own test binary.
func TestLaneReachSkipsOtherPackagesTests(t *testing.T) {
	t.Parallel()
	repo := t.TempDir()
	for name, content := range map[string]string{
		"go.mod":      "module fixture\n\ngo 1.26\n",
		"a/a.go":      "package a\n\nimport \"fixture/b\"\n\nvar A = b.B\n",
		"a/a_test.go": "package a\n",
		"b/b.go":      "package b\n\nvar B = 1\n",
		"b/b_test.go": "package b\n",
		"c/c.go":      "package c\n",
	} {
		path := filepath.Join(repo, filepath.FromSlash(name))
		if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(path, []byte(content), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	candidate := lane{packages: []string{"./a"}}
	for changed, want := range map[string]bool{
		"b/b.go":      true,
		"a/a_test.go": true,
		"b/b_test.go": false,
		"c/c.go":      false,
	} {
		l := &landing{Options: Options{Repo: repo, Log: io.Discard}, paths: []string{changed}}
		reached, err := l.reaches(t.Context(), candidate)
		if err != nil {
			t.Fatal(err)
		}
		if reached != want {
			t.Errorf("a change to %s reached the lane = %t, want %t", changed, reached, want)
		}
	}
}

// TestAudioLaneOptsIntoTheStoreAcceptance holds the audio acceptances to a
// runner: go test skips them without the store-acceptance opt-in, so the lane
// that reaches internal/audioparity must set it.
func TestAudioLaneOptsIntoTheStoreAcceptance(t *testing.T) {
	t.Parallel()
	index := slices.IndexFunc(lanes, func(candidate lane) bool { return slices.Contains(candidate.packages, "./internal/audioparity") })
	if index < 0 {
		t.Fatal("no lane reaches internal/audioparity; its acceptances would run nowhere")
	}
	if !slices.Contains(lanes[index].environment, testskip.StoreAcceptanceEnv+"=1") {
		t.Fatalf("lane %s runs internal/audioparity without %s=1; its acceptances would skip", lanes[index].name, testskip.StoreAcceptanceEnv)
	}
}
