package landing

import (
	"fmt"
	"io"
	"os"
	"path/filepath"
	"slices"
	"strconv"
	"strings"
	"testing"

	"overgo/internal/testskip"
)

// streamChild makes this test binary the command landing.run starts: it
// writes a result and a diagnostic and exits with the code it is given.
const streamChild = "OVERGO_LANDING_STREAM_CHILD"

// TestLandingRunKeepsEachStreamOwned holds the gate's command runner to one
// owner per stream: standard output comes back once, as the command wrote
// it, and a failure carries the diagnostic from standard error.
func TestLandingRunKeepsEachStreamOwned(t *testing.T) {
	if code := os.Getenv(streamChild); code != "" {
		fmt.Print("result")
		fmt.Fprint(os.Stderr, "diagnostic")
		exit, _ := strconv.Atoi(code)
		os.Exit(exit)
	}
	t.Parallel()
	l := &landing{Options: Options{Repo: t.TempDir()}}
	for _, code := range []string{"0", "3"} {
		output, err := l.run(t.Context(), append(os.Environ(), streamChild+"="+code), os.Args[0], "-test.run=^TestLandingRunKeepsEachStreamOwned$")
		if output != "result" {
			t.Errorf("exit %s: standard output = %q, want the result once", code, output)
		}
		if failed := code != "0"; failed != (err != nil) || failed && !strings.Contains(err.Error(), "diagnostic") {
			t.Errorf("exit %s: error = %v, want a failure to carry the diagnostic", code, err)
		}
	}
}

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
