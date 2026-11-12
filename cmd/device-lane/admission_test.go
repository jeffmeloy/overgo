package main

import (
	"os"
	"slices"
	"strings"
	"testing"

	"overgo/internal/testcheck"
	"overgo/internal/testutil"
)

// TestDeviceLaneStepsExitWithTheirOwnStatus pins the step form the lane
// relies on: a `go run` step becomes a built binary, so its exit status is
// the command's own, while a `go test` step keeps its form.
func TestDeviceLaneStepsExitWithTheirOwnStatus(t *testing.T) {
	buildDir := t.TempDir()
	test := deviceTestStep("./internal/cuda/kernel")
	if command, err := executableStep(t.Context(), testutil.RepoRoot(t), buildDir, test); err != nil || !slices.Equal(command, test) {
		t.Fatalf("go test step = %v, %v; want it unchanged", command, err)
	}
	command := testcheck.Must(executableStep(t.Context(), testutil.RepoRoot(t), buildDir, []string{"go", "run", "./cmd/cuda-info"}))(t)
	if len(command) != 1 || !strings.HasPrefix(command[0], buildDir) {
		t.Fatalf("go run step = %v, want one binary under %s", command, buildDir)
	}
	if _, err := os.Stat(command[0]); err != nil {
		t.Fatal(err)
	}
}
