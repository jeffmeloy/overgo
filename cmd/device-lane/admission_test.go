package main

import (
	"bytes"
	"context"
	"errors"
	"os"
	"slices"
	"strings"
	"testing"
	"time"

	"overgo/internal/processcontrol"
	"overgo/internal/testutil"
)

// TestDeviceLaneAdmissionWaitsWithinBudget pins the lane's step admission:
// a step refused by a foreign holder is run again until admitted, the wait
// is reported like the gate's shared lease, an exhausted budget names the
// contention and its cause, and any other outcome returns at once.
func TestDeviceLaneAdmissionWaitsWithinBudget(t *testing.T) {
	var output bytes.Buffer
	runs := 0
	step := func() (int, error) {
		runs++
		if runs <= 3 {
			return processcontrol.ResourceBusyExitCode, nil
		}
		return 0, nil
	}
	code, err := runStepAdmitted(t.Context(), &output, "GPU-test", step)
	if err != nil || code != 0 {
		t.Fatalf("admitted step = code %d, %v", code, err)
	}
	if runs != 4 {
		t.Fatalf("step ran %d times before admission, want 4", runs)
	}
	lines := strings.Split(strings.TrimSpace(output.String()), "\n")
	if len(lines) != 2 || !strings.Contains(lines[0], "mode=exclusive state=waiting budget=") || !strings.Contains(lines[1], "state=admitted wait=") {
		t.Fatalf("admission report = %q", output.String())
	}
	output.Reset()
	if code, err := runStepAdmitted(t.Context(), &output, "GPU-test", func() (int, error) { return 0, nil }); err != nil || code != 0 || output.Len() != 0 {
		t.Fatalf("uncontended step = code %d, %v, report %q", code, err, output.String())
	}
	errTestDeadline := errors.New("test deadline")
	ctx, cancel := context.WithTimeoutCause(t.Context(), 120*time.Millisecond, errTestDeadline)
	defer cancel()
	code, err = runStepAdmitted(ctx, &output, "GPU-test", func() (int, error) { return processcontrol.ResourceBusyExitCode, nil })
	if code != processcontrol.ResourceBusyExitCode || !errors.Is(err, processcontrol.ErrResourceBusy) || !errors.Is(err, errTestDeadline) {
		t.Fatalf("exhausted budget = code %d, %v, want the contention and the deadline", code, err)
	}
	if strings.Contains(output.String(), "state=admitted") {
		t.Fatal("an exhausted budget reported admission")
	}
	if code, err := runStepAdmitted(t.Context(), &output, "GPU-test", func() (int, error) { return 1, nil }); err != nil || code != 1 {
		t.Fatalf("failed step = code %d, %v, want its own exit code at once", code, err)
	}
	launch := errors.New("launch failed")
	if _, err := runStepAdmitted(t.Context(), &output, "GPU-test", func() (int, error) { return 0, launch }); !errors.Is(err, launch) {
		t.Fatalf("launch failure = %v, want it returned at once", err)
	}
}

// TestDeviceLaneStepsExitWithTheirOwnStatus pins the step form the admission
// relies on: a `go run` step becomes a built binary, so its exit status is
// the command's own, while a `go test` step keeps its form.
func TestDeviceLaneStepsExitWithTheirOwnStatus(t *testing.T) {
	buildDir := t.TempDir()
	test := deviceTestStep("./internal/cuda/kernel")
	if command, err := executableStep(t.Context(), testutil.RepoRoot(t), buildDir, test); err != nil || !slices.Equal(command, test) {
		t.Fatalf("go test step = %v, %v; want it unchanged", command, err)
	}
	command, err := executableStep(t.Context(), testutil.RepoRoot(t), buildDir, []string{"go", "run", "./cmd/cuda-info"})
	if err != nil {
		t.Fatal(err)
	}
	if len(command) != 1 || !strings.HasPrefix(command[0], buildDir) {
		t.Fatalf("go run step = %v, want one binary under %s", command, buildDir)
	}
	if _, err := os.Stat(command[0]); err != nil {
		t.Fatal(err)
	}
}
