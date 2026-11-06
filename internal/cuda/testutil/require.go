// Package testutil admits CUDA tests and isolated device measurements.
package testutil

import (
	"bytes"
	"context"
	"os"
	"path/filepath"
	"regexp"
	"testing"
	"time"

	"overgo/internal/cuda/driver"
	"overgo/internal/processcontrol"
	"overgo/internal/testevidence"
	"overgo/internal/testskip"
)

const environmentVariable = "OVERGO_CUDA_TEST"

// probeEnvironmentVariable gates LONG host-reference experiment probes
// (multi-minute composition and optimizer experiments). They are not device
// tests: the device lane's bounded per-package runs must never inherit them,
// so they answer to their own switch.
const probeEnvironmentVariable = "OVERGO_PROBE_TEST"

func Require(t testing.TB) {
	t.Helper()
	requireEnvironment(t, environmentVariable, "CUDA integration tests")
	// go test's cache keys a test's result on the files it opens or stats and the environment it reads, so statting
	// the driver library reruns the device tests when a driver update replaces it.
	_, _ = os.Stat(filepath.Join(os.Getenv(systemRootVariable), systemDirectory, driverLibrary))
}

// driverLibrary is the CUDA driver library the driver package loads by name, which Windows finds in its system
// directory, systemDirectory under the root systemRootVariable names.
const (
	driverLibrary      = "nvcuda.dll"
	systemDirectory    = "System32"
	systemRootVariable = "SystemRoot"
)

// MeasurementProcess runs this test in an exclusive child process. The caller
// returns when true; false means the child owns admission and runs the test body.
// Call after fixture admission and before creating contexts or changing inputs.
func MeasurementProcess(t *testing.T, ordinal int) bool {
	t.Helper()
	ctx := t.Context()
	deadline, bounded := t.Deadline()
	if bounded {
		var cancel context.CancelFunc
		ctx, cancel = context.WithDeadlineCause(ctx, deadline, context.DeadlineExceeded)
		defer cancel()
	}
	const childEnvironment = "OVERGO_CUDA_MEASUREMENT_TEST"
	if os.Getenv(childEnvironment) != t.Name() {
		var stdout, stderr bytes.Buffer
		args := []string{"-test.run=^" + regexp.QuoteMeta(t.Name()) + "$", "-test.v"}
		if bounded {
			remaining := time.Until(deadline)
			if remaining <= 0 {
				t.Fatal("measurement deadline exhausted before launch")
			}
			args = append(args, "-test.timeout="+remaining.String())
		}
		receipt, err := processcontrol.Run(ctx, processcontrol.Command{
			Path:   os.Args[0],
			Args:   args,
			Env:    append(os.Environ(), childEnvironment+"="+t.Name()),
			Stdout: &stdout, Stderr: &stderr,
		})
		output := stdout.String() + stderr.String()
		t.Log(output)
		if receipt.WallNS > 0 {
			t.Logf("resource: task=%s state=not_busy scope=measurement process_tree=exited", t.Name())
		}
		if err != nil || receipt.ExitCode != 0 {
			t.Fatalf("isolated measurement exit %d: %v", receipt.ExitCode, err)
		}
		if err := testevidence.VerifyOutput("go test", output); err != nil {
			t.Fatal(err)
		}
		return true
	}
	library, err := driver.Open()
	if err != nil {
		t.Fatal(err)
	}
	defer library.Close()
	if err := processcontrol.AwaitResource(ctx, func() error {
		_, err := library.ReserveDevice(ordinal)
		return err
	}); err != nil {
		t.Fatal(err)
	}
	return false
}

// requireEnvironment gates a test lane on one opt-in environment
// variable; both gates share the short-mode skip and the hint shape.
func requireEnvironment(t testing.TB, variable, lane string) {
	t.Helper()
	testskip.Short(t)
	if os.Getenv(variable) == "" {
		// Inapplicable wherever the variable is unset, so a complete run
		// classifies the skip and the package still earns its receipt.
		testskip.NotApplicable(t, "set "+variable+"=1 to run "+lane)
	}
}

// RequireProbe gates a long-running experiment probe: it runs only under
// OVERGO_PROBE_TEST=1, keeping multi-minute host-reference experiments out of
// the device lane's bounded package runs.
func RequireProbe(t testing.TB) {
	t.Helper()
	requireEnvironment(t, probeEnvironmentVariable, "long experiment probes")
}
