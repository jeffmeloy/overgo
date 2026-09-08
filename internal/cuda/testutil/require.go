package testutil

import (
	"bytes"
	"os"
	"regexp"
	"testing"

	"overgo/internal/cuda/driver"
	"overgo/internal/processcontrol"
	"overgo/internal/testevidence"
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
}

// MeasurementProcess runs this test in an exclusive child process. The caller
// returns when true; false means the child owns admission and runs the test body.
// Call after fixture admission and before creating contexts or changing inputs.
func MeasurementProcess(t *testing.T, ordinal int) bool {
	t.Helper()
	const childEnvironment = "OVERGO_CUDA_MEASUREMENT_TEST"
	if os.Getenv(childEnvironment) != t.Name() {
		var stdout, stderr bytes.Buffer
		receipt, err := processcontrol.Run(t.Context(), processcontrol.Command{
			Path:   os.Args[0],
			Args:   []string{"-test.run=^" + regexp.QuoteMeta(t.Name()) + "$", "-test.v"},
			Env:    append(os.Environ(), childEnvironment+"="+t.Name()),
			Stdout: &stdout, Stderr: &stderr,
		})
		output := stdout.String() + stderr.String()
		t.Log(output)
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
	if _, err := library.ReserveDevice(ordinal); err != nil {
		t.Fatal(err)
	}
	return false
}

// requireEnvironment gates a test lane on one opt-in environment
// variable; both gates share the short-mode skip and the hint shape.
func requireEnvironment(t testing.TB, variable, lane string) {
	t.Helper()
	if testing.Short() {
		t.Skip(testevidence.ShortIntegrationSkip)
	}
	if os.Getenv(variable) == "" {
		t.Skip("set " + variable + "=1 to run " + lane)
	}
}

// RequireProbe gates a long-running experiment probe: it runs only under
// OVERGO_PROBE_TEST=1, keeping multi-minute host-reference experiments out of
// the device lane's bounded package runs.
func RequireProbe(t testing.TB) {
	t.Helper()
	requireEnvironment(t, probeEnvironmentVariable, "long experiment probes")
}
