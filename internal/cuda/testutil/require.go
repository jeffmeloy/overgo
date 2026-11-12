// Package testutil admits CUDA tests and isolated device measurements.
package testutil

import (
	"bytes"
	"os"
	"regexp"
	"testing"

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
	// The driver a lane runs on, which go test then keys the test on: a driver update reruns it.
	_ = os.Getenv(driver.IdentityEnvironment)
}

// MeasurementProcess runs this test in a child process of its own, whose
// contexts hold the card's turn while it measures, as every card user's do
// (card-turn), so it measures on the card alone. The caller returns when
// true; false means this is the child, which runs the test body. Call after
// fixture admission and before creating contexts or changing inputs. The
// ordinal is no longer read: the turn is the card's, taken as a context is.
func MeasurementProcess(t *testing.T, _ int) bool {
	t.Helper()
	const childEnvironment = "OVERGO_CUDA_MEASUREMENT_TEST"
	if os.Getenv(childEnvironment) != t.Name() {
		var stdout, stderr bytes.Buffer
		args := []string{"-test.run=^" + regexp.QuoteMeta(t.Name()) + "$", "-test.v", "-test.timeout=0"}
		receipt, err := processcontrol.Run(t.Context(), processcontrol.Command{
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
