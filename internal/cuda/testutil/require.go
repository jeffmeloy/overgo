package testutil

import (
	"os"
	"testing"

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
