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
	if testing.Short() {
		t.Skip(testevidence.ShortIntegrationSkip)
	}
	if os.Getenv(environmentVariable) == "" {
		t.Skip("set " + environmentVariable + "=1 to run CUDA integration tests")
	}
}

// RequireProbe gates a long-running experiment probe: it runs only under
// OVERGO_PROBE_TEST=1, keeping multi-minute host-reference experiments out of
// the device lane's bounded package runs.
func RequireProbe(t testing.TB) {
	t.Helper()
	if testing.Short() {
		t.Skip(testevidence.ShortIntegrationSkip)
	}
	if os.Getenv(probeEnvironmentVariable) == "" {
		t.Skip("set " + probeEnvironmentVariable + "=1 to run long experiment probes")
	}
}
