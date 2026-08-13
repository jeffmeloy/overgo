package testutil

import (
	"os"
	"testing"

	"overgo/internal/testevidence"
)

const environmentVariable = "OVERGO_CUDA_TEST"

func Require(t testing.TB) {
	t.Helper()
	if testing.Short() {
		t.Skip(testevidence.ShortIntegrationSkip)
	}
	if os.Getenv(environmentVariable) == "" {
		t.Skip("set " + environmentVariable + "=1 to run CUDA integration tests")
	}
}
