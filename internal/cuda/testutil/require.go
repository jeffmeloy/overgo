package testutil

import (
	"os"
	"testing"
)

const environmentVariable = "OVERGO_CUDA_TEST"

func Require(t testing.TB) {
	t.Helper()
	if os.Getenv(environmentVariable) == "" {
		t.Skip("set " + environmentVariable + "=1 to run CUDA integration tests")
	}
}
