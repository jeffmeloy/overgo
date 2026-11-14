//go:build modeltest

package server

import (
	"testing"

	cudatest "overgo/internal/cuda/testutil"
)

// TestTranscriptionWorkspaceRecordsResidentRunsOnTheDevice holds the runs a
// workspace serves for a resident recipe to the CUDA device its encoder binds. The cycle runs it with the package's
// other device tests, behind modeltest, so an edit to the server's journeys reaches no device lane of the gate.
func TestTranscriptionWorkspaceRecordsResidentRunsOnTheDevice(t *testing.T) {
	t.Parallel()
	cudatest.Require(t)
	if device := workspaceDevice(t, true); device != "cuda:0" {
		t.Fatalf("a resident recipe's runs are recorded on device %q, want cuda:0", device)
	}
}
