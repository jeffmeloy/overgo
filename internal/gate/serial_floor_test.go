package gate

import (
	"testing"

	"overgo/internal/testutil"
)

// TestGateSuiteSerialFloorBudget holds every top-level test that uses the
// shared live-checkout fixture to running in parallel. Go runs the parallel
// tests only after every serial one has finished, and the fixture's one-time
// build -- a git worktree and the package input graph, about 50 s -- is paid
// by whichever test asks for it first. One serial user put that build in the
// serial phase, where nothing overlapped it: measured 2026-09-23 at 50.8 s of
// the suite's 54.1 s serial floor in a 177.6 s run.
func TestGateSuiteSerialFloorBudget(t *testing.T) {
	testutil.RequireParallel(t, "liveRepositoryFixture(")
}
