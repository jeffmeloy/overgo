package plan

import (
	"testing"

	"overgo/internal/testutil"
)

// TestPlanSuiteWallBudget holds every top-level test in this package to
// running in parallel unless it changes process-wide state, which Go allows
// only serially. The suite is selected by every control-plane landing; with
// 86 of its 88 tests serial it measured 74.6 s, and with them parallel 41 to
// 51 s over four runs (2026-09-23). A new serial test would quietly bring the
// wall back.
func TestPlanSuiteWallBudget(t *testing.T) {
	testutil.RequireParallel(t)
}
