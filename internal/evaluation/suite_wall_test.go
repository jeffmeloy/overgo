package evaluation

import (
	"testing"

	"overgo/internal/testutil"
)

// TestEvaluationSuiteWallBudget holds every top-level test in this package to
// running in parallel unless it changes process-wide state. All 135 ran
// serially -- 50.3 s -- though none needed to; in parallel the suite measured
// 11.6 to 11.8 s over three runs (2026-09-23).
func TestEvaluationSuiteWallBudget(t *testing.T) {
	testutil.RequireParallel(t)
}
