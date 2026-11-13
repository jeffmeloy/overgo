package plan

import (
	"strings"
	"testing"
	"time"
)

// TestARunsReportSaysHowLongAPhaseRan holds a measurement run's report to what a phase's wall is, the time it ran, not a
// limit it was held to (owner, 2026-11-13): "preflight did not pass in 10m0s" read as a clock that ended it. A phase
// that did not pass and one that passed each say how long they ran, then their verdict.
func TestARunsReportSaysHowLongAPhaseRan(t *testing.T) {
	t.Parallel()
	began := time.Date(2026, 11, 8, 6, 15, 0, 0, time.UTC)
	report := MeasurementRun{Commit: "fdd4ab1cea37", Process: 1, Started: began, Ended: began.Add(2 * time.Hour), Phases: []MeasurementPhase{
		{Name: "preflight", Started: began, Ended: began.Add(10 * time.Minute)},
		{Name: "lanes", Started: began.Add(10 * time.Minute), Ended: began.Add(111 * time.Minute), Passed: true},
	}}.Report()
	if !strings.Contains(report, "preflight ran 10m0s and did not pass") || !strings.Contains(report, "lanes ran 1h41m0s and passed") || strings.Contains(report, " in 10m") {
		t.Fatalf("the run reads %q, want each phase's wall said as how long it ran", report)
	}
}
