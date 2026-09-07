package gate

import (
	"strings"
	"testing"
	"time"

	"overgo/internal/overgodb"
	"overgo/internal/runrecord"
)

// TestBatchedGateCostReportScope pins: the cost separates total gate wall,
// failed steps of any phase, executed non-acceptance phases, accepted
// checkpoint cost and the estimated saving; each is labelled on the audit
// line and the saving is never folded into the wall.
func TestBatchedGateCostReportScope(t *testing.T) {
	second := uint64(time.Second)
	steps := []runrecord.GateStep{
		{Name: "protection", Phase: runrecord.PhasePackage, Outcome: runrecord.StepSucceeded, DurationNS: 9},
		{Name: "test", Phase: runrecord.PhaseTest, Outcome: runrecord.StepFailed, DurationNS: 2 * second},
		{Name: "docs", Phase: runrecord.PhasePackage, Outcome: runrecord.StepInapplicable, DurationNS: 1},
		{Name: "acceptance-producer", Phase: runrecord.PhaseTest, Outcome: runrecord.StepReused, DurationNS: 1},
		{Name: "acceptance-consumer", Phase: runrecord.PhaseTest, Outcome: runrecord.StepFailed, DurationNS: 3 * second},
		{Name: "acceptance", Phase: runrecord.PhaseTest, Outcome: runrecord.StepSucceeded, DurationNS: 7},
	}
	cost := batchCostOf(steps)
	if cost.StepNS != 5*second+18 {
		t.Fatalf("step time = %s, want every step summed", time.Duration(cost.StepNS))
	}
	if cost.Failed != 2 || cost.FailedNS != 5*second {
		t.Fatalf("failed = %d/%s, want the failed test and acceptance steps", cost.Failed, time.Duration(cost.FailedNS))
	}
	if cost.Other != 1 || cost.OtherNS != 9 {
		t.Fatalf("other phases = %d/%s, want the executed protection step only", cost.Other, time.Duration(cost.OtherNS))
	}
	if cost.Accepted != 2 || cost.Reused != 1 || cost.ExecutedNS != 7 {
		t.Fatalf("accepted cost = %+v, want the reused producer and the parent", cost)
	}

	store, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	commitGateAttempt(t, store, "flush", "scope", costSteps(runrecord.StepSucceeded, runrecord.StepSucceeded, 5*second, 3*second))
	g := &gateContext{planRef: "flush/scope"}
	g.batchCostAudit(t.Context(), store, steps, 3*second)
	if len(g.audit) != 1 {
		t.Fatalf("audit = %v", g.audit)
	}
	line := g.audit[0]
	for _, label := range []string{
		"total_wall=3s summed_step_time=5.000000018s", "failed=2/5s", "other_phases=1/9ns",
		"accepted=2 reused=1 accepted_executed=7ns", "estimated_step_time_avoided=5s", "prior_runs=1",
	} {
		if !strings.Contains(line, label) {
			t.Fatalf("audit lacks %q: %s", label, line)
		}
	}
	if strings.Contains(line, "total_wall=5") || strings.Contains(line, "estimated_saved=") {
		t.Fatalf("saving folded into the wall: %s", line)
	}
}
