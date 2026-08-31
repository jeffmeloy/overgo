package runrecord

import (
	"strings"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/testutil"
)

// TestInteractionEfficiencyGate pins the claim gate's core semantics: a
// claim must cover the complete closed counter surface, equivalent task
// completion is checked before any counter, a Pareto improvement wins, a
// regression on any counter blocks the win unless an explicit tradeoff
// decision names it, and shifted work — better on one counter, silently
// worse on another — refuses.
func TestInteractionEfficiencyGate(t *testing.T) {
	result := testutil.ArtifactID(t, artifact.KindOutput, "gate-result")
	evidence := testutil.ArtifactID(t, artifact.KindEvidence, "gate-evidence")
	baseline := EfficiencyTrace{
		Surface: SurfaceStorage, Task: "catalog-read",
		Work:   InteractionWork{ScannedFacts: 80, Bytes: 4000, WallNS: 900},
		Result: result, Evidence: evidence,
	}
	candidate := baseline
	candidate.Work = InteractionWork{ScannedFacts: 8, Bytes: 4000, WallNS: 700}
	covered := EfficiencyCounterNames()

	comparison, err := CompareEfficiencyTraces(candidate, baseline, covered, nil)
	if err != nil || !comparison.Win || comparison.Tradeoff || len(comparison.Worsened) != 0 {
		t.Fatalf("pareto claim = (%+v, %v)", comparison, err)
	}
	if _, err := CompareEfficiencyTraces(candidate, baseline, covered[1:], nil); err == nil ||
		!strings.Contains(err.Error(), "is not covered") {
		t.Fatalf("uncovered counter admitted: %v", err)
	}
	foreign := baseline
	foreign.Task = "another-task"
	if _, err := CompareEfficiencyTraces(candidate, foreign, covered, nil); err == nil ||
		!strings.Contains(err.Error(), "not comparable") {
		t.Fatalf("foreign task compared: %v", err)
	}

	shifted := candidate
	shifted.Work.Retries = 6
	comparison, err = CompareEfficiencyTraces(shifted, baseline, covered, nil)
	if err != nil || comparison.Win || len(comparison.Worsened) != 1 {
		t.Fatalf("shifted work won without a tradeoff: (%+v, %v)", comparison, err)
	}
	if _, err := CompareEfficiencyTraces(shifted, baseline, covered, &EfficiencyTradeoff{
		Decider: "owner", Rationale: "retries accepted for the scan reduction", Accepted: []string{"waits"},
	}); err == nil || !strings.Contains(err.Error(), "work was shifted, not saved") {
		t.Fatalf("unnamed worsened counter accepted: %v", err)
	}
	comparison, err = CompareEfficiencyTraces(shifted, baseline, covered, &EfficiencyTradeoff{
		Decider: "owner", Rationale: "retries accepted for the scan reduction", Accepted: []string{"retries"},
	})
	if err != nil || !comparison.Win || !comparison.Tradeoff {
		t.Fatalf("explicit tradeoff refused: (%+v, %v)", comparison, err)
	}
}
