package workflowruntime

import (
	"context"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/runrecord"
	"overgo/internal/testutil"
)

// TestInteractionEfficiencyGate proves the coalescing claim from the
// coalescer's own behavior: a flurry of duplicate signals reconciles once
// where the uncoalesced path runs once per signal, and the measured claim
// wins the gate — while a variant that lowered wakeups by silently retrying
// refuses without an explicit tradeoff decision.
func TestInteractionEfficiencyGate(t *testing.T) {
	ctx := t.Context()
	coalescer := NewReconcileCoalescer()
	signals := 6
	runs := 0
	for range signals {
		coalescer.Signal("gate-boundary")
	}
	for range signals {
		if _, err := coalescer.Reconcile(ctx, "gate-boundary", func(context.Context) error {
			runs++
			return nil
		}); err != nil {
			t.Fatal(err)
		}
	}
	if runs != 1 {
		t.Fatalf("coalesced flurry ran %d reconciliations", runs)
	}

	evidence := testutil.ArtifactID(t, artifact.KindEvidence, "efficiency-wakeup-evidence")
	result := testutil.ArtifactID(t, artifact.KindOutput, "efficiency-wakeup-result")
	baseline := runrecord.EfficiencyTrace{
		Surface: runrecord.SurfaceWorkflow, Task: "duplicate-wakeup-flurry",
		Work:   runrecord.InteractionWork{Wakeups: uint64(signals), Commits: 1},
		Result: result, Evidence: evidence,
	}
	candidate := baseline
	candidate.Work.Wakeups = uint64(runs)
	comparison, err := runrecord.CompareEfficiencyTraces(
		candidate, baseline, runrecord.EfficiencyCounterNames(), nil,
	)
	if err != nil || !comparison.Win || len(comparison.Worsened) != 0 {
		t.Fatalf("measured coalescing claim = (%+v, %v)", comparison, err)
	}

	shifted := candidate
	shifted.Work.Retries = 3
	comparison, err = runrecord.CompareEfficiencyTraces(
		shifted, baseline, runrecord.EfficiencyCounterNames(), nil,
	)
	if err != nil || comparison.Win {
		t.Fatalf("wakeup reduction that shifted work into retries won: (%+v, %v)", comparison, err)
	}
}
