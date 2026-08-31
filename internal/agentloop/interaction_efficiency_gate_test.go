package agentloop

import (
	"encoding/json"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/runrecord"
	"overgo/internal/testutil"
)

// TestInteractionEfficiencyGate proves the incremental-context claim from
// the coordinator's own handoffs: the same repeated attempt costs zero fresh
// context bytes on the continuous path where the stateless path re-transfers
// everything, both handoffs carry the same admitted stimulus, and the
// measured claim wins the gate without shifting work to another counter.
func TestInteractionEfficiencyGate(t *testing.T) {
	ctx := t.Context()
	coordinator, _ := coordinatorFixture(t)
	arguments := json.RawMessage(`{"step":1}`)
	continuous := &Session{ID: "efficiency-continuous"}
	if _, err := coordinator.Propose(ctx, continuous, "probe.read", arguments); err != nil {
		t.Fatal(err)
	}
	baselineBytes := uint64(0)
	for _, arc := range continuous.handoff.fresh {
		baselineBytes += arc.Bytes
	}
	if baselineBytes == 0 {
		t.Fatalf("stateless handoff transferred nothing: %+v", continuous.handoff)
	}
	if _, err := coordinator.Propose(ctx, continuous, "probe.read", arguments); err != nil {
		t.Fatal(err)
	}
	candidateBytes := uint64(0)
	for _, arc := range continuous.handoff.fresh {
		candidateBytes += arc.Bytes
	}
	if candidateBytes != 0 || len(continuous.handoff.reused) != 1 {
		t.Fatalf("continuous handoff = %+v", continuous.handoff)
	}

	evidence := testutil.ArtifactID(t, artifact.KindEvidence, "efficiency-agent-evidence")
	result := testutil.ArtifactID(t, artifact.KindOutput, "efficiency-agent-result")
	baseline := runrecord.EfficiencyTrace{
		Surface: runrecord.SurfaceAgent, Task: "repeated-attempt-context",
		Work: runrecord.InteractionWork{
			ContextBytes: baselineBytes, RepeatedContextIDs: 0, ModelTurns: 1, ToolCalls: 1,
		},
		Result: result, Evidence: evidence,
	}
	candidate := baseline
	candidate.Work.ContextBytes = candidateBytes
	candidate.Work.RepeatedContextIDs = 0
	comparison, err := runrecord.CompareEfficiencyTraces(
		candidate, baseline, runrecord.EfficiencyCounterNames(), nil,
	)
	if err != nil || !comparison.Win || len(comparison.Worsened) != 0 {
		t.Fatalf("measured context claim = (%+v, %v)", comparison, err)
	}
}
