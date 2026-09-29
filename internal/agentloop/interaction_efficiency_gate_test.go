package agentloop

import (
	"encoding/json"
	"testing"
)

// TestContinuousHandoffReusesContext proves the incremental-context claim
// from the coordinator's own handoffs: the same repeated attempt costs zero
// fresh context bytes on the continuous path where the stateless path
// re-transfers everything.
func TestContinuousHandoffReusesContext(t *testing.T) {
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
}
