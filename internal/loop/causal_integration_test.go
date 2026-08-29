package loop

import (
	"encoding/json"
	"testing"

	"overgo/internal/runrecord"
)

// TestRSICausalChainClosure pins the loop's side of the chain: comparison
// candidates carry only immutable IDs, and the exact stored attempt selected
// by that ID still answers to the proposal root that motivated it.
func TestRSICausalChainClosure(t *testing.T) {
	fixture := newStrategyExperimentTestFixture(t)
	bound := fixture.candidates[0]
	_, comparison, err := CompareStrategyExperiment(
		fixture.ctx, fixture.store, fixture.task, fixture.baseline, fixture.candidates, nil,
	)
	if err != nil {
		t.Fatal(err)
	}
	found := false
	for _, evidence := range comparison.Evidence {
		found = found || evidence.Strategy == bound.Strategy && evidence.Attempt == bound.Attempt
	}
	if !found {
		t.Fatalf("comparison omitted the bound attempt: %+v", comparison.Evidence)
	}

	encoded, err := json.Marshal(bound)
	if err != nil {
		t.Fatal(err)
	}
	var decoded StrategyExperimentCandidate
	if err := json.Unmarshal(encoded, &decoded); err != nil {
		t.Fatal(err)
	}
	attempt, err := runrecord.RequireAttemptRecord(fixture.ctx, fixture.store, decoded.Attempt)
	if err != nil {
		t.Fatal(err)
	}
	causal := attempt.Causal
	if causal == nil || causal.Root != fixture.causalRoot.Root || causal.Trigger != runrecord.TriggerControllerProposal ||
		len(causal.Motivation) != 1 || causal.Motivation[0] != fixture.causalRoot.Motivation[0] {
		t.Fatalf("stored candidate lost the causal chain: %+v", causal)
	}
}
