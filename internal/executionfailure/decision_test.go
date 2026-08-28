package executionfailure

import "testing"

// TestRetryAndSessionDecisionMatrix pins the deterministic decision
// matrix: every canonical decision is reachable, precedence between
// health, interruption, budget, and cause class holds, and the
// published disposition carries the rule and the exact situation.
func TestRetryAndSessionDecisionMatrix(t *testing.T) {
	budget := uint32(3)
	matrix := []struct {
		name      string
		situation Situation
		decision  Decision
		rule      string
	}{
		{"poisoned history retires over everything",
			Situation{Cause: CauseNetwork, Interrupted: true, Attempts: 1, MaxAttempts: budget,
				Health: SessionHealth{ContextFull: true, HistoryPoisoned: true}},
			DecisionRetireSession, "poisoned-history"},
		{"interrupted healthy session resumes",
			Situation{Cause: CauseUnknown, Interrupted: true, Attempts: 1, MaxAttempts: budget},
			DecisionResume, "interrupted-healthy"},
		{"interrupted full session does not resume",
			Situation{Cause: CauseUnknown, Interrupted: true, Attempts: 1, MaxAttempts: budget,
				Health: SessionHealth{ContextFull: true}},
			DecisionFreshSession, "context-rollover"},
		{"context exhaustion rolls to a fresh session within budget",
			Situation{Cause: CauseContextExhaustion, Attempts: 1, MaxAttempts: budget},
			DecisionFreshSession, "context-rollover"},
		{"context exhaustion retires beyond budget",
			Situation{Cause: CauseContextExhaustion, Attempts: budget, MaxAttempts: budget},
			DecisionRetireSession, "context-budget-spent"},
		{"full session rolls even on a transient cause",
			Situation{Cause: CauseNetwork, Attempts: 1, MaxAttempts: budget,
				Health: SessionHealth{ContextFull: true}},
			DecisionFreshSession, "context-rollover"},
		{"transient retries within budget",
			Situation{Cause: CauseNetwork, Attempts: 2, MaxAttempts: budget},
			DecisionRetry, "transient-within-budget"},
		{"unknown is budget-bounded transient",
			Situation{Cause: CauseUnknown, Attempts: 1, MaxAttempts: budget},
			DecisionRetry, "transient-within-budget"},
		{"transient refuses beyond budget",
			Situation{Cause: CauseTimeout, Attempts: budget, MaxAttempts: budget},
			DecisionRefuse, "budget-exhausted"},
		{"configuration never retries",
			Situation{Cause: CauseConfiguration, Attempts: 0, MaxAttempts: budget},
			DecisionRefuse, "deterministic-local"},
		{"authorization never retries",
			Situation{Cause: CauseAuthorization, Attempts: 0, MaxAttempts: budget},
			DecisionRefuse, "deterministic-local"},
		{"missing executable never retries",
			Situation{Cause: CauseMissingExecutable, Attempts: 0, MaxAttempts: budget},
			DecisionRefuse, "deterministic-local"},
		{"process failure never retries",
			Situation{Cause: CauseProcessFailure, Attempts: 0, MaxAttempts: budget},
			DecisionRefuse, "deterministic-local"},
		{"zero budget refuses transients immediately",
			Situation{Cause: CauseNetwork, Attempts: 0},
			DecisionRefuse, "budget-exhausted"},
	}
	reached := map[Decision]bool{}
	for _, sample := range matrix {
		disposition := Decide(sample.situation)
		if disposition != Decide(sample.situation) {
			t.Fatalf("%s: decision is not deterministic", sample.name)
		}
		if disposition.Decision != sample.decision || disposition.Rule != sample.rule {
			t.Fatalf("%s: disposition = %+v", sample.name, disposition)
		}
		if disposition.Situation != sample.situation {
			t.Fatalf("%s: published situation differs from the one read", sample.name)
		}
		if !ValidDecision(disposition.Decision) {
			t.Fatalf("%s: foreign decision %q", sample.name, disposition.Decision)
		}
		reached[disposition.Decision] = true
	}
	for _, decision := range CanonicalDecisions {
		if !reached[decision] {
			t.Fatalf("matrix never reaches canonical decision %q", decision)
		}
	}
	if ValidDecision(Decision("abandon")) {
		t.Fatal("foreign decision accepted")
	}
}
