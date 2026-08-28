package agentloop

import (
	"testing"

	"overgo/internal/executionfailure"
)

// TestRetryAndSessionDecisionMatrix pins the coordinator bridge into
// the failure decision matrix: the step budget is the context signal,
// untrusted recovery history poisons the session, and the published
// disposition names the rule and the situation the coordinator read.
func TestRetryAndSessionDecisionMatrix(t *testing.T) {
	coordinator, _ := coordinatorFixture(t)
	transient := executionfailure.Normalize(executionfailure.Evidence{
		Message: "dial tcp 127.0.0.1:9| connection refused"})
	if transient.Cause != executionfailure.CauseNetwork {
		t.Fatalf("fixture classification = %+v", transient)
	}
	local := executionfailure.Normalize(executionfailure.Evidence{
		Message: "flag provided but not defined: -modl"})

	healthy := &Session{ID: "policy-healthy", Steps: fixtureSessionSteps - 1}
	full := &Session{ID: "policy-full", Steps: fixtureSessionSteps}

	matrix := []struct {
		name         string
		session      *Session
		class        executionfailure.Classification
		interrupted  bool
		attempts     uint32
		trustHistory bool
		decision     executionfailure.Decision
	}{
		{"transient in a healthy session retries", healthy, transient, false, 1, true, executionfailure.DecisionRetry},
		{"transient past budget refuses", healthy, transient, false, 3, true, executionfailure.DecisionRefuse},
		{"local deterministic failure refuses immediately", healthy, local, false, 0, true, executionfailure.DecisionRefuse},
		{"interrupted healthy session resumes", healthy, transient, true, 1, true, executionfailure.DecisionResume},
		{"step budget spent rolls to a fresh session", full, transient, false, 1, true, executionfailure.DecisionFreshSession},
		{"step budget spent past attempts retires", full, transient, false, 3, true, executionfailure.DecisionRetireSession},
		{"untrusted recovery history retires", healthy, transient, false, 1, false, executionfailure.DecisionRetireSession},
	}
	for _, sample := range matrix {
		disposition := coordinator.SessionDisposition(
			sample.session, sample.class, sample.interrupted, sample.attempts, 3, sample.trustHistory)
		if disposition.Decision != sample.decision {
			t.Fatalf("%s: disposition = %+v", sample.name, disposition)
		}
		if disposition.Rule == "" || disposition.Situation.Cause != sample.class.Cause {
			t.Fatalf("%s: disposition evidence incomplete: %+v", sample.name, disposition)
		}
		wantHealth := executionfailure.SessionHealth{
			ContextFull: sample.session.Steps >= fixtureSessionSteps, HistoryPoisoned: !sample.trustHistory,
		}
		if disposition.Situation.Health != wantHealth {
			t.Fatalf("%s: derived health = %+v", sample.name, disposition.Situation.Health)
		}
	}
	if health := (*Coordinator)(nil).SessionHealth(healthy, true); !health.HistoryPoisoned {
		t.Fatal("absent coordinator must not report a trustworthy session")
	}
}
