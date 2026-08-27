package runrecord

import (
	"overgo/internal/artifact"
	"overgo/internal/testutil"
	"testing"
)

func TestAttemptHistory(t *testing.T) {
	strategy := testutil.ArtifactID(t, artifact.KindProfile, "strategy")
	trajectory := testutil.ArtifactID(t, artifact.KindEvidence, "trajectory")
	first := fixtureAttempt(t)
	first.Strategy = strategy
	first.Trajectory = trajectory
	first.CostUnits = 5
	first.Recovered = true
	first, err := NewAttemptRecord(first)
	if err != nil {
		t.Fatal(err)
	}
	second := fixtureAttempt(t)
	second.Strategy = strategy
	second.Outcome = OutcomeFailed
	second.Failure = "tests"
	second.CostUnits = 3
	second, err = NewAttemptRecord(second)
	if err != nil {
		t.Fatal(err)
	}
	history := AttemptHistory([]AttemptRecord{second, first}, AttemptHistoryFilter{Strategy: strategy})
	if len(history) != 1 || history[0].Attempts != 2 || history[0].Succeeded != 1 || history[0].CostUnits != 8 || history[0].Recoveries != 1 || history[0].Trajectories[0] != trajectory {
		t.Fatalf("history = %+v", history)
	}
}
