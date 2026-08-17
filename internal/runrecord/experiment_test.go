package runrecord

import (
	"testing"
	"time"

	"overgo/internal/artifact"
	"overgo/internal/testutil"
)

// TestExperimentLifecycleIsIdempotentAndRecoverable pins the runtime
// experiment state machine: legal transitions chain through immutable
// content-addressed records; replaying a transition reproduces the identical
// identity (idempotence); an expired lease or runner reconciles to a
// recoverable status whose recovery is an explicit retry-incremented
// transition carrying the checkpoint; illegal transitions and diverged chains
// are refused, never guessed.
func TestExperimentLifecycleIsIdempotentAndRecoverable(t *testing.T) {
	experiment := testutil.ArtifactID(t, artifact.KindEvidence, "experiment")
	evidence := func(name string) artifact.ID {
		return testutil.ArtifactID(t, artifact.KindEvidence, name)
	}
	checkpoint := testutil.ArtifactID(t, artifact.KindCheckpoint, "checkpoint")
	now := time.Date(2026, 8, 17, 0, 0, 0, 0, time.UTC)
	expiry := func(d time.Duration) string { return now.Add(d).Format(time.RFC3339Nano) }

	proposed, err := NewExperimentLifecycle(ExperimentLifecycle{
		State: ExperimentProposed, Experiment: experiment, Evidence: evidence("proposal"),
	}, nil)
	if err != nil {
		t.Fatal(err)
	}
	replayed, err := NewExperimentLifecycle(ExperimentLifecycle{
		State: ExperimentProposed, Experiment: experiment, Evidence: evidence("proposal"),
	}, nil)
	if err != nil || replayed.ID != proposed.ID {
		t.Fatalf("replayed proposal = (%v, %v), want identical identity %v", replayed.ID, err, proposed.ID)
	}
	admitted, err := NewExperimentLifecycle(ExperimentLifecycle{
		State: ExperimentAdmitted, Experiment: experiment, Evidence: evidence("admission"),
	}, &proposed)
	if err != nil {
		t.Fatal(err)
	}
	leased, err := NewExperimentLifecycle(ExperimentLifecycle{
		State: ExperimentLeased, Experiment: experiment, Evidence: evidence("lease"),
		HeartbeatExpiry: expiry(-2 * time.Minute),
	}, &admitted)
	if err != nil {
		t.Fatal(err)
	}

	status, err := ReconcileExperiment([]ExperimentLifecycle{proposed, admitted, leased, leased}, now)
	if err != nil {
		t.Fatal(err)
	}
	if status.Tip.ID != leased.ID || !status.Expired || status.NextRetry != 1 {
		t.Fatalf("expired lease reconciles to %+v, want recoverable at retry 1", status)
	}

	recovered, err := NewExperimentLifecycle(ExperimentLifecycle{
		State: ExperimentLeased, Experiment: experiment, Evidence: evidence("lease-retry"),
		Retry: 1, HeartbeatExpiry: expiry(time.Hour), Checkpoint: &checkpoint,
	}, &leased)
	if err != nil {
		t.Fatal(err)
	}
	running, err := NewExperimentLifecycle(ExperimentLifecycle{
		State: ExperimentRunning, Experiment: experiment, Evidence: evidence("run"),
		Retry: 1, HeartbeatExpiry: expiry(time.Hour), Checkpoint: &checkpoint,
	}, &recovered)
	if err != nil {
		t.Fatal(err)
	}
	evaluated, err := NewExperimentLifecycle(ExperimentLifecycle{
		State: ExperimentEvaluated, Experiment: experiment, Evidence: evidence("verdict"), Retry: 1,
	}, &running)
	if err != nil {
		t.Fatal(err)
	}
	refused, err := NewExperimentLifecycle(ExperimentLifecycle{
		State: ExperimentRefused, Experiment: experiment, Evidence: evidence("refusal"), Retry: 1,
	}, &evaluated)
	if err != nil {
		t.Fatal(err)
	}
	chain := []ExperimentLifecycle{proposed, admitted, leased, recovered, running, evaluated, refused}
	status, err = ReconcileExperiment(chain, now)
	if err != nil {
		t.Fatal(err)
	}
	if status.Tip.ID != refused.ID || status.Expired {
		t.Fatalf("terminal chain reconciles to %+v, want refused tip", status)
	}
	if status.Tip.Retry != 1 || running.Checkpoint == nil || *running.Checkpoint != checkpoint {
		t.Fatal("recovery lost the retry identity or checkpoint")
	}
	roundtrip, err := refused.Content()
	if err != nil {
		t.Fatal(err)
	}
	parsed, err := ParseExperimentLifecycle(roundtrip.Data)
	if err != nil || parsed.ID != refused.ID || parsed.Prior == nil || *parsed.Prior != evaluated.ID {
		t.Fatalf("lifecycle roundtrip = (%+v, %v)", parsed, err)
	}

	if _, err := NewExperimentLifecycle(ExperimentLifecycle{
		State: ExperimentRunning, Experiment: experiment, Evidence: evidence("skip"),
		HeartbeatExpiry: expiry(time.Hour),
	}, &proposed); err == nil {
		t.Fatal("proposed -> running accepted")
	}
	if _, err := NewExperimentLifecycle(ExperimentLifecycle{
		State: ExperimentPromoted, Experiment: experiment, Evidence: evidence("undead"), Retry: 1,
	}, &refused); err == nil {
		t.Fatal("refused -> promoted accepted")
	}
	if _, err := NewExperimentLifecycle(ExperimentLifecycle{
		State: ExperimentLeased, Experiment: experiment, Evidence: evidence("same-retry"),
		Retry: leased.Retry, HeartbeatExpiry: expiry(time.Hour),
	}, &leased); err == nil {
		t.Fatal("recovery without retry increment accepted")
	}
	if _, err := NewExperimentLifecycle(ExperimentLifecycle{
		State: ExperimentLeased, Experiment: experiment, Evidence: evidence("no-heartbeat"), Retry: 1,
	}, &admitted); err == nil {
		t.Fatal("lease without heartbeat expiry accepted")
	}

	fork, err := NewExperimentLifecycle(ExperimentLifecycle{
		State: ExperimentContained, Experiment: experiment, Evidence: evidence("containment"), Retry: 1,
	}, &running)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := ReconcileExperiment(append(chain, fork), now); err == nil {
		t.Fatal("diverged chain reconciled without error")
	}
	contained := []ExperimentLifecycle{proposed, admitted, leased, recovered, running, fork}
	status, err = ReconcileExperiment(contained, now)
	if err != nil || status.Tip.ID != fork.ID {
		t.Fatalf("containment chain = (%+v, %v)", status, err)
	}
	rolledBack, err := NewExperimentLifecycle(ExperimentLifecycle{
		State: ExperimentRolledBack, Experiment: experiment, Evidence: evidence("rollback"), Retry: 1,
	}, &fork)
	if err != nil {
		t.Fatal(err)
	}
	status, err = ReconcileExperiment(append(contained, rolledBack), now)
	if err != nil || status.Tip.State != ExperimentRolledBack {
		t.Fatalf("rollback chain = (%+v, %v)", status, err)
	}
}
