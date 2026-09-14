package runrecord

import (
	"strings"
	"testing"
	"time"

	"overgo/internal/artifact"
	"overgo/internal/testutil"
)

// TestGateCheckpointOutcome holds the checkpoint publication outcome: it
// ends like a success without a commit, refuses a failure code and a
// terminal step, and finalizes a prepared lifecycle.
func TestGateCheckpointOutcome(t *testing.T) {
	t.Parallel()
	recipe := testutil.ArtifactID(t, artifact.KindRecipe, "checkpoint gate recipe")
	environment := testutil.ArtifactID(t, artifact.KindEvidence, "checkpoint gate environment")
	commit := strings.Repeat("c", 40)
	steps := []GateStep{
		{Name: "build", Phase: PhaseBuild, Outcome: StepSucceeded, DurationNS: 1},
		{Name: "acceptance-producer", Phase: PhaseTest, Outcome: StepReused, DurationNS: 1},
	}
	record, err := NewGateRecord(recipe, environment, commit, OutcomeCheckpoint, "", 2, steps)
	if err != nil {
		t.Fatal(err)
	}
	if record.Result.Outcome != OutcomeCheckpoint || record.Run.Outcome != OutcomeCheckpoint {
		t.Fatalf("checkpoint outcome lost: %+v", record.Result)
	}
	if _, err := NewGateRecord(recipe, environment, commit, OutcomeCheckpoint, "planning", 2, steps); err == nil {
		t.Fatal("checkpoint gate accepted a failure code")
	}
	failed := []GateStep{steps[0], {Name: "acceptance-consumer", Phase: PhaseTest, Outcome: StepFailed, DurationNS: 1}}
	if _, err := NewGateRecord(recipe, environment, commit, OutcomeCheckpoint, "", 2, failed); err == nil {
		t.Fatal("checkpoint gate accepted a failed step")
	}
	preparation, err := NewGatePreparation(strings.Repeat("a", 64), environment, time.Unix(100, 0))
	if err != nil {
		t.Fatal(err)
	}
	finalized, err := NewGateFinalization(preparation, commit, record.Result.ID, OutcomeCheckpoint)
	if err != nil {
		t.Fatal(err)
	}
	if finalized.State != GateFinalized || finalized.Outcome != OutcomeCheckpoint {
		t.Fatalf("checkpoint finalization: %+v", finalized)
	}
}
