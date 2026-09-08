package gate

import (
	"strings"
	"testing"
	"time"

	"overgo/internal/artifact"
	"overgo/internal/overgodb"
	"overgo/internal/runrecord"
	"overgo/internal/testutil"
)

func costSteps(producer, consumer runrecord.StepOutcome, producerNS, consumerNS uint64) []runrecord.GateStep {
	return []runrecord.GateStep{
		{Name: "protection", Phase: runrecord.PhasePackage, Outcome: runrecord.StepSucceeded, DurationNS: 9},
		{Name: "acceptance-producer", Phase: runrecord.PhaseTest, Outcome: producer, DurationNS: producerNS},
		{Name: "acceptance-consumer", Phase: runrecord.PhaseTest, Outcome: consumer, DurationNS: consumerNS},
		{Name: "acceptance", Phase: runrecord.PhaseTest, Outcome: runrecord.StepSucceeded, DurationNS: 7},
	}
}

func commitGateAttempt(t *testing.T, store *overgodb.Store, item, step string, steps []runrecord.GateStep) {
	t.Helper()
	record, err := runrecord.NewGateRecord(
		testutil.ArtifactID(t, artifact.KindRecipe, "cost gate recipe"),
		testutil.ArtifactID(t, artifact.KindEvidence, "cost environment"),
		strings.Repeat("ab", 20), runrecord.OutcomeSucceeded, "", 100, steps,
	)
	if err != nil {
		t.Fatal(err)
	}
	batch, err := record.Batch("cost/gate/" + record.Result.ID.String())
	if err != nil {
		t.Fatal(err)
	}
	attempt, err := runrecord.NewAttemptRecord(runrecord.AttemptRecord{
		PlanItem: item, PlanStep: step, Result: record.Result.ID, Recipe: record.Result.Recipe,
		CodeCommit: strings.Repeat("ab", 20), Outcome: runrecord.OutcomeSucceeded, WallNS: 100,
		Selection: runrecord.AttemptSelection{Defined: 4, Selected: 4}, Diff: runrecord.AttemptDiff{Files: 1, Insertions: 1},
	})
	if err != nil {
		t.Fatal(err)
	}
	content, err := attempt.Content()
	if err != nil {
		t.Fatal(err)
	}
	// Register the recipe and environment identities the record's lineage
	// names, as the gate's own final record batch does.
	batch.Artifacts = append(batch.Artifacts,
		artifact.Descriptor{ID: record.Result.Recipe}, artifact.Descriptor{ID: record.Result.Environment}, content.Descriptor)
	batch.Contents = append(batch.Contents, content)
	if _, err := store.Commit(t.Context(), batch); err != nil {
		t.Fatal(err)
	}
}

// TestBatchedGateCostEvidence pins: cost from result steps counts accepted,
// reused and executed wall over acceptance steps only; reuse savings take the
// median prior executed duration and name never-executed checkpoints; prior
// costs come from attempts of the same row; the audit line records both.
func TestBatchedGateCostEvidence(t *testing.T) {
	executed := batchCostOf(costSteps(runrecord.StepSucceeded, runrecord.StepSucceeded, 5*uint64(time.Second), 3*uint64(time.Second)))
	if executed.Accepted != 3 || executed.Reused != 0 || executed.ExecutedNS != 8*uint64(time.Second)+7 || len(executed.Checkpoints) != 3 {
		t.Fatalf("executed cost = %+v", executed)
	}
	if executed.Checkpoints[0].Name != "acceptance" || executed.Checkpoints[1].Name != "acceptance-consumer" {
		t.Fatalf("checkpoints are not sorted by name: %+v", executed.Checkpoints)
	}
	reused := batchCostOf(costSteps(runrecord.StepReused, runrecord.StepSucceeded, 1, 4*uint64(time.Second)))
	if reused.Accepted != 3 || reused.Reused != 1 || reused.ExecutedNS != 4*uint64(time.Second)+7 {
		t.Fatalf("reused cost = %+v", reused)
	}
	slower := batchCostOf(costSteps(runrecord.StepSucceeded, runrecord.StepFailed, 9*uint64(time.Second), 1))
	saved, unmeasured := reuseSavings([]batchCost{executed, slower}, reused)
	if saved != 5*uint64(time.Second) || len(unmeasured) != 0 {
		t.Fatalf("savings = %s %v, want the 5s median of 5s and 9s", time.Duration(saved), unmeasured)
	}
	if saved, unmeasured := reuseSavings(nil, reused); saved != 0 || len(unmeasured) != 1 || unmeasured[0] != "acceptance-producer" {
		t.Fatalf("savings without prior = %d %v", saved, unmeasured)
	}
	if saved, unmeasured := reuseSavings([]batchCost{executed}, executed); saved != 0 || unmeasured != nil {
		t.Fatalf("savings without reuse = %d %v", saved, unmeasured)
	}

	store, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	commitGateAttempt(t, store, "flush", "cost", costSteps(runrecord.StepSucceeded, runrecord.StepSucceeded, 5*uint64(time.Second), 3*uint64(time.Second)))
	commitGateAttempt(t, store, "flush", "other", costSteps(runrecord.StepSucceeded, runrecord.StepSucceeded, 1, 1))
	prior, err := priorBatchCosts(t.Context(), store, "flush/cost")
	if err != nil || len(prior) != 1 || prior[0].ExecutedNS != 8*uint64(time.Second)+7 {
		t.Fatalf("prior costs = %+v, %v; want the one attempt of this row", prior, err)
	}
	if _, err := priorBatchCosts(t.Context(), store, "flush"); err == nil {
		t.Fatal("plan reference without a step was accepted")
	}
	g := &gateContext{planRef: "flush/cost"}
	g.batchCostAudit(t.Context(), store, costSteps(runrecord.StepReused, runrecord.StepSucceeded, 1, 4*uint64(time.Second)), uint64(time.Second))
	if len(g.audit) != 1 || !strings.Contains(g.audit[0], "accepted=3 reused=1") || !strings.Contains(g.audit[0], "estimated_step_time_avoided=5s") || !strings.Contains(g.audit[0], "prior_runs=1") {
		t.Fatalf("audit = %v", g.audit)
	}
	g.audit = nil
	g.batchCostAudit(t.Context(), store, []runrecord.GateStep{{Name: "protection", Phase: runrecord.PhasePackage, Outcome: runrecord.StepSucceeded, DurationNS: 1}}, 1)
	if len(g.audit) != 0 {
		t.Fatalf("audit without acceptance steps = %v", g.audit)
	}
}
