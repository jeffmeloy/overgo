package loop

import (
	"strings"
	"testing"
	"time"

	"overgo/internal/artifact"
	"overgo/internal/plan"
	"overgo/internal/recipe"
	"overgo/internal/runrecord"
	"overgo/internal/testutil"
)

func TestStrategyExperiment(t *testing.T) {
	worker, _ := recipe.NewAgentDefinition(recipe.AgentDefinition{Name: "experiment-worker", Prompt: testutil.ArtifactID(t, artifact.KindFile, "prompt"), ModelRecipe: testutil.ArtifactID(t, artifact.KindRecipe, "model"), Policies: []artifact.ID{testutil.ArtifactID(t, artifact.KindProfile, "policy")}})
	task, _ := recipe.NewAgentTaskContract(recipe.AgentTaskContract{Agent: worker.ID, Objective: "Compare strategies.", Scope: []string{"internal/loop"}, AllowedEffects: []string{"workspace:mutation"}, Acceptance: []recipe.AcceptanceCriterion{{Name: "tests", Scope: "internal/loop", Verifier: testutil.ArtifactID(t, artifact.KindRecipe, "verify")}}, Verification: []artifact.ID{testutil.ArtifactID(t, artifact.KindEvidence, "gate")}, Budget: testutil.ArtifactID(t, artifact.KindEvidence, "budget"), PauseConditions: []string{"no evidence"}})
	baseline := strings.Repeat("a", 40)
	strategyA, _ := NewStrategy(worker, testutil.ArtifactID(t, artifact.KindProfile, "catalog"), Config{MaxAttemptsPerStep: 2, MaxInvocations: 5})
	strategyB := strategyA
	strategyB.Loop.MaxInvocations++
	strategyB, _ = strategyCodec.New(strategyB)
	candidate := func(name string, strategy Strategy, success bool, cost uint64) StrategyExperimentCandidate {
		lease, _ := plan.NewWorkLease(plan.WorkLease{Task: name, Worktree: "C:/repo/" + name, Branch: "codex/" + name, Role: "experiment", TargetHead: baseline, ConflictsWith: []string{}, Resources: plan.Resources{CPUThreads: 1, HostRAMGiB: 1}, ExpiresAt: time.Now().Add(time.Hour).UTC().Format(time.RFC3339Nano)})
		attempt, _ := runrecord.NewAttemptRecord(runrecord.AttemptRecord{PlanItem: "experiment", PlanStep: name, Result: testutil.ArtifactID(t, artifact.KindEvidence, "result-"+name), Recipe: testutil.ArtifactID(t, artifact.KindRecipe, "recipe"), CodeCommit: baseline, Outcome: runrecord.OutcomeSucceeded, WallNS: 1, Strategy: strategy.ID, CostUnits: cost})
		return StrategyExperimentCandidate{Strategy: strategy, Lease: lease, Attempt: attempt, Trajectory: testutil.ArtifactID(t, artifact.KindEvidence, "trajectory-"+name), EvaluationPlan: testutil.ArtifactID(t, artifact.KindProfile, "plan-"+name), EvaluationEvidence: testutil.ArtifactID(t, artifact.KindEvidence, "evaluation-"+name), Hard: StrategyHardOutcome{TaskSucceeded: success, EvidenceComplete: success, GatePassed: success}}
	}
	experiment, comparison, err := CompareStrategyExperiment(task, baseline, []StrategyExperimentCandidate{candidate("a", strategyA, true, 2), candidate("b", strategyB, false, 1)})
	if err != nil {
		t.Fatal(err)
	}
	if !experiment.ID.Valid() || comparison.Winner == nil || *comparison.Winner != strategyA.ID || len(comparison.Counterexamples) != 1 {
		t.Fatalf("comparison = %+v", comparison)
	}
}
