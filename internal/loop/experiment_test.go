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
	task, baseline, strategyA, strategyB, candidate := strategyExperimentFixture(t)
	experiment, comparison, err := CompareStrategyExperiment(task, baseline, []StrategyExperimentCandidate{candidate("a", strategyA, true, 2), candidate("b", strategyB, false, 1)})
	if err != nil {
		t.Fatal(err)
	}
	if !experiment.ID.Valid() || comparison.Winner == nil || *comparison.Winner != strategyA.ID || len(comparison.Counterexamples) != 1 {
		t.Fatalf("comparison = %+v", comparison)
	}
}

func TestStrategyComparisonRequiresIdentity(t *testing.T) {
	task, baseline, strategyA, strategyB, candidate := strategyExperimentFixture(t)
	validA, validB := candidate("a", strategyA, true, 2), candidate("b", strategyB, false, 1)
	_, comparison, err := CompareStrategyExperiment(task, baseline, []StrategyExperimentCandidate{validA, validB})
	if err != nil {
		t.Fatalf("exact strategies refused: %v", err)
	}
	if len(comparison.Evidence) != 2 || comparison.Evidence[0].Strategy != strategyA.ID ||
		comparison.Evidence[0].Attempt != validA.Attempt.ID || comparison.Evidence[0].Trajectory != validA.Trajectory ||
		comparison.Evidence[0].EvaluationPlan != validA.EvaluationPlan || comparison.Evidence[0].EvaluationEvidence != validA.EvaluationEvidence {
		t.Fatalf("strategy evidence bindings = %+v", comparison.Evidence)
	}

	recordable := func(base StrategyExperimentCandidate, label string, strategy artifact.ID) StrategyExperimentCandidate {
		base.Attempt.Strategy, base.Attempt.StrategyID = label, strategy
		var err error
		base.Attempt, err = runrecord.NewAttemptRecord(base.Attempt)
		if err != nil {
			t.Fatalf("recordable attempt refused: %v", err)
		}
		return base
	}
	for name, invalid := range map[string]StrategyExperimentCandidate{
		"anonymous":  recordable(validA, "", artifact.ID{}),
		"label-only": recordable(validA, "display-label", artifact.ID{}),
		"mismatched": recordable(validA, "", strategyB.ID),
	} {
		if _, _, err := CompareStrategyExperiment(task, baseline, []StrategyExperimentCandidate{invalid, validB}); err == nil {
			t.Fatalf("%s attempt entered comparison", name)
		}
	}

	forged := validA
	forged.Strategy = strategyB
	forged.Attempt.StrategyID = strategyB.ID
	if _, _, err := CompareStrategyExperiment(task, baseline, []StrategyExperimentCandidate{forged, candidate("c", strategyA, true, 3)}); err == nil {
		t.Fatal("post-identification strategy mutation entered comparison")
	}

	duplicate := candidate("d", strategyA, false, 1)
	if _, _, err := CompareStrategyExperiment(task, baseline, []StrategyExperimentCandidate{validA, duplicate}); err == nil {
		t.Fatal("one strategy was admitted as a cross-strategy comparison")
	}
	detached := validA
	detached.Trajectory = testutil.ArtifactID(t, artifact.KindEvidence, "detached-trajectory")
	if _, _, err := CompareStrategyExperiment(task, baseline, []StrategyExperimentCandidate{detached, validB}); err == nil {
		t.Fatal("trajectory outside the strategy attempt entered comparison")
	}
}

func strategyExperimentFixture(t *testing.T) (recipe.AgentTaskContract, string, Strategy, Strategy, func(string, Strategy, bool, uint64) StrategyExperimentCandidate) {
	t.Helper()
	worker, _ := recipe.NewAgentDefinition(recipe.AgentDefinition{Name: "experiment-worker", Prompt: testutil.ArtifactID(t, artifact.KindFile, "prompt"), ModelRecipe: testutil.ArtifactID(t, artifact.KindRecipe, "model"), Policies: []artifact.ID{testutil.ArtifactID(t, artifact.KindProfile, "policy")}})
	task, _ := recipe.NewAgentTaskContract(recipe.AgentTaskContract{Agent: worker.ID, Objective: "Compare strategies.", Scope: []string{"internal/loop"}, AllowedEffects: []string{"workspace:mutation"}, Acceptance: []recipe.AcceptanceCriterion{{Name: "tests", Scope: "internal/loop", Verifier: testutil.ArtifactID(t, artifact.KindRecipe, "verify")}}, Verification: []artifact.ID{testutil.ArtifactID(t, artifact.KindEvidence, "gate")}, Budget: testutil.ArtifactID(t, artifact.KindEvidence, "budget"), PauseConditions: []string{"no evidence"}})
	baseline := strings.Repeat("a", 40)
	strategyA, _ := NewStrategy(worker, testutil.ArtifactID(t, artifact.KindProfile, "catalog"), Config{MaxAttemptsPerStep: 2, MaxInvocations: 5})
	strategyB := strategyA
	strategyB.Loop.MaxInvocations++
	strategyB, _ = strategyCodec.New(strategyB)
	candidate := func(name string, strategy Strategy, success bool, cost uint64) StrategyExperimentCandidate {
		lease, _ := plan.NewWorkLease(plan.WorkLease{Task: name, Worktree: "C:/repo/" + name, Branch: "codex/" + name, Role: "experiment", TargetHead: baseline, ConflictsWith: []string{}, Resources: plan.Resources{CPUThreads: 1, HostRAMGiB: 1}, ExpiresAt: time.Now().Add(time.Hour).UTC().Format(time.RFC3339Nano)})
		trajectory := testutil.ArtifactID(t, artifact.KindEvidence, "trajectory-"+name)
		attempt, _ := runrecord.NewAttemptRecord(runrecord.AttemptRecord{PlanItem: "experiment", PlanStep: name, Result: testutil.ArtifactID(t, artifact.KindEvidence, "result-"+name), Recipe: testutil.ArtifactID(t, artifact.KindRecipe, "recipe"), CodeCommit: baseline, Outcome: runrecord.OutcomeSucceeded, WallNS: 1, StrategyID: strategy.ID, Trajectory: trajectory, CostUnits: cost})
		return StrategyExperimentCandidate{Strategy: strategy, Lease: lease, Attempt: attempt, Trajectory: trajectory, EvaluationPlan: testutil.ArtifactID(t, artifact.KindProfile, "plan-"+name), EvaluationEvidence: testutil.ArtifactID(t, artifact.KindEvidence, "evaluation-"+name), Hard: StrategyHardOutcome{TaskSucceeded: success, EvidenceComplete: success, GatePassed: success}}
	}
	return task, baseline, strategyA, strategyB, candidate
}
