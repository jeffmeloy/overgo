package loop

import (
	"encoding/json"
	"strings"
	"testing"
	"time"

	"overgo/internal/artifact"
	"overgo/internal/plan"
	"overgo/internal/recipe"
	"overgo/internal/runrecord"
	"overgo/internal/testutil"
)

// TestRSICausalChainClosure pins the loop's side of the chain: a
// strategy experiment runs over causally bound attempts, and the
// candidate the loop compares and serializes still answers to the
// proposal root that motivated it -- no parallel origin bookkeeping.
func TestRSICausalChainClosure(t *testing.T) {
	proposal := testutil.ArtifactID(t, artifact.KindEvidence, "loop-proposal")
	motivation := testutil.ArtifactID(t, artifact.KindEvidence, "loop-motivation")
	root, err := runrecord.NewCausalRoot(runrecord.TriggerControllerProposal, proposal, motivation)
	if err != nil {
		t.Fatal(err)
	}
	worker, err := recipe.NewAgentDefinition(recipe.AgentDefinition{
		Name: "causal-worker", Prompt: testutil.ArtifactID(t, artifact.KindFile, "prompt"),
		ModelRecipe: testutil.ArtifactID(t, artifact.KindRecipe, "model"),
		Policies:    []artifact.ID{testutil.ArtifactID(t, artifact.KindProfile, "policy")},
	})
	if err != nil {
		t.Fatal(err)
	}
	task, err := recipe.NewAgentTaskContract(recipe.AgentTaskContract{
		Agent: worker.ID, Objective: "Close the causal chain.", Scope: []string{"internal/loop"},
		AllowedEffects: []string{"workspace:mutation"},
		Acceptance: []recipe.AcceptanceCriterion{{Name: "tests", Scope: "internal/loop",
			Verifier: testutil.ArtifactID(t, artifact.KindRecipe, "verify")}},
		Verification:    []artifact.ID{testutil.ArtifactID(t, artifact.KindEvidence, "gate")},
		Budget:          testutil.ArtifactID(t, artifact.KindEvidence, "budget"),
		PauseConditions: []string{"no evidence"},
	})
	if err != nil {
		t.Fatal(err)
	}
	baseline := strings.Repeat("b", 40)
	strategy, err := NewStrategy(worker, testutil.ArtifactID(t, artifact.KindProfile, "catalog"),
		Config{MaxAttemptsPerStep: 2, MaxInvocations: 5})
	if err != nil {
		t.Fatal(err)
	}
	lease, err := plan.NewWorkLease(plan.WorkLease{
		Task: "bound", Worktree: "C:/repo/bound", Branch: "codex/bound", Role: "experiment",
		TargetHead: baseline, ConflictsWith: []string{}, Resources: plan.Resources{CPUThreads: 1, HostRAMGiB: 1},
		ExpiresAt: time.Now().Add(time.Hour).UTC().Format(time.RFC3339Nano),
	})
	if err != nil {
		t.Fatal(err)
	}
	attempt, err := runrecord.NewAttemptRecord(runrecord.AttemptRecord{
		PlanItem: "causal-execution", PlanStep: "bound",
		Result:     testutil.ArtifactID(t, artifact.KindEvidence, "result-bound"),
		Recipe:     testutil.ArtifactID(t, artifact.KindRecipe, "recipe"),
		CodeCommit: baseline, Outcome: runrecord.OutcomeSucceeded, WallNS: 1,
		StrategyID: strategy.ID, CostUnits: 1, Causal: &root,
	})
	if err != nil {
		t.Fatal(err)
	}
	bound := StrategyExperimentCandidate{
		Strategy: strategy, Lease: lease, Attempt: attempt,
		Trajectory:         testutil.ArtifactID(t, artifact.KindEvidence, "trajectory-bound"),
		EvaluationPlan:     testutil.ArtifactID(t, artifact.KindProfile, "plan-bound"),
		EvaluationEvidence: testutil.ArtifactID(t, artifact.KindEvidence, "evaluation-bound"),
		Hard:               StrategyHardOutcome{TaskSucceeded: true, EvidenceComplete: true, GatePassed: true},
	}
	rival := strategy
	rival.Loop.MaxInvocations++
	rival, err = strategyCodec.New(rival)
	if err != nil {
		t.Fatal(err)
	}
	rivalLease, err := plan.NewWorkLease(plan.WorkLease{
		Task: "rival", Worktree: "C:/repo/rival", Branch: "codex/rival", Role: "experiment",
		TargetHead: baseline, ConflictsWith: []string{}, Resources: plan.Resources{CPUThreads: 1, HostRAMGiB: 1},
		ExpiresAt: time.Now().Add(time.Hour).UTC().Format(time.RFC3339Nano),
	})
	if err != nil {
		t.Fatal(err)
	}
	rivalAttempt, err := runrecord.NewAttemptRecord(runrecord.AttemptRecord{
		PlanItem: "causal-execution", PlanStep: "rival",
		Result:     testutil.ArtifactID(t, artifact.KindEvidence, "result-rival"),
		Recipe:     testutil.ArtifactID(t, artifact.KindRecipe, "recipe"),
		CodeCommit: baseline, Outcome: runrecord.OutcomeFailed, Failure: "tests", WallNS: 1,
		StrategyID: rival.ID, CostUnits: 2, Causal: &root,
	})
	if err != nil {
		t.Fatal(err)
	}
	competitor := StrategyExperimentCandidate{
		Strategy: rival, Lease: rivalLease, Attempt: rivalAttempt,
		Trajectory:         testutil.ArtifactID(t, artifact.KindEvidence, "trajectory-rival"),
		EvaluationPlan:     testutil.ArtifactID(t, artifact.KindProfile, "plan-rival"),
		EvaluationEvidence: testutil.ArtifactID(t, artifact.KindEvidence, "evaluation-rival"),
		Hard:               StrategyHardOutcome{},
	}
	if _, _, err := CompareStrategyExperiment(task, baseline, []StrategyExperimentCandidate{bound, competitor}); err != nil {
		t.Fatal(err)
	}

	encoded, err := json.Marshal(bound)
	if err != nil {
		t.Fatal(err)
	}
	var decoded StrategyExperimentCandidate
	if err := json.Unmarshal(encoded, &decoded); err != nil {
		t.Fatal(err)
	}
	causal := decoded.Attempt.Causal
	if causal == nil || causal.Root != proposal || causal.Trigger != runrecord.TriggerControllerProposal ||
		len(causal.Motivation) != 1 || causal.Motivation[0] != motivation {
		t.Fatalf("candidate lost the causal chain: %+v", causal)
	}
}
