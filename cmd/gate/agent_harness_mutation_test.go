package main

import (
	"context"
	"encoding/json"
	"testing"

	"overgo/internal/agentloop"
	"overgo/internal/agenttool"
	"overgo/internal/artifact"
	"overgo/internal/automationcheck"
	"overgo/internal/overgodb"
	"overgo/internal/plan"
	"overgo/internal/recipe"
	"overgo/internal/runrecord"
	"overgo/internal/testutil"
)

func TestAgentHarnessMutations(t *testing.T) {
	cases := []struct {
		name, check, invariant string
		caught                 func(*testing.T) bool
	}{
		{"effect underclassification", "agenttool-effect", "unresolved executable writer is opaque", mutationEffectUnderclassification},
		{"stale obligation", "agent-contract", "later mutation epoch invalidates evidence", mutationStaleObligation},
		{"proxy bypass", "capability-proxy", "inactive manual cannot dispatch", mutationProxyBypass},
		{"grant widening", "delegation", "read-only grant cannot contain mutation", mutationGrantWidening},
		{"claim overlap", "workspace-claims", "nested writer scopes conflict", mutationClaimOverlap},
		{"checkpoint gap", "agent-checkpoint", "incomplete capture cannot restore", mutationCheckpointGap},
		{"resume drift", "session-recovery", "worker mismatch fails closed", mutationSessionDrift},
		{"trajectory omission", "trajectory", "task authority cannot be omitted", mutationTrajectoryOmission},
		{"stale policy", "policy-lifecycle", "active state cannot bypass lifecycle", mutationPolicyActivation},
		{"wrong exclusion", "gate-impact", "unknown impact cannot exclude checks", mutationGateExclusion},
	}
	for _, fixture := range cases {
		t.Run(fixture.name, func(t *testing.T) {
			if !fixture.caught(t) {
				t.Fatalf("check %s missed injected defect: %s", fixture.check, fixture.invariant)
			}
		})
	}
}

func mutationEffectUnderclassification(t *testing.T) bool {
	manual, err := agenttool.NewManual(agenttool.Manual{Name: "host.write", Description: "Run an unconstrained writer.", Effect: agenttool.EffectMutation, Transport: agenttool.Transport{Kind: agenttool.TransportArgv, Program: "git"}})
	if err != nil {
		t.Fatal(err)
	}
	effect, err := agenttool.DeriveInvocationEffect(manual, json.RawMessage(`{}`), nil)
	return err == nil && effect.OpaqueMutation && !effect.Known
}
func mutationStaleObligation(t *testing.T) bool {
	obligation, _ := runrecord.NewAgentObligation(runrecord.AgentObligation{Task: testutil.ArtifactID(t, artifact.KindRecipe, "task"), Name: "tests", Scope: "internal", MutationEpoch: 2, Sources: []artifact.ID{testutil.ArtifactID(t, artifact.KindEvidence, "source")}})
	resolution, _ := runrecord.NewAgentObligationResolution(runrecord.AgentObligationResolution{Obligation: obligation.ID, Scope: obligation.Scope, MutationEpoch: 1, Evidence: []artifact.ID{testutil.ArtifactID(t, artifact.KindEvidence, "pass")}})
	return !runrecord.AgentObligationSatisfied(obligation, []runrecord.AgentObligationResolution{resolution})
}
func mutationProxyBypass(t *testing.T) bool {
	ctx := context.Background()
	store, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	allowed, _ := agenttool.NewManual(agenttool.Manual{Name: "allowed.read", Description: "Allowed.", Effect: agenttool.EffectInspection, Transport: agenttool.Transport{Kind: agenttool.TransportBuiltin}})
	excluded, _ := agenttool.NewManual(agenttool.Manual{Name: "excluded.read", Description: "Excluded.", Effect: agenttool.EffectInspection, Transport: agenttool.Transport{Kind: agenttool.TransportBuiltin}})
	if _, err := agenttool.PublishManualCatalog(ctx, store, []agenttool.Manual{allowed, excluded}); err != nil {
		t.Fatal(err)
	}
	snapshot, _ := agenttool.NewCatalogSnapshot([]agenttool.Manual{allowed})
	content, _ := snapshot.ArtifactContent()
	batch, _ := artifact.NewDocumentBatch("mutation-proxy", []artifact.Content{content}, nil, []artifact.AliasBinding{{Name: agenttool.ActiveCatalogAlias, Target: snapshot.ID}})
	if _, err := artifact.CommitBatch(ctx, store, batch); err != nil {
		t.Fatal(err)
	}
	_, err = agenttool.ResolveRegisteredManual(ctx, store, excluded.Name)
	return err != nil
}
func mutationGrantWidening(t *testing.T) bool {
	_, err := recipe.NewDelegatedAgentInvocation(recipe.DelegatedAgentInvocation{Task: testutil.ArtifactID(t, artifact.KindRecipe, "task"), Worker: testutil.ArtifactID(t, artifact.KindRecipe, "worker"), Grant: recipe.AgentCapabilityGrant{Manuals: []artifact.ID{testutil.ArtifactID(t, artifact.KindRecipe, "manual")}, AllowedEffects: []string{"mutation"}, ReadOnly: true}, Scheduler: recipe.AgentSchedulerPolicy{MaxSteps: 1, MaxParallel: 1}, Catalog: testutil.ArtifactID(t, artifact.KindProfile, "catalog"), Model: testutil.ArtifactID(t, artifact.KindModel, "model")})
	return err != nil
}
func mutationClaimOverlap(t *testing.T) bool {
	left := plan.WorkLease{Worktree: "C:/repo", Claims: plan.WorkspaceClaims{Write: []string{"C:/repo/internal"}}}
	right := plan.WorkLease{Worktree: "C:/repo", Claims: plan.WorkspaceClaims{Read: []string{"C:/repo/internal/agentloop"}}}
	return plan.WorkspaceClaimsConflict(left, right)
}
func mutationCheckpointGap(t *testing.T) bool {
	checkpoint, err := runrecord.NewAgentMutationCheckpoint(runrecord.AgentMutationCheckpoint{Operation: testutil.ArtifactID(t, artifact.KindEvidence, "operation"), Entries: []runrecord.AgentCheckpointEntry{{Target: "missing", Mode: "unknown", Encoding: "none", Gap: "unreadable"}}})
	if err != nil {
		t.Fatal(err)
	}
	store, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	runtime, err := agentloop.NewMutationCheckpointRuntime(t.TempDir(), store)
	if err != nil {
		t.Fatal(err)
	}
	_, err = runtime.RestoreCheckpoint(context.Background(), checkpoint, agenttool.InvocationEffect{}, testutil.ArtifactID(t, artifact.KindRecipe, "recipe"))
	return checkpoint.Gaps != 0 && err != nil
}
func mutationSessionDrift(t *testing.T) bool {
	ctx := context.Background()
	store, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	recipeID := testutil.ArtifactID(t, artifact.KindRecipe, "recipe")
	model := testutil.ArtifactID(t, artifact.KindModel, "model")
	coordinator, err := agentloop.New(store, agenttool.NewOperatorExecutor(), agentloop.Identity{Recipe: recipeID, Model: model, Node: "agent"}, 2)
	if err != nil {
		t.Fatal(err)
	}
	policy := testutil.ArtifactID(t, artifact.KindProfile, "policy")
	worker, _ := recipe.NewAgentDefinition(recipe.AgentDefinition{Name: "worker", Prompt: testutil.ArtifactID(t, artifact.KindFile, "prompt"), ModelRecipe: recipeID, Policies: []artifact.ID{policy}})
	other, _ := recipe.NewAgentDefinition(recipe.AgentDefinition{Name: "other", Prompt: testutil.ArtifactID(t, artifact.KindFile, "other-prompt"), ModelRecipe: recipeID, Policies: []artifact.ID{policy}})
	task, _ := recipe.NewAgentTaskContract(recipe.AgentTaskContract{Agent: worker.ID, Objective: "Recover.", Scope: []string{"internal"}, AllowedEffects: []string{"inspection"}, Acceptance: []recipe.AcceptanceCriterion{{Name: "tests", Scope: "internal", Verifier: testutil.ArtifactID(t, artifact.KindRecipe, "verify")}}, Verification: []artifact.ID{testutil.ArtifactID(t, artifact.KindEvidence, "gate")}, Budget: testutil.ArtifactID(t, artifact.KindEvidence, "budget"), PauseConditions: []string{"drift"}})
	_, err = coordinator.RecoverAgentSession(ctx, "session", agentloop.SessionRecoveryAuthority{Task: task, Agent: other, Model: model, Catalog: testutil.ArtifactID(t, artifact.KindProfile, "catalog"), Policies: []artifact.ID{policy}})
	return err != nil
}
func mutationTrajectoryOmission(t *testing.T) bool {
	base, err := runrecord.NewInteractionTrace(runrecord.Interaction{Recipe: testutil.ArtifactID(t, artifact.KindRecipe, "recipe"), Model: testutil.ArtifactID(t, artifact.KindModel, "model")}, testutil.ArtifactID(t, artifact.KindEvidence, "request"), []runrecord.InteractionMessage{{Role: "user", Content: "task"}}, nil)
	if err != nil {
		t.Fatal(err)
	}
	base.Terminal = runrecord.OutcomeSucceeded
	_, err = runrecord.NewAgentTrajectory(base)
	return err != nil
}
func mutationPolicyActivation(t *testing.T) bool {
	store, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	value := runrecord.AutomationPolicyLifecycle{Name: "policy", Policy: testutil.ArtifactID(t, artifact.KindProfile, "candidate"), State: runrecord.AutomationPolicyActive, Incumbent: testutil.ArtifactID(t, artifact.KindProfile, "incumbent"), Rollback: testutil.ArtifactID(t, artifact.KindProfile, "incumbent"), EvaluationPlan: testutil.ArtifactID(t, artifact.KindProfile, "plan"), EvaluationEvidence: testutil.ArtifactID(t, artifact.KindEvidence, "evaluation"), Trajectories: []artifact.ID{testutil.ArtifactID(t, artifact.KindEvidence, "trajectory")}, Decision: testutil.ArtifactID(t, artifact.KindEvidence, "decision")}
	_, err = runrecord.PublishAutomationPolicyTransition(context.Background(), store, value)
	return err != nil
}
func mutationGateExclusion(t *testing.T) bool {
	check := bootstrapCheck("agent-contract", "owner:agent-contract")
	impact := automationcheck.OwnershipImpact([]automationcheck.Check{check}, automationcheck.Surface{Identity: "candidate", Unknown: []string{"analysis failed"}})
	planned, err := automationcheck.Plan([]automationcheck.Check{check}, impact)
	return err == nil && len(planned) == 1 && len(impact.Exclusions) == 0
}
