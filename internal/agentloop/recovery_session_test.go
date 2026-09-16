package agentloop

import (
	"encoding/json"
	"testing"
	"time"

	"overgo/internal/agenttool"
	"overgo/internal/artifact"
	"overgo/internal/recipe"
	"overgo/internal/runrecord"
	"overgo/internal/testutil"
	"overgo/internal/worklease"
)

func TestAgentSessionRecovery(t *testing.T) {
	ctx := t.Context()
	coordinator, store := coordinatorFixture(t)
	manual, err := agenttool.ResolveRegisteredManual(ctx, store, "probe.read")
	if err != nil {
		t.Fatal(err)
	}
	mutation, err := agenttool.ResolveRegisteredManual(ctx, store, "probe.write")
	if err != nil {
		t.Fatal(err)
	}
	snapshot, err := agenttool.NewCatalogSnapshot([]agenttool.Manual{manual, mutation})
	if err != nil {
		t.Fatal(err)
	}
	content, _ := snapshot.ArtifactContent()
	batch, _ := artifact.NewDocumentBatch("recovery-catalog", []artifact.Content{content}, nil, []artifact.AliasBinding{{Name: agenttool.ActiveCatalogAlias, Target: snapshot.ID}})
	if _, err := artifact.CommitBatch(ctx, store, batch); err != nil {
		t.Fatal(err)
	}
	policy := testutil.ArtifactID(t, artifact.KindProfile, "policy")
	agent, err := recipe.NewAgentDefinition(recipe.AgentDefinition{Name: "recovery-agent", Prompt: testutil.ArtifactID(t, artifact.KindFile, "prompt"), ModelRecipe: coordinator.identity.Recipe, ToolManuals: []artifact.ID{manual.ID, mutation.ID}, Policies: []artifact.ID{policy}})
	if err != nil {
		t.Fatal(err)
	}
	task, err := recipe.NewAgentTaskContract(recipe.AgentTaskContract{Agent: agent.ID, Objective: "Resume exact work.", Scope: []string{"internal/agentloop"}, AllowedEffects: []string{"inspection"},
		Acceptance: []recipe.AcceptanceCriterion{{Name: "tests", Scope: "internal/agentloop", Verifier: testutil.ArtifactID(t, artifact.KindRecipe, "verify")}}, Verification: []artifact.ID{testutil.ArtifactID(t, artifact.KindEvidence, "gate")}, Budget: testutil.ArtifactID(t, artifact.KindEvidence, "budget"), PauseConditions: []string{"authority mismatch"}})
	if err != nil {
		t.Fatal(err)
	}
	obligation, _ := runrecord.NewAgentObligation(runrecord.AgentObligation{Task: task.ID, Name: "tests", Scope: "internal/agentloop", Sources: []artifact.ID{task.ID}})
	resolution, _ := runrecord.NewAgentObligationResolution(runrecord.AgentObligationResolution{Obligation: obligation.ID, Scope: obligation.Scope, Evidence: []artifact.ID{testutil.ArtifactID(t, artifact.KindEvidence, "pass")}})
	leaseData, _ := json.Marshal(worklease.Lease{Version: 1, Task: "recovery/session", Worktree: "C:/repo/recovery", Branch: "codex/recovery", Role: "developer", TargetHead: "0123456789abcdef0123456789abcdef01234567", ConflictsWith: []string{}, Resources: worklease.Resources{CPUThreads: 1, HostRAMGiB: 1}, ExpiresAt: time.Now().Add(time.Hour).UTC().Format(time.RFC3339Nano)})
	lease, err := worklease.Record(ctx, store, leaseData)
	if err != nil {
		t.Fatal(err)
	}
	session := &Session{ID: "recover-session"}
	if _, err := coordinator.Propose(ctx, session, manual.Name, json.RawMessage(`{}`)); err != nil {
		t.Fatal(err)
	}
	recovered, err := coordinator.RecoverAgentSession(ctx, session.ID, SessionRecoveryAuthority{Task: task, Agent: agent, Model: coordinator.identity.Model, Catalog: snapshot.ID, Policies: []artifact.ID{policy}, Lease: &lease, Obligations: []runrecord.AgentObligation{obligation}, Resolutions: []runrecord.AgentObligationResolution{resolution}})
	if err != nil {
		t.Fatal(err)
	}
	if recovered.Session.Steps != session.Steps || recovered.NextAction != "recover-session-step-2" || len(recovered.Outstanding) != 0 {
		t.Fatalf("recovery = %+v", recovered)
	}
	bad := policy
	bad, _ = artifact.IdentifyBytes(artifact.KindProfile, []byte("other"))
	if _, err := coordinator.RecoverAgentSession(ctx, session.ID, SessionRecoveryAuthority{Task: task, Agent: agent, Model: coordinator.identity.Model, Catalog: snapshot.ID, Policies: []artifact.ID{bad}, Lease: &lease}); err == nil {
		t.Fatal("policy drift was accepted")
	}
}
