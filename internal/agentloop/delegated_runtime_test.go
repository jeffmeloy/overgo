package agentloop

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"overgo/internal/agenttool"
	"overgo/internal/artifact"
	"overgo/internal/overgodb"
	"overgo/internal/recipe"
	"overgo/internal/testutil"
)

// TestDelegatedCapabilityRuntimeConsumesStagedOwners pins the production
// join of the delegation chain: compiled grant, delegated coordinator,
// grant-derived proxy admission, mutation checkpointing, and task-contract
// obligations all wire through one constructor, and the grant policy can
// only narrow -- a read-only grant refuses every non-inspection dispatch.
func TestDelegatedCapabilityRuntimeConsumesStagedOwners(t *testing.T) {
	ctx := t.Context()
	store, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	manual, err := agenttool.NewManual(agenttool.Manual{
		Name: "repo.read", Description: "Read repository state.",
		Effect: agenttool.EffectInspection, Transport: agenttool.Transport{Kind: agenttool.TransportBuiltin},
	})
	if err != nil {
		t.Fatal(err)
	}
	snapshot, err := agenttool.NewCatalogSnapshot([]agenttool.Manual{manual})
	if err != nil {
		t.Fatal(err)
	}
	snapshotContent, err := snapshot.ArtifactContent()
	if err != nil {
		t.Fatal(err)
	}
	if _, err := agenttool.PublishManualCatalog(ctx, store, []agenttool.Manual{manual}); err != nil {
		t.Fatal(err)
	}
	batch, err := artifact.NewDocumentBatch("delegated-runtime-fixture", []artifact.Content{snapshotContent}, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := artifact.CommitBatch(ctx, store, batch); err != nil {
		t.Fatal(err)
	}
	worker, err := recipe.NewAgentDefinition(recipe.AgentDefinition{
		Name: "worker", Prompt: testutil.ArtifactID(t, artifact.KindFile, "prompt"),
		ModelRecipe: testutil.ArtifactID(t, artifact.KindRecipe, "model-recipe"),
		ToolManuals: []artifact.ID{manual.ID},
		Policies:    []artifact.ID{testutil.ArtifactID(t, artifact.KindProfile, "policy")},
	})
	if err != nil {
		t.Fatal(err)
	}
	task, err := recipe.NewAgentTaskContract(recipe.AgentTaskContract{
		Agent: worker.ID, Objective: "Inspect and report.", Scope: []string{"internal"},
		AllowedEffects: []string{"inspection"},
		Acceptance: []recipe.AcceptanceCriterion{
			{Name: "tests", Scope: "internal", Verifier: testutil.ArtifactID(t, artifact.KindRecipe, "verify tests")},
			{Name: "report", Scope: "internal", Verifier: testutil.ArtifactID(t, artifact.KindRecipe, "verify report")},
		},
		Verification:    []artifact.ID{testutil.ArtifactID(t, artifact.KindEvidence, "gate")},
		Budget:          testutil.ArtifactID(t, artifact.KindEvidence, "budget"),
		PauseConditions: []string{"drift"},
	})
	if err != nil {
		t.Fatal(err)
	}
	invocation, err := recipe.NewDelegatedAgentInvocation(recipe.DelegatedAgentInvocation{
		Task: task.ID, Worker: worker.ID,
		Grant:           recipe.AgentCapabilityGrant{Manuals: []artifact.ID{manual.ID}, AllowedEffects: []string{"inspection"}, ReadOnly: true},
		StartingContext: []artifact.ID{testutil.ArtifactID(t, artifact.KindEvidence, "context")},
		Scheduler:       recipe.AgentSchedulerPolicy{MaxSteps: 4, MaxParallel: 1, Preemptible: true},
		Catalog:         snapshot.ID, Model: testutil.ArtifactID(t, artifact.KindModel, "model"),
	})
	if err != nil {
		t.Fatal(err)
	}
	runtime, err := NewDelegatedCapabilityRuntime(
		ctx, store, agenttool.NewOperatorExecutor(), invocation, worker, task, "agent", t.TempDir(),
	)
	if err != nil {
		t.Fatal(err)
	}
	if runtime.Coordinator == nil || runtime.Checkpoints == nil ||
		len(runtime.Compiled.ManualIDs()) != 1 || runtime.Compiled.ManualIDs()[0] != manual.ID {
		t.Fatalf("runtime join = %+v", runtime)
	}
	if len(runtime.Obligations) != 2 {
		t.Fatalf("obligations = %+v", runtime.Obligations)
	}
	byName := map[string]int{}
	for index, obligation := range runtime.Obligations {
		byName[obligation.Name] = index
		if obligation.Scope != "internal" || obligation.MutationEpoch != 1 || len(obligation.Sources) != 1 {
			t.Fatalf("obligation %q = %+v", obligation.Name, obligation)
		}
	}
	if _, tests := byName["tests"]; !tests {
		t.Fatalf("obligations lack the tests criterion: %+v", runtime.Obligations)
	}
	if _, report := byName["report"]; !report {
		t.Fatalf("obligations lack the report criterion: %+v", runtime.Obligations)
	}

	admission := GrantEffectAdmission(invocation.Grant)
	inspect, err := agenttool.DeriveInvocationEffect(manual, json.RawMessage(`{}`), nil)
	if err != nil {
		t.Fatal(err)
	}
	if err := admission(inspect); err != nil {
		t.Fatalf("granted inspection refused: %v", err)
	}
	mutation, err := agenttool.NewManual(agenttool.Manual{
		Name: "repo.write", Description: "Write state.",
		Effect: agenttool.EffectMutation, Transport: agenttool.Transport{Kind: agenttool.TransportBuiltin},
	})
	if err != nil {
		t.Fatal(err)
	}
	mutate, err := agenttool.DeriveInvocationEffect(mutation, json.RawMessage(`{}`), nil)
	if err != nil {
		t.Fatal(err)
	}
	if err := admission(mutate); err == nil || !strings.Contains(err.Error(), "read-only") {
		t.Fatalf("read-only grant admitted mutation: %v", err)
	}

	resolved, done, err := runtime.ResolveObligation("tests", 1,
		[]artifact.ID{testutil.ArtifactID(t, artifact.KindEvidence, "tests pass")}, nil)
	if err != nil || done {
		t.Fatalf("first resolution = done=%t, %v", done, err)
	}
	stale, done, err := runtime.ResolveObligation("report", 2,
		[]artifact.ID{testutil.ArtifactID(t, artifact.KindEvidence, "stale report")}, resolved)
	if err != nil || done {
		t.Fatalf("stale-epoch resolution satisfied the contract: done=%t, %v", done, err)
	}
	_, done, err = runtime.ResolveObligation("report", 1,
		[]artifact.ID{testutil.ArtifactID(t, artifact.KindEvidence, "report done")}, stale)
	if err != nil || !done {
		t.Fatalf("exact resolutions = done=%t, %v", done, err)
	}
	if _, _, err := runtime.ResolveObligation("absent", 1, nil, nil); err == nil {
		t.Fatal("unknown obligation resolved")
	}

	wrongTask, err := recipe.NewAgentTaskContract(recipe.AgentTaskContract{
		Agent: worker.ID, Objective: "Different task.", Scope: []string{"internal"},
		AllowedEffects: []string{"inspection"},
		Acceptance: []recipe.AcceptanceCriterion{
			{Name: "tests", Scope: "internal", Verifier: testutil.ArtifactID(t, artifact.KindRecipe, "other verify")},
		},
		Verification:    []artifact.ID{testutil.ArtifactID(t, artifact.KindEvidence, "gate")},
		Budget:          testutil.ArtifactID(t, artifact.KindEvidence, "budget"),
		PauseConditions: []string{"drift"},
	})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := NewDelegatedCapabilityRuntime(
		ctx, store, agenttool.NewOperatorExecutor(), invocation, worker, wrongTask, "agent", t.TempDir(),
	); err == nil {
		t.Fatal("mismatched task contract admitted")
	}
}

// TestAgentDelegationStagedSurfaceRetired pins the retirement: the staged
// surface catalog no longer parks any agent-delegation owner -- every one
// gained its production consumer in the delegated capability runtime and
// the peer invocation path.
func TestAgentDelegationStagedSurfaceRetired(t *testing.T) {
	data, err := os.ReadFile(filepath.Join("..", "..", "docs", "staged_surface.json"))
	if err != nil {
		t.Fatal(err)
	}
	var catalog struct {
		Staged []struct {
			Name string `json:"name"`
		} `json:"staged"`
	}
	if err := json.Unmarshal(data, &catalog); err != nil {
		t.Fatal(err)
	}
	retired := map[string]bool{
		"NewMutationCheckpointRuntime": true, "NewDelegatedCoordinator": true,
		"CapabilityProxyManual": true, "RegisterCapabilityProxy": true,
		"BindAgentTrajectoryPlan": true, "NewAgentTaskContract": true,
		"NewDelegatedAgentInvocation": true, "NewAgentObligation": true,
		"NewAgentObligationResolution": true, "CompileDelegatedAgentInvocation": true,
		"PublishAutomationPolicyTransition": true,
	}
	for _, entry := range catalog.Staged {
		if retired[entry.Name] {
			t.Errorf("delegation owner %s is still staged", entry.Name)
		}
	}
}
