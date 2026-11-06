package workflowruntime

import (
	"testing"

	"overgo/internal/agenttool"
	"overgo/internal/artifact"
	"overgo/internal/overgodbtest"
	"overgo/internal/recipe"
	"overgo/internal/testcheck"
	"overgo/internal/testutil"
)

func TestDelegatedAgentInvocation(t *testing.T) {
	ctx := t.Context()
	store := overgodbtest.Open(t, t.TempDir())
	manual, err := agenttool.NewManual(agenttool.Manual{Name: "repo.read", Description: "Read repository state.", Effect: agenttool.EffectInspection, Transport: agenttool.Transport{Kind: agenttool.TransportBuiltin}})
	testcheck.NoError(t, err)
	snapshot, err := agenttool.NewCatalogSnapshot([]agenttool.Manual{manual})
	testcheck.NoError(t, err)
	snapshotContent, _ := snapshot.ArtifactContent()
	if _, err := agenttool.PublishManualCatalog(ctx, store, []agenttool.Manual{manual}); err != nil {
		t.Fatal(err)
	}
	batch, err := artifact.NewDocumentBatch("delegation-fixture", []artifact.Content{snapshotContent}, nil, nil)
	testcheck.NoError(t, err)
	if _, err := artifact.CommitBatch(ctx, store, batch); err != nil {
		t.Fatal(err)
	}
	worker, err := recipe.NewAgentDefinition(recipe.AgentDefinition{Name: "worker", Prompt: testutil.ArtifactID(t, artifact.KindFile, "prompt"), ModelRecipe: testutil.ArtifactID(t, artifact.KindRecipe, "model-recipe"), ToolManuals: []artifact.ID{manual.ID}, Policies: []artifact.ID{testutil.ArtifactID(t, artifact.KindProfile, "policy")}})
	testcheck.NoError(t, err)
	invocation, err := recipe.NewDelegatedAgentInvocation(recipe.DelegatedAgentInvocation{
		Task: testutil.ArtifactID(t, artifact.KindRecipe, "task"), Worker: worker.ID,
		Grant:           recipe.AgentCapabilityGrant{Manuals: []artifact.ID{manual.ID}, AllowedEffects: []string{"inspection"}, ReadOnly: true},
		StartingContext: []artifact.ID{testutil.ArtifactID(t, artifact.KindEvidence, "context")},
		Scheduler:       recipe.AgentSchedulerPolicy{MaxSteps: 4, MaxParallel: 1, Preemptible: true}, Catalog: snapshot.ID, Model: testutil.ArtifactID(t, artifact.KindModel, "model"),
	})
	testcheck.NoError(t, err)
	compiled, err := CompileDelegatedAgentInvocation(ctx, store, invocation, worker)
	testcheck.NoError(t, err)
	if len(compiled.Manuals) != 1 || compiled.Manuals[0].ID != manual.ID {
		t.Fatalf("compiled manuals = %v", compiled.Manuals)
	}
	mutation, _ := agenttool.NewManual(agenttool.Manual{Name: "repo.write", Description: "Write state.", Effect: agenttool.EffectMutation, Transport: agenttool.Transport{Kind: agenttool.TransportBuiltin}})
	worker.ToolManuals = append(worker.ToolManuals, mutation.ID)
	invocation.Grant.Manuals = []artifact.ID{mutation.ID}
	if _, err := CompileDelegatedAgentInvocation(ctx, store, invocation, worker); err == nil {
		t.Fatal("read-only child admitted mutation")
	}
}
