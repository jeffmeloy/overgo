package workflowruntime

import (
	"context"
	"testing"

	"overgo/internal/agenttool"
	"overgo/internal/artifact"
	"overgo/internal/overgodb"
	"overgo/internal/recipe"
	"overgo/internal/testutil"
)

func TestDelegatedAgentInvocation(t *testing.T) {
	ctx := context.Background()
	store, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	manual, err := agenttool.NewManual(agenttool.Manual{Name: "repo.read", Description: "Read repository state.", Effect: agenttool.EffectInspection, Transport: agenttool.Transport{Kind: agenttool.TransportBuiltin}})
	if err != nil {
		t.Fatal(err)
	}
	snapshot, err := agenttool.NewCatalogSnapshot([]agenttool.Manual{manual})
	if err != nil {
		t.Fatal(err)
	}
	snapshotContent, _ := snapshot.ArtifactContent()
	if _, err := agenttool.PublishManualCatalog(ctx, store, []agenttool.Manual{manual}); err != nil {
		t.Fatal(err)
	}
	batch, err := artifact.NewDocumentBatch("delegation-fixture", []artifact.Content{snapshotContent}, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := artifact.CommitBatch(ctx, store, batch); err != nil {
		t.Fatal(err)
	}
	worker, err := recipe.NewAgentDefinition(recipe.AgentDefinition{Name: "worker", Prompt: testutil.ArtifactID(t, artifact.KindFile, "prompt"), ModelRecipe: testutil.ArtifactID(t, artifact.KindRecipe, "model-recipe"), ToolManuals: []artifact.ID{manual.ID}, Policies: []artifact.ID{testutil.ArtifactID(t, artifact.KindProfile, "policy")}})
	if err != nil {
		t.Fatal(err)
	}
	invocation, err := recipe.NewDelegatedAgentInvocation(recipe.DelegatedAgentInvocation{
		Task: testutil.ArtifactID(t, artifact.KindRecipe, "task"), Worker: worker.ID,
		Grant:           recipe.AgentCapabilityGrant{Manuals: []artifact.ID{manual.ID}, AllowedEffects: []string{"inspection"}, ReadOnly: true},
		StartingContext: []artifact.ID{testutil.ArtifactID(t, artifact.KindEvidence, "context")},
		Scheduler:       recipe.AgentSchedulerPolicy{MaxSteps: 4, MaxParallel: 1, Preemptible: true}, Catalog: snapshot.ID, Model: testutil.ArtifactID(t, artifact.KindModel, "model"),
	})
	if err != nil {
		t.Fatal(err)
	}
	compiled, err := CompileDelegatedAgentInvocation(ctx, store, invocation, worker)
	if err != nil {
		t.Fatal(err)
	}
	if len(compiled.ManualIDs()) != 1 || compiled.ManualIDs()[0] != manual.ID {
		t.Fatalf("compiled manuals = %v", compiled.ManualIDs())
	}
	mutation, _ := agenttool.NewManual(agenttool.Manual{Name: "repo.write", Description: "Write state.", Effect: agenttool.EffectMutation, Transport: agenttool.Transport{Kind: agenttool.TransportBuiltin}})
	worker.ToolManuals = append(worker.ToolManuals, mutation.ID)
	invocation.Grant.Manuals = []artifact.ID{mutation.ID}
	if _, err := CompileDelegatedAgentInvocation(ctx, store, invocation, worker); err == nil {
		t.Fatal("read-only child admitted mutation")
	}
}
