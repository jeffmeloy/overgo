package evaluation

import (
	"context"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/overgodb"
	"overgo/internal/recipe"
	"overgo/internal/testutil"
)

func TestActivationCaseRegistry(t *testing.T) {
	ctx := context.Background()
	store, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { store.Close() })
	contract := testutil.ArtifactID(t, artifact.KindRecipe, "activation-task-contract")
	input := testutil.ArtifactID(t, artifact.KindDataset, "activation-case-input")
	check := testutil.ArtifactID(t, artifact.KindProfile, "activation-evidence-check")
	if _, err := store.Commit(ctx, artifact.Batch{Key: "activation/case/parents", Artifacts: []artifact.Descriptor{{ID: contract}, {ID: input}, {ID: check}}}); err != nil {
		t.Fatal(err)
	}
	first := ActivationCase{Name: "generate.exact", Task: recipe.TaskGeneration, Contract: contract, Input: input, EvidenceCheck: check}
	second := ActivationCase{Name: "generate.bounded", Task: recipe.TaskGeneration, Contract: contract, Input: input, EvidenceCheck: check}
	registry, _, err := PublishActivationCaseRegistry(ctx, store, []ActivationCase{first, second})
	if err != nil {
		t.Fatal(err)
	}
	reordered, _, err := PublishActivationCaseRegistry(ctx, store, []ActivationCase{second, first})
	if err != nil || reordered.ID != registry.ID {
		t.Fatalf("reordered registry = (%s, %v), want %s", reordered.ID, err, registry.ID)
	}
	resolved, cases, found, err := ResolveActivationCaseRegistry(ctx, store)
	if err != nil || !found || resolved.Denominator != 2 || len(cases) != 2 {
		t.Fatalf("registry = (%+v, %d, %t, %v)", resolved, len(cases), found, err)
	}
	if _, _, err := PublishActivationCaseRegistry(ctx, store, []ActivationCase{first, first}); err == nil {
		t.Fatal("duplicate behavioral case entered the denominator")
	}
}
