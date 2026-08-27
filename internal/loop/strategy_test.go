package loop

import (
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/recipe"
	"overgo/internal/testutil"
)

func TestStrategyIdentity(t *testing.T) {
	worker, err := recipe.NewAgentDefinition(recipe.AgentDefinition{Name: "strategy-worker", Prompt: testutil.ArtifactID(t, artifact.KindFile, "prompt"), ModelRecipe: testutil.ArtifactID(t, artifact.KindRecipe, "model"), Policies: []artifact.ID{testutil.ArtifactID(t, artifact.KindProfile, "policy")}})
	if err != nil {
		t.Fatal(err)
	}
	config := Config{MaxAttemptsPerStep: 3, MaxInvocations: 10}
	catalog := testutil.ArtifactID(t, artifact.KindProfile, "catalog")
	first, err := NewStrategy(worker, catalog, config)
	if err != nil {
		t.Fatal(err)
	}
	second, err := NewStrategy(worker, catalog, config)
	if err != nil {
		t.Fatal(err)
	}
	if first.ID != second.ID || len(first.Lineage()) == 0 {
		t.Fatalf("strategies differ: %s %s", first.ID, second.ID)
	}
	config.MaxInvocations++
	changed, err := NewStrategy(worker, catalog, config)
	if err != nil || changed.ID == first.ID {
		t.Fatal("loop configuration did not change strategy identity")
	}
}
