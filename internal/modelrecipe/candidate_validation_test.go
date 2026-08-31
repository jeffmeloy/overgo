package modelrecipe

import (
	"strings"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/overgodb"
	"overgo/internal/recipe"
	"overgo/internal/testutil"
)

// TestCandidateRecipeCapabilityClosure pins the validation contract: a
// derived candidate proves every module, port, dependency, placement,
// residency, and session policy against the registered catalogs; a candidate
// with an unresolved dependency or an incoherent policy records exact
// refusals instead of a fallback topology; and both verdicts publish as
// durable evidence citing the candidate.
func TestCandidateRecipeCapabilityClosure(t *testing.T) {
	modelID := testutil.ArtifactID(t, artifact.KindModel, "candidate-closure-model")
	candidate, err := CapabilityDefinition(recipe.TaskForecast, modelID)
	if err != nil {
		t.Fatal(err)
	}
	runnable := ValidateCandidateRecipe(candidate)
	if !runnable.Runnable || len(runnable.Refusals) != 0 || runnable.Definition != candidate.ID {
		t.Fatalf("registered candidate closure = %+v", runnable)
	}

	broken := candidate
	broken.Dependencies = append(broken.Dependencies, recipe.Dependency{Role: recipe.DependencyModel})
	refused := ValidateCandidateRecipe(broken)
	if refused.Runnable || len(refused.Refusals) == 0 {
		t.Fatalf("unresolved dependency validated: %+v", refused)
	}
	for _, refusal := range refused.Refusals {
		if strings.Contains(refusal, "fallback") {
			t.Fatalf("refusal selected a fallback: %q", refusal)
		}
	}

	store, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	ctx := t.Context()
	definitionContent, err := Content(candidate)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := store.Commit(ctx, artifact.Batch{
		Key: "closure/candidate", Contents: []artifact.Content{definitionContent},
	}); err != nil {
		t.Fatal(err)
	}
	evidence, err := PublishCandidateClosure(ctx, store, runnable)
	if err != nil || evidence.Kind() != artifact.KindEvidence {
		t.Fatalf("closure evidence = (%s, %v)", evidence, err)
	}
	if _, err := PublishCandidateClosure(ctx, store, CandidateRecipeClosure{}); err == nil {
		t.Fatal("closure without a definition identity published")
	}
}
