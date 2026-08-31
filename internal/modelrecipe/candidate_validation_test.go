package modelrecipe

import (
	"fmt"
	"slices"
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

func TestCandidateRecipeRejectsUnknownTransformControllerOrCheckpoint(t *testing.T) {
	modelContent, err := artifact.JSONContent(artifact.DocumentContract{
		Kind: artifact.KindModel, MediaType: "application/vnd.overgo.test-model+json", Schema: "overgo/test-model/v1",
	}, struct {
		Name string `json:"name"`
	}{Name: "candidate-capability-model"})
	if err != nil {
		t.Fatal(err)
	}
	definition, err := CapabilityDefinition(recipe.TaskForecast, modelContent.Descriptor.ID)
	if err != nil {
		t.Fatal(err)
	}
	store, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	if _, err := store.Commit(t.Context(), artifact.Batch{
		Key: "candidate/capability-model", Contents: []artifact.Content{modelContent},
	}); err != nil {
		t.Fatal(err)
	}
	if closure := ValidateStoredCandidateRecipe(t.Context(), store, definition); !closure.Runnable {
		t.Fatalf("locally closed candidate refused: %+v", closure)
	}

	unknown := []struct {
		name string
		role recipe.DependencyRole
		kind artifact.Kind
	}{
		{name: "transform", role: recipe.DependencyCandidateComponent, kind: artifact.KindProfile},
		{name: "controller", role: recipe.DependencyDerivationProfile, kind: artifact.KindProfile},
		{name: "checkpoint", role: recipe.DependencyCheckpoint, kind: artifact.KindCheckpoint},
	}
	for slot, capability := range unknown {
		t.Run(capability.name, func(t *testing.T) {
			missing := testutil.ArtifactID(t, capability.kind, "unknown-"+capability.name)
			dependencies := append([]recipe.Dependency(nil), definition.Dependencies...)
			dependencies = append(dependencies, recipe.Dependency{
				Role: capability.role, Slot: uint32(slot), Artifact: missing,
			})
			candidate, err := recipe.NewDefinitionWithDependencies(
				definition.Task, dependencies, definition.Nodes, definition.Edges, definition.Inputs, definition.Outputs,
			)
			if err != nil {
				t.Fatal(err)
			}
			closure := ValidateStoredCandidateRecipe(t.Context(), store, candidate)
			if closure.Runnable || len(closure.Refusals) == 0 {
				t.Fatalf("unknown %s validated: %+v", capability.name, closure)
			}
			want := fmt.Sprintf("capability %s[%d] artifact %s is absent", capability.role, slot, missing)
			if !slices.Contains(closure.Refusals, want) {
				t.Fatalf("unknown %s refusals = %v, want %q", capability.name, closure.Refusals, want)
			}
			for _, refusal := range closure.Refusals {
				if strings.Contains(refusal, "fallback") {
					t.Fatalf("unknown %s selected fallback: %q", capability.name, refusal)
				}
			}
		})
	}
}
