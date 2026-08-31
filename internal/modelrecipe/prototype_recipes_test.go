package modelrecipe

import (
	"slices"
	"strings"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/model"
	"overgo/internal/modelartifact"
	"overgo/internal/recipe"
	"overgo/internal/testutil"
)

func derivePrototypeRecipeFixture(
	resolved ResolvedModelDefinition,
	placement recipe.Placement,
	session DecodeSessionPolicy,
	residency recipe.ResidencyPolicy,
	tasks []recipe.Task,
) (map[recipe.Task]recipe.Definition, error) {
	derived, err := deriveResolvedExecutionRecipes(resolved, CandidateRecipePolicy{
		Placement: placement, Session: session, Residency: residency, Tasks: slices.Clone(tasks),
	}, nil)
	if err != nil {
		return nil, err
	}
	result := make(map[recipe.Task]recipe.Definition, len(derived))
	for _, candidate := range derived {
		result[candidate.Task] = candidate.Definition
	}
	return result, nil
}

func derivationResolvedDefinition(t *testing.T, name string) ResolvedModelDefinition {
	t.Helper()
	profile, _ := model.LookupArchitecture(definitionArchitecture)
	profile.LayerTopology = model.LayerTopologyCausalPostQKNormSkip
	profileDocument, err := NewProfileDocument(profile)
	if err != nil {
		t.Fatal(err)
	}
	tensors, err := modelartifact.NewTensorInventoryDocument(
		testutil.ArtifactID(t, artifact.KindModel, name), modelartifact.TensorFormatGGUF,
		[]modelartifact.TensorFact{{
			Name: "token_embd.weight", Shape: []uint64{definitionTensorWidth, definitionTensorWidth},
			Storage: "f32", Bytes: definitionTensorBytes,
		}},
	)
	if err != nil {
		t.Fatal(err)
	}
	document, err := NewModelDefinitionDocument(profileDocument, tensors, definitionSpec())
	if err != nil {
		t.Fatal(err)
	}
	return ResolvedModelDefinition{
		Document: document, Profile: profileDocument, Tensors: tensors, Spec: definitionSpec(),
	}
}

// TestPrototypeRecipeDerivation pins the compatibility entry to the common
// resolved-model owner: exact model facts become a closed compiled graph and
// unsupported, duplicate, or unresolved requests fail without a fallback.
func TestPrototypeRecipeDerivation(t *testing.T) {
	resolved := derivationResolvedDefinition(t, "prototype-derivation-model")
	candidates, err := derivePrototypeRecipeFixture(
		resolved, recipe.PlacementHybrid, recipe.SessionCapacity, recipe.ResidencyHybridNative,
		[]recipe.Task{recipe.TaskForecast},
	)
	if err != nil {
		t.Fatal(err)
	}
	if len(candidates) != 2 {
		t.Fatalf("candidate set = %d recipes", len(candidates))
	}
	inference := candidates[recipe.TaskInference]
	bound := map[artifact.ID]bool{}
	for _, dependency := range inference.Dependencies {
		bound[dependency.Artifact] = true
	}
	for _, id := range []artifact.ID{
		resolved.Document.Model, resolved.Profile.ID, resolved.Document.ID, resolved.Tensors.ID,
	} {
		if !bound[id] {
			t.Fatalf("inference candidate lost dependency %s: %+v", id, inference.Dependencies)
		}
	}
	if len(inference.Nodes) == 0 || !inference.ID.Valid() {
		t.Fatalf("inference candidate is not a closed module graph: %+v", inference)
	}
	forecast := candidates[recipe.TaskForecast]
	for _, node := range forecast.Nodes {
		if node.Module == "" {
			t.Fatalf("forecast candidate node lacks a registered module: %+v", node)
		}
	}

	if _, err := derivePrototypeRecipeFixture(
		resolved, recipe.PlacementHybrid, recipe.SessionCapacity, recipe.ResidencyHybridNative,
		[]recipe.Task{recipe.Task("weight-surgery")},
	); err == nil || !strings.Contains(err.Error(), "unsupported capability task") {
		t.Fatalf("unregistered task derived a fallback topology: %v", err)
	}
	if _, err := derivePrototypeRecipeFixture(
		resolved, recipe.PlacementHybrid, recipe.SessionCapacity, recipe.ResidencyHybridNative,
		[]recipe.Task{recipe.TaskForecast, recipe.TaskForecast},
	); err == nil || !strings.Contains(err.Error(), "duplicate candidate task") {
		t.Fatalf("duplicate task derived twice: %v", err)
	}
	if _, err := derivePrototypeRecipeFixture(
		ResolvedModelDefinition{}, recipe.PlacementHybrid, recipe.SessionCapacity,
		recipe.ResidencyHybridNative, nil,
	); err == nil || !strings.Contains(err.Error(), "resolved model definition") {
		t.Fatalf("unresolved definition derived: %v", err)
	}
}
