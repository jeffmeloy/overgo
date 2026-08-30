package modelrecipe

import (
	"strings"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/model"
	"overgo/internal/modelartifact"
	"overgo/internal/recipe"
	"overgo/internal/testutil"
)

func derivationResolvedDefinition(t *testing.T) ResolvedModelDefinition {
	t.Helper()
	profile, _ := model.LookupArchitecture(definitionArchitecture)
	profile.LayerTopology = model.LayerTopologyCausalPostQKNormSkip
	profileDocument, err := NewProfileDocument(profile)
	if err != nil {
		t.Fatal(err)
	}
	tensors, err := modelartifact.NewTensorInventoryDocument(
		testutil.ArtifactID(t, artifact.KindModel, "derivation-model"), modelartifact.TensorFormatGGUF,
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

// TestPrototypeRecipeDerivation pins the candidate derivation contract: the
// inference candidate binds the resolved definition's exact model, profile,
// and definition identities as dependencies of a closed module graph, every
// additional task derives through the registered capability catalog, and an
// unregistered task, a duplicate task, or an unresolved definition refuses
// instead of selecting a fallback topology.
func TestPrototypeRecipeDerivation(t *testing.T) {
	resolved := derivationResolvedDefinition(t)
	candidates, err := DerivePrototypeRecipes(
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
	if !bound[resolved.Document.Model] || !bound[resolved.Profile.ID] || !bound[resolved.Document.ID] {
		t.Fatalf("inference candidate lost its definition bindings: %+v", inference.Dependencies)
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

	if _, err := DerivePrototypeRecipes(
		resolved, recipe.PlacementHybrid, recipe.SessionCapacity, recipe.ResidencyHybridNative,
		[]recipe.Task{recipe.Task("weight-surgery")},
	); err == nil || !strings.Contains(err.Error(), "unsupported capability task") {
		t.Fatalf("unregistered task derived a fallback topology: %v", err)
	}
	if _, err := DerivePrototypeRecipes(
		resolved, recipe.PlacementHybrid, recipe.SessionCapacity, recipe.ResidencyHybridNative,
		[]recipe.Task{recipe.TaskForecast, recipe.TaskForecast},
	); err == nil || !strings.Contains(err.Error(), "duplicate candidate task") {
		t.Fatalf("duplicate task derived twice: %v", err)
	}
	if _, err := DerivePrototypeRecipes(
		ResolvedModelDefinition{}, recipe.PlacementHybrid, recipe.SessionCapacity,
		recipe.ResidencyHybridNative, nil,
	); err == nil || !strings.Contains(err.Error(), "resolved model definition") {
		t.Fatalf("unresolved definition derived: %v", err)
	}
}
