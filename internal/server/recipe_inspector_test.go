package server

import (
	"encoding/json"
	"errors"
	"net/http"
	"slices"
	"strings"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/modelrecipe"
	"overgo/internal/recipe"
	"overgo/internal/testutil"
)

type recipeInspectorGenerator struct {
	*fakeGenerator
	description modelrecipe.RuntimeDescription
	err         error
}

func (generator *recipeInspectorGenerator) ModelID() artifact.ID {
	return generator.description.Identity.Model
}

func (generator *recipeInspectorGenerator) RecipeRuntimeDescription(
	task recipe.Task,
) (modelrecipe.RuntimeDescription, error) {
	if generator.err != nil {
		return modelrecipe.RuntimeDescription{}, generator.err
	}
	if task != generator.description.Task {
		return modelrecipe.RuntimeDescription{}, errors.New("task is not active")
	}
	return generator.description, nil
}

func responseRecipeGenerator(t testing.TB, generator *fakeGenerator) *recipeInspectorGenerator {
	t.Helper()
	const (
		nodeID   recipe.NodeID   = "respond"
		moduleID recipe.ModuleID = "test.respond"
		portName recipe.PortName = "messages"
	)
	modelID := testutil.ArtifactID(t, artifact.KindModel, "response-model")
	catalog, err := recipe.NewCatalog(recipe.Module{
		ID: moduleID, Tasks: []recipe.Task{recipe.TaskInference}, Placements: []recipe.Placement{recipe.PlacementHost},
		Outputs: []recipe.Port{{Name: portName, Data: recipe.DataLogits, Cardinality: recipe.CardinalityOne}},
	})
	if err != nil {
		t.Fatal(err)
	}
	definition, err := recipe.NewDefinitionWithDependencies(
		recipe.TaskInference,
		[]recipe.Dependency{{Role: recipe.DependencyModel, Artifact: modelID}},
		[]recipe.Node{{ID: nodeID, Module: moduleID, Placement: recipe.PlacementHost}}, nil, nil,
		[]recipe.Output{{Name: portName, Data: recipe.DataLogits, Source: recipe.Endpoint{Node: nodeID, Port: portName}}},
	)
	if err != nil {
		t.Fatal(err)
	}
	program, err := recipe.CompileProgram(definition, catalog)
	if err != nil {
		t.Fatal(err)
	}
	scope, err := program.InteractionScope(nodeID)
	if err != nil {
		t.Fatal(err)
	}
	return &recipeInspectorGenerator{fakeGenerator: generator, description: modelrecipe.RuntimeDescription{
		Task:     recipe.TaskInference,
		Identity: modelrecipe.ProgramIdentity{Model: modelID, Recipe: definition.ID},
		Stages:   program.Stages(), Outputs: definition.Outputs,
		CacheIdentity: definition.ID, Interaction: scope,
	}}
}

func TestActiveRecipeInspectorUsesCompiledOrder(t *testing.T) {
	modelID := testutil.ArtifactID(t, artifact.KindModel, "inspector-model")
	recipeID := testutil.ArtifactID(t, artifact.KindRecipe, "inspector-recipe")
	profileID := testutil.ArtifactID(t, artifact.KindProfile, "inspector-profile")
	definitionID := testutil.ArtifactID(t, artifact.KindModelDefinition, "inspector-definition")
	stageOrder := []recipe.NodeID{"prepare", "execute", "publish"}
	stages := make([]recipe.Stage, len(stageOrder))
	for index, id := range stageOrder {
		module := recipe.ModuleID("runtime." + string(id))
		stages[index] = recipe.Stage{
			Node: recipe.Node{ID: id, Module: module, Placement: recipe.PlacementHost},
			Module: recipe.Module{
				ID: module, Tasks: []recipe.Task{recipe.TaskInference},
				Placements: []recipe.Placement{recipe.PlacementHost},
			},
		}
	}
	description := modelrecipe.RuntimeDescription{
		Task: recipe.TaskInference,
		Identity: modelrecipe.ProgramIdentity{
			Model: modelID, Profile: profileID, Definition: definitionID,
			Recipe: recipeID, RecipeVersion: recipe.Version,
			Placement: recipe.PlacementHost, Residency: recipe.ResidencyHostCache,
			Runtime: modelrecipe.RuntimeInference,
		},
		Stages: stages, CacheIdentity: recipeID,
		RequiredFacts: []recipe.Dependency{{Role: recipe.DependencyModel, Artifact: modelID}},
	}
	handler := newTestHandler(t, &recipeInspectorGenerator{fakeGenerator: &fakeGenerator{}, description: description})
	response := serveTestRequest(handler, http.MethodGet, "/recipes/active?task=inference", "")
	if response.Code != http.StatusOK {
		t.Fatalf("status = %d body=%s", response.Code, response.Body.String())
	}
	var result activeRecipeResponse
	if err := json.Unmarshal(response.Body.Bytes(), &result); err != nil {
		t.Fatal(err)
	}
	if !result.Admitted || result.Recipe == nil || result.Recipe.CacheIdentity != recipeID {
		t.Fatalf("inspection = %+v", result)
	}
	gotOrder := make([]recipe.NodeID, len(result.Recipe.Stages))
	for index, stage := range result.Recipe.Stages {
		gotOrder[index] = stage.Node.ID
	}
	if !slices.Equal(gotOrder, stageOrder) {
		t.Fatalf("stage order = %v, want compiled %v", gotOrder, stageOrder)
	}

	module := serveTestRequest(handler, http.MethodGet, "/mod/recipe.js", "").Body.String()
	for _, token := range []string{"runtime.stages.entries()", "runtime.required_facts", "runtime.cache_identity"} {
		if !strings.Contains(module, token) {
			t.Errorf("recipe module missing %q", token)
		}
	}
	if strings.Contains(module, "runtime.stages.sort(") {
		t.Error("browser reorders compiled stages")
	}

	refusalText := "compiled module is unavailable"
	refused := newTestHandler(t, &recipeInspectorGenerator{fakeGenerator: &fakeGenerator{}, err: errors.New(refusalText)})
	refusal := serveTestRequest(refused, http.MethodGet, "/recipes/active?task=inference", "")
	var refusedResult activeRecipeResponse
	if err := json.Unmarshal(refusal.Body.Bytes(), &refusedResult); err != nil {
		t.Fatal(err)
	}
	if refusedResult.Admitted || refusedResult.Refusal != refusalText {
		t.Fatalf("refusal = %+v", refusedResult)
	}
}
