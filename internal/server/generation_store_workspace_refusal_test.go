package server

import (
	"context"
	"encoding/json"
	"strings"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/mediacapability"
	"overgo/internal/modelrecipe"
	"overgo/internal/overgodb"
	"overgo/internal/recipe"
)

// sessionlessVQADefinition is the vqa recipe as stores activated it before
// its generate stage declared a session: a valid recipe the store cannot
// compile a component session plan for.
func sessionlessVQADefinition(modelID artifact.ID) (recipe.Definition, error) {
	prepare := recipe.Node{ID: "prepare", Module: modelrecipe.ModuleVQAPrepare, Placement: recipe.PlacementHost}
	generate := recipe.Node{ID: "generate", Module: modelrecipe.ModuleVQAGenerate, Placement: recipe.PlacementDevice}
	return recipe.NewDefinitionWithDependencies(
		recipe.TaskVQA,
		[]recipe.Dependency{{Role: recipe.DependencyModel, Artifact: modelID}},
		[]recipe.Node{prepare, generate},
		[]recipe.Edge{{From: recipe.Endpoint{Node: prepare.ID, Port: "session"}, To: recipe.Endpoint{Node: generate.ID, Port: "session"}}},
		[]recipe.Input{
			{Name: "image", Data: recipe.DataImage, Target: recipe.Endpoint{Node: prepare.ID, Port: "image"}},
			{Name: "question", Data: recipe.DataText, Target: recipe.Endpoint{Node: prepare.ID, Port: "question"}},
		},
		[]recipe.Output{{Name: "answer", Data: recipe.DataText, Source: recipe.Endpoint{Node: generate.ID, Port: "answer"}}},
	)
}

// TestGenerationWorkspaceListsUnresolvableActivationWithRefusal pins the
// listing over an activation the store cannot resolve an execution for
// (a vqa activation whose recipe declares no component session): it lists
// with the reason as its refusal, so the page offers it disabled with why,
// the run refuses it, and the workspace keeps listing.
func TestGenerationWorkspaceListsUnresolvableActivationWithRefusal(t *testing.T) {
	ctx := t.Context()
	store, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	modelID := activateModel(t, store, sessionlessVQADefinition)
	catalog := map[recipe.Task]mediacapability.Capability{recipe.TaskVQA: {
		Execute: func(context.Context, artifact.Repository, string, modelrecipe.CapabilityEvidenceSelection, string) (any, error) {
			t.Fatal("a refused capability executed")
			return nil, nil
		},
	}}
	workspace := NewStoreGenerationWorkspace(store, BindGenerationCatalog(catalog, mediacapability.Controls, mediacapability.OutputContent), 16)
	capabilities, err := workspace.WorkflowCapabilities(ctx, WorkflowGeneration)
	if err != nil || len(capabilities) != 1 {
		t.Fatalf("capabilities = %+v, %v", capabilities, err)
	}
	capability := capabilities[0]
	if capability.Task != recipe.TaskVQA || capability.Model != modelID || !strings.Contains(capability.Refusal, "session") ||
		len(capability.Stages) == 0 || len(capability.Controls) != 0 {
		t.Fatalf("capability = %+v", capability)
	}
	if err := validateWorkflowCapabilities(capabilities); err != nil {
		t.Fatal(err)
	}
	if _, err := workspace.ExecuteWorkflow(ctx, WorkflowGeneration, recipe.TaskVQA, capability.Recipe, json.RawMessage(`{"question":"what?"}`), &recordingReporter{}); err == nil || !strings.Contains(err.Error(), "session") {
		t.Fatalf("a refused capability ran: %v", err)
	}
}
