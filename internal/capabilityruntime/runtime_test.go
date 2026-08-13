package capabilityruntime

import (
	"context"
	"errors"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/recipe"
	"overgo/internal/repodb"
	"overgo/internal/testutil"
	"overgo/internal/workflowruntime"
)

type scalarRequest struct {
	Value int `json:"value"`
}

const scalarModule recipe.ModuleID = "test.scalar"

func capabilityFixture(t *testing.T, name string) (*repodb.Store, artifact.ID, recipe.Program) {
	t.Helper()
	store, err := repodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = store.Close() })
	modelID := testutil.ArtifactID(t, artifact.KindModel, name)
	testutil.PublishArtifact(t, store, modelID)
	node := recipe.Node{ID: "scalar", Module: scalarModule, Placement: recipe.PlacementHost}
	definition, err := recipe.NewDefinitionWithDependencies(
		recipe.TaskImageGen,
		[]recipe.Dependency{{Role: recipe.DependencyModel, Artifact: modelID}},
		[]recipe.Node{node}, nil,
		[]recipe.Input{{Name: "input", Data: recipe.DataTensor, Target: recipe.Endpoint{Node: node.ID, Port: "input"}}},
		[]recipe.Output{{Name: "output", Data: recipe.DataImage, Source: recipe.Endpoint{Node: node.ID, Port: "output"}}},
	)
	if err != nil {
		t.Fatal(err)
	}
	catalog, err := recipe.NewCatalog(recipe.Module{
		ID: scalarModule, Tasks: []recipe.Task{recipe.TaskImageGen}, Placements: []recipe.Placement{recipe.PlacementHost},
		Inputs:  []recipe.Port{{Name: "input", Data: recipe.DataTensor, Cardinality: recipe.CardinalityOne}},
		Outputs: []recipe.Port{{Name: "output", Data: recipe.DataImage, Cardinality: recipe.CardinalityOne}},
	})
	if err != nil {
		t.Fatal(err)
	}
	program, err := recipe.CompileProgram(definition, catalog)
	if err != nil {
		t.Fatal(err)
	}
	return store, modelID, program
}

func TestJSONScalarExecutesIdentityBoundProgram(t *testing.T) {
	store, modelID, program := capabilityFixture(t, "scalar-model")
	execute := JSONScalar[scalarRequest, int, int](
		"scalar", func(request scalarRequest) error {
			if request.Value <= 0 {
				return errors.New("positive value required")
			}
			return nil
		},
		func(_ context.Context, path string, request scalarRequest) (int, error) {
			return len(path) + request.Value, nil
		},
		func(runtime *workflowruntime.Runtime, bound artifact.ID, model int) error {
			return workflowruntime.RegisterJSONStage[scalarRequest, int](
				runtime, scalarModule, bound,
				artifact.JSONContract(artifact.KindOutput, "test.scalar-output.v1"),
				func(request scalarRequest) (int, error) { return request.Value + model, nil },
			)
		},
	)
	got, err := execute(context.Background(), store, "abc", modelID, program, `{"value":4}`)
	if err != nil {
		t.Fatal(err)
	}
	if got != 11 {
		t.Fatalf("output = %v, want 11", got)
	}
	if _, err := execute(context.Background(), store, "abc", modelID, program, `{"value":4,"extra":1}`); err == nil {
		t.Fatal("unknown input field accepted")
	}
	if program.Definition().Task != recipe.TaskImageGen {
		t.Fatalf("task = %s", program.Definition().Task)
	}
}
