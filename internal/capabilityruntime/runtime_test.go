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
		func(_ context.Context, _ artifact.Repository, path string, _ recipe.Program, request scalarRequest) (int, error) {
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

type cachedScalarModel struct {
	bias   int
	closed *int
}

func (m *cachedScalarModel) Close(context.Context) error {
	*m.closed++
	return nil
}

func TestImageSessionCacheReusesResidentRuntime(t *testing.T) {
	store, modelID, program := capabilityFixture(t, "cached-scalar-model")
	loads, resets, closes := 0, 0, 0
	cache, err := NewScalarSessionCache[scalarRequest, *cachedScalarModel, int](
		"scalar", "cuda:0", 1,
		func(request scalarRequest) error {
			if request.Value <= 0 {
				return errors.New("positive value required")
			}
			return nil
		},
		func(scalarRequest) (string, error) { return "shape:scalar", nil },
		func(_ context.Context, _ artifact.Repository, _ string, _ recipe.Program, request scalarRequest) (*cachedScalarModel, error) {
			loads++
			return &cachedScalarModel{bias: request.Value, closed: &closes}, nil
		},
		func(_ context.Context, model *cachedScalarModel, request scalarRequest) error {
			resets++
			model.bias = request.Value
			return nil
		},
		func(runtime *workflowruntime.Runtime, bound artifact.ID, model *cachedScalarModel) error {
			return workflowruntime.RegisterJSONStage[scalarRequest, int](
				runtime, scalarModule, bound,
				artifact.JSONContract(artifact.KindOutput, "test.cached-scalar-output.v1"),
				func(request scalarRequest) (int, error) { return request.Value + model.bias, nil },
			)
		},
	)
	if err != nil {
		t.Fatal(err)
	}
	execute := cache.Executor()
	foreign := testutil.ArtifactID(t, artifact.KindModel, "foreign-cached-model")
	if _, err := execute(context.Background(), store, "model", foreign, program, `{"value":4}`); err == nil {
		t.Fatal("foreign model loaded into session cache")
	}
	if loads != 0 {
		t.Fatalf("foreign admission loaded %d models", loads)
	}
	for raw, want := range map[string]int{`{"value":4}`: 8, `{"value":5}`: 10} {
		got, err := execute(context.Background(), store, "model", modelID, program, raw)
		if err != nil {
			t.Fatal(err)
		}
		if got != want {
			t.Fatalf("output=%v want=%d", got, want)
		}
	}
	if loads != 1 || resets != 1 {
		t.Fatalf("loads=%d resets=%d", loads, resets)
	}
	if err := cache.Close(context.Background()); err != nil {
		t.Fatal(err)
	}
	if closes != 1 {
		t.Fatalf("closes=%d", closes)
	}
}
