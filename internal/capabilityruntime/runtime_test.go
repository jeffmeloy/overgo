package capabilityruntime

import (
	"context"
	"errors"
	"sync/atomic"
	"testing"
	"time"

	"overgo/internal/artifact"
	"overgo/internal/modelrecipe"
	"overgo/internal/overgodb"
	"overgo/internal/recipe"
	"overgo/internal/testutil"
	"overgo/internal/workflowruntime"
)

type scalarRequest struct {
	Value int `json:"value"`
}

func (scalarRequest) SessionKey() (string, error) { return "shape:scalar", nil }

type concurrentScalarModel struct {
	bias    int
	entered chan<- struct{}
	release <-chan struct{}
	closed  *atomic.Int32
}

func (m *concurrentScalarModel) Close(context.Context) error {
	m.closed.Add(1)
	return nil
}

func TestModelSessionDirectorConcurrentKeys(t *testing.T) {
	firstStore, firstID, firstProgram := capabilityFixture(t, "concurrent-scalar-first")
	secondStore, secondID, secondProgram := capabilityFixture(t, "concurrent-scalar-second")
	entered := make(chan struct{}, 2)
	release := make(chan struct{})
	var closed atomic.Int32
	director, err := NewModelSessionDirector[scalarRequest, *concurrentScalarModel, int](
		"scalar", "cuda:0", 2,
		func(request scalarRequest) error {
			if request.Value <= 0 {
				return errors.New("positive value required")
			}
			return nil
		},
		func(_ context.Context, _ artifact.Repository, _ string, _ recipe.Program, request scalarRequest) (*concurrentScalarModel, error) {
			return &concurrentScalarModel{bias: request.Value, entered: entered, release: release, closed: &closed}, nil
		},
		func(_ context.Context, model *concurrentScalarModel, request scalarRequest) error {
			model.bias = request.Value
			return nil
		},
		func(runtime *workflowruntime.Runtime, bound artifact.ID, model *concurrentScalarModel) error {
			return workflowruntime.RegisterJSONStage[scalarRequest, int](
				runtime, scalarModule, bound,
				artifact.JSONContract(artifact.KindOutput, "test.concurrent-scalar-output.v1"),
				func(request scalarRequest) (int, error) {
					model.entered <- struct{}{}
					<-model.release
					return request.Value + model.bias, nil
				},
			)
		},
	)
	if err != nil {
		t.Fatal(err)
	}
	type result struct {
		value any
		err   error
	}
	results := make(chan result, 2)
	run := func(store artifact.Repository, _ artifact.ID, program recipe.Program, raw string) {
		execution := candidateExecution(t, store, program)
		value, err := director.Executor()(t.Context(), store, "model", execution, raw)
		results <- result{value: value, err: err}
	}
	go run(firstStore, firstID, firstProgram, `{"value":2}`)
	go run(secondStore, secondID, secondProgram, `{"value":3}`)
	deadline := time.After(5 * time.Second)
	for range 2 {
		select {
		case <-entered:
		case <-deadline:
			t.Fatal("distinct session keys executed serially")
		}
	}
	close(release)
	values := map[int]bool{}
	for range 2 {
		got := <-results
		if got.err != nil {
			t.Fatal(got.err)
		}
		values[Unwrap(got.value).(int)] = true
	}
	if !values[4] || !values[6] {
		t.Fatalf("outputs=%v", values)
	}
	if err := director.Close(t.Context()); err != nil {
		t.Fatal(err)
	}
	if got := closed.Load(); got != 2 {
		t.Fatalf("closed=%d", got)
	}
}

const scalarModule recipe.ModuleID = "test.scalar"

func capabilityFixture(t *testing.T, name string) (*overgodb.Store, artifact.ID, recipe.Program) {
	t.Helper()
	store, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = store.Close() })
	weights := []byte(name + "-weights")
	weightsID := testutil.ArtifactBytesID(t, artifact.KindTensorSet, weights)
	manifest, err := artifact.NewManifest(artifact.KindModel, []artifact.Component{{
		Role: artifact.ComponentWeights, Name: "weights", Artifact: weightsID,
	}})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := store.Commit(t.Context(), artifact.Batch{
		Key:       "test/session-model/" + manifest.ID.String(),
		Artifacts: []artifact.Descriptor{{ID: weightsID, Size: uint64(len(weights))}}, Manifests: []artifact.Manifest{manifest},
	}); err != nil {
		t.Fatal(err)
	}
	modelID := manifest.ID
	node := recipe.Node{ID: "scalar", Module: scalarModule, Placement: recipe.PlacementHost, Session: recipe.SessionCapacity}
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

func candidateExecution(t *testing.T, store artifact.Reader, program recipe.Program) modelrecipe.CapabilityEvidenceSelection {
	t.Helper()
	execution, err := modelrecipe.CompileCandidateExecution(t.Context(), store, program)
	if err != nil {
		t.Fatal(err)
	}
	return execution
}

func TestJSONScalarExecutesIdentityBoundProgram(t *testing.T) {
	store, _, program := capabilityFixture(t, "scalar-model")
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
	execution := candidateExecution(t, store, program)
	got, err := execute(t.Context(), store, "abc", execution, `{"value":4}`)
	if err != nil {
		t.Fatal(err)
	}
	if Unwrap(got) != 11 {
		t.Fatalf("output = %v, want 11", got)
	}
	if _, err := execute(t.Context(), store, "abc", execution, `{"value":4,"extra":1}`); err == nil {
		t.Fatal("unknown input field accepted")
	}
	if program.Definition().Task != recipe.TaskImageGen {
		t.Fatalf("task = %s", program.Definition().Task)
	}
}

func TestExecutorCatalogUsesCompiledEntryModule(t *testing.T) {
	store, modelID, program := capabilityFixture(t, "executor-catalog")
	execution := candidateExecution(t, store, program)
	catalog := ExecutorCatalog{scalarModule: func(
		_ context.Context, _ artifact.Repository, _ string, bound modelrecipe.CapabilityEvidenceSelection, raw string,
	) (any, error) {
		if bound.Program.Definition().Model != modelID {
			t.Fatal("executor received another model")
		}
		return raw, nil
	}}
	const input = `{"value":4}`
	got, err := catalog.Execute(context.WithoutCancel(t.Context()), store, "model", execution, input)
	if err != nil {
		t.Fatal(err)
	}
	if got != input {
		t.Fatalf("output=%v", got)
	}
	delete(catalog, scalarModule)
	if _, err := catalog.Execute(context.WithoutCancel(t.Context()), store, "model", execution, input); err == nil {
		t.Fatal("missing entry executor accepted")
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

func TestComponentSessionDirectorFollowsCompiledLifetimes(t *testing.T) {
	store, modelID, program := capabilityFixture(t, "cached-scalar-model")
	resources, err := modelrecipe.CompileComponentSessionPlan(t.Context(), store, program)
	if err != nil {
		t.Fatal(err)
	}
	if len(resources.Components) != 1 || resources.Components[0].Node != "scalar" ||
		resources.Components[0].Model != modelID || resources.Components[0].Session != recipe.SessionCapacity {
		t.Fatalf("capacity resources=%+v", resources)
	}
	loads, resets, closes := 0, 0, 0
	director, err := NewModelSessionDirector[scalarRequest, *cachedScalarModel, int](
		"scalar", "cuda:0", recipe.SessionRequestCapacity,
		func(request scalarRequest) error {
			if request.Value <= 0 {
				return errors.New("positive value required")
			}
			return nil
		},
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
	execute := director.Executor()
	foreign := testutil.ArtifactID(t, artifact.KindModel, "foreign-cached-model")
	execution := candidateExecution(t, store, program)
	execution.Activation.Definition.Model = foreign
	if _, err := execute(t.Context(), store, "model", execution, `{"value":4}`); err == nil {
		t.Fatal("foreign model loaded into session director")
	}
	if loads != 0 {
		t.Fatalf("foreign admission loaded %d models", loads)
	}
	for raw, want := range map[string]int{`{"value":4}`: 8, `{"value":5}`: 10} {
		got, err := execute(t.Context(), store, "model", candidateExecution(t, store, program), raw)
		if err != nil {
			t.Fatal(err)
		}
		if Unwrap(got) != want {
			t.Fatalf("output=%v want=%d", got, want)
		}
	}
	if loads != 1 || resets != 1 {
		t.Fatalf("loads=%d resets=%d", loads, resets)
	}
	definition := program.Definition()
	definition.Nodes[0].Session = recipe.SessionRequest
	requestDefinition, err := recipe.NewDefinitionWithDependencies(
		definition.Task, definition.Dependencies, definition.Nodes, definition.Edges, definition.Inputs, definition.Outputs,
	)
	if err != nil {
		t.Fatal(err)
	}
	requestProgram, err := recipe.CompileProgram(requestDefinition, program.Catalog())
	if err != nil {
		t.Fatal(err)
	}
	requestResources, err := modelrecipe.CompileComponentSessionPlan(t.Context(), store, requestProgram)
	if err != nil {
		t.Fatal(err)
	}
	if len(requestResources.Components) != 1 || requestResources.Components[0].Session != recipe.SessionRequest ||
		requestResources.Identity == resources.Identity {
		t.Fatalf("request resources=%+v", requestResources)
	}
	for range 2 {
		if _, err := execute(t.Context(), store, "model", candidateExecution(t, store, requestProgram), `{"value":6}`); err != nil {
			t.Fatal(err)
		}
	}
	if loads != 3 || resets != 1 || closes != 3 {
		t.Fatalf("resource plan loads=%d resets=%d closes=%d", loads, resets, closes)
	}
	if err := director.Close(t.Context()); err != nil {
		t.Fatal(err)
	}
	if closes != 3 {
		t.Fatalf("closes=%d", closes)
	}
}

type mappedVideoRequest struct {
	Condition int `json:"condition"`
	Source    int `json:"source"`
}

func (mappedVideoRequest) SessionKey() (string, error) { return "video:1x1", nil }

type mappedVideoModel struct {
	runs   int
	closed *int
}

func (m *mappedVideoModel) Close(context.Context) error {
	*m.closed++
	return nil
}

func TestVideoProductionActivation(t *testing.T) {
	const module recipe.ModuleID = "test.video-compose"
	store, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	weights := []byte("mapped-video-weights")
	weightsID := testutil.ArtifactBytesID(t, artifact.KindTensorSet, weights)
	manifest, err := artifact.NewManifest(artifact.KindModel, []artifact.Component{{Role: artifact.ComponentWeights, Name: "weights", Artifact: weightsID}})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := store.Commit(t.Context(), artifact.Batch{
		Key: "test/mapped-video-model", Artifacts: []artifact.Descriptor{{ID: weightsID, Size: uint64(len(weights))}},
		Manifests: []artifact.Manifest{manifest},
	}); err != nil {
		t.Fatal(err)
	}
	modelID := manifest.ID
	node := recipe.Node{ID: "compose", Module: module, Placement: recipe.PlacementHost, Session: recipe.SessionCapacity}
	definition, err := recipe.NewDefinitionWithDependencies(
		recipe.TaskVideoGen, []recipe.Dependency{{Role: recipe.DependencyModel, Artifact: modelID}},
		[]recipe.Node{node}, nil,
		[]recipe.Input{
			{Name: "condition", Data: recipe.DataPromptConditioning, Target: recipe.Endpoint{Node: node.ID, Port: "condition"}},
			{Name: "source", Data: recipe.DataVideo, Target: recipe.Endpoint{Node: node.ID, Port: "source"}},
		},
		[]recipe.Output{{Name: "video", Data: recipe.DataVideo, Source: recipe.Endpoint{Node: node.ID, Port: "video"}}},
	)
	if err != nil {
		t.Fatal(err)
	}
	catalog, err := recipe.NewCatalog(recipe.Module{
		ID: module, Tasks: []recipe.Task{recipe.TaskVideoGen}, Placements: []recipe.Placement{recipe.PlacementHost},
		Inputs: []recipe.Port{
			{Name: "condition", Data: recipe.DataPromptConditioning, Cardinality: recipe.CardinalityOne},
			{Name: "source", Data: recipe.DataVideo, Cardinality: recipe.CardinalityOne},
		},
		Outputs: []recipe.Port{{Name: "video", Data: recipe.DataVideo, Cardinality: recipe.CardinalityOne}},
	})
	if err != nil {
		t.Fatal(err)
	}
	program, err := recipe.CompileProgram(definition, catalog)
	if err != nil {
		t.Fatal(err)
	}
	loads, resets, closes := 0, 0, 0
	director, err := NewMappedModelSessionDirector[mappedVideoRequest, *mappedVideoModel, int](
		"video", "cuda:0", recipe.SessionRequestCapacity,
		func(request mappedVideoRequest) error {
			if request.Condition <= 0 || request.Source <= 0 {
				return errors.New("positive video inputs required")
			}
			return nil
		},
		func(context.Context, artifact.Repository, string, recipe.Program, mappedVideoRequest) (*mappedVideoModel, error) {
			loads++
			return &mappedVideoModel{closed: &closes}, nil
		},
		func(context.Context, *mappedVideoModel, mappedVideoRequest) error { resets++; return nil },
		func(runtime *workflowruntime.Runtime, bound artifact.ID, model *mappedVideoModel) error {
			return workflowruntime.RegisterResolvedStage(runtime, module, bound,
				func(_ context.Context, step workflowruntime.StepRequest) (int, error) {
					condition, err := workflowruntime.ScalarInput[int](step, "condition")
					if err != nil {
						return 0, err
					}
					source, err := workflowruntime.ScalarInput[int](step, "source")
					model.runs++
					return condition + source, err
				}, func(value int) (artifact.Content, error) {
					return artifact.JSONContent(artifact.JSONContract(artifact.KindOutput, "test.video-output.v1"), value)
				})
		},
		func(request mappedVideoRequest) (map[recipe.PortName]MappedInput, error) {
			condition, err := artifact.JSONContent(artifact.JSONContract(artifact.KindFile, "test.video-condition.v1"), request.Condition)
			if err != nil {
				return nil, err
			}
			source, err := artifact.JSONContent(artifact.JSONContract(artifact.KindFile, "test.video-source.v1"), request.Source)
			return map[recipe.PortName]MappedInput{
				"condition": {Value: request.Condition, Content: condition},
				"source":    {Value: request.Source, Content: source},
			}, err
		},
	)
	if err != nil {
		t.Fatal(err)
	}
	for range 2 {
		output, err := director.Executor()(t.Context(), store, "model", candidateExecution(t, store, program), `{"condition":2,"source":3}`)
		if err != nil {
			t.Fatal(err)
		}
		if Unwrap(output) != 5 {
			t.Fatalf("video output=%v", output)
		}
	}
	if loads != 1 || resets != 1 {
		t.Fatalf("loads=%d resets=%d", loads, resets)
	}
	if err := director.Close(t.Context()); err != nil {
		t.Fatal(err)
	}
	if closes != 1 {
		t.Fatalf("closes=%d", closes)
	}
}
