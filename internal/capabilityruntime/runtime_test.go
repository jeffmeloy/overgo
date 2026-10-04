package capabilityruntime

import (
	"context"
	"errors"
	"sync/atomic"
	"testing"
	"time"

	"overgo/internal/artifact"
	"overgo/internal/modelrecipe"
	"overgo/internal/modelrecipetest"
	"overgo/internal/overgodb"
	"overgo/internal/processmeasure"
	"overgo/internal/recipe"
	"overgo/internal/runrecord"
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
		execution := modelrecipetest.CandidateExecution(t, store, program)
		value, err := director.Executor()(t.Context(), store, "model", execution, raw)
		results <- result{value: value, err: err}
	}
	go run(firstStore, firstID, firstProgram, `{"value":2}`)
	go run(secondStore, secondID, secondProgram, `{"value":3}`)
	// Both executions enter before either is released; a serial executor
	// never delivers the second entry and the test binary reports the hang.
	for range 2 {
		<-entered
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

func capabilityFixture(t *testing.T, name string, sessions ...recipe.SessionPolicy) (*overgodb.Store, artifact.ID, recipe.Program) {
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
	if len(sessions) != 0 {
		node.Session = sessions[0]
	}
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
	execution := modelrecipetest.CandidateExecution(t, store, program)
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

type terminalScalarModel struct {
	closeErr error
	closed   bool
}

func (model *terminalScalarModel) Close(ctx context.Context) error {
	model.closed = true
	return errors.Join(model.closeErr, ctx.Err())
}

// TestTerminalMeasurementRetainsPublishedRun keeps execution failure and
// cancellation, and a close failure after success, linked to the actual run.
// A failed attempt is not a successful output or a measured zero device peak.
func TestTerminalMeasurementRetainsPublishedRun(t *testing.T) {
	failure := errors.New("fixture execution failed")
	closeFailure := errors.New("fixture close failed")
	for _, shape := range []string{"scalar", "session"} {
		for _, test := range []struct {
			name                 string
			executeErr, closeErr error
			outcome              runrecord.Outcome
		}{
			{"failed", failure, closeFailure, runrecord.OutcomeFailed},
			{"cancelled", context.Canceled, nil, runrecord.OutcomeCancelled},
			{"close failed", nil, closeFailure, runrecord.OutcomeSucceeded},
		} {
			t.Run(shape+"/"+test.name, func(t *testing.T) {
				store, _, program := capabilityFixture(t, shape+test.name, recipe.SessionRequest)
				execution := modelrecipetest.CandidateExecution(t, store, program)
				model := &terminalScalarModel{closeErr: test.closeErr}
				validate := func(scalarRequest) error { return nil }
				load := func(context.Context, artifact.Repository, string, recipe.Program, scalarRequest) (*terminalScalarModel, error) {
					return model, nil
				}
				ctx, cancel := context.WithCancel(t.Context())
				defer cancel()
				bind := func(runtime *workflowruntime.Runtime, bound artifact.ID, _ *terminalScalarModel) error {
					return workflowruntime.RegisterJSONStage[scalarRequest, int](runtime, scalarModule, bound,
						artifact.JSONContract(artifact.KindOutput, "test.terminal-output.v1"),
						func(input scalarRequest) (int, error) {
							if errors.Is(test.executeErr, context.Canceled) {
								cancel()
							}
							return input.Value, test.executeErr
						})
				}
				execute := JSONScalar[scalarRequest, *terminalScalarModel, int]("scalar", validate, load, bind)
				if shape == "session" {
					director, err := NewModelSessionDirector[scalarRequest, *terminalScalarModel, int]("scalar", "host", 1,
						validate, load, func(context.Context, *terminalScalarModel, scalarRequest) error { return nil }, bind)
					if err != nil {
						t.Fatal(err)
					}
					execute = director.Executor()
					t.Cleanup(func() { _ = director.Close(t.Context()) })
				}
				value, err := execute(ctx, store, "model", execution, `{"value":4}`)
				if test.executeErr != nil && !errors.Is(err, test.executeErr) || test.closeErr != nil && !errors.Is(err, test.closeErr) || !model.closed {
					t.Fatalf("terminal execution: error %v, closed %t; want %v and %v after close", err, model.closed, test.executeErr, test.closeErr)
				}
				measured, ok := value.(Measured)
				if !ok {
					t.Fatalf("terminal execution discarded its measurement: %T", value)
				}
				run, err := runrecord.RequireExactRun(t.Context(), store, measured.Run)
				if err != nil || run.Recipe != program.Definition().ID || run.Outcome != test.outcome || len(run.Inputs) != 1 || run.Inputs[0] != measured.Input.Descriptor.ID {
					t.Fatalf("terminal run %+v (%v); want the persisted recipe, request and %s outcome", run, err, test.outcome)
				}
				if test.executeErr != nil && (len(run.Outputs) != 0 || measured.Output != nil || measured.PeakDeviceBytes != 0) {
					t.Fatalf("failed measurement has successful output or peak: %+v; run %+v", measured, run)
				}
			})
		}
	}
}

// TestMeasurementCarriesTheLoadWall runs one recipe on both executor shapes,
// a request-scoped scalar and a session director, whose fixture load takes a
// span the clock can time. The measurement of an execution that loaded its
// model carries the load's wall: every scalar execution's, and the
// director's first. A request the director's resident session serves loaded
// nothing, and its measurement carries no load.
func TestMeasurementCarriesTheLoadWall(t *testing.T) {
	validate := func(scalarRequest) error { return nil }
	load := func(context.Context, artifact.Repository, string, recipe.Program, scalarRequest) (int, error) {
		started, err := processmeasure.Counter()
		for now := started; err == nil && now <= started; now, err = processmeasure.Counter() {
		}
		return 1, err
	}
	bind := func(runtime *workflowruntime.Runtime, bound artifact.ID, model int) error {
		return workflowruntime.RegisterJSONStage[scalarRequest, int](
			runtime, scalarModule, bound, artifact.JSONContract(artifact.KindOutput, "test.loaded-output.v1"),
			func(request scalarRequest) (int, error) { return request.Value + model, nil },
		)
	}
	director, err := NewModelSessionDirector[scalarRequest, int, int](
		"scalar", "host", 1, validate, load, func(context.Context, int, scalarRequest) error { return nil }, bind,
	)
	if err != nil {
		t.Fatal(err)
	}
	defer director.Close(t.Context())
	for name, shape := range map[string]struct {
		execute  Executor
		resident bool
	}{
		"scalar":  {execute: JSONScalar[scalarRequest, int, int]("scalar", validate, load, bind)},
		"session": {execute: director.Executor(), resident: true},
	} {
		t.Run(name, func(t *testing.T) {
			store, _, program := capabilityFixture(t, "loaded-"+name)
			execution := modelrecipetest.CandidateExecution(t, store, program)
			var loads []time.Duration
			for _, raw := range []string{`{"value":1}`, `{"value":2}`} {
				got, err := shape.execute(t.Context(), store, "model", execution, raw)
				if err != nil {
					t.Fatal(err)
				}
				loads = append(loads, got.(Measured).Load)
			}
			if loads[0] <= 0 || shape.resident != (loads[1] == 0) || loads[1] < 0 {
				t.Fatalf("loads = %v; want the first execution's wall, and the second's only when it loaded again (resident session: %t)", loads, shape.resident)
			}
		})
	}
}

// TestFreshRunExecutesWhatAnIdenticalRequestStored executes and recovers
// one request on both executor shapes. Each new fresh operation runs the
// node and reports its wall; retrying that context recovers its receipt.
// The ordinary request's original receipt remains unchanged and recovers.
func TestFreshRunExecutesWhatAnIdenticalRequestStored(t *testing.T) {
	validate := func(scalarRequest) error { return nil }
	load := func(context.Context, artifact.Repository, string, recipe.Program, scalarRequest) (int, error) {
		return 1, nil
	}
	for _, name := range []string{"scalar", "session"} {
		t.Run(name, func(t *testing.T) {
			var runs atomic.Int32
			bind := func(runtime *workflowruntime.Runtime, bound artifact.ID, model int) error {
				return workflowruntime.RegisterJSONStage[scalarRequest, int](
					runtime, scalarModule, bound, artifact.JSONContract(artifact.KindOutput, "test.fresh-output.v1"),
					func(request scalarRequest) (int, error) {
						runs.Add(1)
						return request.Value + model, nil
					},
				)
			}
			execute := JSONScalar[scalarRequest, int, int]("scalar", validate, load, bind)
			if name == "session" {
				director, err := NewModelSessionDirector[scalarRequest, int, int](
					"scalar", "host", 1, validate, load, func(context.Context, int, scalarRequest) error { return nil }, bind,
				)
				if err != nil {
					t.Fatal(err)
				}
				defer director.Close(t.Context())
				execute = director.Executor()
			}
			store, _, program := capabilityFixture(t, "fresh-"+name)
			execution := modelrecipetest.CandidateExecution(t, store, program)
			output, err := artifact.JSONContent(artifact.JSONContract(artifact.KindOutput, "test.fresh-output.v1"), 5)
			if err != nil {
				t.Fatal(err)
			}
			var own artifact.ID
			var fresh context.Context
			for index, step := range []struct {
				fresh bool
				retry bool
				runs  int32
			}{{runs: 1}, {runs: 1}, {fresh: true, runs: 2}, {retry: true, runs: 2}, {fresh: true, runs: 3}, {runs: 3}} {
				ctx := t.Context()
				if step.fresh {
					fresh = FreshRun(ctx)
					ctx = fresh
				} else if step.retry {
					ctx = fresh
				}
				got, err := execute(ctx, store, "model", execution, `{"value":4}`)
				if err != nil {
					t.Fatal(err)
				}
				measured := got.(Measured)
				ran := index == 0 || step.fresh
				if measured.Output != 5 || runs.Load() != step.runs || ran != (len(measured.Phases) == 1) {
					t.Fatalf("execution %d (fresh run: %t) = %v with node walls %v after %d runs of the node; want 5 after %d runs, its wall only when it ran",
						index, step.fresh, measured.Output, measured.Phases, runs.Load(), step.runs)
				}
				definition := program.Definition()
				run, err := runrecord.RequireExactRun(t.Context(), store, measured.Run)
				if err != nil || run.Recipe != definition.ID || run.Outcome != runrecord.OutcomeSucceeded || len(run.Inputs) != 1 || run.Inputs[0] != measured.Input.Descriptor.ID || len(run.Outputs) != 1 || run.Outputs[0] != output.Descriptor.ID {
					t.Fatalf("execution %d has producer run %+v (%v); want its persisted recipe, input and output", index, run, err)
				}
				operation, err := workflowruntime.ExecutionID(definition.ID, RunKey(definition, measured.Input))
				if err != nil {
					t.Fatal(err)
				}
				receipt, found, err := runrecord.ResolveStageReceipt(t.Context(), store, operation, "scalar")
				if err != nil || !found || receipt.Attempt != 1 || receipt.State != runrecord.StageCompleted {
					t.Fatalf("the request's own run after execution %d = %+v (found %t, %v); want its first attempt, completed", index, receipt, found, err)
				}
				if index == 0 {
					own = receipt.ID
				}
				if receipt.ID != own {
					t.Fatalf("execution %d (fresh run: %t) moved the request's own run from receipt %s to %s", index, step.fresh, own, receipt.ID)
				}
			}
		})
	}
}

// decodingModel is a fixture runtime that decodes tokens: its stage gives
// as many as the request's value and keeps the count of its last run.
type decodingModel struct{ decoded uint64 }

func (model *decodingModel) OutputTokens() uint64 { return model.decoded }

// TestMeasurementCarriesTheDecodeOfATokenRuntime runs one recipe on both
// executor shapes over a runtime that reports the tokens its last run
// decoded. The measurement of an execution that ran the output's node
// carries that count, with the node's wall as its decode. An identical
// request recovers the stored stage and decodes nothing: its measurement
// carries neither, though the director's resident model still holds the
// earlier run's count. A runtime that reports no tokens, one whose output is
// whole, carries neither even when its node ran.
func TestMeasurementCarriesTheDecodeOfATokenRuntime(t *testing.T) {
	validate := func(scalarRequest) error { return nil }
	stage := func(runtime *workflowruntime.Runtime, bound artifact.ID, decoded *uint64) error {
		return workflowruntime.RegisterJSONStage[scalarRequest, int](
			runtime, scalarModule, bound, artifact.JSONContract(artifact.KindOutput, "test.decoded-output.v1"),
			func(request scalarRequest) (int, error) {
				// The node takes a span the clock can time.
				started, err := processmeasure.Counter()
				for now := started; err == nil && now <= started; now, err = processmeasure.Counter() {
				}
				*decoded = uint64(request.Value)
				return request.Value, err
			},
		)
	}
	load := func(context.Context, artifact.Repository, string, recipe.Program, scalarRequest) (*decodingModel, error) {
		return &decodingModel{}, nil
	}
	bind := func(runtime *workflowruntime.Runtime, bound artifact.ID, model *decodingModel) error {
		return stage(runtime, bound, &model.decoded)
	}
	director, err := NewModelSessionDirector[scalarRequest, *decodingModel, int](
		"scalar", "host", 1, validate, load, func(context.Context, *decodingModel, scalarRequest) error { return nil }, bind,
	)
	if err != nil {
		t.Fatal(err)
	}
	defer director.Close(t.Context())
	var unreported uint64
	whole := JSONScalar[scalarRequest, int, int]("scalar", validate,
		func(context.Context, artifact.Repository, string, recipe.Program, scalarRequest) (int, error) {
			return 1, nil
		},
		func(runtime *workflowruntime.Runtime, bound artifact.ID, _ int) error {
			return stage(runtime, bound, &unreported)
		})
	for name, shape := range map[string]struct {
		execute Executor
		tokens  bool
	}{
		"scalar":  {execute: JSONScalar[scalarRequest, *decodingModel, int]("scalar", validate, load, bind), tokens: true},
		"session": {execute: director.Executor(), tokens: true},
		"whole":   {execute: whole},
	} {
		t.Run(name, func(t *testing.T) {
			store, _, program := capabilityFixture(t, "decoded-"+name)
			execution := modelrecipetest.CandidateExecution(t, store, program)
			for index, step := range []struct {
				raw    string
				tokens uint64
			}{{raw: `{"value":3}`, tokens: 3}, {raw: `{"value":3}`}, {raw: `{"value":5}`, tokens: 5}} {
				got, err := shape.execute(t.Context(), store, "model", execution, step.raw)
				if err != nil {
					t.Fatal(err)
				}
				measured := got.(Measured)
				ran := step.tokens != 0
				if ran != (len(measured.Phases) == 1) || ran && measured.Phases[0].WallNS == 0 {
					t.Fatalf("execution %d node walls = %v; want the node's timed wall exactly when it ran (%t)", index, measured.Phases, ran)
				}
				var tokens uint64
				var decode time.Duration
				if ran && shape.tokens {
					tokens, decode = step.tokens, time.Duration(measured.Phases[0].WallNS)
				}
				if measured.OutputTokens != tokens || measured.Decode != decode {
					t.Fatalf("execution %d measured %d output tokens over a %s decode; want %d over %s: the count and the output node's wall of a token runtime's executed run, else neither",
						index, measured.OutputTokens, measured.Decode, tokens, decode)
				}
			}
		})
	}
}

func TestExecutorCatalogUsesCompiledEntryModule(t *testing.T) {
	store, modelID, program := capabilityFixture(t, "executor-catalog")
	execution := modelrecipetest.CandidateExecution(t, store, program)
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
	execution := modelrecipetest.CandidateExecution(t, store, program)
	execution.Activation.Definition.Model = foreign
	if _, err := execute(t.Context(), store, "model", execution, `{"value":4}`); err == nil {
		t.Fatal("foreign model loaded into session director")
	}
	if loads != 0 {
		t.Fatalf("foreign admission loaded %d models", loads)
	}
	for raw, want := range map[string]int{`{"value":4}`: 8, `{"value":5}`: 10} {
		got, err := execute(t.Context(), store, "model", modelrecipetest.CandidateExecution(t, store, program), raw)
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
		if _, err := execute(t.Context(), store, "model", modelrecipetest.CandidateExecution(t, store, requestProgram), `{"value":6}`); err != nil {
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
		output, err := director.Executor()(t.Context(), store, "model", modelrecipetest.CandidateExecution(t, store, program), `{"condition":2,"source":3}`)
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
