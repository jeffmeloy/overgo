package workflowruntime

import (
	"context"
	"errors"
	"strings"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/overgodb"
	"overgo/internal/recipe"
	"overgo/internal/runrecord"
	"overgo/internal/testutil"
	"overgo/internal/workflowrecipe"
)

func TestRuntimeExecutesWorkflowAndPublishesRun(t *testing.T) {
	ctx := context.Background()
	store, program := runtimeFixture(t)
	defer store.Close()
	runtime, err := NewForProgram(store, program)
	if err != nil {
		t.Fatal(err)
	}
	registerGenerationAdapters(t, runtime, false)
	prompt := fixtureContent(t, artifact.KindFile, "hello")
	const key = "runtime/success"
	operation := runtimeExecutionID(t, program, key)
	inputs := map[recipe.PortName]Value{
		"prompt": {Kind: recipe.DataText, Items: []Datum{{Content: &prompt, Value: "hello"}}},
	}
	result, err := runtime.ExecuteProgram(ctx, key, operation, nil, program, inputs)
	if err != nil {
		t.Fatal(err)
	}
	if result.Run.Outcome != runrecord.OutcomeSucceeded || !result.Commit.Valid() {
		t.Fatalf("result = %+v", result)
	}
	stored, ok, err := artifact.ReadContent(ctx, store, result.Run.ID)
	if err != nil || !ok {
		t.Fatalf("stored run = (%v, %v)", ok, err)
	}
	parsed, err := runrecord.ParseRun(stored.Data)
	if err != nil || parsed.ID != result.Run.ID || len(parsed.Outputs) != 1 {
		t.Fatalf("parsed run = (%+v, %v)", parsed, err)
	}
	replayed, err := runtime.ExecuteProgram(ctx, key, operation, nil, program, inputs)
	if err != nil || replayed.Run.ID != result.Run.ID {
		t.Fatalf("replayed run = (%s, %v), want %s", replayed.Run.ID, err, result.Run.ID)
	}
	parents, err := store.Parents(ctx, result.Run.ID)
	if err != nil || len(parents) != 2 {
		t.Fatalf("run parents = (%+v, %v)", parents, err)
	}
}

func TestRuntimePublishesFailedRun(t *testing.T) {
	ctx := context.Background()
	store, program := runtimeFixture(t)
	defer store.Close()
	runtime, err := NewForProgram(store, program)
	if err != nil {
		t.Fatal(err)
	}
	registerGenerationAdapters(t, runtime, true)
	result, err := runtime.ExecuteProgram(ctx, "runtime/failure", runtimeExecutionID(t, program, "runtime/failure"), nil, program, map[recipe.PortName]Value{
		"prompt": {Kind: recipe.DataText, Items: []Datum{{Value: "hello"}}},
	})
	if err == nil || result.Run.Outcome != runrecord.OutcomeFailed || result.Run.Failure != executionFailureCode {
		t.Fatalf("failed result = (%+v, %v)", result, err)
	}
	if _, ok, loadErr := artifact.ReadContent(ctx, store, result.Run.ID); loadErr != nil || !ok {
		t.Fatalf("failed run was not published: %v", loadErr)
	}
}

func TestRuntimePublishesCancelledRun(t *testing.T) {
	store, program := runtimeFixture(t)
	defer store.Close()
	runtime, err := NewForProgram(store, program)
	if err != nil {
		t.Fatal(err)
	}
	registerGenerationAdapters(t, runtime, false)
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	result, err := runtime.ExecuteProgram(ctx, "runtime/cancelled", runtimeExecutionID(t, program, "runtime/cancelled"), nil, program, map[recipe.PortName]Value{
		"prompt": {Kind: recipe.DataText, Items: []Datum{{Value: "hello"}}},
	})
	if !errors.Is(err, context.Canceled) || result.Run.Outcome != runrecord.OutcomeCancelled {
		t.Fatalf("cancelled result = (%+v, %v)", result, err)
	}
	if _, ok, loadErr := artifact.ReadContent(context.Background(), store, result.Run.ID); loadErr != nil || !ok {
		t.Fatalf("cancelled run was not published: %v", loadErr)
	}
}

func TestWorkflowRestartSkipsCompletedStages(t *testing.T) {
	ctx := context.Background()
	store, program := runtimeFixture(t)
	defer store.Close()
	operation := runtimeExecutionID(t, program, "runtime/restart")
	first, err := NewForProgram(store, program)
	if err != nil {
		t.Fatal(err)
	}
	registerGenerationAdapters(t, first, true)
	prompt := fixtureContent(t, artifact.KindFile, "restart")
	inputs := map[recipe.PortName]Value{
		"prompt": {Kind: recipe.DataText, Items: []Datum{{Content: &prompt, Value: "restart"}}},
	}
	if _, err := first.ExecuteProgram(ctx, "runtime/restart", operation, nil, program, inputs); err == nil {
		t.Fatal("interrupted workflow succeeded")
	}
	second, err := NewForProgram(store, program)
	if err != nil {
		t.Fatal(err)
	}
	tokenizeRuns := 0
	if err := second.Register(workflowrecipe.ModuleTokenize, AdapterFunc(func(context.Context, StepRequest) (map[recipe.PortName]Value, error) {
		tokenizeRuns++
		return nil, errors.New("completed tokenize stage reran")
	})); err != nil {
		t.Fatal(err)
	}
	copyTokens := AdapterFunc(func(_ context.Context, request StepRequest) (map[recipe.PortName]Value, error) {
		return map[recipe.PortName]Value{"tokens": {Kind: recipe.DataTokens, Items: request.Inputs["tokens"].Items}}, nil
	})
	if err := second.Register(workflowrecipe.ModuleGenerate, copyTokens); err != nil {
		t.Fatal(err)
	}
	if err := second.Register(workflowrecipe.ModuleDetokenize, AdapterFunc(func(_ context.Context, request StepRequest) (map[recipe.PortName]Value, error) {
		text := request.Inputs["tokens"].Items[0]
		content := fixtureContent(t, artifact.KindOutput, strings.ToUpper(text.Value.(string)))
		text.Content, text.Artifact = &content, content.Descriptor
		return map[recipe.PortName]Value{"text": {Kind: recipe.DataText, Items: []Datum{text}}}, nil
	})); err != nil {
		t.Fatal(err)
	}
	result, err := second.ExecuteProgram(ctx, "runtime/restart", operation, nil, program, inputs)
	if err != nil || tokenizeRuns != 0 || result.Run.Outcome != runrecord.OutcomeSucceeded {
		t.Fatalf("recovered workflow = (runs=%d, outcome=%s, err=%v)", tokenizeRuns, result.Run.Outcome, err)
	}
}

func TestWorkflowRecoveryRejectsRecipeDrift(t *testing.T) {
	ctx := context.Background()
	store, program := runtimeFixture(t)
	defer store.Close()
	operation := runtimeExecutionID(t, program, "runtime/drift")
	runtime, err := NewForProgram(store, program)
	if err != nil {
		t.Fatal(err)
	}
	registerGenerationAdapters(t, runtime, true)
	if _, err := runtime.ExecuteProgram(ctx, "runtime/drift", operation, nil, program, map[recipe.PortName]Value{
		"prompt": {Kind: recipe.DataText, Items: []Datum{{Value: "drift"}}},
	}); err == nil {
		t.Fatal("drift fixture did not stop")
	}
	definition := program.Definition()
	definition.Dependencies[0].Artifact = testutil.ArtifactID(t, artifact.KindModel, "replacement-model")
	drifted, err := recipe.NewDefinitionWithDependencies(
		definition.Task, definition.Dependencies, definition.Nodes, definition.Edges, definition.Inputs, definition.Outputs,
	)
	if err != nil {
		t.Fatal(err)
	}
	driftedProgram, err := recipe.CompileProgram(drifted, workflowrecipe.Catalog())
	if err != nil {
		t.Fatal(err)
	}
	driftedRuntime, err := NewForProgram(store, driftedProgram)
	if err != nil {
		t.Fatal(err)
	}
	registerGenerationAdapters(t, driftedRuntime, false)
	if _, err := driftedRuntime.ExecuteProgram(ctx, "runtime/drift", operation, nil, driftedProgram, map[recipe.PortName]Value{
		"prompt": {Kind: recipe.DataText, Items: []Datum{{Value: "drift"}}},
	}); err == nil || !strings.Contains(err.Error(), "recovery recipe differs") {
		t.Fatalf("recipe drift error = %v", err)
	}
}

func TestExecuteProgramPreservesOrchestrationBoundary(t *testing.T) {
	store, _ := runtimeFixture(t)
	defer store.Close()
	batch := recipe.Node{ID: "batch", Module: workflowrecipe.ModuleBatchDataset, Placement: recipe.PlacementHost}
	definition, err := recipe.NewDefinitionWithDependencies(
		recipe.TaskTraining,
		[]recipe.Dependency{
			{Role: recipe.DependencyModel, Artifact: testutil.ArtifactID(t, artifact.KindModel, "training-model")},
			{Role: recipe.DependencyDataset, Artifact: testutil.ArtifactID(t, artifact.KindDataset, "training-dataset")},
		},
		[]recipe.Node{batch}, nil, nil,
		[]recipe.Output{{Name: "batch", Data: recipe.DataBatch, Source: recipe.Endpoint{Node: batch.ID, Port: "batch"}}},
	)
	if err != nil {
		t.Fatal(err)
	}
	program, err := recipe.CompileProgram(definition, workflowrecipe.Catalog())
	if err != nil {
		t.Fatal(err)
	}
	runtime, err := NewForProgram(store, program)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := runtime.ExecuteProgram(context.Background(), "runtime/training", runtimeExecutionID(t, program, "runtime/training"), nil, program, nil); err == nil {
		t.Fatal("orchestration-only program executed")
	}
}

func TestExecuteProgramRejectsAnotherCatalogAuthority(t *testing.T) {
	store, program := runtimeFixture(t)
	defer store.Close()
	runtime, err := NewForProgram(store, program)
	if err != nil {
		t.Fatal(err)
	}
	foreign, err := recipe.NewCatalog(workflowrecipe.Catalog().Modules()...)
	if err != nil {
		t.Fatal(err)
	}
	program, err = recipe.CompileProgram(program.Definition(), foreign)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := runtime.ExecuteProgram(context.Background(), "runtime/foreign", runtimeExecutionID(t, program, "runtime/foreign"), nil, program, nil); err == nil {
		t.Fatal("program compiled by another catalog accepted")
	}
}

func runtimeExecutionID(t testing.TB, program recipe.Program, key string) artifact.ID {
	t.Helper()
	id, err := ExecutionID(program.Definition().ID, key)
	if err != nil {
		t.Fatal(err)
	}
	return id
}

func TestRuntimeCollapsesRepeatedArtifactFacts(t *testing.T) {
	content := fixtureContent(t, artifact.KindFile, "shared")
	values := map[recipe.PortName]Value{
		"left":  {Kind: recipe.DataText, Items: []Datum{{Content: &content}}},
		"right": {Kind: recipe.DataText, Items: []Datum{{Artifact: content.Descriptor}}},
	}
	ids, facts, err := externalFacts(values, true)
	if err != nil {
		t.Fatal(err)
	}
	if len(ids) != 1 || len(facts.descriptors) != 1 || len(facts.contents) != 1 {
		t.Fatalf("facts = (%d IDs, %d descriptors, %d contents)", len(ids), len(facts.descriptors), len(facts.contents))
	}
}

func TestValidateOutputsAllowsAbsentOptionalPort(t *testing.T) {
	module := recipe.Module{Outputs: []recipe.Port{
		{Name: "required", Data: recipe.DataText, Cardinality: recipe.CardinalityOne},
		{Name: "optional", Data: recipe.DataMetrics, Cardinality: recipe.CardinalityOptional},
	}}
	outputs, err := validateOutputs(module, map[recipe.PortName]Value{
		"required": {Kind: recipe.DataText, Items: []Datum{{Value: "value"}}},
	})
	if err != nil || len(outputs) != 1 {
		t.Fatalf("outputs = (%+v, %v)", outputs, err)
	}
}

func runtimeFixture(t *testing.T) (*overgodb.Store, recipe.Program) {
	t.Helper()
	store, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	modelID := testutil.ArtifactID(t, artifact.KindModel, "model")
	tokenizerID := testutil.ArtifactID(t, artifact.KindTokenizer, "tokenizer")
	if _, err := store.Commit(context.Background(), artifact.Batch{
		Key:       "runtime/dependencies",
		Artifacts: []artifact.Descriptor{{ID: modelID}, {ID: tokenizerID}},
	}); err != nil {
		store.Close()
		t.Fatal(err)
	}
	tokenize := recipe.Node{ID: "tokenize", Module: workflowrecipe.ModuleTokenize, Placement: recipe.PlacementHost}
	generate := recipe.Node{ID: "generate", Module: workflowrecipe.ModuleGenerate, Placement: recipe.PlacementHost}
	detokenize := recipe.Node{ID: "detokenize", Module: workflowrecipe.ModuleDetokenize, Placement: recipe.PlacementHost}
	definition, err := recipe.NewDefinitionWithDependencies(
		recipe.TaskGeneration,
		[]recipe.Dependency{
			{Role: recipe.DependencyModel, Artifact: modelID},
			{Role: recipe.DependencyTokenizer, Artifact: tokenizerID},
		},
		[]recipe.Node{tokenize, generate, detokenize},
		[]recipe.Edge{
			{From: recipe.Endpoint{Node: tokenize.ID, Port: "tokens"}, To: recipe.Endpoint{Node: generate.ID, Port: "tokens"}},
			{From: recipe.Endpoint{Node: generate.ID, Port: "tokens"}, To: recipe.Endpoint{Node: detokenize.ID, Port: "tokens"}},
		},
		[]recipe.Input{{Name: "prompt", Data: recipe.DataText, Target: recipe.Endpoint{Node: tokenize.ID, Port: "text"}}},
		[]recipe.Output{{Name: "text", Data: recipe.DataText, Source: recipe.Endpoint{Node: detokenize.ID, Port: "text"}}},
	)
	if err != nil {
		store.Close()
		t.Fatal(err)
	}
	program, err := recipe.CompileProgram(definition, workflowrecipe.Catalog())
	if err != nil {
		store.Close()
		t.Fatal(err)
	}
	return store, program
}

func registerGenerationAdapters(t *testing.T, runtime *Runtime, fail bool) {
	t.Helper()
	copyAdapter := func(output recipe.DataKind, port recipe.PortName) AdapterFunc {
		return func(_ context.Context, request StepRequest) (map[recipe.PortName]Value, error) {
			for _, input := range request.Inputs {
				return map[recipe.PortName]Value{port: {Kind: output, Items: input.Items}}, nil
			}
			return nil, errors.New("missing input")
		}
	}
	if err := runtime.Register(workflowrecipe.ModuleTokenize, copyAdapter(recipe.DataTokens, "tokens")); err != nil {
		t.Fatal(err)
	}
	generate := copyAdapter(recipe.DataTokens, "tokens")
	if fail {
		generate = func(context.Context, StepRequest) (map[recipe.PortName]Value, error) {
			return nil, errors.New("fixture failure")
		}
	}
	if err := runtime.Register(workflowrecipe.ModuleGenerate, generate); err != nil {
		t.Fatal(err)
	}
	if err := runtime.Register(workflowrecipe.ModuleDetokenize, AdapterFunc(
		func(_ context.Context, request StepRequest) (map[recipe.PortName]Value, error) {
			text := request.Inputs["tokens"].Items[0]
			content := fixtureContent(t, artifact.KindOutput, strings.ToUpper(text.Value.(string)))
			text.Content, text.Artifact = &content, content.Descriptor
			return map[recipe.PortName]Value{
				"text": {Kind: recipe.DataText, Items: []Datum{text}},
			}, nil
		},
	)); err != nil {
		t.Fatal(err)
	}
}

func fixtureContent(t *testing.T, kind artifact.Kind, value string) artifact.Content {
	t.Helper()
	id := testutil.ArtifactID(t, kind, value)
	return artifact.Content{
		Descriptor: artifact.Descriptor{ID: id, Size: uint64(len(value)), MediaType: "text/plain"},
		Data:       []byte(value),
	}
}
