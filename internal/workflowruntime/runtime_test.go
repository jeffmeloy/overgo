package workflowruntime

import (
	"context"
	"errors"
	"strings"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/recipe"
	"overgo/internal/repodb"
	"overgo/internal/runrecord"
	"overgo/internal/workflowrecipe"
)

func TestRuntimeExecutesWorkflowAndPublishesRun(t *testing.T) {
	ctx := context.Background()
	store, definition := runtimeFixture(t)
	defer store.Close()
	runtime, err := New(store)
	if err != nil {
		t.Fatal(err)
	}
	registerGenerationAdapters(t, runtime, false)
	prompt := fixtureContent(t, artifact.KindFile, "hello")
	result, err := runtime.Execute(ctx, "runtime/success", definition, map[recipe.PortName]Value{
		"prompt": {Kind: recipe.DataText, Items: []Datum{{Content: &prompt, Value: "hello"}}},
	})
	if err != nil {
		t.Fatal(err)
	}
	if result.Run.Outcome != runrecord.OutcomeSucceeded || !result.Commit.Valid() {
		t.Fatalf("result = %+v", result)
	}
	stored, ok, err := store.Content(ctx, result.Run.ID)
	if err != nil || !ok {
		t.Fatalf("stored run = (%v, %v)", ok, err)
	}
	parsed, err := runrecord.ParseRun(stored.Data)
	if err != nil || parsed.ID != result.Run.ID || len(parsed.Outputs) != 1 {
		t.Fatalf("parsed run = (%+v, %v)", parsed, err)
	}
	parents, err := store.Parents(ctx, result.Run.ID)
	if err != nil || len(parents) != 2 {
		t.Fatalf("run parents = (%+v, %v)", parents, err)
	}
}

func TestRuntimePublishesFailedRun(t *testing.T) {
	ctx := context.Background()
	store, definition := runtimeFixture(t)
	defer store.Close()
	runtime, err := New(store)
	if err != nil {
		t.Fatal(err)
	}
	registerGenerationAdapters(t, runtime, true)
	result, err := runtime.Execute(ctx, "runtime/failure", definition, map[recipe.PortName]Value{
		"prompt": {Kind: recipe.DataText, Items: []Datum{{Value: "hello"}}},
	})
	if err == nil || result.Run.Outcome != runrecord.OutcomeFailed || result.Run.Failure != executionFailureCode {
		t.Fatalf("failed result = (%+v, %v)", result, err)
	}
	if _, ok, loadErr := store.Content(ctx, result.Run.ID); loadErr != nil || !ok {
		t.Fatalf("failed run was not published: %v", loadErr)
	}
}

func TestRuntimePublishesCancelledRun(t *testing.T) {
	store, definition := runtimeFixture(t)
	defer store.Close()
	runtime, err := New(store)
	if err != nil {
		t.Fatal(err)
	}
	registerGenerationAdapters(t, runtime, false)
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	result, err := runtime.Execute(ctx, "runtime/cancelled", definition, map[recipe.PortName]Value{
		"prompt": {Kind: recipe.DataText, Items: []Datum{{Value: "hello"}}},
	})
	if !errors.Is(err, context.Canceled) || result.Run.Outcome != runrecord.OutcomeCancelled {
		t.Fatalf("cancelled result = (%+v, %v)", result, err)
	}
	if _, ok, loadErr := store.Content(context.Background(), result.Run.ID); loadErr != nil || !ok {
		t.Fatalf("cancelled run was not published: %v", loadErr)
	}
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

func runtimeFixture(t *testing.T) (*repodb.Store, recipe.Definition) {
	t.Helper()
	store, err := repodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	modelID := fixtureID(t, artifact.KindModel, "model")
	tokenizerID := fixtureID(t, artifact.KindTokenizer, "tokenizer")
	if _, err := store.Commit(context.Background(), artifact.Batch{
		Key:       "runtime/dependencies",
		Artifacts: []artifact.Descriptor{{ID: modelID}, {ID: tokenizerID}},
	}); err != nil {
		store.Close()
		t.Fatal(err)
	}
	definition, err := workflowrecipe.Generation(workflowrecipe.Bindings{
		Model: modelID, Tokenizer: tokenizerID,
	}, recipe.PlacementHost)
	if err != nil {
		store.Close()
		t.Fatal(err)
	}
	return store, definition
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
	id := fixtureID(t, kind, value)
	return artifact.Content{
		Descriptor: artifact.Descriptor{ID: id, Size: uint64(len(value)), MediaType: "text/plain"},
		Data:       []byte(value),
	}
}

func fixtureID(t *testing.T, kind artifact.Kind, value string) artifact.ID {
	t.Helper()
	id, err := artifact.IdentifyBytes(kind, []byte(value))
	if err != nil {
		t.Fatal(err)
	}
	return id
}
