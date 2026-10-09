package workflowruntime

import (
	"context"
	"errors"
	"strings"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/recipe"
	"overgo/internal/runrecord"
	"overgo/internal/workflowrecipe"
)

// countingAdapters registers the fixture's three stages on a runtime, each
// counting its runs: tokenize and generate hand on values the store does not
// keep, and detokenize publishes its text, unless it is made to fail.
func countingAdapters(t *testing.T, runtime *Runtime, runs map[recipe.NodeID]int, failDetokenize bool) {
	t.Helper()
	hand := func(node recipe.NodeID) AdapterFunc {
		return func(_ context.Context, request StepRequest) (map[recipe.PortName]Value, error) {
			runs[node]++
			for _, input := range request.Inputs {
				return map[recipe.PortName]Value{"tokens": {Kind: recipe.DataTokens, Items: []Datum{{Value: input.Items[0].Value}}}}, nil
			}
			return nil, errors.New("missing input")
		}
	}
	for module, adapter := range map[recipe.ModuleID]Adapter{
		workflowrecipe.ModuleTokenize: hand("tokenize"),
		workflowrecipe.ModuleGenerate: hand("generate"),
		workflowrecipe.ModuleDetokenize: AdapterFunc(func(_ context.Context, request StepRequest) (map[recipe.PortName]Value, error) {
			runs["detokenize"]++
			if failDetokenize {
				return nil, errors.New("fixture failure")
			}
			content := fixtureContent(t, artifact.KindOutput, strings.ToUpper(request.Inputs["tokens"].Items[0].Value.(string)))
			return map[recipe.PortName]Value{"text": {Kind: recipe.DataText, Items: []Datum{{Content: &content, Artifact: content.Descriptor, Value: string(content.Data)}}}}, nil
		}),
	} {
		if err := runtime.Register(module, adapter); err != nil {
			t.Fatal(err)
		}
	}
}

// TestReplayRunsNoStageItsOutputDoesNotNeed holds a request the store already
// answers to costing no stage. The fixture's first two stages hand on values
// the store does not keep and its last publishes the answer, the shape of a
// picture's program. Served again, the request ran its first two stages each
// time and dropped what they computed, the answer being taken from the store.
// It now runs nothing; a runtime with no adapter registered replays it, as a
// server does before it loads a model; and an operation whose last stage
// failed is not a replay: the store holds no answer, every stage runs, and
// asking only for a replay publishes nothing.
func TestReplayRunsNoStageItsOutputDoesNotNeed(t *testing.T) {
	ctx := t.Context()
	store, program := runtimeFixture(t)
	defer store.Close()
	prompt := fixtureContent(t, artifact.KindFile, "hello")
	inputs := map[recipe.PortName]Value{"prompt": {Kind: recipe.DataText, Items: []Datum{{Content: &prompt, Value: "hello"}}}}
	served := func(runs map[recipe.NodeID]int, fail bool) *Runtime {
		runtime, err := NewForProgram(store, program)
		if err != nil {
			t.Fatal(err)
		}
		countingAdapters(t, runtime, runs, fail)
		return runtime
	}

	const key = "runtime/replay"
	operation := runtimeExecutionID(t, program, key)
	runs := map[recipe.NodeID]int{}
	first, err := served(runs, false).ExecuteProgram(ctx, key, operation, nil, program, inputs)
	if err != nil || len(first.NodeWalls) != 3 || runs["tokenize"] != 1 || runs["generate"] != 1 || runs["detokenize"] != 1 {
		t.Fatalf("the first serving ran %v over %+v (%v); want each stage once", runs, first.NodeWalls, err)
	}
	again, err := served(runs, false).ExecuteProgram(ctx, key, operation, nil, program, inputs)
	if err != nil || again.Run.ID != first.Run.ID || len(again.NodeWalls) != 0 || runs["tokenize"] != 1 || runs["generate"] != 1 || runs["detokenize"] != 1 {
		t.Fatalf("the request served again ran %v over %+v, run %s (%v); want no stage run and the first run %s", runs, again.NodeWalls, again.Run.ID, err, first.Run.ID)
	}
	unbound, err := NewForProgram(store, program)
	if err != nil {
		t.Fatal(err)
	}
	replayed, held, err := unbound.Replay(ctx, key, operation, program, inputs)
	text, single := replayed.Outputs["text"].Single()
	if err != nil || !held || replayed.Run.ID != first.Run.ID || !single || text.Content == nil || string(text.Content.Data) != "HELLO" {
		t.Fatalf("a runtime with no adapter replayed run %s, answer %+v (held %t, %v); want the first run %s and its text", replayed.Run.ID, text, held, err, first.Run.ID)
	}

	// An operation whose last stage failed left no answer.
	const failed = "runtime/replay-failed"
	failedOperation := runtimeExecutionID(t, program, failed)
	runs = map[recipe.NodeID]int{}
	if _, err := served(runs, true).ExecuteProgram(ctx, failed, failedOperation, nil, program, inputs); err == nil {
		t.Fatal("the failing fixture succeeded")
	}
	head, sequence := store.Head()
	if result, held, err := unbound.Replay(ctx, failed, failedOperation, program, inputs); err != nil || held || result.Run.ID.Valid() {
		t.Fatalf("an operation with no answer replayed as %+v (held %t, %v)", result.Run, held, err)
	}
	if after, afterSequence := store.Head(); after != head || afterSequence != sequence {
		t.Fatal("asking for a replay the store does not hold published something")
	}
	result, err := served(runs, false).ExecuteProgram(ctx, failed, failedOperation, nil, program, inputs)
	if err != nil || result.Run.Outcome != runrecord.OutcomeSucceeded || len(result.NodeWalls) != 3 || runs["tokenize"] != 2 || runs["generate"] != 2 || runs["detokenize"] != 2 {
		t.Fatalf("serving the failed request again ran %v over %+v (%v); want every stage run a second time", runs, result.NodeWalls, err)
	}
}
