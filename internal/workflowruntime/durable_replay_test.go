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

// TestDurableLogReplayAcceptance pins: an execution interrupted after a
// memoised stage leaves that stage's completion in the attempt log under
// its invocation; a resumed execution opens a newer invocation, reads the
// memoised stage from the log instead of running it, completes the rest,
// and the interrupted invocation is stale from then on.
func TestDurableLogReplayAcceptance(t *testing.T) {
	ctx := t.Context()
	store, program := runtimeFixture(t)
	defer store.Close()
	operation := runtimeExecutionID(t, program, "runtime/durable-replay")
	first, err := NewForProgram(store, program)
	if err != nil {
		t.Fatal(err)
	}
	registerGenerationAdapters(t, first, true)
	prompt := fixtureContent(t, artifact.KindFile, "replay")
	inputs := map[recipe.PortName]Value{
		"prompt": {Kind: recipe.DataText, Items: []Datum{{Content: &prompt, Value: "replay"}}},
	}
	if _, err := first.ExecuteProgram(ctx, "runtime/durable-replay", operation, nil, program, inputs); err == nil {
		t.Fatal("interrupted workflow succeeded")
	}
	var tokenizeNode recipe.NodeID
	for _, node := range program.Definition().Nodes {
		if node.Module == workflowrecipe.ModuleTokenize {
			tokenizeNode = node.ID
		}
	}
	if tokenizeNode == "" {
		t.Fatal("fixture program has no tokenize stage")
	}
	interrupted := runrecord.DurableAttempt{Unit: operation, Invocation: 1}
	memo, found, err := interrupted.Lookup(ctx, store, runrecord.DurableMemo, string(tokenizeNode))
	if err != nil || !found || memo.Invocation != 1 || !memo.Result.Valid() {
		t.Fatalf("memoised tokenize stage = %+v found=%t, %v", memo, found, err)
	}
	receipt, found, err := runrecord.ResolveStageReceipt(ctx, store, operation, tokenizeNode)
	if err != nil || !found || receipt.ID != memo.Result || receipt.State != runrecord.StageCompleted {
		t.Fatalf("memo result is not the completed stage receipt: %+v found=%t, %v", receipt, found, err)
	}

	second, err := NewForProgram(store, program)
	if err != nil {
		t.Fatal(err)
	}
	tokenizeRuns := 0
	if err := second.Register(workflowrecipe.ModuleTokenize, AdapterFunc(func(context.Context, StepRequest) (map[recipe.PortName]Value, error) {
		tokenizeRuns++
		return nil, errors.New("memoised tokenize stage was recomputed")
	})); err != nil {
		t.Fatal(err)
	}
	if err := second.Register(workflowrecipe.ModuleGenerate, AdapterFunc(func(_ context.Context, request StepRequest) (map[recipe.PortName]Value, error) {
		return map[recipe.PortName]Value{"tokens": {Kind: recipe.DataTokens, Items: request.Inputs["tokens"].Items}}, nil
	})); err != nil {
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
	result, err := second.ExecuteProgram(ctx, "runtime/durable-replay", operation, nil, program, inputs)
	if err != nil || tokenizeRuns != 0 || result.Run.Outcome != runrecord.OutcomeSucceeded {
		t.Fatalf("resumed workflow = (runs=%d, outcome=%s, err=%v)", tokenizeRuns, result.Run.Outcome, err)
	}
	resumed := runrecord.DurableAttempt{Unit: operation, Invocation: 2}
	if replayed, found, err := resumed.Lookup(ctx, store, runrecord.DurableMemo, string(tokenizeNode)); err != nil || !found || replayed.ID != memo.ID {
		t.Fatalf("resumed invocation read a different memo: %+v found=%t, %v", replayed, found, err)
	}
	if _, _, err := interrupted.Lookup(ctx, store, runrecord.DurableMemo, string(tokenizeNode)); !errors.Is(err, runrecord.ErrStaleInvocation) {
		t.Fatalf("interrupted invocation still reads the log: %v", err)
	}
}
