package workflowruntime

import (
	"encoding/json"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/recipe"
	"overgo/internal/runrecord"
)

// earlierStageReceiptSchema is the schema stores hold the stage receipts of
// requests served before a receipt could bind an invocation.
const earlierStageReceiptSchema = "overgo/stage-receipt/v1"

// TestOperationRunsOnFromAnEarlierFormReceipt serves a request whose earlier
// serving left its stage receipts in a store under the contract before
// invocation bindings: two stages completed and the last one failed. The
// repeat was refused before any stage ran, the receipts unreadable. It runs:
// each stage takes its next attempt from the earlier receipt, the chain of
// receipts goes back to it, and the request is answered.
func TestOperationRunsOnFromAnEarlierFormReceipt(t *testing.T) {
	ctx := t.Context()
	store, program := runtimeFixture(t)
	defer store.Close()
	const key = "runtime/earlier-receipts"
	definition := program.Definition()
	operation := runtimeExecutionID(t, program, key)
	prompt := fixtureContent(t, artifact.KindFile, "hello")
	earlier := map[recipe.NodeID]artifact.ID{}
	for _, written := range []runrecord.StageReceipt{
		{Node: "tokenize", State: runrecord.StageCompleted,
			Inputs:  []runrecord.StageBinding{{Port: "text", Artifacts: []artifact.ID{prompt.Descriptor.ID}}},
			Outputs: []runrecord.StageBinding{{Port: "tokens"}}},
		{Node: "generate", State: runrecord.StageCompleted,
			Inputs: []runrecord.StageBinding{{Port: "tokens"}}, Outputs: []runrecord.StageBinding{{Port: "tokens"}}},
		{Node: "detokenize", State: runrecord.StageFailed, Failure: executionFailureCode,
			Inputs: []runrecord.StageBinding{{Port: "tokens"}}},
	} {
		written.Recipe, written.Operation, written.Attempt = definition.ID, operation, 1
		receipt, err := runrecord.NewStageReceipt(written)
		if err != nil {
			t.Fatal(err)
		}
		data, err := json.Marshal(receipt)
		if err != nil {
			t.Fatal(err)
		}
		content, err := artifact.DocumentContract{
			Kind: artifact.KindEvidence, MediaType: runrecord.StageReceiptMediaType, Schema: earlierStageReceiptSchema,
		}.ContentBytes(data)
		if err != nil {
			t.Fatal(err)
		}
		if content.Descriptor.ID != receipt.ID {
			t.Fatalf("the %s receipt's bytes are %s, the receipt %s; the fixture must write a receipt's own bytes", written.Node, content.Descriptor.ID, receipt.ID)
		}
		if _, err := store.Commit(ctx, artifact.Batch{
			Key: "test/earlier-receipt/" + receipt.ID.String(), Contents: []artifact.Content{content},
			Aliases: []artifact.AliasBinding{artifact.AliasMove(runrecord.StageReceiptAliasRoot+operation.String()+"/"+string(written.Node), receipt.ID, artifact.ID{})},
		}); err != nil {
			t.Fatal(err)
		}
		earlier[written.Node] = receipt.ID
	}

	runtime, err := NewForProgram(store, program)
	if err != nil {
		t.Fatal(err)
	}
	registerGenerationAdapters(t, runtime, false)
	result, err := runtime.ExecuteProgram(ctx, key, operation, nil, program, map[recipe.PortName]Value{
		"prompt": {Kind: recipe.DataText, Items: []Datum{{Content: &prompt, Value: "hello"}}},
	})
	answer := fixtureContent(t, artifact.KindOutput, "HELLO")
	if err != nil || result.Run.Outcome != runrecord.OutcomeSucceeded || len(result.Run.Outputs) != 1 || result.Run.Outputs[0] != answer.Descriptor.ID || len(result.NodeWalls) != len(earlier) {
		t.Fatalf("the repeated request gave run %+v over nodes %+v (%v); want it served, every stage run, its output %s", result.Run, result.NodeWalls, err, answer.Descriptor.ID)
	}
	for node, first := range earlier {
		receipt, found, err := runrecord.ResolveStageReceipt(ctx, store, operation, node)
		if err != nil || !found || receipt.Attempt != 2 || receipt.State != runrecord.StageCompleted {
			t.Fatalf("%s ends at %+v (found %t, %v); want its second attempt, completed", node, receipt, found, err)
		}
		// Completed, running and admitted are the second attempt's three receipts; before them stands the earlier one.
		for range 3 {
			content, held, err := artifact.ReadContent(ctx, store, receipt.Previous)
			if err != nil || !held {
				t.Fatalf("%s: the receipt before %s is not in the store (%v)", node, receipt.ID, err)
			}
			if receipt, err = runrecord.ParseStageReceipt(content.Data); err != nil {
				t.Fatal(err)
			}
		}
		if receipt.ID != first || receipt.Attempt != 1 {
			t.Fatalf("%s: the second attempt's receipts go back to %s, attempt %d; want the earlier receipt %s", node, receipt.ID, receipt.Attempt, first)
		}
	}
}
