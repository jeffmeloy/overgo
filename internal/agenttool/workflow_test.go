package agenttool

import (
	"encoding/json"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/recipe"
	"overgo/internal/testutil"
)

func TestCompileToolWorkflowUsesExactManualDependencies(t *testing.T) {
	inspect := workflowManual(t, "state.inspect", EffectInspection)
	mutate := workflowManual(t, "state.update", EffectMutation)
	model := testutil.ArtifactID(t, artifact.KindModel, "workflow-model")
	proposal := ToolWorkflowProposal{Model: model, Steps: []ToolWorkflowStep{
		{ID: "update", Manual: mutate.ID, After: []recipe.NodeID{"inspect"}},
		{ID: "inspect", Manual: inspect.ID},
	}}
	first, err := CompileToolWorkflow(proposal, []Manual{mutate, inspect})
	if err != nil {
		t.Fatal(err)
	}
	second, err := CompileToolWorkflow(proposal, []Manual{inspect, mutate})
	if err != nil {
		t.Fatal(err)
	}
	if first.Program().Definition().ID != second.Program().Definition().ID ||
		len(first.Program().Stages()) != 2 || first.Program().Stages()[0].Node.ID != "inspect" {
		t.Fatalf("compiled workflows differ: %+v %+v", first.Program().Definition(), second.Program().Definition())
	}
	calls, err := first.Calls(map[recipe.NodeID]json.RawMessage{"inspect": json.RawMessage(`{}`), "update": json.RawMessage(`{}`)})
	if err != nil || len(calls) != 2 {
		t.Fatalf("calls = %+v, %v", calls, err)
	}
}

func TestCompileToolWorkflowRefusesUngroundedMutation(t *testing.T) {
	mutate := workflowManual(t, "state.update", EffectMutation)
	model := testutil.ArtifactID(t, artifact.KindModel, "workflow-model")
	if _, err := CompileToolWorkflow(ToolWorkflowProposal{Model: model, Steps: []ToolWorkflowStep{
		{ID: "update", Manual: mutate.ID},
	}}, []Manual{mutate}); err == nil {
		t.Fatal("mutation without inspection compiled")
	}
}

func workflowManual(t *testing.T, name string, effect Effect) Manual {
	t.Helper()
	manual, err := NewManual(Manual{
		Name: name, Description: "Workflow test tool.", Effect: effect,
		Transport: Transport{Kind: TransportHTTP, URL: "https://example.test/" + name},
	})
	if err != nil {
		t.Fatal(err)
	}
	return manual
}
