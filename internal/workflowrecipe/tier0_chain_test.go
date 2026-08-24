// Package workflowrecipe_test verifies public recipe execution contracts.
package workflowrecipe_test

import (
	"context"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/overgodb"
	"overgo/internal/recipe"
	"overgo/internal/runrecord"
	"overgo/internal/testutil"
	"overgo/internal/workflowrecipe"
	"overgo/internal/workflowruntime"
)

// TestTier0ChainExecutesThroughTypedPorts pins the Tier-0 chain contract: one
// content-addressed recipe artifact composes two validated models by wiring
// the first model's text output port into the second model's text input port
// -- typed data kinds end to end, no latent bridging -- and the chain executes
// through the generic workflow runtime with every node dispatched against the
// model its declared slot resolves to.
func TestTier0ChainExecutesThroughTypedPorts(t *testing.T) {
	ctx := context.Background()
	modelA := testutil.ArtifactID(t, artifact.KindModel, "chain-model-a")
	modelB := testutil.ArtifactID(t, artifact.KindModel, "chain-model-b")
	definition := chainDefinition(t, modelA, modelB)
	if definition.Model != modelA {
		t.Fatalf("primary model = %v, want slot-0 model %v", definition.Model, modelA)
	}
	replay := chainDefinition(t, modelA, modelB)
	if definition.ID != replay.ID {
		t.Fatalf("chain artifact is not content-addressed: %v != %v", definition.ID, replay.ID)
	}
	program, err := recipe.CompileProgram(definition, workflowrecipe.Catalog())
	if err != nil {
		t.Fatal(err)
	}

	store, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	if _, err := store.Commit(ctx, artifact.Batch{
		Key:       "tier0/chain-models",
		Artifacts: []artifact.Descriptor{{ID: modelA}, {ID: modelB}},
	}); err != nil {
		t.Fatal(err)
	}
	runtime, err := workflowruntime.NewForProgram(store, program)
	if err != nil {
		t.Fatal(err)
	}
	seen := map[recipe.ModuleID][]artifact.ID{}
	register := func(
		module recipe.ModuleID,
		port recipe.PortName,
		adapter func(workflowruntime.StepRequest) workflowruntime.Value,
	) {
		err := runtime.Register(module, workflowruntime.AdapterFunc(func(
			_ context.Context, request workflowruntime.StepRequest,
		) (map[recipe.PortName]workflowruntime.Value, error) {
			seen[module] = append(seen[module], request.Model)
			return map[recipe.PortName]workflowruntime.Value{port: adapter(request)}, nil
		}))
		if err != nil {
			t.Fatal(err)
		}
	}
	register(workflowrecipe.ModuleTokenize, "tokens", func(request workflowruntime.StepRequest) workflowruntime.Value {
		return workflowruntime.Value{Kind: recipe.DataTokens, Items: request.Inputs["text"].Items}
	})
	register(workflowrecipe.ModuleGenerate, "tokens", func(request workflowruntime.StepRequest) workflowruntime.Value {
		return workflowruntime.Value{Kind: recipe.DataTokens, Items: request.Inputs["tokens"].Items}
	})
	register(workflowrecipe.ModuleDetokenize, "text", func(request workflowruntime.StepRequest) workflowruntime.Value {
		datum := request.Inputs["tokens"].Items[0]
		text := datum.Value.(string) + "+model"
		content := artifact.Content{
			Descriptor: artifact.Descriptor{
				ID:        testutil.ArtifactID(t, artifact.KindOutput, text),
				Size:      uint64(len(text)),
				MediaType: "text/plain",
			},
			Data: []byte(text),
		}
		return workflowruntime.ArtifactValue(recipe.DataText, text, content)
	})

	operation, err := workflowruntime.ExecutionID(definition.ID, "tier0/chain")
	if err != nil {
		t.Fatal(err)
	}
	result, err := runtime.ExecuteProgram(ctx, "tier0/chain", operation, nil, program, map[recipe.PortName]workflowruntime.Value{
		"prompt": {Kind: recipe.DataText, Items: []workflowruntime.Datum{{Value: "prompt"}}},
	})
	if err != nil {
		t.Fatal(err)
	}
	if result.Run.Outcome != runrecord.OutcomeSucceeded || !result.Commit.Valid() {
		t.Fatalf("chain run = %+v", result.Run)
	}
	text, ok := result.Outputs["text"].Single()
	if !ok || text.Value != "prompt+model+model" {
		t.Fatalf("chain output = (%+v, %v), want text traversing both models", text, ok)
	}
	for module, models := range seen {
		if len(models) != 2 || models[0] != modelA || models[1] != modelB {
			t.Fatalf("module %q dispatched against %v, want [%v %v]", module, models, modelA, modelB)
		}
	}

	nodes := replay.Nodes
	for index := range nodes {
		if nodes[index].ID == "b-generate" {
			nodes[index].ModelSlot = 7
		}
	}
	if _, err := recipe.NewDefinitionWithDependencies(
		recipe.TaskGeneration, replay.Dependencies, nodes, replay.Edges, replay.Inputs, replay.Outputs,
	); err == nil {
		t.Fatal("node bound to an absent model slot was accepted")
	}
}

// chainDefinition composes two models in one recipe: model A's generation
// text feeds model B's tokenizer over a typed text edge.
func chainDefinition(t *testing.T, modelA, modelB artifact.ID) recipe.Definition {
	t.Helper()
	node := func(id recipe.NodeID, module recipe.ModuleID, slot uint32) recipe.Node {
		return recipe.Node{ID: id, Module: module, Placement: recipe.PlacementHost, ModelSlot: slot}
	}
	edge := func(from recipe.NodeID, fromPort recipe.PortName, to recipe.NodeID, toPort recipe.PortName) recipe.Edge {
		return recipe.Edge{
			From: recipe.Endpoint{Node: from, Port: fromPort},
			To:   recipe.Endpoint{Node: to, Port: toPort},
		}
	}
	definition, err := recipe.NewDefinitionWithDependencies(
		recipe.TaskGeneration,
		[]recipe.Dependency{
			{Role: recipe.DependencyModel, Slot: 0, Artifact: modelA},
			{Role: recipe.DependencyModel, Slot: 1, Artifact: modelB},
		},
		[]recipe.Node{
			node("a-tokenize", workflowrecipe.ModuleTokenize, 0),
			node("a-generate", workflowrecipe.ModuleGenerate, 0),
			node("a-detokenize", workflowrecipe.ModuleDetokenize, 0),
			node("b-tokenize", workflowrecipe.ModuleTokenize, 1),
			node("b-generate", workflowrecipe.ModuleGenerate, 1),
			node("b-detokenize", workflowrecipe.ModuleDetokenize, 1),
		},
		[]recipe.Edge{
			edge("a-tokenize", "tokens", "a-generate", "tokens"),
			edge("a-generate", "tokens", "a-detokenize", "tokens"),
			edge("a-detokenize", "text", "b-tokenize", "text"),
			edge("b-tokenize", "tokens", "b-generate", "tokens"),
			edge("b-generate", "tokens", "b-detokenize", "tokens"),
		},
		[]recipe.Input{{
			Name: "prompt", Data: recipe.DataText,
			Target: recipe.Endpoint{Node: "a-tokenize", Port: "text"},
		}},
		[]recipe.Output{{
			Name: "text", Data: recipe.DataText,
			Source: recipe.Endpoint{Node: "b-detokenize", Port: "text"},
		}},
	)
	if err != nil {
		t.Fatal(err)
	}
	return definition
}
