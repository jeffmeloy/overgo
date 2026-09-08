package runrecord

import (
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/overgodb"
	"overgo/internal/recipe"
	"overgo/internal/testutil"
)

const interactionTestNode recipe.NodeID = "respond"

func TestInteractionTerminalReservation(t *testing.T) {
	store, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	value := Interaction{Response: "reserved", Node: interactionTestNode,
		Recipe: testutil.ArtifactID(t, artifact.KindRecipe, "reserved-recipe"),
		Model:  testutil.ArtifactID(t, artifact.KindModel, "reserved-model")}
	if _, err := store.Commit(t.Context(), artifact.Batch{Key: "reserved-authority", Artifacts: []artifact.Descriptor{{ID: value.Recipe}}}); err != nil {
		t.Fatal(err)
	}
	prompt := []InteractionMessage{{Role: "user", Content: "original prompt"}}
	reserved, err := PublishInteraction(t.Context(), store, value, prompt, OutcomeInconclusive)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := PublishInteraction(t.Context(), store, value,
		[]InteractionMessage{{Role: "user", Content: "another prompt"}, {Role: "assistant", Content: "replacement"}}, OutcomeSucceeded); err == nil {
		t.Fatal("a different prompt took over the reserved response")
	}
	messages := append(prompt, InteractionMessage{Role: "assistant", Content: "partial"})
	if _, err := PublishInteraction(t.Context(), store, value, messages, Outcome("invented")); err == nil {
		t.Fatal("invalid outcome was published")
	}
	final, err := PublishInteraction(t.Context(), store, value, messages, OutcomeCancelled)
	if err != nil {
		t.Fatal(err)
	}
	if final.ID == reserved.ID {
		t.Fatal("terminal publication mutated the initial record")
	}
	repeated, err := PublishInteraction(t.Context(), store, value, messages, OutcomeCancelled)
	if err != nil || repeated.ID != final.ID {
		t.Fatalf("terminal publication is not idempotent: %s (%v)", repeated.ID, err)
	}
	trace, err := RequireInteractionTrace(t.Context(), store, final.Trace)
	if err != nil || trace.Terminal != OutcomeCancelled {
		t.Fatalf("terminal = %+v (%v)", trace, err)
	}
	if _, err := PublishInteraction(t.Context(), store, value, messages, OutcomeSucceeded); err == nil {
		t.Fatal("late completion overwrote a cancelled response")
	}
	current, found, err := ResolveInteraction(t.Context(), store, value.Response)
	if err != nil || !found || current.ID != final.ID {
		t.Fatal("terminal alias changed after a refused replacement")
	}
}

func TestInteractionIdentity(t *testing.T) {
	recipeID := testutil.ArtifactID(t, artifact.KindRecipe, "interaction-identity-recipe")
	modelID := testutil.ArtifactID(t, artifact.KindModel, "interaction-identity-model")
	messages := []InteractionMessage{{Role: "user", Content: "hello"}}
	first, err := NewInteractionTranscript(messages)
	if err != nil {
		t.Fatal(err)
	}
	second, err := NewInteractionTranscript(messages)
	if err != nil || first.ID != second.ID {
		t.Fatalf("transcript identity = (%s, %s, %v)", first.ID, second.ID, err)
	}
	interaction, err := NewInteraction(Interaction{
		Response: "resp_fixture", Recipe: recipeID, Model: modelID, Node: interactionTestNode, Message: first.ID,
		Trace: testutil.ArtifactID(t, artifact.KindEvidence, "interaction trace"),
	})
	if err != nil || interaction.ID.Kind() != artifact.KindEvidence {
		t.Fatalf("interaction identity = (%s, %v)", interaction.ID, err)
	}
}

func TestInteractionLineage(t *testing.T) {
	store, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	recipeID := testutil.ArtifactID(t, artifact.KindRecipe, "interaction-recipe")
	modelID := testutil.ArtifactID(t, artifact.KindModel, "interaction-model")
	parentID := testutil.ArtifactID(t, artifact.KindEvidence, "parent-interaction")
	if _, err := store.Commit(t.Context(), artifact.Batch{
		Key: "interaction/parents", Artifacts: []artifact.Descriptor{{ID: recipeID}, {ID: parentID}},
	}); err != nil {
		t.Fatal(err)
	}
	published, err := PublishInteraction(t.Context(), store, Interaction{
		Response: "resp_lineage", Recipe: recipeID, Model: modelID, Node: interactionTestNode, Parent: parentID,
	}, []InteractionMessage{{Role: "assistant", Content: "answer"}}, "")
	if err != nil {
		t.Fatal(err)
	}
	parents, err := store.Parents(t.Context(), published.ID)
	if err != nil || len(parents) != 3 {
		t.Fatalf("interaction parents = (%v, %v)", parents, err)
	}
	resolved, found, err := ResolveInteraction(t.Context(), store, published.Response)
	if err != nil || !found || resolved.ID != published.ID {
		t.Fatalf("resolved interaction = (%s, %v, %v)", resolved.ID, found, err)
	}
}

func TestCurrentTurnProjection(t *testing.T) {
	store, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	modelID := testutil.ArtifactID(t, artifact.KindModel, "interaction-projection-model")
	const (
		moduleID recipe.ModuleID = "interaction.respond"
		portName recipe.PortName = "messages"
	)
	catalog, err := recipe.NewCatalog(recipe.Module{
		ID: moduleID, Tasks: []recipe.Task{recipe.TaskInference}, Placements: []recipe.Placement{recipe.PlacementHost},
		Outputs: []recipe.Port{{Name: portName, Data: recipe.DataLogits, Cardinality: recipe.CardinalityOne}},
	})
	if err != nil {
		t.Fatal(err)
	}
	definition, err := recipe.NewDefinitionWithDependencies(
		recipe.TaskInference,
		[]recipe.Dependency{{Role: recipe.DependencyModel, Artifact: modelID}},
		[]recipe.Node{{ID: interactionTestNode, Module: moduleID, Placement: recipe.PlacementHost}}, nil, nil,
		[]recipe.Output{{Name: portName, Data: recipe.DataLogits, Source: recipe.Endpoint{Node: interactionTestNode, Port: portName}}},
	)
	if err != nil {
		t.Fatal(err)
	}
	program, err := recipe.CompileProgram(definition, catalog)
	if err != nil {
		t.Fatal(err)
	}
	scope, err := program.InteractionScope(interactionTestNode)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := store.Commit(t.Context(), artifact.Batch{
		Key: "interaction/projection/recipe", Artifacts: []artifact.Descriptor{{ID: definition.ID}},
	}); err != nil {
		t.Fatal(err)
	}
	rootMessages := []InteractionMessage{{Role: "user", Content: "root"}, {Role: "assistant", Content: "first"}}
	root, err := PublishInteraction(t.Context(), store, Interaction{
		Response: "resp_root", Recipe: definition.ID, Model: modelID, Node: interactionTestNode,
	}, rootMessages, "")
	if err != nil {
		t.Fatal(err)
	}
	turnMessages := []InteractionMessage{{Role: "user", Content: "child"}, {Role: "assistant", Content: "second"}}
	child, err := PublishInteraction(t.Context(), store, Interaction{
		Response: "resp_child", Recipe: definition.ID, Model: modelID, Node: interactionTestNode, Parent: root.ID,
	}, turnMessages, "")
	if err != nil {
		t.Fatal(err)
	}
	stored, err := RequireInteractionTranscript(t.Context(), store, child.Message)
	if err != nil || len(stored.Messages) != len(turnMessages) {
		t.Fatalf("stored current turn = (%+v, %v)", stored.Messages, err)
	}
	visible, err := VisibleInteractionMessages(t.Context(), store, scope, child)
	if err != nil || len(visible) != len(rootMessages)+len(turnMessages) || visible[0].Content != rootMessages[0].Content {
		t.Fatalf("visible history = (%+v, %v)", visible, err)
	}
}
