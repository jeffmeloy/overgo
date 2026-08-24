package runrecord

import (
	"context"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/recipe"
	"overgo/internal/repodb"
	"overgo/internal/testutil"
)

const interactionTestNode recipe.NodeID = "respond"

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
	store, err := repodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	recipeID := testutil.ArtifactID(t, artifact.KindRecipe, "interaction-recipe")
	modelID := testutil.ArtifactID(t, artifact.KindModel, "interaction-model")
	parentID := testutil.ArtifactID(t, artifact.KindEvidence, "parent-interaction")
	if _, err := store.Commit(context.Background(), artifact.Batch{
		Key: "interaction/parents", Artifacts: []artifact.Descriptor{{ID: recipeID}, {ID: parentID}},
	}); err != nil {
		t.Fatal(err)
	}
	published, err := PublishInteraction(context.Background(), store, Interaction{
		Response: "resp_lineage", Recipe: recipeID, Model: modelID, Node: interactionTestNode, Parent: parentID,
	}, []InteractionMessage{{Role: "assistant", Content: "answer"}})
	if err != nil {
		t.Fatal(err)
	}
	parents, err := store.Parents(context.Background(), published.ID)
	if err != nil || len(parents) != 3 {
		t.Fatalf("interaction parents = (%v, %v)", parents, err)
	}
	resolved, found, err := ResolveInteraction(context.Background(), store, published.Response)
	if err != nil || !found || resolved.ID != published.ID {
		t.Fatalf("resolved interaction = (%s, %v, %v)", resolved.ID, found, err)
	}
}

func TestCurrentTurnProjection(t *testing.T) {
	store, err := repodb.Open(t.TempDir())
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
	if _, err := store.Commit(context.Background(), artifact.Batch{
		Key: "interaction/projection/recipe", Artifacts: []artifact.Descriptor{{ID: definition.ID}},
	}); err != nil {
		t.Fatal(err)
	}
	rootMessages := []InteractionMessage{{Role: "user", Content: "root"}, {Role: "assistant", Content: "first"}}
	root, err := PublishInteraction(context.Background(), store, Interaction{
		Response: "resp_root", Recipe: definition.ID, Model: modelID, Node: interactionTestNode,
	}, rootMessages)
	if err != nil {
		t.Fatal(err)
	}
	turnMessages := []InteractionMessage{{Role: "user", Content: "child"}, {Role: "assistant", Content: "second"}}
	child, err := PublishInteraction(context.Background(), store, Interaction{
		Response: "resp_child", Recipe: definition.ID, Model: modelID, Node: interactionTestNode, Parent: root.ID,
	}, turnMessages)
	if err != nil {
		t.Fatal(err)
	}
	stored, err := RequireInteractionTranscript(context.Background(), store, child.Message)
	if err != nil || len(stored.Messages) != len(turnMessages) {
		t.Fatalf("stored current turn = (%+v, %v)", stored.Messages, err)
	}
	visible, err := VisibleInteractionMessages(context.Background(), store, scope, child)
	if err != nil || len(visible) != len(rootMessages)+len(turnMessages) || visible[0].Content != rootMessages[0].Content {
		t.Fatalf("visible history = (%+v, %v)", visible, err)
	}
}
