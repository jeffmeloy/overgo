package server

import (
	"net/http"
	"strings"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/recipe"
	"overgo/internal/runrecord"
	"overgo/internal/testutil"
)

// TestWorkflowDAG pins the live DAG projection: the recipe graph joins
// the durable stage receipts, a node with a receipt carries its
// lifecycle state and attempt, a node without one is pending, and the
// edges are the definition's own. An operation with neither receipts
// nor live status is a typed not-found.
func TestWorkflowDAG(t *testing.T) {
	handler := newTestHandlerWithRepository(t, &fakeGenerator{})
	store := handler.repository
	ctx := t.Context()

	moduleFirst := recipe.ModuleID("dag.first")
	moduleSecond := recipe.ModuleID("dag.second")
	port := recipe.PortName("out")
	modelID := testutil.ArtifactID(t, artifact.KindModel, "dag-model")
	definition, err := recipe.NewDefinitionWithDependencies(
		recipe.TaskInference,
		[]recipe.Dependency{{Role: recipe.DependencyModel, Artifact: modelID}},
		[]recipe.Node{
			{ID: "first", Module: moduleFirst, Placement: recipe.PlacementHost},
			{ID: "second", Module: moduleSecond, Placement: recipe.PlacementHost},
		},
		[]recipe.Edge{{From: recipe.Endpoint{Node: "first", Port: port}, To: recipe.Endpoint{Node: "second", Port: port}}},
		nil,
		[]recipe.Output{{Name: port, Data: recipe.DataLogits, Source: recipe.Endpoint{Node: "second", Port: port}}},
	)
	if err != nil {
		t.Fatal(err)
	}
	definitionContent, err := definition.ArtifactContent()
	if err != nil {
		t.Fatal(err)
	}
	operationID := testutil.ArtifactID(t, artifact.KindEvidence, "dag-operation")
	if _, err := artifact.CommitBatch(ctx, store, artifact.Batch{
		Key:       "workflow-dag-fixture",
		Contents:  []artifact.Content{definitionContent},
		Artifacts: []artifact.Descriptor{{ID: modelID}, {ID: operationID}},
	}); err != nil {
		t.Fatal(err)
	}
	base := runrecord.StageReceipt{
		Recipe: definition.ID, Node: "first", Operation: operationID,
		Attempt: 1, State: runrecord.StageAdmitted,
	}
	if _, err := runrecord.PublishStageReceipt(ctx, store, base, nil, nil); err != nil {
		t.Fatal(err)
	}
	base.State = runrecord.StageRunning
	if _, err := runrecord.PublishStageReceipt(ctx, store, base, nil, nil); err != nil {
		t.Fatal(err)
	}
	base.State = runrecord.StageWaiting
	if _, err := runrecord.PublishStageReceipt(ctx, store, base, nil, nil); err != nil {
		t.Fatal(err)
	}

	dag := serveTestRequest(handler, http.MethodGet, "/operations/dag?id="+operationID.String(), "")
	body := dag.Body.String()
	if dag.Code != http.StatusOK ||
		!strings.Contains(body, `"id":"first"`) || !strings.Contains(body, `"state":"waiting"`) ||
		!strings.Contains(body, `"id":"second"`) || !strings.Contains(body, `"state":"pending"`) ||
		!strings.Contains(body, `"from":"first"`) || !strings.Contains(body, `"to":"second"`) {
		t.Fatalf("dag status=%d body=%s", dag.Code, body)
	}

	absent := testutil.ArtifactID(t, artifact.KindEvidence, "dag-unknown")
	missing := serveTestRequest(handler, http.MethodGet, "/operations/dag?id="+absent.String(), "")
	if missing.Code != http.StatusNotFound {
		t.Fatalf("unknown operation status=%d body=%s", missing.Code, missing.Body.String())
	}
}
