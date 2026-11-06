package server

import (
	"encoding/json"
	"net/http"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/operatoraction"
	"overgo/internal/overgodbtest"
	"overgo/internal/runrecord"
	"overgo/internal/testcheck"
	"overgo/internal/testutil"
)

// TestOperatorDecisionInteractionContract pins the operator's two surfaces:
// one pending-decision view listing committed approval requests with their
// exact action surface, and one causal timeline citing canonical evidence in
// introduction order. Deciding through the existing decision authority
// empties the view; the endpoints add no decision authority of their own.
func TestOperatorDecisionInteractionContract(t *testing.T) {
	t.Parallel()
	root := t.TempDir()
	store := overgodbtest.Create(t, root)
	ctx := t.Context()
	operationID := testutil.ArtifactID(t, artifact.KindEvidence, "server-decision-operation")
	recipeID := testutil.ArtifactID(t, artifact.KindRecipe, "server-decision-recipe")
	request := testcheck.Must(operatoraction.NewApprovalRequest(operationID, recipeID, operatoraction.Action{
		Code: "rollback", Argv: []string{"--target", "prior"},
	}, artifact.ID{}))(t)
	requestContent := testcheck.Must(request.Content())(t)
	if _, err := store.Commit(ctx, artifact.Batch{
		Key:       "operator/approval/" + request.ID.String(),
		Artifacts: []artifact.Descriptor{{ID: operationID}, {ID: recipeID}},
		Contents:  []artifact.Content{requestContent},
	}); err != nil {
		t.Fatal(err)
	}
	if _, err := runrecord.PublishStageReceipt(ctx, store, runrecord.StageReceipt{
		Recipe: recipeID, Node: "respond", Operation: operationID,
		Attempt: 1, State: runrecord.StageAdmitted,
	}, nil, nil); err != nil {
		t.Fatal(err)
	}
	handler := testcheck.Must(New(Config{ModelID: testModelID, MaxTokens: testMaxTokens, OvergoDBPath: root}, &fakeGenerator{}))(t)
	t.Cleanup(func() { _ = handler.Close() })

	response := serveTestRequest(handler, http.MethodGet, "/operations/decisions", "")
	if response.Code != http.StatusOK {
		t.Fatalf("status=%d body=%s", response.Code, response.Body.String())
	}
	var view struct {
		Approvals []runrecord.PendingOperatorDecision `json:"approvals"`
		Blocked   []json.RawMessage                   `json:"blocked"`
	}
	testcheck.NoError(t, json.Unmarshal(response.Body.Bytes(), &view))
	if len(view.Approvals) != 1 || view.Approvals[0].Request != request.ID ||
		view.Approvals[0].Tool != "rollback" || len(view.Blocked) != 0 {
		t.Fatalf("pending-decision view = %+v", view)
	}

	decision := testcheck.Must(runrecord.NewHumanDecision(request, operatoraction.AnswerDecline))(t)
	testcheck.NoError(t, runrecord.PublishHumanDecision(ctx, store, request, decision))
	response = serveTestRequest(handler, http.MethodGet, "/operations/decisions", "")
	if response.Code != http.StatusOK {
		t.Fatalf("status=%d body=%s", response.Code, response.Body.String())
	}
	testcheck.NoError(t, json.Unmarshal(response.Body.Bytes(), &view))
	if len(view.Approvals) != 0 {
		t.Fatalf("decided request stayed in the view: %+v", view.Approvals)
	}

	response = serveTestRequest(handler, http.MethodGet, "/operations/timeline?id="+operationID.String(), "")
	if response.Code != http.StatusOK {
		t.Fatalf("status=%d body=%s", response.Code, response.Body.String())
	}
	var timeline struct {
		Operation artifact.ID                       `json:"operation"`
		Events    []runrecord.OperatorTimelineEvent `json:"events"`
	}
	testcheck.NoError(t, json.Unmarshal(response.Body.Bytes(), &timeline))
	if timeline.Operation != operationID || len(timeline.Events) != 2 ||
		timeline.Events[0].Kind != "stage-receipt" || timeline.Events[1].Kind != "human-decision" ||
		timeline.Events[0].Sequence >= timeline.Events[1].Sequence {
		t.Fatalf("timeline = %+v", timeline)
	}
}
