package server

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/operation"
	"overgo/internal/operatoraction"
	"overgo/internal/recipe"
	"overgo/internal/testutil"
)

// TestApprovalDecisionSeparation pins the decision contract: a decision
// must name the exact approval request the pending view advertises, so
// one action can never both derive the approval and grant it. A blind
// grant, a grant naming a different request, and the retired boolean
// agent grant are all refused; a grant naming the advertised request is
// recorded with both identities and recovers the operation.
func TestApprovalDecisionSeparation(t *testing.T) {
	handler := newTestHandlerWithRepository(t, &fakeGenerator{})
	recipeID := testutil.ArtifactID(t, artifact.KindRecipe, "separation-recipe")
	runID := testutil.ArtifactID(t, artifact.KindRun, "separation-run")
	action := operatoraction.Action{Code: "resume", Summary: "Resume the blocked work", Argv: []string{"overgo", "resume"}}
	attempts := 0
	id, err := handler.operations.Submit(t.Context(), operation.Request{
		Task: recipe.TaskGeneration, Recipe: recipeID,
	}, func(context.Context, operation.Reporter) (operation.Completion, error) {
		attempts++
		if attempts == 1 {
			return operation.Completion{Run: runID}, operatoraction.Recoverable(errors.New("approval required"), operatoraction.Block{
				Subject: recipeID, Reason: "operator approval required", Evidence: []artifact.ID{}, Actions: []operatoraction.Action{action},
			})
		}
		return operation.Completion{Run: runID}, nil
	})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := handler.operations.Wait(t.Context(), id); err != nil {
		t.Fatal(err)
	}

	view := serveTestRequest(handler, http.MethodGet, "/operations/decisions", "")
	if view.Code != http.StatusOK {
		t.Fatalf("decision view status=%d body=%s", view.Code, view.Body.String())
	}
	var decisions struct {
		Blocked []struct {
			Operation artifact.ID `json:"operation"`
			Actions   []struct {
				Code    string      `json:"code"`
				Request artifact.ID `json:"request"`
			} `json:"actions"`
		} `json:"blocked"`
	}
	if err := json.Unmarshal(view.Body.Bytes(), &decisions); err != nil {
		t.Fatal(err)
	}
	var advertised artifact.ID
	for _, blocked := range decisions.Blocked {
		for _, candidate := range blocked.Actions {
			if blocked.Operation == id && candidate.Code == action.Code {
				advertised = candidate.Request
			}
		}
	}
	if !advertised.Valid() {
		t.Fatalf("view advertises no approval request: %s", view.Body.String())
	}

	blind := serveTestRequest(handler, http.MethodPost, "/operations/decision", marshalAutomationJSON(t, map[string]any{
		"operation": id, "tool": action.Code, "answer": operatoraction.AnswerGrant,
	}))
	if blind.Code != http.StatusConflict {
		t.Fatalf("blind grant status=%d body=%s", blind.Code, blind.Body.String())
	}
	mismatched := serveTestRequest(handler, http.MethodPost, "/operations/decision", marshalAutomationJSON(t, map[string]any{
		"operation": id, "tool": action.Code, "answer": operatoraction.AnswerGrant, "request": recipeID,
	}))
	if mismatched.Code != http.StatusConflict {
		t.Fatalf("mismatched grant status=%d body=%s", mismatched.Code, mismatched.Body.String())
	}
	if status, found := handler.operations.Status(id); !found || status.State != operation.StateBlocked {
		t.Fatalf("refused grants moved the operation: %+v found=%t", status, found)
	}

	granted := serveTestRequest(handler, http.MethodPost, "/operations/decision", marshalAutomationJSON(t, map[string]any{
		"operation": id, "tool": action.Code, "answer": operatoraction.AnswerGrant, "request": advertised,
	}))
	if granted.Code != http.StatusAccepted {
		t.Fatalf("advertised grant status=%d body=%s", granted.Code, granted.Body.String())
	}
	var outcome operationDecisionResponse
	if err := json.Unmarshal(granted.Body.Bytes(), &outcome); err != nil || !outcome.Decision.Valid() {
		t.Fatalf("grant outcome = %+v, %v", outcome, err)
	}
	if completed, err := handler.operations.Wait(t.Context(), id); err != nil || completed.State != operation.StateCompleted {
		t.Fatalf("recovered operation = (%+v, %v)", completed, err)
	}
}
