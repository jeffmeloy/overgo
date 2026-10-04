package server

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"strings"
	"sync/atomic"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/operation"
	"overgo/internal/operatoraction"
	"overgo/internal/recipe"
	"overgo/internal/testutil"
)

// TestApprovalDecisionSeparation pins the decision contract and the
// pending-work projection: a blocked operation appears in the inbox with its
// reason and advertised actions, and a decision must name the exact approval
// request the pending view advertises, so one action can never both derive
// the approval and grant it. A blind grant and a grant naming a different
// request are refused; a grant naming the advertised request is recorded,
// recovers the operation, leaves the inbox, shows in the operation's
// evidence, and cannot be replayed.
func TestApprovalDecisionSeparation(t *testing.T) {
	t.Parallel()
	handler := newTestHandlerWithRepository(t, &fakeGenerator{})
	empty := serveTestRequest(handler, http.MethodGet, "/operations/inbox", "")
	if empty.Code != http.StatusOK || !strings.Contains(empty.Body.String(), `"waiting":[]`) {
		t.Fatalf("empty inbox status=%d body=%s", empty.Code, empty.Body.String())
	}
	recipeID := testutil.ArtifactID(t, artifact.KindRecipe, "separation-recipe")
	runID := testutil.ArtifactID(t, artifact.KindRun, "separation-run")
	action := operatoraction.Action{Code: "resume", Summary: "Resume the blocked work", Argv: []string{"overgo", "resume"}}
	var attempts atomic.Int32
	id, err := handler.operations.Submit(t.Context(), operation.Request{
		Task: recipe.TaskGeneration, Recipe: recipeID,
	}, func(context.Context, operation.Reporter) (operation.Completion, error) {
		if attempts.Add(1) == 1 {
			return operation.Completion{Run: runID}, operatoraction.Recoverable(errors.New("approval required"), operatoraction.Block{
				Subject: recipeID, Reason: "operator approval required", Evidence: []artifact.ID{}, Actions: []operatoraction.Action{action},
			})
		}
		return operation.Completion{Run: runID}, nil
	})
	if err != nil {
		t.Fatal(err)
	}
	if blocked, err := handler.operations.Wait(t.Context(), id); err != nil || blocked.State != operation.StateBlocked {
		t.Fatalf("blocked operation=(%+v, %v)", blocked, err)
	}
	waiting := serveTestRequest(handler, http.MethodGet, "/operations/inbox", "")
	if waiting.Code != http.StatusOK || !strings.Contains(waiting.Body.String(), id.String()) ||
		!strings.Contains(waiting.Body.String(), "operator approval required") || !strings.Contains(waiting.Body.String(), `"resume"`) {
		t.Fatalf("waiting inbox status=%d body=%s", waiting.Code, waiting.Body.String())
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

	blind := serveTestRequest(handler, http.MethodPost, "/operations/decision", marshalTestJSON(t, map[string]any{
		"operation": id, "tool": action.Code, "answer": operatoraction.AnswerGrant,
	}))
	if blind.Code != http.StatusConflict {
		t.Fatalf("blind grant status=%d body=%s", blind.Code, blind.Body.String())
	}
	mismatched := serveTestRequest(handler, http.MethodPost, "/operations/decision", marshalTestJSON(t, map[string]any{
		"operation": id, "tool": action.Code, "answer": operatoraction.AnswerGrant, "request": recipeID,
	}))
	if mismatched.Code != http.StatusConflict {
		t.Fatalf("mismatched grant status=%d body=%s", mismatched.Code, mismatched.Body.String())
	}
	if status, found := handler.operations.Status(id); !found || status.State != operation.StateBlocked {
		t.Fatalf("refused grants moved the operation: %+v found=%t", status, found)
	}

	grant := marshalTestJSON(t, map[string]any{
		"operation": id, "tool": action.Code, "answer": operatoraction.AnswerGrant, "request": advertised,
	})
	granted := serveTestRequest(handler, http.MethodPost, "/operations/decision", grant)
	if granted.Code != http.StatusAccepted {
		t.Fatalf("advertised grant status=%d body=%s", granted.Code, granted.Body.String())
	}
	var outcome operationDecisionResponse
	if err := json.Unmarshal(granted.Body.Bytes(), &outcome); err != nil || !outcome.Decision.Valid() {
		t.Fatalf("grant outcome = %+v, %v", outcome, err)
	}
	if completed, err := handler.operations.Wait(t.Context(), id); err != nil || completed.State != operation.StateCompleted || attempts.Load() != 2 {
		t.Fatalf("recovered operation = (%+v, %v), attempts=%d", completed, err, attempts.Load())
	}
	if replayed := serveTestRequest(handler, http.MethodPost, "/operations/decision", grant); replayed.Code != http.StatusConflict {
		t.Fatalf("replay status=%d body=%s", replayed.Code, replayed.Body.String())
	}
	if drained := serveTestRequest(handler, http.MethodGet, "/operations/inbox", ""); drained.Code != http.StatusOK || strings.Contains(drained.Body.String(), id.String()) {
		t.Fatalf("recovered operation still in inbox: %s", drained.Body.String())
	}
	if projection := serveTestRequest(handler, http.MethodGet, "/operations/evidence?id="+id.String(), ""); projection.Code != http.StatusOK || !strings.Contains(projection.Body.String(), `"answer":"grant"`) {
		t.Fatalf("recovery evidence status=%d body=%s", projection.Code, projection.Body.String())
	}
}
