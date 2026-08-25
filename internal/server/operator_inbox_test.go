package server

import (
	"context"
	"errors"
	"net/http"
	"strings"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/operation"
	"overgo/internal/operatoraction"
	"overgo/internal/recipe"
	"overgo/internal/testutil"
)

// TestOperatorInbox pins the pending-work projection: a blocked
// operation appears in the inbox with its reason and advertised
// actions, granting through the decision endpoint recovers it, and the
// recovered operation leaves the inbox -- one list answers "what is
// waiting on me", wired to the durable decision chain.
func TestOperatorInbox(t *testing.T) {
	handler := newTestHandlerWithRepository(t, &fakeGenerator{})
	empty := serveTestRequest(handler, http.MethodGet, "/operations/inbox", "")
	if empty.Code != http.StatusOK || !strings.Contains(empty.Body.String(), `"waiting":[]`) {
		t.Fatalf("empty inbox status=%d body=%s", empty.Code, empty.Body.String())
	}
	recipeID := testutil.ArtifactID(t, artifact.KindRecipe, "inbox-recipe")
	runID := testutil.ArtifactID(t, artifact.KindRun, "inbox-run")
	action := operatoraction.Action{Code: "resume", Summary: "Resume the blocked work", Argv: []string{"overgo", "resume"}}
	attempts := 0
	id, err := handler.operations.Submit(context.Background(), operation.Request{
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
	if _, err := handler.operations.Wait(context.Background(), id); err != nil {
		t.Fatal(err)
	}
	waiting := serveTestRequest(handler, http.MethodGet, "/operations/inbox", "")
	if waiting.Code != http.StatusOK ||
		!strings.Contains(waiting.Body.String(), id.String()) ||
		!strings.Contains(waiting.Body.String(), "operator approval required") ||
		!strings.Contains(waiting.Body.String(), `"resume"`) {
		t.Fatalf("waiting inbox status=%d body=%s", waiting.Code, waiting.Body.String())
	}
	grant := serveTestRequest(handler, http.MethodPost, "/operations/decision", marshalAutomationJSON(t, map[string]any{
		"operation": id, "tool": action.Code, "answer": operatoraction.AnswerGrant,
	}))
	if grant.Code != http.StatusAccepted {
		t.Fatalf("grant status=%d body=%s", grant.Code, grant.Body.String())
	}
	if _, err := handler.operations.Wait(context.Background(), id); err != nil {
		t.Fatal(err)
	}
	drained := serveTestRequest(handler, http.MethodGet, "/operations/inbox", "")
	if drained.Code != http.StatusOK || strings.Contains(drained.Body.String(), id.String()) {
		t.Fatalf("recovered operation still in inbox: %s", drained.Body.String())
	}
}
