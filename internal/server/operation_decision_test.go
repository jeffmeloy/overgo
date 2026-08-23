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
	"overgo/internal/repodb"
	"overgo/internal/testutil"
)

func TestDecisionAPI(t *testing.T) {
	store, err := repodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { store.Close() })
	handler, err := New(Config{Repository: store}, &fakeGenerator{})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { handler.Close() })
	recipeID := testutil.ArtifactID(t, artifact.KindRecipe, "decision api recipe")
	runID := testutil.ArtifactID(t, artifact.KindRun, "decision api run")
	action := operatoraction.Action{Code: "resume", Summary: "Resume exact work", Argv: []string{"overgo", "resume"}}
	attempts := 0
	operationID, err := handler.operations.Submit(context.Background(), operation.Request{
		Task: recipe.TaskTraining, Recipe: recipeID,
	}, func(context.Context, operation.Reporter) (operation.Completion, error) {
		attempts++
		if attempts == 1 {
			return operation.Completion{Run: runID}, operatoraction.Recoverable(errors.New("approval required"), operatoraction.Block{
				Subject: recipeID, Reason: "operator approval required", Evidence: []artifact.ID{runID},
				Actions: []operatoraction.Action{action},
			})
		}
		return operation.Completion{Run: runID}, nil
	})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := handler.operations.Wait(context.Background(), operationID); err != nil {
		t.Fatal(err)
	}
	body, err := json.Marshal(operationDecisionRequest{
		Operation: operationID, Tool: action.Code, Answer: operatoraction.AnswerGrant,
	})
	if err != nil {
		t.Fatal(err)
	}
	response := serveTestRequest(handler, http.MethodPost, "/operations/decision", string(body))
	if response.Code != http.StatusAccepted {
		t.Fatalf("decision status=%d body=%s", response.Code, response.Body.String())
	}
	status, err := handler.operations.Wait(context.Background(), operationID)
	if err != nil || status.State != operation.StateCompleted || attempts != 2 {
		t.Fatalf("decision recovery = (%+v, %v), attempts=%d", status, err, attempts)
	}
	replayed := serveTestRequest(handler, http.MethodPost, "/operations/decision", string(body))
	if replayed.Code != http.StatusConflict {
		t.Fatalf("replay status=%d body=%s", replayed.Code, replayed.Body.String())
	}
}
