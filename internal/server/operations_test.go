package server

import (
	"bytes"
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/operation"
	"overgo/internal/recipe"
	"overgo/internal/testutil"
)

func TestOperationRuntimeLifecycle(t *testing.T) {
	handler := newTestHandler(t, &fakeGenerator{})
	recipeID := testutil.ArtifactID(t, artifact.KindRecipe, "server-operation-recipe")
	runID := testutil.ArtifactID(t, artifact.KindRun, "server-operation-run")
	release := make(chan struct{})
	id, err := handler.operations.Submit(t.Context(), operation.Request{
		Task: recipe.TaskGeneration, Recipe: recipeID,
	}, func(ctx context.Context, _ operation.Reporter) (operation.Completion, error) {
		select {
		case <-ctx.Done():
			return operation.Completion{Run: runID}, ctx.Err()
		case <-release:
			return operation.Completion{Run: runID}, nil
		}
	})
	if err != nil {
		t.Fatal(err)
	}

	statusResponse := httptest.NewRecorder()
	handler.ServeHTTP(statusResponse, httptest.NewRequest(http.MethodGet, "/operations?id="+id.String(), nil))
	if statusResponse.Code != http.StatusOK {
		t.Fatalf("status code = %d, body=%s", statusResponse.Code, statusResponse.Body.String())
	}
	var status operation.Status
	if err := json.Unmarshal(statusResponse.Body.Bytes(), &status); err != nil {
		t.Fatal(err)
	}
	if status.ID != id || status.Recipe != recipeID || status.State == operation.StateCompleted {
		t.Fatalf("status = %+v", status)
	}

	body, err := json.Marshal(operationCancelRequest{ID: id})
	if err != nil {
		t.Fatal(err)
	}
	cancelResponse := httptest.NewRecorder()
	handler.ServeHTTP(cancelResponse, httptest.NewRequest(http.MethodPost, "/operations/cancel", bytes.NewReader(body)))
	if cancelResponse.Code != http.StatusAccepted {
		t.Fatalf("cancel code = %d, body=%s", cancelResponse.Code, cancelResponse.Body.String())
	}
	status, err = handler.operations.Wait(t.Context(), id)
	if err != nil {
		t.Fatal(err)
	}
	if status.State != operation.StateCancelled || status.Run == nil || *status.Run != runID {
		t.Fatalf("cancelled status = %+v", status)
	}
	close(release)
}
