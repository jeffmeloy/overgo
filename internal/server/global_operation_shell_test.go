package server

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/operation"
	"overgo/internal/recipe"
	"overgo/internal/testutil"
)

// TestGlobalOperationShellSSE: the runtime stream is an event stream that
// opens with its snapshots, operations among them, and carries a later
// operation's transitions.
func TestGlobalOperationShellSSE(t *testing.T) {
	t.Parallel()
	handler := newTestHandler(t, &fakeGenerator{})
	defer handler.Close()
	ctx, cancel := context.WithCancel(t.Context())
	recorder := &countingRecorder{ResponseRecorder: httptest.NewRecorder(), flushes: make(chan struct{}, 8)}
	done := make(chan struct{})
	go func() {
		handler.ServeHTTP(recorder, httptest.NewRequest(http.MethodGet, "/runtime/activity/stream", nil).WithContext(ctx))
		close(done)
	}()
	// The stream's four snapshots: sessions, activity, operations and downloads.
	for range 4 {
		<-recorder.flushes
	}
	recipeID := testutil.ArtifactID(t, artifact.KindRecipe, "global-shell-sse-recipe")
	runID := testutil.ArtifactID(t, artifact.KindRun, "global-shell-sse-run")
	id, err := handler.operations.Submit(t.Context(), operation.Request{
		Task: recipe.TaskGeneration, Recipe: recipeID,
	}, func(context.Context, operation.Reporter) (operation.Completion, error) {
		return operation.Completion{Run: runID}, nil
	})
	if err != nil {
		cancel()
		<-done
		t.Fatal(err)
	}
	if _, err := handler.operations.Wait(t.Context(), id); err != nil {
		cancel()
		<-done
		t.Fatal(err)
	}
	<-recorder.flushes
	cancel()
	<-done
	if recorder.Header().Get("Content-Type") != "text/event-stream" ||
		!strings.Contains(recorder.Body.String(), "event: operation.snapshot") ||
		!strings.Contains(recorder.Body.String(), "event: operation\n") ||
		!strings.Contains(recorder.Body.String(), id.String()) {
		t.Fatalf("operation SSE headers=%v body=%s", recorder.Header(), recorder.Body.String())
	}
}

func TestGlobalOperationTerminalState(t *testing.T) {
	t.Parallel()
	handler := newTestHandlerWithRepository(t, &fakeGenerator{})
	recipeID := testutil.ArtifactID(t, artifact.KindRecipe, "global-shell-terminal-recipe")
	runID := testutil.ArtifactID(t, artifact.KindRun, "global-shell-terminal-run")
	outputID := testutil.ArtifactID(t, artifact.KindOutput, "global-shell-terminal-output")
	id, err := handler.operations.Submit(t.Context(), operation.Request{
		Task: recipe.TaskGeneration, Recipe: recipeID,
	}, func(context.Context, operation.Reporter) (operation.Completion, error) {
		return operation.Completion{Run: runID, Outputs: []artifact.ID{outputID}}, nil
	})
	if err != nil {
		t.Fatal(err)
	}
	if status, err := handler.operations.Wait(t.Context(), id); err != nil || status.State != operation.StateCompleted {
		t.Fatalf("terminal operation=(%+v, %v)", status, err)
	}
	response := serveTestRequest(handler, http.MethodGet, "/operations/evidence?id="+id.String(), "")
	var projection operationEvidenceProjection
	if response.Code != http.StatusOK || json.Unmarshal(response.Body.Bytes(), &projection) != nil ||
		projection.Operation == nil || projection.Operation.State != operation.StateCompleted ||
		len(projection.Operation.Outputs) != 1 || projection.Operation.Outputs[0] != outputID {
		t.Fatalf("terminal projection status=%d value=%+v body=%s", response.Code, projection, response.Body.String())
	}
	cancel := serveTestRequest(handler, http.MethodPost, "/operations/cancel", marshalTestJSON(t, map[string]any{"id": id}))
	if cancel.Code != http.StatusConflict {
		t.Fatalf("terminal cancellation status=%d body=%s", cancel.Code, cancel.Body.String())
	}
}
