package server

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"overgo/internal/artifact"
	"overgo/internal/operation"
	"overgo/internal/operatoraction"
	"overgo/internal/recipe"
	"overgo/internal/testutil"
)

func TestGlobalOperationShell(t *testing.T) {
	handler := newTestHandler(t, &fakeGenerator{})
	defer handler.Close()
	app := serveTestRequest(handler, http.MethodGet, "/app.html", "").Body.String()
	boot := serveTestRequest(handler, http.MethodGet, "/boot.js", "").Body.String()
	shell := serveTestRequest(handler, http.MethodGet, "/operations_shell.js", "").Body.String()
	if !strings.Contains(app, "global-operation-shell") {
		t.Error("app shell lacks the global operation shell host")
	}
	if !strings.Contains(boot, "/operations_shell.js") {
		t.Error("boot.js does not load the operations shell library")
	}
	for _, expected := range []string{
		"runtimeEvents.subscribe", "/operations/evidence?id=", "/operations/cancel", "/operations/decision",
		"searchParams.set(\"operation\"", "Recovery decision", "Outputs and traces", "/artifacts/content?id=",
	} {
		if !strings.Contains(shell, expected) {
			t.Errorf("global operation shell lacks %q", expected)
		}
	}
	for _, forbidden := range []string{"api.events(", "overgo.poller", "setInterval("} {
		if strings.Contains(shell, forbidden) {
			t.Errorf("global operation shell creates a second lifecycle source with %q", forbidden)
		}
	}
}

func TestGlobalOperationShellSSE(t *testing.T) {
	handler := newTestHandler(t, &fakeGenerator{})
	defer handler.Close()
	ctx, cancel := context.WithCancel(t.Context())
	recorder := &countingRecorder{ResponseRecorder: httptest.NewRecorder(), flushes: make(chan struct{}, 8)}
	done := make(chan struct{})
	go func() {
		handler.ServeHTTP(recorder, httptest.NewRequest(http.MethodGet, "/runtime/activity/stream", nil).WithContext(ctx))
		close(done)
	}()
	for range 3 {
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
	select {
	case <-recorder.flushes:
	case <-time.After(time.Second):
		cancel()
		<-done
		t.Fatal("operation event was not flushed")
	}
	cancel()
	<-done
	if !strings.Contains(recorder.Body.String(), "event: operation") ||
		!strings.Contains(recorder.Body.String(), id.String()) {
		t.Fatalf("operation SSE body=%s", recorder.Body.String())
	}
}

func TestGlobalOperationDecisionRecovery(t *testing.T) {
	handler := newTestHandlerWithRepository(t, &fakeGenerator{})
	recipeID := testutil.ArtifactID(t, artifact.KindRecipe, "global-shell-recovery-recipe")
	runID := testutil.ArtifactID(t, artifact.KindRun, "global-shell-recovery-run")
	action := operatoraction.Action{Code: "resume", Summary: "Resume operation", Argv: []string{"overgo", "resume"}}
	var executions atomic.Uint32
	id, err := handler.operations.Submit(t.Context(), operation.Request{
		Task: recipe.TaskGeneration, Recipe: recipeID,
	}, func(context.Context, operation.Reporter) (operation.Completion, error) {
		if executions.Add(1) == 1 {
			return operation.Completion{Run: runID}, operatoraction.Recoverable(errors.New("approval required"), operatoraction.Block{
				Subject: recipeID, Reason: "operator approval required", Evidence: []artifact.ID{}, Actions: []operatoraction.Action{action},
			})
		}
		return operation.Completion{Run: runID}, nil
	})
	if err != nil {
		t.Fatal(err)
	}
	blocked, err := handler.operations.Wait(t.Context(), id)
	if err != nil || blocked.State != operation.StateBlocked {
		t.Fatalf("blocked operation=(%+v, %v)", blocked, err)
	}
	advertised, err := operatoraction.NewApprovalRequest(id, recipeID, action, artifact.ID{})
	if err != nil {
		t.Fatal(err)
	}
	response := serveTestRequest(handler, http.MethodPost, "/operations/decision", marshalAutomationJSON(t, map[string]any{
		"operation": id, "tool": action.Code, "answer": operatoraction.AnswerGrant, "request": advertised.ID,
	}))
	if response.Code != http.StatusAccepted {
		t.Fatalf("decision status=%d body=%s", response.Code, response.Body.String())
	}
	completed, err := handler.operations.Wait(t.Context(), id)
	if err != nil || completed.State != operation.StateCompleted || executions.Load() != 2 {
		t.Fatalf("recovered operation=(%+v, executions=%d, %v)", completed, executions.Load(), err)
	}
	projection := serveTestRequest(handler, http.MethodGet, "/operations/evidence?id="+id.String(), "")
	if projection.Code != http.StatusOK || !strings.Contains(projection.Body.String(), `"answer":"grant"`) {
		t.Fatalf("recovery evidence status=%d body=%s", projection.Code, projection.Body.String())
	}
}

func TestGlobalOperationTerminalState(t *testing.T) {
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
	var projection OperationEvidenceProjection
	if response.Code != http.StatusOK || json.Unmarshal(response.Body.Bytes(), &projection) != nil ||
		projection.Operation == nil || projection.Operation.State != operation.StateCompleted ||
		len(projection.Operation.Outputs) != 1 || projection.Operation.Outputs[0] != outputID {
		t.Fatalf("terminal projection status=%d value=%+v body=%s", response.Code, projection, response.Body.String())
	}
	cancel := serveTestRequest(handler, http.MethodPost, "/operations/cancel", marshalAutomationJSON(t, map[string]any{"id": id}))
	if cancel.Code != http.StatusConflict {
		t.Fatalf("terminal cancellation status=%d body=%s", cancel.Code, cancel.Body.String())
	}
}
