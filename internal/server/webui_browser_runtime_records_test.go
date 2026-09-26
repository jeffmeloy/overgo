package server

import (
	"context"
	"errors"
	"net/http"
	"net/http/httptest"
	"os"
	"strconv"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/operation"
	"overgo/internal/operatoraction"
	"overgo/internal/recipe"
	"overgo/internal/runrecord"
	"overgo/internal/testskip"
	"overgo/internal/testutil"
	"overgo/internal/webuilane"
)

// TestWebUIBrowserRuntimeRecords drives the runtime tab over the records it
// projects: a workflow stage receipt shows in the stages table, a serving
// attempt that spilled over to a peer on compatibility evidence is marked as
// the peer's, an operation blocked for approval offers its recovery action
// and granting it completes the operation and lists the decision, and a
// stored interaction replays its trace.
func TestWebUIBrowserRuntimeRecords(t *testing.T) {
	t.Parallel()
	if os.Getenv("OVERGO_WEBUI_LANE") != "1" {
		t.Skip(testskip.Inapplicable + ": the runtime records leg uses Chromium through cmd/webui-lane")
	}
	ctx := t.Context()
	handler, store := servingStreamFixture(t, 8)
	id := func(kind artifact.Kind, label string) artifact.ID { return testutil.ArtifactID(t, kind, label) }
	recipeID, operationID := id(artifact.KindRecipe, "runtime records recipe"), id(artifact.KindEvidence, "runtime records operation")
	model, environment := id(artifact.KindModel, "runtime records model"), id(artifact.KindEvidence, "runtime records environment")
	compatibility := id(artifact.KindEvidence, "runtime records peer compatibility")
	var authorities []artifact.Descriptor
	for _, authority := range []artifact.ID{recipeID, operationID, model, environment, compatibility} {
		authorities = append(authorities, artifact.Descriptor{ID: authority})
	}
	if _, err := store.Commit(ctx, artifact.Batch{Key: "runtime-records/authorities", Artifacts: authorities}); err != nil {
		t.Fatal(err)
	}
	if _, err := runrecord.PublishStageReceipt(ctx, store, runrecord.StageReceipt{
		Recipe: recipeID, Node: "respond", Operation: operationID, Attempt: 1, State: runrecord.StageAdmitted,
	}, nil, nil); err != nil {
		t.Fatal(err)
	}
	// A local attempt, then its spillover to a peer on compatibility evidence.
	local, err := runrecord.PublishServingObservation(ctx, store, runrecord.ServingObservation{
		Model: model, Recipe: recipeID, Environment: environment, Operation: operationID, Task: recipe.TaskInference,
		Outcome: runrecord.OutcomeFailed, Failure: "execution_failed", StartedUnixNS: 1, MeasuredNS: 1,
	})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := runrecord.PublishServingObservation(ctx, store, runrecord.ServingObservation{
		Model: model, Recipe: recipeID, Environment: environment, Operation: operationID, Task: recipe.TaskInference,
		Previous: local.ID, Attempt: local.Attempt + 1, Compatibility: compatibility,
		Outcome: runrecord.OutcomeSucceeded, StartedUnixNS: local.StartedUnixNS + int64(local.MeasuredNS), MeasuredNS: 1,
	}); err != nil {
		t.Fatal(err)
	}
	// A stored interaction to replay.
	if turn := serveTestRequest(handler, http.MethodPost, "/v1/responses", `{"input":"runtime records","max_output_tokens":1}`); turn.Code != http.StatusOK {
		t.Fatalf("interaction turn status=%d body=%s", turn.Code, turn.Body)
	}
	// An operation blocked for approval, which the decision resumes.
	action := operatoraction.Action{Code: "resume", Summary: "Resume the runtime records work", Argv: []string{"overgo", "resume"}}
	runID := id(artifact.KindRun, "runtime records run")
	if _, err := store.Commit(ctx, artifact.Batch{Key: "runtime-records/run", Artifacts: []artifact.Descriptor{{ID: runID}}}); err != nil {
		t.Fatal(err)
	}
	attempts := 0
	blocked, err := handler.operations.Submit(ctx, operation.Request{Task: recipe.TaskTraining, Recipe: recipeID},
		func(context.Context, operation.Reporter) (operation.Completion, error) {
			if attempts++; attempts == 1 {
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
	if status, err := handler.operations.Wait(ctx, blocked); err != nil || status.State != operation.StateBlocked {
		t.Fatalf("fixture operation = %+v, %v", status, err)
	}

	server := httptest.NewServer(handler)
	defer server.Close()
	path, err := webuilane.FindBrowser(os.Getenv("OVERGO_BROWSER"))
	if err != nil {
		t.Fatal(err)
	}
	browser, err := webuilane.Open(ctx, path, server.URL)
	if err != nil {
		t.Fatal(err)
	}
	defer browser.Close()
	check := func(expression string) { t.Helper(); assertBrowserPredicate(t, ctx, browser, expression) }
	settle := func(expression string) {
		t.Helper()
		if err := browser.Eventually(ctx, expression); err != nil {
			t.Fatalf("%s: %v", expression, err)
		}
	}
	settle(`!!document.querySelector('.composer textarea')`)
	check(`(() => { location.hash = 'activity'; return true; })()`)
	settle(`!!document.querySelector('#panel-activity.active')`)
	check(`(() => {
  window.runtimeTable = (first) => [...document.querySelectorAll('#panel-activity table')].find(table => table.querySelector('th')?.textContent === first);
  window.runtimeRows = (first) => { const table = runtimeTable(first); return table ? [...table.rows].slice(1).map(row => [...row.cells].map(cell => cell.textContent)) : []; };
  return true;
})()`)
	// The stage receipt and the peer's attempt, as the tab renders them.
	settle(`runtimeRows('stage').some(row => row[0] === 'respond' && row[1] === 'admitted' && row[2] === '1')`)
	settle(`runtimeRows('started').some(row => row[1] === 'inference / peer') && runtimeRows('started').some(row => row[1] === 'inference')`)
	// The blocked operation offers its recovery action; granting it completes the work.
	settle(`[...document.querySelectorAll('#panel-activity button')].some(button => button.textContent === 'Grant resume')`)
	check(`(() => { [...document.querySelectorAll('#panel-activity button')].find(button => button.textContent === 'Grant resume').click(); return true; })()`)
	settle(`runtimeRows('state').some(row => row[0] === 'completed' && row[1] === 'training')`)
	if status, err := handler.operations.Wait(ctx, blocked); err != nil || status.State != operation.StateCompleted || attempts != 2 {
		t.Fatalf("granted operation = %+v, %v, attempts=%d", status, err, attempts)
	}
	// The recorded decision reaches the open tab without a reload.
	settle(`runtimeRows('answer').some(row => row[0] === 'grant' && row[1] === 'resume')`)
	// The stored interaction replays its trace.
	settle(`runtimeRows('response').length > 0`)
	check(`(() => { runtimeTable('response').querySelector('button').click(); return true; })()`)
	settle(`[...document.querySelectorAll('#panel-activity pre')].some(pre => pre.textContent.includes(` + strconv.Quote(`"trace"`) + `))`)
	webuilane.Leg(t, "runtime records leg", "a stage receipt, a peer spillover attempt, a granted recovery decision and an interaction replay rendered and acted on in the runtime tab")
}
