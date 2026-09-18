package server

import (
	"context"
	"net/http"
	"net/http/httptest"
	"os"
	"strconv"
	"sync/atomic"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/operation"
	"overgo/internal/recipe"
	"overgo/internal/testskip"
	"overgo/internal/testutil"
	"overgo/internal/webuilane"
)

func TestWebUIBrowserServingStream(t *testing.T) {
	if os.Getenv("OVERGO_WEBUI_LANE") != "1" {
		t.Skip(testskip.ShortIntegration + ": serving stream uses Chromium through cmd/webui-lane")
	}
	handler, store := servingStreamFixture(t, 2)
	handler.config.APIKey = testAPIKey
	if response := serveTestRequest(handler, http.MethodGet, "/runtime/activity/stream", ""); response.Code != http.StatusUnauthorized {
		t.Fatalf("unauthenticated activity stream status=%d", response.Code)
	}
	var polls, streams atomic.Int64
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/runtime/activity" {
			polls.Add(1)
		}
		if r.URL.Path == "/runtime/activity/stream" && r.Header.Get("Authorization") == "Bearer "+testAPIKey {
			streams.Add(1)
		}
		handler.ServeHTTP(w, r)
	}))
	defer server.Close()
	path, err := webuilane.FindBrowser(os.Getenv("OVERGO_BROWSER"))
	if err != nil {
		t.Fatal(err)
	}
	ctx := t.Context()
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
	check(`(() => {
  window.servingPage = document;
  window.servingSnapshot = null;
  window.servingSnapshots = 0;
  window.servingStop = overgo.runtimeEvents.subscribe((name, value) => { if (name === 'runtime.activity') { servingSnapshot = value; servingSnapshots++; } });
  document.getElementById('settings-toggle').click();
  const key = document.getElementById('api-key'); key.value = ` + strconv.Quote(testAPIKey) + `;
  key.dispatchEvent(new Event('change')); document.getElementById('settings-dialog').close();
  return true;
})()`)
	settle(`servingSnapshot && servingSnapshot.count === 0 && servingSnapshot.limit === 2`)
	turn := `(async () => { await overgo.api.post('/v1/completions', {prompt:'browser serving turn', max_tokens:1}); return true; })()`
	check(turn)
	settle(`servingSnapshot.count === 1 && servingSnapshot.cursor === '1'`)
	check(`(() => { window.servingFirstID = servingSnapshot.activity[0].id; location.hash = 'activity'; return true; })()`)
	settle(`!!document.querySelector('#panel-activity.active')`)
	check(`(() => {
  window.servingRows = () => {
    const table = [...document.querySelectorAll('#panel-activity table')].find(value => value.querySelector('th')?.textContent === 'started');
    return table ? table.rows.length - 1 : -1;
  };
  return true;
})()`)
	settle(`servingRows() === 1`)
	// A new serving row must not restore the operation state from the older
	// initial activity snapshot over a more recent operation event.
	_, recipeID, ok := handler.servingIdentity(recipe.TaskInference)
	if !ok {
		t.Fatal("fixture has no serving identity")
	}
	runID := testutil.ArtifactID(t, artifact.KindRun, "serving stream operation")
	if _, err := store.Commit(ctx, artifact.Batch{Key: "stream/run", Artifacts: []artifact.Descriptor{{ID: runID}}}); err != nil {
		t.Fatal(err)
	}
	operationID, err := handler.operations.Submit(ctx, operation.Request{Task: recipe.TaskInference, Recipe: recipeID},
		func(context.Context, operation.Reporter) (operation.Completion, error) {
			return operation.Completion{Run: runID}, nil
		})
	if err != nil {
		t.Fatal(err)
	}
	if status, err := handler.operations.Wait(ctx, operationID); err != nil || status.State != operation.StateCompleted {
		t.Fatalf("fixture operation=%+v, %v", status, err)
	}
	settle(`document.querySelector('#panel-activity').textContent.includes('completed')`)
	check(turn)
	settle(`servingSnapshot.count === 2 && servingRows() === 2 && servingSnapshot.cursor === '2'`)
	check(turn)
	settle(`servingSnapshot.count === 2 && servingRows() === 2 && servingSnapshot.cursor === '3' && servingSnapshot.truncated`)
	check(`new Set(servingSnapshot.activity.map(value => value.id)).size === 2 && !servingSnapshot.activity.some(value => value.id === servingFirstID)`)
	check(`document.querySelector('#panel-activity').textContent.includes('completed') && servingSnapshot.operations.some(value => value.state === 'completed')`)
	check(`(() => {
  window.servingBeforeReconnect = servingSnapshots;
  window.servingRetainedIDs = servingSnapshot.activity.map(value => value.id).join(',');
  overgo.runtimeEvents.restart(); return true;
})()`)
	settle(`servingSnapshots > servingBeforeReconnect && servingSnapshot.cursor === '3' && servingRows() === 2`)
	check(`servingSnapshot.activity.map(value => value.id).join(',') === servingRetainedIDs && document === servingPage`)
	check(turn)
	settle(`servingSnapshot.cursor === '4' && servingRows() === 2`)
	if polls.Load() != 0 || streams.Load() < 2 {
		t.Fatalf("activity GETs=%d authenticated streams=%d", polls.Load(), streams.Load())
	}
	t.Logf("serving stream leg: live turns, late Activity tab, bounded rows, reconnect, bearer stream; activity GETs=%d", polls.Load())
}
