package server

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"strconv"
	"sync"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/operation"
	"overgo/internal/recipe"
	"overgo/internal/testskip"
	"overgo/internal/testutil"
	"overgo/internal/webuilane"
)

// streamLifecycleGenerator serves a model (so the shell's runtime stream
// carries operations) with the automation and peer workspaces beside it.
type streamLifecycleGenerator struct {
	*recipeInspectorGenerator
	*AutomationWorkspace
	*PeerWorkspace
}

// TestWebUIBrowserStreamLifecycle remounts the stream-owning workspaces in a
// real browser: every stream a remount replaces is aborted and the inbox opens
// none of its own (each peers mount once leaked a runtime listener, and the
// inbox held a stream of its own), one operation reads its peer evidence once, a stream
// the server ends cleanly is opened again, and a hidden page stops probing
// health and the catalog.
func TestWebUIBrowserStreamLifecycle(t *testing.T) {
	if os.Getenv("OVERGO_WEBUI_LANE") != "1" {
		t.Skip(testskip.Inapplicable + ": the stream lifecycle runs through cmd/webui-lane")
	}
	fixture := newAutomationServerFixture(t)
	defer fixture.store.Close()
	defer fixture.handler.Close()
	automation := fixture.handler.generator.(*automationWorkspaceGenerator).AutomationWorkspace
	generator := &streamLifecycleGenerator{recipeInspectorGenerator: responseRecipeGenerator(t, &fakeGenerator{}), AutomationWorkspace: automation}
	handler := newTestHandlerForRepository(t, fixture.store, generator)
	defer handler.Close()
	// The peer workspace answers the handler's own inventory bound.
	var err error
	generator.PeerWorkspace, err = (PeerWorkspaceConfig{Store: fixture.store, Backend: peerWorkspaceBackend{}, Limit: handler.config.MaxStoredResponses}).Open()
	if err != nil {
		t.Fatal(err)
	}
	var mu sync.Mutex
	hits := map[string]int{}
	endedOnce := false
	server := httptest.NewServer(http.HandlerFunc(func(response http.ResponseWriter, request *http.Request) {
		path := request.URL.Path
		if path == "/__lifecycle" {
			mu.Lock()
			counts := map[string]int{"health": hits["/health"], "catalog": hits["/catalog/models"]}
			mu.Unlock()
			_ = json.NewEncoder(response).Encode(counts)
			return
		}
		mu.Lock()
		hits[path]++
		endNow := path == "/automations/stream" && !endedOnce
		endedOnce = endedOnce || endNow
		mu.Unlock()
		if endNow {
			// The first automation stream ends cleanly at once, the way a server restart ends it.
			response.Header().Set("Content-Type", "text/event-stream")
			response.WriteHeader(http.StatusOK)
			return
		}
		handler.ServeHTTP(response, request)
	}))
	defer server.Close()
	path, err := webuilane.FindBrowser(os.Getenv("OVERGO_BROWSER"))
	if err != nil {
		t.Fatal(err)
	}
	ctx := t.Context()
	browser, err := webuilane.Open(ctx, path, server.URL+"/app.html#chat")
	if err != nil {
		t.Fatal(err)
	}
	defer browser.Close()
	settle := func(expression string) {
		t.Helper()
		if err := browser.Eventually(ctx, expression); err != nil {
			t.Fatalf("%s: %v", expression, err)
		}
	}
	settle(`!!document.querySelector('#panel-chat.active')`)
	// The page counts the stream requests it makes and its peer evidence reads.
	assertBrowserPredicate(t, ctx, browser, `(() => {
  window.laneStreams = {};
  window.laneEvidence = 0;
  window.laneOperations = 0;
  overgo.runtimeEvents.subscribe((event) => { if (event === 'operation') window.laneOperations++; });
  const original = window.fetch;
  window.fetch = function (path) {
    if (typeof path === 'string' && path.endsWith('/stream')) (window.laneStreams[path] = window.laneStreams[path] || []).push(arguments[1] && arguments[1].signal);
    if (typeof path === 'string' && path.startsWith('/peers/evidence')) window.laneEvidence++;
    return original.apply(this, arguments);
  };
  return true;
})()`)
	visit := func() {
		t.Helper()
		for _, tab := range []string{"automations", "peers", "inbox"} {
			assertBrowserPredicate(t, ctx, browser, `(() => { location.hash = '`+tab+`'; return true; })()`)
			settle(`(() => { const panel = document.querySelector('#panel-` + tab + `.active'); return !!panel && panel.children.length > 0 && !panel.querySelector('.workspace-refusal'); })()`)
		}
		assertBrowserPredicate(t, ctx, browser, `(() => { location.hash = 'chat'; return true; })()`)
	}
	visit()
	// The server ended the first automation stream cleanly; the tab opens it again on its own.
	settle(`(window.laneStreams['/automations/stream'] || []).length >= 2`)
	// Each key change remounts every workspace and restarts the shared stream.
	for range 3 {
		assertBrowserPredicate(t, ctx, browser, `(() => { const key = document.getElementById('api-key'); key.value = key.value === '' ? 'lane-remount' : ''; key.dispatchEvent(new Event('change')); return true; })()`)
		visit()
	}
	// Every stream a remount replaced was aborted: each owner keeps only its latest request live
	// (the first automation stream, which the server ended, is past), and the inbox opened none.
	// A tab opens its stream after its first reads, so the state is awaited, not sampled.
	settle(`(() => {
  const streams = window.laneStreams;
  const replacedAborted = (path, skip) => (streams[path] || []).slice(skip, -1).every((signal) => signal.aborted);
  return !streams['/agents/stream'] && (streams['/peers/stream'] || []).length >= 2 &&
    replacedAborted('/automations/stream', 1) && replacedAborted('/peers/stream', 0) && replacedAborted('/runtime/activity/stream', 0);
})()`)
	// One finished operation reaches the page through the shared runtime stream.
	var readsBefore, eventsBefore int
	if err := browser.Evaluate(ctx, "window.laneEvidence", &readsBefore); err != nil {
		t.Fatal(err)
	}
	if err := browser.Evaluate(ctx, "window.laneOperations", &eventsBefore); err != nil {
		t.Fatal(err)
	}
	if _, err := handler.operations.Submit(ctx, operation.Request{Task: recipe.TaskGeneration, Recipe: testutil.ArtifactID(t, artifact.KindRecipe, "stream-lifecycle")},
		func(context.Context, operation.Reporter) (operation.Completion, error) {
			return operation.Completion{Run: testutil.ArtifactID(t, artifact.KindRun, "stream-lifecycle-run")}, nil
		}); err != nil {
		t.Fatal(err)
	}
	// Each operation event the page receives reads its evidence exactly once; a listener a remount
	// leaked reads it again in the same dispatch, so reads and events would never agree.
	settle("window.laneOperations > " + strconv.Itoa(eventsBefore) + " && window.laneEvidence - " + strconv.Itoa(readsBefore) + " === window.laneOperations - " + strconv.Itoa(eventsBefore))
	// A hidden page does not probe health or the catalog through a whole status interval.
	// Requests a remount started finish first, so what follows is the hidden page alone.
	settle(`overgo.api.inFlight() === 0`)
	var hiddenFrom, hidden map[string]float64
	if err := browser.Evaluate(ctx, `(async () => {
  Object.defineProperty(document, 'hidden', { configurable: true, get: () => true });
  document.dispatchEvent(new Event('visibilitychange'));
  return await fetch('/__lifecycle').then((answer) => answer.json());
})()`, &hiddenFrom); err != nil {
		t.Fatal(err)
	}
	if err := browser.Evaluate(ctx, `new Promise((resolve) => setTimeout(resolve, 11000)).then(() => fetch('/__lifecycle')).then((answer) => answer.json())`, &hidden); err != nil {
		t.Fatal(err)
	}
	if hidden["health"] != hiddenFrom["health"] || hidden["catalog"] != hiddenFrom["catalog"] {
		t.Errorf("a hidden page probed the server: before %v, after %v", hiddenFrom, hidden)
	}
	assertBrowserPredicate(t, ctx, browser, `overgo.errors.length === 0`)
	t.Log("stream lifecycle leg: remounting every stream-owning workspace aborts every stream it replaces and reads each operation's evidence once, a cleanly ended stream reconnects, and a hidden page stops probing")
}
