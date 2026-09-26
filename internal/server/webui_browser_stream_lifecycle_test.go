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

// TestWebUIBrowserStreamLifecycle remounts the live workspaces in a real
// browser over one event stream: the automations, peers, inbox and library
// tabs open no stream and poll nothing of their own (each once held its own
// stream or poller), every runtime stream a remount replaces is aborted, the
// runtime stream the server ends cleanly is opened again, one operation reads
// its peer evidence once, and a hidden page stops probing health and the
// catalog and probes health at once when shown again.
func TestWebUIBrowserStreamLifecycle(t *testing.T) {
	t.Parallel()
	if os.Getenv("OVERGO_WEBUI_LANE") != "1" {
		testskip.NotApplicable(t, "the stream lifecycle runs through cmd/webui-lane")
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
			counts := map[string]int{"health": hits["/health"], "catalog": hits["/catalog/models"], "runtime": hits["/runtime/activity/stream"], "peers": hits["/peers"],
				"retired": hits["/automations/stream"] + hits["/peers/stream"] + hits["/agents/stream"] + hits["/hub/downloads"]}
			mu.Unlock()
			_ = json.NewEncoder(response).Encode(counts)
			return
		}
		mu.Lock()
		hits[path]++
		endNow := path == "/runtime/activity/stream" && !endedOnce
		endedOnce = endedOnce || endNow
		mu.Unlock()
		if endNow {
			// The first runtime stream ends cleanly at once, the way a server restart ends it.
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
  window.laneAnswered = new WeakSet();
  window.laneEvidence = 0;
  window.laneOperations = 0;
  window.laneSeen = new Set();
  overgo.runtimeEvents.subscribe((event, value) => {
    if (event === 'operation') { window.laneOperations++; window.laneSeen.add(value.status.id); if (value.status.state === 'completed') window.laneSeen.add('completed:' + value.status.id); }
    if (event === 'operation.snapshot') for (const status of value) window.laneSeen.add(status.id);
  });
  const original = window.fetch;
  window.fetch = function (path) {
    const signal = arguments[1] && arguments[1].signal;
    if (typeof path === 'string' && path.endsWith('/stream')) (window.laneStreams[path] = window.laneStreams[path] || []).push(signal);
    if (typeof path === 'string' && path.startsWith('/peers/evidence')) window.laneEvidence++;
    const answer = original.apply(this, arguments);
    if (signal) answer.then(() => window.laneAnswered.add(signal), () => {});
    return answer;
  };
  return true;
})()`)
	visit := func() {
		t.Helper()
		for _, tab := range []string{"automations", "peers", "inbox", "library"} {
			assertBrowserPredicate(t, ctx, browser, `(() => { location.hash = '`+tab+`'; return true; })()`)
			settle(`(() => { const panel = document.querySelector('#panel-` + tab + `.active'); return !!panel && panel.children.length > 0 && !panel.querySelector('.workspace-refusal'); })()`)
		}
		assertBrowserPredicate(t, ctx, browser, `(() => { location.hash = 'chat'; return true; })()`)
	}
	visit()
	// The server ended the page's first runtime stream cleanly; the page opens it again on its own.
	lifecycle := func() map[string]float64 {
		t.Helper()
		var counts map[string]float64
		if err := browser.Evaluate(ctx, `fetch('/__lifecycle').then((answer) => answer.json())`, &counts); err != nil {
			t.Fatal(err)
		}
		return counts
	}
	settle(`fetch('/__lifecycle').then((answer) => answer.json()).then((counts) => counts.runtime >= 2)`)
	// Each key change remounts every workspace and restarts the shared stream.
	const remounts = 3
	for range remounts {
		assertBrowserPredicate(t, ctx, browser, `(() => { const key = document.getElementById('api-key'); key.value = key.value === '' ? 'lane-remount' : ''; key.dispatchEvent(new Event('change')); return true; })()`)
		// The remount reads the manifest, then releases every tab; tabs visited before it
		// settles would be released under the visit.
		settle(`overgo.api.inFlight() === 0`)
		visit()
	}
	// The runtime stream is the page's only stream: every one a remount replaced was aborted and
	// the latest stays live. A remount reopens it after its first reads, so the state is awaited.
	settle(`(() => {
  const streams = window.laneStreams;
  const runtime = streams['/runtime/activity/stream'] || [];
  return Object.keys(streams).every((path) => path === '/runtime/activity/stream') && runtime.length >= ` + strconv.Itoa(remounts) + ` &&
    runtime.slice(0, -1).every((signal) => signal.aborted) && !runtime.at(-1).aborted;
})()`)
	if counts := lifecycle(); counts["retired"] != 0 {
		t.Errorf("the page reached a retired tab stream or the download poll %v times", counts["retired"])
	}
	// One finished operation reaches the page through the shared runtime stream. The server
	// subscribes a stream before answering it, so the operation waits for the latest stream's
	// answer. A transition while a remount reopens the stream arrives in the new stream's
	// snapshot instead of as an event, so the operation finishes only once the page has seen it:
	// its last transition is then always an event.
	settle(`(() => { const latest = (window.laneStreams['/runtime/activity/stream'] || []).at(-1); return !!latest && !latest.aborted && window.laneAnswered.has(latest); })()`)
	var readsBefore, eventsBefore int
	if err := browser.Evaluate(ctx, "window.laneEvidence", &readsBefore); err != nil {
		t.Fatal(err)
	}
	if err := browser.Evaluate(ctx, "window.laneOperations", &eventsBefore); err != nil {
		t.Fatal(err)
	}
	seen := make(chan struct{})
	submitted, err := handler.operations.Submit(ctx, operation.Request{Task: recipe.TaskGeneration, Recipe: testutil.ArtifactID(t, artifact.KindRecipe, "stream-lifecycle")},
		func(ctx context.Context, _ operation.Reporter) (operation.Completion, error) {
			select {
			case <-seen:
			case <-ctx.Done():
				return operation.Completion{}, context.Cause(ctx)
			}
			return operation.Completion{Run: testutil.ArtifactID(t, artifact.KindRun, "stream-lifecycle-run")}, nil
		})
	if err != nil {
		t.Fatal(err)
	}
	settle(`window.laneSeen.has(` + strconv.Quote(submitted.String()) + `)`)
	close(seen)
	// Each operation event the page receives reads its evidence exactly once; a listener a remount
	// leaked reads it again in the same dispatch, so reads and events would disagree. The counts
	// are compared once the finished operation's last event arrived and its reads were made.
	settle(`window.laneSeen.has(` + strconv.Quote("completed:"+submitted.String()) + `) && overgo.api.inFlight() === 0`)
	var delivered struct {
		Events, Reads int
		Page          string
	}
	if err := browser.Evaluate(ctx, `({Events: window.laneOperations, Reads: window.laneEvidence, Page: JSON.stringify({hash: location.hash, peers: (document.querySelector('#panel-peers') || {innerText: 'absent'}).innerText.slice(0, 200), errors: overgo.errors})})`, &delivered); err != nil {
		t.Fatal(err)
	}
	if events, reads := delivered.Events-eventsBefore, delivered.Reads-readsBefore; events == 0 || reads != events {
		t.Errorf("the page read peer evidence %d times for %d operation events; page %s", reads, events, delivered.Page)
	}
	// A changed inventory reads again in the open tab that shows it, with no reload and no
	// action of its own; a change to another inventory leaves that tab's read alone.
	assertBrowserPredicate(t, ctx, browser, `(() => { location.hash = 'peers'; return true; })()`)
	settle(`!!document.querySelector('#panel-peers.active') && overgo.api.inFlight() === 0`)
	peersBefore := lifecycle()["peers"]
	handler.events.publish("workspace.changed", workspaceChange{Inventory: "/datasets"})
	handler.events.publish("workspace.changed", workspaceChange{Inventory: "/peers"})
	settle(`fetch('/__lifecycle').then((answer) => answer.json()).then((counts) => counts.peers > ` + strconv.FormatFloat(peersBefore, 'f', -1, 64) + `)`)
	settle(`overgo.api.inFlight() === 0`)
	if peersAfter := lifecycle()["peers"]; peersAfter != peersBefore+1 {
		t.Errorf("an inventory change read the peers %v times, want once", peersAfter-peersBefore)
	}
	assertBrowserPredicate(t, ctx, browser, `(() => { location.hash = 'chat'; return true; })()`)
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
	// Shown again, the page probes health at once rather than at its next interval.
	var resumed map[string]float64
	if err := browser.Evaluate(ctx, `(async () => {
  const pageFetch = window.fetch;
  const probed = new Promise((resolve) => {
    window.fetch = (input, init) => {
      const answer = pageFetch(input, init);
      if (String(input instanceof Request ? input.url : input).endsWith('/health')) answer.finally(resolve);
      return answer;
    };
  });
  Object.defineProperty(document, 'hidden', { configurable: true, get: () => false });
  document.dispatchEvent(new Event('visibilitychange'));
  await probed;
  window.fetch = pageFetch;
  return await pageFetch('/__lifecycle').then((answer) => answer.json());
})()`, &resumed); err != nil {
		t.Fatalf("a page shown again never probed health (hidden counts %v): %v", hidden, err)
	}
	if resumed["health"] <= hidden["health"] {
		t.Errorf("a page shown again did not probe health: hidden %v, shown %v", hidden, resumed)
	}
	assertBrowserPredicate(t, ctx, browser, `overgo.errors.length === 0`)
	webuilane.Leg(t, "one event stream leg", "the automations, peers, inbox and library tabs remounted %d times, each remount settled before its tabs were visited, over the runtime stream alone (no tab stream, no download poll), every replaced stream aborted, a cleanly ended stream reconnected, each operation read its evidence once, a peers change read the open peers tab again once and another inventory's change left it alone, a hidden page stopped probing, and a page shown again resumes probing", remounts)
}
