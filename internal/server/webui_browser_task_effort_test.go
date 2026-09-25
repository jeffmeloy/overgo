package server

import (
	"context"
	"encoding/json"
	"net/http/httptest"
	"os"
	"strconv"
	"testing"

	"overgo/internal/testskip"
	"overgo/internal/webuilane"
)

// effortStep is one thing a person does: click a control (by selector and,
// when several match, its visible text), fill a field, or press a key.
type effortStep struct {
	action, target, text string
}

// taskEffort is what one task costs the person doing it.
type taskEffort struct {
	Clicks, Fields, Keys, TabSwitches int
}

// effortTask is a common task as the shortest path the workbench offers,
// and the page state that says it is done.
type effortTask struct {
	name  string
	steps []effortStep
	done  string
}

// effortCeilings holds every measured task at its count: a change that adds
// a step fails, and one that removes a step lowers the ceiling with it, so
// the effort a task takes only falls.
var effortCeilings = map[string]taskEffort{
	"ask a question":            {Fields: 1, Keys: 1},
	"decide a pending approval": {Clicks: 2, TabSwitches: 1},
	"inspect the served model":  {Clicks: 2, TabSwitches: 1},
	"add a model from the hub":  {Clicks: 1, Fields: 1, Keys: 1},
}

// commonTasks are the tasks the task effort leg drives from a fresh landing on
// the conversation; the onboarding leg drives adding a model from the Library.
var commonTasks = []effortTask{
	{"ask a question", []effortStep{
		{"type", ".composer textarea", "What does this model answer?"},
		{"key", "Enter", ""},
	}, `!!document.querySelector(".chat-log .msg.assistant .body") && document.querySelector(".chat-log .msg.assistant .body").textContent.trim().length > 0`},
	{"decide a pending approval", []effortStep{
		{"click", "#inbox-count", ""},
		{"click", "#panel-inbox button", "Grant retry-browser"},
	}, `document.querySelector("#panel-inbox").textContent.includes("grant recorded")`},
	{"inspect the served model", []effortStep{
		{"click", "#workbench-toggle", ""},
		{"click", "button.tab", "Inspect"},
	}, `!!document.querySelector("#panel-inspect.active .section-title")`},
}

// navigationCeiling holds the entries a person scans in the navigation, its
// places and their tabs: 30 before the analysis tabs became Inspect's views.
const navigationCeiling = 25

// TestWebUIBrowserTaskEffort drives the workbench's common tasks by their
// shortest paths in a real browser and counts what each costs: clicks,
// typed fields, key presses and tab switches, with the requests the page
// made beside them. The owner asked for less effort per common task; the
// effort rows are judged by lowering these counts.
func TestWebUIBrowserTaskEffort(t *testing.T) {
	t.Parallel()
	if os.Getenv("OVERGO_WEBUI_LANE") != "1" {
		t.Skip(testskip.Inapplicable + ": the task effort leg runs through cmd/webui-lane")
	}
	fixture := newAutomationServerFixture(t)
	defer fixture.store.Close()
	defer fixture.handler.Close()
	automation := fixture.handler.generator.(*automationWorkspaceGenerator).AutomationWorkspace
	generator := &streamLifecycleGenerator{recipeInspectorGenerator: responseRecipeGenerator(t, &fakeGenerator{}), AutomationWorkspace: automation}
	handler := newTestHandlerForRepository(t, fixture.store, generator)
	defer handler.Close()
	var err error
	generator.PeerWorkspace, err = (PeerWorkspaceConfig{Store: fixture.store, Backend: peerWorkspaceBackend{}, Limit: handler.config.MaxStoredResponses}).Open()
	if err != nil {
		t.Fatal(err)
	}
	publishBrowserLaneOperations(t, handler)
	server := httptest.NewServer(handler)
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
			var page string
			_ = browser.Evaluate(ctx, `JSON.stringify({errors: overgo.errors, hash: location.hash, text: document.body.innerText.slice(0, 600)})`, &page)
			t.Fatalf("%s: %v; %s", expression, err, page)
		}
	}
	// Effort is measured at the desktop layout, where the navigation stands beside the conversation.
	if err := webuilane.SettleViewport(ctx, browser, webuilane.ScreenViewports[0]); err != nil {
		t.Fatal(err)
	}
	settle(`!!document.querySelector("#panel-chat.active .composer textarea") && overgo.errors.length === 0`)
	measured := map[string]taskEffort{}
	requests := map[string]int{}
	for index, task := range commonTasks {
		// Every task starts where a person lands: the page freshly opened on the conversation (the
		// query makes each opening a full load, not a move within the page).
		if err := browser.Call(ctx, "Page.navigate", map[string]any{"url": server.URL + "/app.html?effort=" + strconv.Itoa(index) + "#chat"}, nil); err != nil {
			t.Fatal(err)
		}
		settle(`document.readyState === "complete" && !window.effortSwitches && !!document.querySelector("#panel-chat.active .composer textarea") && overgo.api.inFlight() === 0`)
		measured[task.name], requests[task.name] = measureEffort(t, ctx, browser, settle, task)
	}
	var places struct{ Sections, Entries int }
	if err := browser.Evaluate(ctx, `({Sections: document.querySelectorAll("#sections button.section").length, Entries: document.querySelectorAll("#sections button").length})`, &places); err != nil {
		t.Fatal(err)
	}
	if places.Sections != 5 || places.Entries != navigationCeiling {
		t.Errorf("navigation holds %d places and %d entries, want 5 places and %d entries", places.Sections, places.Entries, navigationCeiling)
	}
	assertBrowserPredicate(t, ctx, browser, `overgo.errors.length === 0`)
	encoded, _ := json.Marshal(measured)
	webuilane.Leg(t, "task effort leg", "%d common tasks at their shortest paths %s, with requests %v; five workspace places with %d navigation entries", len(commonTasks), encoded, requests, places.Entries)
}

// effortCounters makes the page count its tab switches and the requests it
// makes, streams aside.
const effortCounters = `(() => {
  window.effortSwitches = 0; window.effortRequests = 0;
  window.addEventListener("overgo-panel-change", () => window.effortSwitches++);
  const pageFetch = window.fetch;
  window.fetch = function (input) { if (!String(input).endsWith("/stream")) window.effortRequests++; return pageFetch.apply(this, arguments); };
  return true;
})()`

// measureEffort drives one task's steps on the page as it stands, awaits the
// state that says the task is done, holds the effort at the task's ceiling,
// and answers the effort and the requests the page made.
func measureEffort(t *testing.T, ctx context.Context, browser *webuilane.Browser, settle func(string), task effortTask) (taskEffort, int) {
	t.Helper()
	assertBrowserPredicate(t, ctx, browser, effortCounters)
	var effort taskEffort
	for _, step := range task.steps {
		switch step.action {
		case "click":
			find := `[...document.querySelectorAll(` + strconv.Quote(step.target) + `)].find((node) => !node.hidden && !node.disabled && node.getClientRects().length && (!` + strconv.Quote(step.text) + ` || node.textContent.trim() === ` + strconv.Quote(step.text) + `))`
			settle("!!" + find)
			assertBrowserPredicate(t, ctx, browser, `(() => { `+find+`.click(); return true; })()`)
			effort.Clicks++
		case "type":
			settle(`!!document.querySelector(` + strconv.Quote(step.target) + `)`)
			assertBrowserPredicate(t, ctx, browser, `(() => { const field = document.querySelector(`+strconv.Quote(step.target)+`); field.focus(); field.value = `+strconv.Quote(step.text)+`; field.dispatchEvent(new Event("input", { bubbles: true })); return true; })()`)
			effort.Fields++
		case "key":
			pressKey(t, ctx, browser, step.target, 13)
			effort.Keys++
		default:
			t.Fatalf("%s: unknown step %q", task.name, step.action)
		}
	}
	settle(task.done)
	var counted struct{ Switches, Requests int }
	if err := browser.Evaluate(ctx, `({Switches: window.effortSwitches, Requests: window.effortRequests})`, &counted); err != nil {
		t.Fatal(err)
	}
	effort.TabSwitches = counted.Switches
	if ceiling, found := effortCeilings[task.name]; !found || effort != ceiling {
		t.Errorf("%s: effort %+v, ceiling %+v (held %v); a task's effort only falls, and a fall lowers its ceiling", task.name, effort, ceiling, found)
	}
	return effort, counted.Requests
}

// TestEffortCeilingsNameMeasuredTasks holds every ceiling to a task a leg drives.
func TestEffortCeilingsNameMeasuredTasks(t *testing.T) {
	t.Parallel()
	driven := map[string]bool{onboardingTask.name: true}
	for _, task := range commonTasks {
		driven[task.name] = true
	}
	for name := range effortCeilings {
		if !driven[name] {
			t.Errorf("the effort ceiling %q names a task no leg drives", name)
		}
	}
	if len(driven) != len(effortCeilings) {
		t.Errorf("legs drive %d tasks and %d carry a ceiling", len(driven), len(effortCeilings))
	}
}
