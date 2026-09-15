package server

import (
	"context"
	"errors"
	"net/http/httptest"
	"os"
	"strconv"
	"strings"
	"sync/atomic"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/operation"
	"overgo/internal/operatoraction"
	"overgo/internal/overgodb"
	"overgo/internal/recipe"
	"overgo/internal/remoteprovider"
	"overgo/internal/testskip"
	"overgo/internal/testutil"
	"overgo/internal/webuilane"
)

func TestWebUIBrowserAcceptance(t *testing.T) {
	if os.Getenv("OVERGO_WEBUI_LANE") != "1" {
		if _, err := webuilane.FindBrowser(os.Getenv("OVERGO_BROWSER")); err != nil {
			t.Fatalf("browser acceptance prerequisite is absent; run go run ./cmd/webui-lane: %v", err)
		}
		// A skip, not a pass: a verify naming this test outside the lane runner is vacuous evidence.
		t.Skip(testskip.ShortIntegration + ": browser acceptance runs through cmd/webui-lane")
	}
	browserPath, err := webuilane.FindBrowser(os.Getenv("OVERGO_BROWSER"))
	if err != nil {
		t.Fatal(err)
	}
	fixture := newAutomationServerFixture(t)
	defer fixture.store.Close()
	running, blocked := publishBrowserLaneOperations(t, fixture.handler)
	httpServer := httptest.NewServer(fixture.handler)
	defer httpServer.Close()

	ctx := t.Context()
	browser, err := webuilane.Open(ctx, browserPath, httpServer.URL+"/app.html#automations")
	if err != nil {
		t.Fatal(err)
	}
	defer browser.Close()
	if err := browser.Eventually(ctx, `!!document.querySelector("#panel-automations.active .schema-form") &&
        document.querySelectorAll(".operation-chip").length >= 2`); err != nil {
		var page string
		_ = browser.Evaluate(ctx, `JSON.stringify({errors: window.overgo && window.overgo.errors, html: document.documentElement.outerHTML.slice(0, 1500)})`, &page)
		t.Fatalf("%v; page: %s", err, page)
	}

	// Every library and module parsed and registered: a script that fails to
	// parse leaves window.overgo without its surface and a window error behind.
	assertBrowserPredicate(t, ctx, browser, `window.overgo.errors.length === 0 && typeof window.overgo.composer === "function" && typeof window.overgo.thread === "function"`)

	assertBrowserPredicate(t, ctx, browser, `(() => {
      location.hash = "recipe";
      return true;
    })()`)
	if err := browser.Eventually(ctx, `!!document.querySelector("#panel-recipe.active")`); err != nil {
		t.Fatal(err)
	}
	assertBrowserPredicate(t, ctx, browser, `(() => { location.hash = "automations"; return true; })()`)
	if err := browser.Eventually(ctx, `!!document.querySelector("#panel-automations.active .schema-form")`); err != nil {
		t.Fatal(err)
	}

	// The empty store: the catalog lists nothing servable, so the welcome
	// card's control opens the picker, the picker names the two ways in, and
	// the hosted one opens the Library tab on the provider form with its
	// first field focused.
	assertBrowserPredicate(t, ctx, browser, `(() => { location.hash = "chat"; return true; })()`)
	if err := browser.Eventually(ctx, `!!document.querySelector("#panel-chat.active .card.front-empty button")`); err != nil {
		t.Fatal(err)
	}
	assertChatAttachmentModeReset(t, ctx, browser)
	assertBrowserPredicate(t, ctx, browser, `(() => { const add = [...document.querySelectorAll("#panel-chat .card.front-empty button")].find((b) => b.textContent === "Choose a model"); if (add) add.click(); return !!add; })()`)
	if err := browser.Eventually(ctx, `[...document.querySelectorAll(".topbar .card .row button")].map((b) => b.textContent).join("|") === "register a local model|declare a hosted provider"`); err != nil {
		t.Fatal(err)
	}
	// The picker over an empty store, captured and audited at both viewports.
	for _, viewport := range webuilane.ScreenViewports {
		findings, err := webuilane.CaptureState(ctx, browser, os.Getenv("OVERGO_WEBUI_LANE_SCREENS"), viewport, "picker-empty")
		if err != nil {
			t.Fatal(err)
		}
		for _, finding := range findings {
			t.Error(finding)
		}
	}
	t.Logf("states leg: picker-empty captured at %d viewports", len(webuilane.ScreenViewports))
	assertBrowserPredicate(t, ctx, browser, `(() => { [...document.querySelectorAll(".topbar .card button")].pop().click(); return true; })()`)
	if err := browser.Eventually(ctx, `!!document.querySelector("#panel-library.active") && document.activeElement.getAttribute("aria-label") === "provider name"`); err != nil {
		t.Fatal(err)
	}
	assertBrowserPredicate(t, ctx, browser, `(() => { location.hash = "automations"; return !document.querySelector(".topbar dialog[open]"); })()`)
	if err := browser.Eventually(ctx, `!!document.querySelector("#panel-automations.active .schema-form")`); err != nil {
		t.Fatal(err)
	}
	// Every control mounted so far (the chat, library and automations tabs, the
	// shell) has an accessible name: its label, aria-label, text, placeholder or
	// title; the status strip announces politely and a banner is an alert.
	assertBrowserPredicate(t, ctx, browser, `(() => {
      const named = (node) => node.getAttribute("aria-label") || node.textContent.trim() || node.placeholder || node.title ||
        (node.closest("label") && node.closest("label").textContent.trim()) || (node.id && document.querySelector("label[for='" + node.id + "']"));
      const unnamed = [...document.querySelectorAll("input:not([type=file]):not([type=hidden]), select, textarea, button")].filter((node) => !named(node));
      if (unnamed.length) { console.error("unnamed controls", unnamed.map((node) => node.outerHTML.slice(0, 80))); return false; }
      return document.querySelector(".operation-strip").getAttribute("aria-live") === "polite";
    })()`)

	if err := browser.SetViewport(ctx, 1280, 900); err != nil {
		t.Fatal(err)
	}
	assertBrowserPredicate(t, ctx, browser, `getComputedStyle(document.querySelector(".shell")).flexDirection === "row" &&
      document.querySelector(".sidebar").getBoundingClientRect().width < innerWidth`)
	if err := browser.SetViewport(ctx, 640, 900); err != nil {
		t.Fatal(err)
	}
	if err := browser.Eventually(ctx, `getComputedStyle(document.querySelector(".sidebar")).display === "none" &&
      document.querySelector("#navigation-toggle").getBoundingClientRect().width > 0`); err != nil {
		t.Fatal(err)
	}

	assertBrowserPredicate(t, ctx, browser, `(() => {
      const input = document.querySelector("#panel-automations .schema-form input");
      input.value = "browser-lane";
      input.dispatchEvent(new Event("input", {bubbles:true}));
      const event = new Event("beforeunload", {cancelable:true});
      window.dispatchEvent(event);
      return event.defaultPrevented && input.required && input.pattern.length > 0;
    })()`)
	assertBrowserPredicate(t, ctx, browser, `(() => {
      const nav = document.querySelector("nav.sidebar");
      const main = document.querySelector("main");
      const operations = document.querySelector("aside[aria-label='Operations']");
      const labels = [...document.querySelectorAll("#panel-automations label.control")];
      const buttons = [...document.querySelectorAll("button")];
      return !!nav && !!main && !!operations && labels.length > 0 &&
        labels.every((label) => label.querySelector("input,select,textarea")) &&
        buttons.every((button) => button.textContent.trim().length > 0 || button.getAttribute("aria-label"));
    })()`)

	blockedID := strconv.Quote(blocked.String())
	assertBrowserPredicate(t, ctx, browser, `(() => {document.querySelector('.activity-summary').click();return document.querySelector('.activity-dialog').open;})()`)
	assertBrowserPredicate(t, ctx, browser, `(() => {
      const chip = [...document.querySelectorAll(".operation-chip.blocked")].find((item) => item.title.includes(`+blockedID+`));
      if (!chip) return false;
      chip.click();
      return true;
    })()`)
	if err := browser.Eventually(ctx, `document.querySelector(".operation-detail").textContent.includes("Recovery decision")`); err != nil {
		t.Fatal(err)
	}
	assertBrowserPredicate(t, ctx, browser, `(() => {
      const grant = [...document.querySelectorAll(".operation-detail button")].find((item) => item.textContent.startsWith("Grant "));
      if (!grant) return false;
      grant.click();
      return true;
    })()`)
	if err := browser.Eventually(ctx, `document.querySelector(".operation-chip.completed") !== null`); err != nil {
		t.Fatal(err)
	}
	if status, err := fixture.handler.operations.Wait(ctx, blocked); err != nil || status.State != operation.StateCompleted {
		t.Fatalf("browser recovery=(%+v, %v)", status, err)
	}

	runningID := strconv.Quote(running.String())
	assertBrowserPredicate(t, ctx, browser, `(() => {
      const chip = [...document.querySelectorAll(".operation-chip.running")].find((item) => item.title.includes(`+runningID+`));
      if (!chip) return false;
      chip.click();
      return true;
    })()`)
	if err := browser.Eventually(ctx, `[...document.querySelectorAll(".operation-detail button")].some((item) => item.textContent === "Cancel")`); err != nil {
		t.Fatal(err)
	}
	assertBrowserPredicate(t, ctx, browser, `(() => {
      const cancel = [...document.querySelectorAll(".operation-detail button")].find((item) => item.textContent === "Cancel");
      if (!cancel) return false;
      cancel.click();
      return true;
    })()`)
	if err := browser.Eventually(ctx, `document.querySelector(".operation-chip.cancelled") !== null`); err != nil {
		t.Fatal(err)
	}
	if status, err := fixture.handler.operations.Wait(ctx, running); err != nil || status.State != operation.StateCancelled {
		t.Fatalf("browser cancellation=(%+v, %v)", status, err)
	}
	if err := browser.Eventually(ctx, `document.getElementById('global-operation-shell').hidden`); err != nil {
		t.Fatal(err)
	}
	assertBrowserPredicate(t, ctx, browser, `(() => {document.querySelector('[aria-label="Close activity"]').click();return !document.querySelector('.activity-dialog').open;})()`)

	// The API key must live in memory for the page session: browser storage
	// stays empty until the operator opts in through the remember control,
	// and opting back out removes the stored key.
	assertBrowserPredicate(t, ctx, browser, `(() => {
      const key = document.querySelector("#api-key");
      const remember = document.querySelector("#api-key-remember");
      if (!key || !remember || remember.checked) return false;
      key.value = "browser-lane-secret";
      key.dispatchEvent(new Event("change", {bubbles:true}));
      if (localStorage.getItem("overgo.apiKey") !== null) return false;
      remember.checked = true;
      remember.dispatchEvent(new Event("change", {bubbles:true}));
      if (localStorage.getItem("overgo.apiKey") !== "browser-lane-secret") return false;
      remember.checked = false;
      remember.dispatchEvent(new Event("change", {bubbles:true}));
      return localStorage.getItem("overgo.apiKey") === null;
    })()`)
	// Model output renders through the sanitizing DOM builder: markup in the
	// text becomes text, never elements, while the markdown subset still
	// produces real formatting nodes.
	assertBrowserPredicate(t, ctx, browser, `(() => {
      const rendered = window.overgo.md("<script>window.pwned=1<\/script> **bold** <img src=x onerror=window.pwned=1>");
      const host = document.createElement("div");
      host.appendChild(rendered);
      return !host.querySelector("script,img") && !window.pwned &&
        host.textContent.includes("<script>") && !!host.querySelector("strong,b");
    })()`)
}

// assertChatAttachmentModeReset exercises fresh uploads after leaving a media slot.
func assertChatAttachmentModeReset(t *testing.T, ctx context.Context, browser *webuilane.Browser) {
	t.Helper()
	assertBrowserPredicate(t, ctx, browser, `(() => {
      const o = window.overgo;
      window.modeResetProbe = {capabilities: o.capabilities, get: o.api.get, upload: o.api.upload, uploads: 0};
      o.capabilities = () => ({...window.modeResetProbe.capabilities(), modes: [{id:"chat", label:"Chat", enabled:true}, {id:"vqa", label:"Image question", enabled:true}]});
      o.api.get = (path, ...args) => path === "/generation/capabilities" ? Promise.resolve([{task:"vqa", recipe:"probe", name:"probe", controls:[{name:"image", type:"artifact", label:"image", media:"image"}]}]) : window.modeResetProbe.get(path, ...args);
      o.api.upload = () => { window.modeResetProbe.uploads++; return Promise.resolve({id:"probe"}); };
      o.openConversation(null); return true;
    })()`)
	defer func() {
		assertBrowserPredicate(t, ctx, browser, `(() => {
          const o = window.overgo, saved = window.modeResetProbe;
          o.capabilities = saved.capabilities; o.api.get = saved.get; o.api.upload = saved.upload;
          delete window.modeResetProbe; o.openConversation(null); return true;
        })()`)
		if err := browser.Eventually(ctx, `!!document.querySelector("#panel-chat.active .card.front-empty button")`); err != nil {
			t.Fatal(err)
		}
	}()
	if err := browser.Eventually(ctx, `!!document.querySelector('.composer select[aria-label="mode"] option[value="vqa"]')`); err != nil {
		t.Fatal(err)
	}
	assertBrowserPredicate(t, ctx, browser, `(() => {
      const mode = document.querySelector('.composer select[aria-label="mode"]');
      mode.value = "vqa"; mode.dispatchEvent(new Event("change")); return true;
    })()`)
	if err := browser.Eventually(ctx, `!!document.querySelector('.mode-controls label.control input')`); err != nil {
		t.Fatal(err)
	}
	assertBrowserPredicate(t, ctx, browser, `(() => {
      const mode = document.querySelector('.composer select[aria-label="mode"]');
      mode.value = "chat"; mode.dispatchEvent(new Event("change"));
      const picker = document.querySelector('.composer input[type="file"]'), files = new DataTransfer();
      files.items.add(new File(["mode reset"], "mode-reset.txt", {type:"text/plain"}));
      picker.files = files.files; picker.dispatchEvent(new Event("change"));
      return window.modeResetProbe.uploads === 0;
    })()`)
	if err := browser.Eventually(ctx, `(() => {
      const attachment = [...document.querySelectorAll('.composer .attachment-row[data-state=ready]')].find(row => row.textContent.includes('mode-reset.txt'));
      return !!attachment && window.modeResetProbe.uploads === 0 && ![...attachment.querySelectorAll('a')].some(link => link.textContent === 'Stored file');
    })()`); err != nil {
		t.Fatal(err)
	}
}

func publishBrowserLaneOperations(t *testing.T, handler *Handler) (artifact.ID, artifact.ID) {
	t.Helper()
	recipeID := testutil.ArtifactID(t, artifact.KindRecipe, "browser-lane-recipe")
	runID := testutil.ArtifactID(t, artifact.KindRun, "browser-lane-recovered-run")
	runningRunID := testutil.ArtifactID(t, artifact.KindRun, "browser-lane-cancelled-run")
	runningEntered := make(chan struct{})
	running, err := handler.operations.Submit(t.Context(), operation.Request{
		Task: recipe.TaskGeneration, Recipe: recipeID,
	}, func(ctx context.Context, reporter operation.Reporter) (operation.Completion, error) {
		close(runningEntered)
		<-ctx.Done()
		return operation.Completion{Run: runningRunID}, ctx.Err()
	})
	if err != nil {
		t.Fatal(err)
	}
	<-runningEntered
	var attempts atomic.Uint32
	blocked, err := handler.operations.Submit(t.Context(), operation.Request{
		Task: recipe.TaskGeneration, Recipe: recipeID,
	}, func(context.Context, operation.Reporter) (operation.Completion, error) {
		if attempts.Add(1) == 1 {
			return operation.Completion{Run: runID}, operatoraction.Recoverable(errors.New("browser approval required"), operatoraction.Block{
				Subject: recipeID, Reason: "approve browser recovery", Evidence: []artifact.ID{},
				Actions: []operatoraction.Action{{Code: "retry-browser", Summary: "Retry browser operation", Argv: []string{"overgo", "retry-browser"}}},
			})
		}
		return operation.Completion{Run: runID}, nil
	})
	if err != nil {
		t.Fatal(err)
	}
	waitBrowserOperationState(t, handler, blocked, operation.StateBlocked)
	return running, blocked
}

// waitBrowserOperationState follows the operation manager's own events
// until the operation reaches the state, subscribing before the first look
// so no transition between the look and the events is lost.
func waitBrowserOperationState(t *testing.T, handler *Handler, id artifact.ID, state operation.State) {
	t.Helper()
	events, stop, err := handler.operations.Subscribe()
	if err != nil {
		t.Fatal(err)
	}
	defer stop()
	reached := func(current operation.Status) bool {
		t.Helper()
		if current.State == state {
			return true
		}
		if current.State == operation.StateCompleted || current.State == operation.StateCancelled || current.State == operation.StateFailed {
			t.Fatalf("browser operation reached %s while waiting for %s: %s", current.State, state, current.Failure)
		}
		return false
	}
	if current, found := handler.operations.Status(id); found && reached(current) {
		return
	}
	for {
		select {
		case event := <-events:
			if event.Status.ID == id && reached(event.Status) {
				return
			}
		case <-t.Context().Done():
			t.Fatal(t.Context().Err())
		}
	}
}

func assertBrowserPredicate(t *testing.T, ctx context.Context, browser *webuilane.Browser, expression string) {
	t.Helper()
	var accepted bool
	if err := browser.Evaluate(ctx, expression, &accepted); err != nil || !accepted {
		t.Fatalf("browser predicate refused: %s: %v", strings.TrimSpace(expression), err)
	}
}

// TestWebUIBrowserFrontPage drives the front page in a real browser
// (professional GUI campaign, gui-quality/accessibility-responsive): every
// control a keyboard user needs is a native focusable element, a real Tab
// key press moves focus on with a visible ring, the inspector lands focus
// on its close control and Escape closes it, the reduced-motion preference
// stops transitions, the light and dark schemes come from the token set,
// and the layout holds without horizontal overflow at desktop, tablet and
// phone widths.
func TestWebUIBrowserFrontPage(t *testing.T) {
	if os.Getenv("OVERGO_WEBUI_LANE") != "1" {
		t.Skip(testskip.ShortIntegration + ": browser acceptance runs through cmd/webui-lane")
	}
	browserPath, err := webuilane.FindBrowser(os.Getenv("OVERGO_BROWSER"))
	if err != nil {
		t.Fatal(err)
	}
	// A declared hosted model gives the picker a row (its key set, so the
	// row serves); the generator holds every turn open until released, so
	// a running turn is there for the Escape key to stop.
	repository, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = repository.Close() })
	t.Setenv("OVERGO_FRONT_PAGE_TEST_KEY", "front-key")
	if _, err := remoteprovider.Declare(t.Context(), repository, remoteprovider.Provider{
		Name: "front", Endpoint: "https://front.example/api/v1", KeyEnvironment: "OVERGO_FRONT_PAGE_TEST_KEY", Model: "vendor/front-model",
	}, transcriptionHTTPCommit); err != nil {
		t.Fatal(err)
	}
	blocking := &fakeGenerator{started: make(chan struct{}), release: make(chan struct{})}
	t.Cleanup(func() { close(blocking.release) })
	handler := newTestHandlerForRepository(t, repository, responseRecipeGenerator(t, blocking))
	defer handler.Close()
	httpServer := httptest.NewServer(handler)
	defer httpServer.Close()
	ctx := t.Context()
	browser, err := webuilane.Open(ctx, browserPath, httpServer.URL+"/")
	if err != nil {
		t.Fatal(err)
	}
	defer browser.Close()
	if err := browser.Eventually(ctx, `!!document.querySelector("#panel-chat.active .composer textarea") && window.overgo.errors.length === 0`); err != nil {
		var page string
		_ = browser.Evaluate(ctx, `JSON.stringify({errors: window.overgo && window.overgo.errors, text: document.body.innerText.slice(0, 600)})`, &page)
		t.Fatalf("%v; page: %s", err, page)
	}
	// The conversation rail renders after its own request, so the check waits for every control.
	if err := browser.Eventually(ctx, `["#workbench-toggle", "#model-pill", ".composer textarea", ".composer .btn", "#inbox-count", "#conversation-list button"]
      .every((selector) => { const node = document.querySelector(selector); return !!node && node.tabIndex >= 0 && ["BUTTON", "TEXTAREA", "SELECT", "INPUT", "A"].includes(node.tagName); })`); err != nil {
		t.Fatal(err)
	}
	assertBrowserPredicate(t, ctx, browser, `(() => { document.querySelector(".composer textarea").focus(); return document.activeElement.tagName === "TEXTAREA"; })()`)
	pressKey(t, ctx, browser, "Tab", 9)
	assertBrowserPredicate(t, ctx, browser, `document.activeElement !== document.body && document.activeElement.tagName !== "TEXTAREA" &&
      getComputedStyle(document.activeElement).outlineStyle !== "none" && parseFloat(getComputedStyle(document.activeElement).outlineWidth) > 0`)
	assertBrowserPredicate(t, ctx, browser, `(() => { window.overgo.inspectTurn("resp_missing"); return true; })()`)
	if err := browser.Eventually(ctx, `!document.querySelector("#inspector").hidden && document.activeElement.getAttribute("aria-label") === "close the inspector"`); err != nil {
		t.Fatal(err)
	}
	pressKey(t, ctx, browser, "Escape", 27)
	if err := browser.Eventually(ctx, `document.querySelector("#inspector").hidden`); err != nil {
		t.Fatal(err)
	}
	// Keyboard path to a model: Alt+M opens the picker, its first row takes
	// focus, Enter serves it (without the proxy the page says a relaunch is
	// the way), Escape stops a running turn from anywhere on the page.
	pressKey(t, ctx, browser, "m", 77, keyModifierAlt)
	if err := browser.Eventually(ctx, `document.activeElement.classList.contains("row") && document.activeElement.querySelector(".mono").textContent === "front-model"`); err != nil {
		t.Fatal(err)
	}
	pressKey(t, ctx, browser, "Enter", 13)
	if err := browser.Eventually(ctx, `document.querySelector(".topbar .card").textContent.includes("without the swap proxy")`); err != nil {
		t.Fatal(err)
	}
	assertBrowserPredicate(t, ctx, browser, `(() => { document.querySelector("#model-pill").click(); return !document.querySelector(".topbar .card"); })()`)
	say(t, ctx, browser, "hello")
	if err := browser.Eventually(ctx, `document.querySelector(".composer .btn").disabled`); err != nil {
		t.Fatal(err)
	}
	pressKey(t, ctx, browser, "Escape", 27)
	if err := browser.Eventually(ctx, `!document.querySelector(".composer .btn").disabled`); err != nil {
		t.Fatal(err)
	}
	emulateMedia(t, ctx, browser, "prefers-reduced-motion", "reduce")
	assertBrowserPredicate(t, ctx, browser, `matchMedia("(prefers-reduced-motion: reduce)").matches && parseFloat(getComputedStyle(document.querySelector("#server-dot")).transitionDuration) < 0.001`)
	for scheme, background := range map[string]string{"light": "rgb(250, 250, 247)", "dark": "rgb(21, 24, 25)"} {
		emulateMedia(t, ctx, browser, "prefers-color-scheme", scheme)
		if err := browser.Eventually(ctx, `matchMedia("(prefers-color-scheme: `+scheme+`)").matches && getComputedStyle(document.body).backgroundColor === "`+background+`"`); err != nil {
			t.Fatalf("scheme %s: %v", scheme, err)
		}
	}
	for _, width := range []int{1280, 820, 390} {
		if err := browser.SetViewport(ctx, width, 900); err != nil {
			t.Fatal(err)
		}
		if err := browser.Eventually(ctx, `document.documentElement.scrollWidth <= innerWidth &&
        document.querySelector(".composer textarea").getBoundingClientRect().right <= innerWidth &&
        document.querySelector(".topbar").getBoundingClientRect().right <= innerWidth`); err != nil {
			t.Fatalf("width %d: %v", width, err)
		}
	}
	// Phone width with a conversation: an unbroken line wraps inside its
	// bubble and nothing scrolls sideways; Escape stops the turn it opened.
	say(t, ctx, browser, strings.Repeat("overgo", 60))
	if err := browser.Eventually(ctx, `document.querySelectorAll("#panel-chat .msg.user").length === 2 && document.documentElement.scrollWidth <= innerWidth &&
      [...document.querySelectorAll("#panel-chat .msg")].every((message) => message.getBoundingClientRect().right <= innerWidth)`); err != nil {
		var page string
		_ = browser.Evaluate(ctx, `JSON.stringify({width: innerWidth, scroll: document.documentElement.scrollWidth, messages: [...document.querySelectorAll("#panel-chat .msg")].map((m) => m.getBoundingClientRect().right)})`, &page)
		t.Fatalf("phone conversation: %v; %s", err, page)
	}
	pressKey(t, ctx, browser, "Escape", 27)
	if err := browser.Eventually(ctx, `!document.querySelector(".composer .btn").disabled`); err != nil {
		t.Fatal(err)
	}
}

// keyModifierAlt is the DevTools key event modifier bit for Alt.
const keyModifierAlt = 1

// pressKey sends one real key press through the browser's input domain, so
// focus-visible and default key handling behave as they do for a person;
// modifiers are the DevTools modifier bits held during the press.
func pressKey(t *testing.T, ctx context.Context, browser *webuilane.Browser, key string, code int, modifiers ...int) {
	t.Helper()
	modifier := 0
	for _, bit := range modifiers {
		modifier |= bit
	}
	for _, kind := range []string{"keyDown", "keyUp"} {
		event := map[string]any{
			"type": kind, "key": key, "code": key, "windowsVirtualKeyCode": code, "nativeVirtualKeyCode": code, "modifiers": modifier,
		}
		if key == "Enter" && kind == "keyDown" {
			event["text"] = "\r" // native button activation requires the character event
		}
		if err := browser.Call(ctx, "Input.dispatchKeyEvent", event, nil); err != nil {
			t.Fatal(err)
		}
	}
}

// emulateMedia sets one media feature the page's stylesheet responds to.
func emulateMedia(t *testing.T, ctx context.Context, browser *webuilane.Browser, name, value string) {
	t.Helper()
	if err := browser.Call(ctx, "Emulation.setEmulatedMedia", map[string]any{
		"features": []map[string]string{{"name": name, "value": value}},
	}, nil); err != nil {
		t.Fatal(err)
	}
}
