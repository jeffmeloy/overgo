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
	"time"

	"overgo/internal/artifact"
	"overgo/internal/operation"
	"overgo/internal/operatoraction"
	"overgo/internal/recipe"
	"overgo/internal/testutil"
	"overgo/internal/webuilane"
)

func TestWebUIBrowserAcceptance(t *testing.T) {
	if os.Getenv("OVERGO_WEBUI_LANE") != "1" {
		if _, err := webuilane.FindBrowser(os.Getenv("OVERGO_BROWSER")); err != nil {
			t.Fatalf("browser acceptance prerequisite is absent; run go run ./cmd/webui-lane: %v", err)
		}
		return
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

	ctx, cancel := context.WithTimeout(t.Context(), 90*time.Second)
	defer cancel()
	browser, err := webuilane.Open(ctx, browserPath, httpServer.URL+"/app.html#automations")
	if err != nil {
		t.Fatal(err)
	}
	defer browser.Close()
	probeCtx, probeCancel := context.WithTimeoutCause(ctx, 30*time.Second, errors.New("webui lane: the shell did not present the automations panel"))
	defer probeCancel()
	if err := browser.Eventually(probeCtx, `!!document.querySelector("#panel-automations.active .schema-form") &&
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

	if err := browser.SetViewport(ctx, 1280, 900); err != nil {
		t.Fatal(err)
	}
	assertBrowserPredicate(t, ctx, browser, `getComputedStyle(document.querySelector(".shell")).flexDirection === "row" &&
      document.querySelector(".sidebar").getBoundingClientRect().width < innerWidth`)
	if err := browser.SetViewport(ctx, 640, 900); err != nil {
		t.Fatal(err)
	}
	if err := browser.Eventually(ctx, `getComputedStyle(document.querySelector(".shell")).flexDirection === "column" &&
      document.querySelector(".sidebar").getBoundingClientRect().width <= innerWidth`); err != nil {
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
	if err := browser.Eventually(ctx, `document.querySelector(".operation-strip").textContent.includes("0 active")`); err != nil {
		t.Fatal(err)
	}

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

func waitBrowserOperationState(t *testing.T, handler *Handler, id artifact.ID, state operation.State) {
	t.Helper()
	ticker := time.Tick(time.Millisecond)
	for {
		if current, found := handler.operations.Status(id); found {
			if current.State == state {
				return
			}
			if current.State == operation.StateCompleted || current.State == operation.StateCancelled || current.State == operation.StateFailed {
				t.Fatalf("browser operation reached %s while waiting for %s: %s", current.State, state, current.Failure)
			}
		}
		select {
		case <-t.Context().Done():
			t.Fatal(t.Context().Err())
		case <-ticker:
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
