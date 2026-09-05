package server

import (
	"context"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
	"time"

	"overgo/internal/modelswap"
	"overgo/internal/webuilane"
)

// TestWebUIBrowserFirstRun drives the front page end to end against a
// served model through the real swap proxy (professional GUI campaign,
// gui-quality/acceptance-lane). cmd/webui-lane prepares the journey: the
// server binary, the store and the smallest servable models; without them
// the journey reports UNAVAILABLE and is skipped. The journey: the page boots once
// the default model serves (the pill names it, the proxy dot is on), the
// first message receives a streamed reply with the context meter filled,
// an image attachment is either taken to a grounded reply (a vision-capable
// model) or refused with the declared reason, the turn's inspector opens,
// an agent created through the API runs a tool step and a chat turn from
// the same composer, the served model switches through the picker and the
// composer re-derives, and a running turn is stopped from the composer.
func TestWebUIBrowserFirstRun(t *testing.T) {
	if os.Getenv("OVERGO_WEBUI_LANE") != "1" {
		return
	}
	binary, store := os.Getenv("OVERGO_WEBUI_LANE_SERVER"), os.Getenv("OVERGO_WEBUI_LANE_STORE")
	modelName, modelLocation := os.Getenv("OVERGO_WEBUI_LANE_MODEL"), os.Getenv("OVERGO_WEBUI_LANE_MODEL_LOCATION")
	if binary == "" || store == "" || modelName == "" || modelLocation == "" {
		t.Skip("first-run journey UNAVAILABLE: cmd/webui-lane prepared no served model")
	}
	t.Logf("first-run journey: model %s at %s", modelName, modelLocation)
	browserPath, err := webuilane.FindBrowser(os.Getenv("OVERGO_BROWSER"))
	if err != nil {
		t.Fatal(err)
	}
	supervisor, err := modelswap.New(modelswap.ServerLauncher{Binary: binary, Store: store, Dir: filepath.Dir(store)}, 0)
	if err != nil {
		t.Fatal(err)
	}
	defer supervisor.Close()
	proxy := &modelswap.Proxy{
		Supervisor: supervisor, Resolver: &modelswap.CatalogResolver{Store: store, Limit: 256},
		Default: modelswap.Servable{Name: modelName, Location: modelLocation},
	}
	front := httptest.NewServer(proxy)
	defer front.Close()
	ctx, cancel := context.WithTimeoutCause(t.Context(), 5*time.Minute, errors.New("webui lane: the first-run journey did not complete"))
	defer cancel()
	browser, err := webuilane.Open(ctx, browserPath, front.URL+"/")
	if err != nil {
		t.Fatal(err)
	}
	defer browser.Close()
	settle := func(what, expression string) {
		t.Helper()
		if err := browser.Eventually(ctx, expression); err != nil {
			var page string
			dump, done := context.WithTimeoutCause(t.Context(), 10*time.Second, errors.New("webui lane: page dump timed out"))
			defer done()
			_ = browser.Evaluate(dump, `JSON.stringify({errors: window.overgo && window.overgo.errors, text: document.body.innerText.slice(0, 800)})`, &page)
			t.Fatalf("%s: %v; page: %s", what, err, page)
		}
	}

	// 1. First run: the default model serves, the pill names it, the proxy dot is on.
	settle("front page over the served model", `!!document.querySelector("#panel-chat.active .composer textarea") &&
      document.querySelector("#model-pill").textContent === `+strconv.Quote(modelName)+` &&
      document.querySelector("#proxy-dot").classList.contains("ok") && window.overgo.errors.length === 0`)

	// 2. The first message streams a reply and fills the context meter.
	say(t, ctx, browser, "Reply with the single word hello.")
	settle("streamed first reply", `document.querySelectorAll("#panel-chat .msg.assistant").length === 1 &&
      document.querySelector("#panel-chat .msg.assistant .body").textContent.trim().length > 0 &&
      document.querySelector('[aria-label="context meter"]').textContent.includes("Input") && !document.querySelector(".composer .btn").disabled`)

	// 3. An image attachment: a grounded reply from a vision-capable model, or the declared refusal.
	var vision bool
	if err := browser.Evaluate(ctx, `!!(window.overgo.capabilities().modalities || {}).image`, &vision); err != nil {
		t.Fatal(err)
	}
	assertBrowserPredicate(t, ctx, browser, `(() => {
      const picker = document.querySelector(".composer input[type=file]");
      const transfer = new DataTransfer();
      transfer.items.add(new File([new Uint8Array([137, 80, 78, 71, 13, 10, 26, 10])], "tiny.png", { type: "image/png" }));
      picker.files = transfer.files;
      picker.dispatchEvent(new Event("change"));
      return true;
    })()`)
	if vision {
		settle("image attached", `!!document.querySelector(".composer .card:not(.refused)")`)
		say(t, ctx, browser, "Describe the attached image in one sentence.")
		settle("grounded reply", `document.querySelectorAll("#panel-chat .msg.assistant").length === 2 && !document.querySelector(".composer .btn").disabled`)
	} else {
		settle("image refused with the declared reason", `(() => {
        const card = document.querySelector(".composer .card.refused");
        const reason = ((window.overgo.capabilities().media || {}).refusals || {}).image || "";
        return !!card && reason !== "" && card.textContent.includes(reason);
      })()`)
		assertBrowserPredicate(t, ctx, browser, `(() => { document.querySelector(".composer .card.refused button").click(); return !document.querySelector(".composer .card.refused"); })()`)
		t.Log("vision leg: the served model accepts no images, so the refusal contract was proven instead")
	}

	// 4. The inspector opens over the first turn with its run record.
	assertBrowserPredicate(t, ctx, browser, `(() => { const button = document.querySelector("#panel-chat .msg.assistant .role button.link-button"); if (!button) return false; button.click(); return true; })()`)
	settle("inspector over the turn", `!document.querySelector("#inspector").hidden && document.querySelector("#inspector").textContent.includes("done")`)
	pressKey(t, ctx, browser, "Escape", 27)
	settle("inspector closed", `document.querySelector("#inspector").hidden`)

	// 5. An agent created through the API runs a tool step and a chat turn from this composer.
	ensureLaneAgent(t, front.URL)
	assertBrowserPredicate(t, ctx, browser, `(() => { window.overgo.openConversation(null); return true; })()`)
	settle("agent mode offered", `(() => {
      const select = document.querySelector('.composer select[aria-label="mode"]');
      return !!select && [...select.options].some((option) => option.value === "agent");
    })()`)
	assertBrowserPredicate(t, ctx, browser, `(() => {
      const select = document.querySelector('.composer select[aria-label="mode"]');
      select.value = "agent";
      select.dispatchEvent(new Event("change"));
      return !document.querySelector(".agent-session").hidden;
    })()`)
	assertBrowserPredicate(t, ctx, browser, `(() => {
      const picker = document.querySelector('.agent-session select[aria-label="agent"]');
      picker.value = "lane-agent";
      picker.dispatchEvent(new Event("change"));
      return picker.value === "lane-agent";
    })()`)
	assertBrowserPredicate(t, ctx, browser, `[...document.querySelector(".agent-session select:not([aria-label])").options].some((option) => option.value === "store.head")`)
	assertBrowserPredicate(t, ctx, browser, `(() => {
      const execute = [...document.querySelectorAll(".agent-session button")].find((button) => button.textContent === "Execute inspection");
      if (!execute || execute.disabled) return false;
      execute.click();
      return true;
    })()`)
	settle("tool step card", `[...document.querySelectorAll("#panel-chat .tool-call .tag")].some((tag) => tag.textContent.startsWith("done")) &&
      document.querySelector('[aria-label="guardrails"]').textContent.includes("remaining")`)
	say(t, ctx, browser, "Say hi in one word.")
	settle("agent chat reply", `document.querySelectorAll("#panel-chat .msg.assistant").length >= 1 &&
      [...document.querySelectorAll("#panel-chat .msg.assistant .body")].at(-1).textContent.trim().length > 0 && !document.querySelector(".composer .btn").disabled`)

	// 6. Switching the served model through the picker re-derives the composer: the
	// picker offers every other servable model; the first one is served.
	assertBrowserPredicate(t, ctx, browser, `(() => { document.querySelector("#model-pill").click(); return true; })()`)
	settle("picker lists the servable models", `document.querySelectorAll(".topbar .card .row .mono").length > 0`)
	var switchName string
	if err := browser.Evaluate(ctx, `(() => {
      const row = [...document.querySelectorAll(".topbar .card .row")].find((node) => node.querySelector(".mono") && node.querySelector(".mono").textContent !== `+strconv.Quote(modelName)+` && [...node.querySelectorAll("button")].some((button) => button.textContent === "serve"));
      if (!row) return "";
      const name = row.querySelector(".mono").textContent;
      [...row.querySelectorAll("button")].find((button) => button.textContent === "serve").click();
      return name;
    })()`, &switchName); err != nil {
		t.Fatal(err)
	}
	if switchName != "" {
		settle("model switched and the composer re-derived", `document.querySelector("#model-pill").textContent === `+strconv.Quote(switchName)+` &&
        !!document.querySelector("#panel-chat.active .composer textarea") && window.overgo.capabilities().id !== "" &&
        document.querySelector(".front-empty h2").textContent !== `+strconv.Quote(modelName)+``)
		t.Logf("switched from %s to %s", modelName, switchName)
	} else {
		t.Log("switch leg not taken: the store holds one servable model")
	}

	// 7. A running turn stops from the composer.
	assertBrowserPredicate(t, ctx, browser, `(() => {
      const select = document.querySelector('.composer select[aria-label="mode"]');
      if (select) { select.value = "chat"; select.dispatchEvent(new Event("change")); }
      const limit = [...document.querySelectorAll('.composer input[type="number"]')].at(-1);
      limit.value = "400";
      return true;
    })()`)
	say(t, ctx, browser, "Write a very long story about a lighthouse, at least twenty paragraphs.")
	settle("turn running", `!document.querySelector(".composer .btn").disabled === false`)
	assertBrowserPredicate(t, ctx, browser, `(() => { const stop = [...document.querySelectorAll(".composer .btn")].find((button) => button.textContent === "stop"); if (!stop) return false; stop.click(); return true; })()`)
	settle("turn stopped", `!document.querySelector(".composer .btn").disabled &&
      ([...document.querySelectorAll("#panel-chat .msg.assistant .body")].at(-1).textContent.includes("[stopped]") || !!document.querySelector("#panel-chat .msg.error"))`)
}

// say types one message into the composer through the browser's input
// domain and presses Enter, the way a person sends a turn.
func say(t *testing.T, ctx context.Context, browser *webuilane.Browser, text string) {
	t.Helper()
	assertBrowserPredicate(t, ctx, browser, `(() => { const input = document.querySelector(".composer textarea"); input.focus(); input.value = ""; return document.activeElement === input; })()`)
	if err := browser.Call(ctx, "Input.insertText", map[string]any{"text": text}, nil); err != nil {
		t.Fatal(err)
	}
	pressKey(t, ctx, browser, "Enter", 13)
}

// ensureLaneAgent creates the journey's agent through the API once; a store
// that already holds it from an earlier run keeps its active definition.
func ensureLaneAgent(t *testing.T, base string) {
	t.Helper()
	listed, err := http.Get(base + "/agents")
	if err != nil {
		t.Fatal(err)
	}
	inventory, _ := io.ReadAll(listed.Body)
	listed.Body.Close()
	if strings.Contains(string(inventory), `"name":"lane-agent"`) {
		return
	}
	created, err := http.Post(base+"/agents/create", "application/json",
		strings.NewReader(`{"name":"lane-agent","instructions":"You are the lane agent. Answer briefly.","tools":["store.head"]}`))
	if err != nil {
		t.Fatal(err)
	}
	body, _ := io.ReadAll(created.Body)
	created.Body.Close()
	if created.StatusCode != http.StatusOK {
		t.Fatalf("agent creation status = %d body=%s", created.StatusCode, body)
	}
}
