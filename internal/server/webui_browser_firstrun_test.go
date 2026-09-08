package server

import (
	"bytes"
	"context"
	"debug/buildinfo"
	"encoding/base64"
	"errors"
	"image"
	"image/color"
	"image/png"
	"io"
	"math"
	"net/http"
	"net/http/httptest"
	"os"
	"path"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
	"time"

	"overgo/internal/modelswap"
	"overgo/internal/overgodb"
	"overgo/internal/remoteprovider"
	"overgo/internal/remoterelay"
	"overgo/internal/remoterelay/relaytest"
	"overgo/internal/runrecord"
	"overgo/internal/testevidence"
	"overgo/internal/testutil"
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
// composer re-derives, a running turn is stopped from the composer, an
// image comes out of the cheapest declared image model as an artifact
// card and goes back in as the next attachment, and a speech clip comes
// out of the declared speech model with a voice its artifact exports
// (gui-journey-modalities). When the default model refuses images and the
// lane names a model whose store declares a projector, that model serves
// the image-in leg.
func TestWebUIBrowserFirstRun(t *testing.T) {
	if os.Getenv("OVERGO_WEBUI_LANE") != "1" {
		t.Skip(testevidence.ShortIntegrationSkip + ": the journey runs through cmd/webui-lane")
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
	// Leg 13's remote model: fake loopback provider declared in the lane
	// store before the proxy child opens it; key in this environment (child
	// inherits); retired after the journey.
	remoteName := declareLaneRemote(t, store, "webui-lane", "OVERGO_WEBUI_LANE_REMOTE_KEY", "lane-key", []string{"Hello", " from the relay"})
	// Leg 14's keyless model: declared with its variable empty; the page
	// enters the key, which the proxy (this process) and the child take.
	t.Setenv("OVERGO_WEBUI_LANE_ENTRY_KEY", "")
	// Leg 15's page-declared provider: its key set before any child launches, so the running child can list and serve it.
	t.Setenv("OVERGO_WEBUI_LANE_PAGE_KEY", "lane-key")
	entryName := declareLaneRemote(t, store, "webui-lane-entry", "OVERGO_WEBUI_LANE_ENTRY_KEY", "", []string{"Hello", " after the key"})
	supervisor, err := modelswap.New(modelswap.ServerLauncher{Binary: binary, Store: store, Dir: filepath.Dir(store)}, 0)
	if err != nil {
		t.Fatal(err)
	}
	defer supervisor.Close()
	repository, err := overgodb.Open(store)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = repository.Close() })
	resolver := &modelswap.CatalogResolver{Store: repository, Limit: 256}
	// No default: the journey opens on the cold proxy, as overgo_gui.bat
	// without a model does, and chooses the first model from the picker.
	proxy := &modelswap.Proxy{
		Supervisor: supervisor, Resolver: resolver, Keys: resolver,
		Idle: &IdleShell{Catalog: resolver.Catalog, Repository: repository, Intake: laneIdleIntake()},
	}
	front := httptest.NewServer(proxy)
	defer front.Close()
	ctx, cancel := context.WithTimeoutCause(t.Context(), 8*time.Minute, errors.New("webui lane: the first-run journey did not complete"))
	defer cancel()
	browser, err := webuilane.Open(ctx, browserPath, front.URL+"/")
	if err != nil {
		t.Fatal(err)
	}
	defer browser.Close()
	// Each step settles within its own bound so a leg that cannot settle
	// fails with the page's state rather than spending the journey's budget.
	settle := func(what, expression string) {
		t.Helper()
		step, done := context.WithTimeoutCause(ctx, 2*time.Minute, errors.New("webui lane: the step did not settle"))
		defer done()
		if err := browser.Eventually(step, expression); err != nil {
			var page string
			dump, finish := context.WithTimeoutCause(t.Context(), 10*time.Second, errors.New("webui lane: page dump timed out"))
			defer finish()
			_ = browser.Evaluate(dump, `JSON.stringify({errors: window.overgo && window.overgo.errors, pill: (document.querySelector("#model-pill") || {}).textContent,
        capability: (window.overgo.capabilities() || {}).id, composers: [...document.querySelectorAll(".composer")].map((node) => JSON.stringify(node.dataset)),
        chips: [...document.querySelectorAll(".operation-chip")].map((chip) => chip.textContent),
        assistant: document.querySelectorAll("#panel-chat .msg.assistant").length, busy: (document.querySelector(".composer .btn") || {}).disabled,
        failures: [...document.querySelectorAll("#panel-chat .msg.error .body")].map((node) => node.textContent.slice(0, 200)),
        last: ([...document.querySelectorAll("#panel-chat .msg.assistant .body")].at(-1) || {}).textContent, text: document.body.innerText.slice(0, 120),
        notes: [...document.querySelectorAll(".note, .err-banner")].map((node) => node.textContent.slice(0, 160)).filter(Boolean),
        predicate: (() => { try { return String(`+expression+`); } catch (failure) { return "throws: " + failure; } })()})`, &page)
			t.Fatalf("%s: %v; page: %s", what, err, page)
		}
	}
	// captureStates: the page as it stands at this leg, captured and audited
	// at both viewports (written when the lane writes screens); the journey
	// continues at the desktop size.
	var layoutFindings []webuilane.StateFinding
	defer func() {
		// Keep nonfatal layout failures beside the terminal verdict: the
		// gate reports a bounded tail after this long real-model journey.
		for _, finding := range layoutFindings {
			t.Error(finding)
		}
	}()
	captureStates := func(name string) {
		t.Helper()
		for _, viewport := range webuilane.ScreenViewports {
			findings, err := webuilane.CaptureState(ctx, browser, os.Getenv("OVERGO_WEBUI_LANE_SCREENS"), viewport, name)
			if err != nil {
				t.Fatal(err)
			}
			layoutFindings = append(layoutFindings, findings...)
		}
		if err := browser.SetViewport(ctx, webuilane.ScreenViewports[0].Width, webuilane.ScreenViewports[0].Height); err != nil {
			t.Fatal(err)
		}
		t.Logf("states leg: %s captured at %d viewports", name, len(webuilane.ScreenViewports))
	}
	// openPicker: picker open over the catalog rows. A picker left open after
	// a swap shows the swap's note, not rows: a click closes it, a second
	// opens it afresh.
	openPicker := func() {
		t.Helper()
		assertBrowserPredicate(t, ctx, browser, `(() => {
      const pill = document.querySelector("#model-pill");
      if (!document.querySelector(".topbar .card .row .mono")) { pill.click(); if (!document.querySelector(".topbar .card")) pill.click(); }
      return true;
    })()`)
		settle("picker lists the servable models", `document.querySelectorAll(".topbar .card .row .mono").length > 0`)
	}
	// attachTinyPNG: a real image a projector can encode (a small red
	// square) handed to the composer's file input as file bytes.
	attachTinyPNG := func() {
		t.Helper()
		assertBrowserPredicate(t, ctx, browser, `(() => {
      const picker = document.querySelector(".composer input[type=file]");
      const transfer = new DataTransfer();
      transfer.items.add(new File([Uint8Array.from(atob(`+strconv.Quote(tinyPNGBase64(t))+`), (char) => char.charCodeAt(0))], "tiny.png", { type: "image/png" }));
      picker.files = transfer.files;
      picker.dispatchEvent(new Event("change"));
      return true;
    })()`)
	}
	// switchTo serves another model through the picker and waits for the
	// pill to name it and the composer to re-derive from its capabilities.
	switchTo := func(name string) {
		t.Helper()
		var previous string
		if err := browser.Evaluate(ctx, `window.overgo.capabilities().id`, &previous); err != nil {
			t.Fatal(err)
		}
		// The page re-derives by remounting: the composer in place before the
		// switch is marked, and the switch has landed once a new one stands.
		assertBrowserPredicate(t, ctx, browser, `(() => { document.querySelector(".composer").dataset.laneBefore = "1"; return true; })()`)
		openPicker()
		assertBrowserPredicate(t, ctx, browser, `(() => {
      const row = [...document.querySelectorAll(".topbar .card .row")].find((node) => node.querySelector(".mono") && node.querySelector(".mono").textContent === `+strconv.Quote(name)+`);
      if (!row) return false;
      [...row.querySelectorAll("button")].find((button) => button.textContent === "serve").click();
      return true;
    })()`)
		// The pill names the served file; the welcome card, when the
		// conversation is empty, names the model by its own declared name.
		settle("model switched and the composer re-derived", `document.querySelector("#model-pill").textContent === `+strconv.Quote(name)+` &&
      !!document.querySelector("#panel-chat.active .composer textarea") && !document.querySelector(".composer").dataset.laneBefore &&
      !window.overgo.modelSwitching() && !document.querySelector('dialog[aria-label="Choose a model"][open]') && !document.querySelector('.send-button').disabled &&
      (window.overgo.capabilities() || {}).id !== `+strconv.Quote(previous)+``)
	}

	// 0. Cold start: the proxy serves the shell with no child; the pill names
	// no model, the front page says how to choose, and the picker's serve
	// launches the first child.
	settle("cold page over the proxy with no child", `document.querySelector("#model-pill").textContent === "no model serves" &&
      !!document.querySelector("#cold-start") && !document.querySelector("#panel-chat.active") &&
      document.querySelector("#proxy-dot").classList.contains("ok") && window.overgo.errors.length === 0`)
	// 0b. The Library is the one tab the cold page serves: the page's provider
	// (leg 15's) declares from here, before any child runs, through the same
	// intake the launcher binds; a modified tree sees the refusal, the gate's
	// clean tree sees the declared location, which the journey serves later.
	pageFake, _ := relaytest.ServeListing(t, "", "lane-key", []string{"Hello", " from the page's provider"}, []relaytest.Model{{ID: "lane/page-model", Name: "Page model", ContextLength: 4096}})
	assertBrowserPredicate(t, ctx, browser, `(() => { [...document.querySelectorAll("#cold-start button")].find((button) => button.textContent === "Open the Library").click(); return true; })()`)
	settle("the cold page's library shows the provider form and the catalog", `!document.querySelector("#cold-start") && !!document.querySelector("#panel-library.active input[aria-label='provider name']") &&
      document.querySelectorAll("#panel-library tr").length > 1`)
	t.Cleanup(func() { retireLaneRemote(t, store, "remote://webui-lane-page/lane/page-model") })
	assertBrowserPredicate(t, ctx, browser, `(() => {
      const fill = (label, value) => { document.querySelector("input[aria-label='provider " + label + "']").value = value; };
      fill("name", "webui-lane-page"); fill("endpoint", `+strconv.Quote(pageFake.URL)+`); fill("key variable", "OVERGO_WEBUI_LANE_PAGE_KEY");
      [...document.querySelectorAll("button")].find((button) => button.textContent === "list the provider's models").click(); return true; })()`)
	settle("the provider lists its model on the cold page", `[...document.querySelectorAll("button")].some((button) => button.textContent === "lane/page-model · 4096")`)
	assertBrowserPredicate(t, ctx, browser, `(() => {
      [...document.querySelectorAll("button")].find((button) => button.textContent === "lane/page-model · 4096").click();
      const value = (label) => document.querySelector("input[aria-label='provider " + label + "']").value;
      if (value("model ids (comma-separated)") !== "lane/page-model" || value("context length (optional)") !== "4096") return false;
      [...document.querySelectorAll("button")].find((button) => button.textContent === "declare a hosted provider").click(); return true; })()`)
	const declaredTag = `[...document.querySelectorAll(".tag")].some((tag) => tag.textContent.startsWith("remote://webui-lane-page/lane/page-model"))`
	settle("the cold-page declaration answers with its location or the modified-tree refusal", declaredTag+` ||
      [...document.querySelectorAll(".note")].some((note) => note.textContent.includes("worktree is dirty"))`)
	var declaredOnPage bool
	if err := browser.Evaluate(ctx, declaredTag, &declaredOnPage); err != nil {
		t.Fatal(err)
	}
	t.Logf("cold-page declaration leg: declared on the page %v", declaredOnPage)
	// The hash no longer names the Library, so the served model's remount lands on the front page.
	assertBrowserPredicate(t, ctx, browser, `(() => { history.replaceState(null, "", location.pathname); document.querySelector("#model-pill").click(); return true; })()`)
	settle("picker lists the store's models on the cold page", `document.querySelectorAll(".topbar .card .row .mono").length > 0`)
	assertBrowserPredicate(t, ctx, browser, `(() => {
      const row = [...document.querySelectorAll(".topbar .card .row")].find((node) => node.querySelector(".mono") && node.querySelector(".mono").textContent === `+strconv.Quote(modelName)+`);
      if (!row) return false;
      [...row.querySelectorAll("button")].find((button) => button.textContent === "serve").click();
      return true;
    })()`)

	// 1. First run: the chosen model serves, the pill names it, the proxy dot is on, the cold card is gone.
	settle("front page over the served model", `!document.querySelector("#cold-start") && !!document.querySelector("#panel-chat.active .composer textarea") &&
      document.querySelector("#model-pill").textContent === `+strconv.Quote(modelName)+` &&
      !window.overgo.modelSwitching() && !document.querySelector('dialog[aria-label="Choose a model"][open]') && !document.querySelector('.send-button').disabled &&
      document.querySelector("#proxy-dot").classList.contains("ok") && window.overgo.errors.length === 0`)
	t.Logf("cold-start leg: the picker served %s from the cold page and the front page followed", modelName)

	// 2. The first message streams a reply and fills the context meter.
	say(t, ctx, browser, "Reply with the single word hello.")
	settle("streamed first reply", `document.querySelectorAll("#panel-chat .msg.assistant").length === 1 &&
      document.querySelector("#panel-chat .msg.assistant .body").textContent.trim().length > 0 &&
      document.querySelector('[aria-label="context meter"]').textContent.includes("Input") && !document.querySelector(".composer .btn").disabled`)

	// 2b. The conversation renames in place: the rail's rename control turns the
	// title into a field, Enter saves through the label route, the rail relists.
	settle("the rail lists the conversation", `!!document.querySelector('#history-rows .conversation')`)
	assertBrowserPredicate(t, ctx, browser, `(() => {
      const row=document.querySelector('#history-rows .conversation');row.querySelector('.history-options').click();row.querySelector('[aria-label="rename conversation"]').click();
      const field=row.querySelector('input');field.value='renamed by the lane';field.focus();return true;
    })()`)
	pressKey(t, ctx, browser, "Enter", 13)
	settle("the rail shows the new title", `[...document.querySelectorAll('#conversation-list .conversation-title')].some(node=>node.textContent==='renamed by the lane')`)
	t.Log("rename leg: the conversation renamed in place from the rail")
	captureStates("thread-reply")

	// 3. The inspector opens over the first turn with its run record.
	assertBrowserPredicate(t, ctx, browser, `(() => { const button = document.querySelector("#panel-chat .msg.assistant .role button.link-button"); if (!button) return false; button.click(); return true; })()`)
	settle("inspector over the turn", `!document.querySelector("#inspector").hidden && document.querySelector("#inspector").textContent.includes("done")`)
	pressKey(t, ctx, browser, "Escape", 27)
	settle("inspector closed", `document.querySelector("#inspector").hidden`)

	// 4. An image attachment: a grounded reply from a vision-capable model, or the declared refusal.
	// When the default model refuses images and the lane named a model whose
	// store declares a projector, that model serves this leg (multimodal in);
	// the leg runs after the agent leg so the agent's turns keep the default
	// model, and the switch it makes precedes the picker leg.
	imageLeg := func() {
		var vision bool
		if err := browser.Evaluate(ctx, `!!(window.overgo.capabilities().modalities || {}).image`, &vision); err != nil {
			t.Fatal(err)
		}
		if multimodal := os.Getenv("OVERGO_WEBUI_LANE_MULTIMODAL_MODEL"); !vision && multimodal != "" && multimodal != modelName {
			switchTo(multimodal)
			if err := browser.Evaluate(ctx, `!!(window.overgo.capabilities().modalities || {}).image`, &vision); err != nil {
				t.Fatal(err)
			}
			t.Logf("image-in leg: switched to %s, image input declared %v", multimodal, vision)
		}
		attachTinyPNG()
		if vision {
			settle("image attached", `!!document.querySelector('.composer .attachment-row[data-state=ready]')`)
			// A switch may have opened a fresh conversation: the reply is the one
			// assistant turn beyond what the thread held before the image was sent.
			var before int
			if err := browser.Evaluate(ctx, `document.querySelectorAll("#panel-chat .msg.assistant").length`, &before); err != nil {
				t.Fatal(err)
			}
			say(t, ctx, browser, "Describe the attached image in one sentence.")
			settle("grounded reply", `document.querySelectorAll("#panel-chat .msg.assistant").length === `+strconv.Itoa(before+1)+` &&
      [...document.querySelectorAll("#panel-chat .msg.assistant .body")].at(-1).textContent.trim().length > 0 && !document.querySelector(".composer .btn").disabled`)
			t.Log("image-in leg: the multimodal model answered the attached image")
		} else {
			settle("image refused with the declared reason", `(() => {
        const card = document.querySelector('.composer .attachment-row[data-state=refused]');
        const reason = ((window.overgo.capabilities().media || {}).refusals || {}).image || "";
        return !!card && reason !== "" && card.textContent.includes(reason);
      })()`)
			assertBrowserPredicate(t, ctx, browser, `(() => { document.querySelector('.composer .attachment-row[data-state=refused] button[aria-label^="Remove "]').click(); return !document.querySelector('.composer .attachment-row[data-state=refused]'); })()`)
			t.Log("vision leg: the served model accepts no images, so the refusal contract was proven instead")
		}
	}

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
	assertBrowserPredicate(t, ctx, browser, `[...document.querySelector(".agent-session select[aria-label='tool']").options].some((option) => option.value === "store.head")`)
	assertBrowserPredicate(t, ctx, browser, `(() => {
      const execute = [...document.querySelectorAll(".agent-session button")].find((button) => button.textContent === "Execute inspection");
      if (!execute || execute.disabled) return false;
      execute.click();
      return true;
    })()`)
	settle("tool step card", `[...document.querySelectorAll("#panel-chat .tool-call .tag")].some((tag) => tag.textContent.startsWith("done")) &&
      document.querySelector('[aria-label="guardrails"]').textContent.includes("remaining")`)
	captureStates("thread-tool")
	say(t, ctx, browser, "Say hi in one word.")
	settle("agent chat reply", `document.querySelectorAll("#panel-chat .msg.assistant").length >= 1 &&
      [...document.querySelectorAll("#panel-chat .msg.assistant .body")].at(-1).textContent.trim().length > 0 && !document.querySelector(".composer .btn").disabled`)

	imageLeg()

	// 6. Switching the served model through the picker re-derives the composer: the
	// picker offers every other servable model; the default model is served
	// again when another serves now, else the first other model.
	assertBrowserPredicate(t, ctx, browser, `(() => {
      const pill = document.querySelector("#model-pill");
      if (!document.querySelector(".topbar .card .row .mono")) { pill.click(); if (!document.querySelector(".topbar .card")) pill.click(); }
      return true;
    })()`)
	settle("picker lists the servable models", `document.querySelectorAll(".topbar .card .row .mono").length > 0`)
	var switchName string
	if err := browser.Evaluate(ctx, `(() => {
      const current = document.querySelector("#model-pill").textContent;
      const rows = [...document.querySelectorAll(".topbar .card .row")].filter((node) => node.querySelector(".mono") && node.querySelector(".mono").textContent !== current && [...node.querySelectorAll("button")].some((button) => button.textContent === "serve"));
      const preferred = rows.find((node) => node.querySelector(".mono").textContent === `+strconv.Quote(modelName)+`) || rows[0];
      return preferred ? preferred.querySelector(".mono").textContent : "";
    })()`, &switchName); err != nil {
		t.Fatal(err)
	}
	if switchName != "" {
		switchTo(switchName)
		t.Logf("switched to %s", switchName)
		// 7a. The picker leaves once the swap serves; opened again, Escape closes it.
		settle("the picker left after the served swap", `!document.querySelector(".topbar .card")`)
		openPicker()
		pressKey(t, ctx, browser, "Escape", 27)
		settle("Escape closes the picker", `!document.querySelector(".topbar .card")`)
		t.Log("picker leg: the picker closed after the served swap and on Escape")
	} else {
		t.Log("switch leg not taken: the store holds one servable model")
	}

	// 7. A running turn stops from the composer.
	assertBrowserPredicate(t, ctx, browser, `(() => {
      const select = document.querySelector('.composer select[aria-label="mode"]');
      if (select) { select.value = "chat"; select.dispatchEvent(new Event("change")); }
      document.getElementById('settings-toggle').click();
      const limit = document.querySelector('#conversation-settings input[aria-label="max tokens"]');
      limit.value = "400";
      document.querySelector('#settings-dialog [data-close-dialog]').click();
      return true;
    })()`)
	say(t, ctx, browser, "Write a very long story about a lighthouse, at least twenty paragraphs.")
	settle("turn running", `!document.querySelector(".composer .btn").disabled === false`)
	assertBrowserPredicate(t, ctx, browser, `(() => { const stop = document.querySelector(".stop-button"); if (!stop) return false; stop.click(); return true; })()`)
	settle("turn stopped", `!document.querySelector(".composer .btn").disabled &&
      (document.querySelector("#panel-chat .chat-log").textContent.includes("Stopped") || !!document.querySelector("#panel-chat .msg.error"))`)
	captureStates("thread-stopped")

	// 8. Media out: an image from the cheapest declared image model, chosen from
	// the generation capabilities by its host oscillator entry module, through
	// the image mode's declared controls; the output lands as an artifact card.
	// 9. Speech out: a clip from the declared speech model with a voice the
	// model's artifact exports, offered as a choice.
	// 10. Media back in: the image card offers itself as the next turn's input.
	generationMode := func(mode, entryModule string) bool {
		t.Helper()
		var recipeID string
		if err := browser.Evaluate(ctx, `(async () => {
      const capabilities = await window.overgo.api.get("/generation/capabilities");
      const declared = capabilities.find((capability) => capability.task === `+strconv.Quote(mode)+` && !capability.refusal &&
        (`+strconv.Quote(entryModule)+` === "" || (capability.stages[0] && capability.stages[0].module.id === `+strconv.Quote(entryModule)+`)));
      return declared ? declared.recipe : "";
    })()`, &recipeID); err != nil {
			t.Fatal(err)
		}
		if recipeID == "" {
			return false
		}
		assertBrowserPredicate(t, ctx, browser, `(() => {
      const select = document.querySelector('.composer select[aria-label="mode"]');
      if (!select || ![...select.options].some((option) => option.value === `+strconv.Quote(mode)+`)) return false;
      select.value = `+strconv.Quote(mode)+`; select.dispatchEvent(new Event("change"));
      return true;
    })()`)
		settle(mode+" mode renders its declaration", `(() => {
      const picker = document.querySelector('.composer select[aria-label="generation model"]');
      if (!picker || ![...picker.options].some((option) => option.value === `+strconv.Quote(recipeID)+`)) return false;
      picker.value = `+strconv.Quote(recipeID)+`; picker.dispatchEvent(new Event("change"));
      for (const label of document.querySelectorAll(".mode-controls label.control")) {
        const input = label.querySelector("input, select, textarea");
        if (input.tagName === "SELECT" && !input.value) input.value = [...input.options].map((option) => option.value).find(Boolean) || "";
        else if (input.type === "number" && !input.value) input.value = label.textContent.trim().startsWith("seed") ? String(Date.now() % 100000) : "1";
      }
      return true;
    })()`)
		return true
	}
	mediaCards := `document.querySelectorAll("#panel-chat .msg.media").length`
	if generationMode("image-gen", "model.oscillator-image-prepare") {
		// 7b. Enhance the prompt first: the served chat model rewrites the
		// typed prompt under the server's instruction, the rewrite stands
		// beside the original, and accepting it fills the composer; the run
		// that follows cites the enhancement record as its source.
		assertBrowserPredicate(t, ctx, browser, `(() => { const input = document.querySelector(".composer textarea"); input.value = "a test image"; input.focus(); return true; })()`)
		assertBrowserPredicate(t, ctx, browser, `(() => { document.getElementById('settings-toggle').click(); const enhance = [...document.querySelectorAll("#conversation-settings button")].find((button) => button.textContent === "Enhance current prompt"); if (!enhance) return false; enhance.click(); return true; })()`)
		settle("the rewrite stands beside the original", `(() => { const card = document.querySelector("#panel-chat .enhancement .card");
      return !!card && card.textContent.includes("original: a test image") && (card.querySelector(".rewrite") || {}).textContent.trim().length > 0; })()`)
		var rewrite string
		if err := browser.Evaluate(ctx, `(() => { document.querySelector("#panel-chat .enhancement .card .btn").click(); return document.querySelector(".composer textarea").value; })()`, &rewrite); err != nil || rewrite == "" || rewrite == "a test image" {
			t.Fatalf("accepted rewrite = %q, %v", rewrite, err)
		}
		t.Logf("enhance leg: the served model rewrote the prompt to %q", rewrite)
		pressKey(t, ctx, browser, "Enter", 13)
		settle("image out as an artifact card", mediaCards+` === 1 && !!document.querySelector("#panel-chat .msg.media img") &&
      !!document.querySelector("#panel-chat .msg.media .note a") && !document.querySelector(".composer .btn").disabled`)
		t.Log("media-out leg: an image landed as an artifact card")
		captureStates("thread-media")
		assertBrowserPredicate(t, ctx, browser, `(() => { const again = [...document.querySelectorAll("#panel-chat .msg.media button")].find((button) => button.textContent === "use as input"); if (!again) return false; again.click(); return true; })()`)
		settle("image back in as an attachment", `document.querySelectorAll('.composer .attachment-row[data-state=ready], .composer .attachment-row[data-state=refused]').length === 1`)
		var refused bool
		if err := browser.Evaluate(ctx, `!!document.querySelector('.composer .attachment-row[data-state=refused]')`, &refused); err != nil {
			t.Fatal(err)
		}
		t.Logf("media-in leg: the generated image re-entered the composer, refused=%v", refused)
		captureStates("composer-attachment")
		assertBrowserPredicate(t, ctx, browser, `(() => { document.querySelector('.composer .attachment-row button[aria-label^="Remove "]').click(); return !document.querySelector('.composer .attachment-row') && !document.querySelector('.send-button').disabled; })()`)
		// 10b. Gallery after a reload: the image the store holds reappears
		// newest first in the image mode's gallery rail, and its thumb opens the
		// record that made it (the request's seed) beside the image.
		var generated string
		if err := browser.Evaluate(ctx, `new URL(document.querySelector("#panel-chat .msg.media .note a").href).searchParams.get("id")`, &generated); err != nil || generated == "" {
			t.Fatalf("generated image id = %q, %v", generated, err)
		}
		if err := browser.Evaluate(ctx, `location.reload()`, nil); err != nil {
			t.Fatal(err)
		}
		settle("page back after the reload", `document.readyState === "complete" && !!window.overgo && !!document.querySelector('.composer select[aria-label="mode"]') && `+mediaCards+` === 0`)
		if !generationMode("image-gen", "model.oscillator-image-prepare") {
			t.Fatal("the image mode is gone after the reload")
		}
		settle("gallery rail lists the generated image newest first", `(document.querySelector(".mode-controls .gallery .intake-thumb") || {}).dataset && document.querySelector(".mode-controls .gallery .intake-thumb").dataset.id === `+strconv.Quote(generated))
		assertBrowserPredicate(t, ctx, browser, `(() => { document.querySelector(".mode-controls .gallery .intake-thumb").click(); return true; })()`)
		settle("gallery thumb opens the image with its record", mediaCards+` === 1 && !!document.querySelector("#panel-chat .msg.media img") &&
      [...document.querySelectorAll("#panel-chat .msg.media .record td")].some((cell) => cell.textContent === "seed") &&
      !!document.querySelector("#panel-chat .msg.media .record a[download]") && [...document.querySelectorAll("#panel-chat .msg.media button")].some((button) => button.textContent === "use as input")`)
		t.Log("gallery leg: the generated image reappeared in the gallery after a reload and opened its record")
		captureStates("thread-record")
		// 10c. Vary and regenerate from the record: the card's "vary" resubmits
		// the stored request with a fresh seed and the new card names its
		// parent; "regenerate" resubmits it unchanged and the card says the
		// store memoized it to the same output.
		assertBrowserPredicate(t, ctx, browser, `(() => { const vary = [...document.querySelectorAll("#panel-chat .msg.media button")].find((button) => button.textContent === "vary"); if (!vary) return false; vary.click(); return true; })()`)
		settle("vary makes a new image naming its parent", mediaCards+` === 2 && !document.querySelector(".composer .btn").disabled &&
      [...document.querySelectorAll("#panel-chat .msg.media")].at(-1).querySelector(".note").textContent.includes("vary of ") && !!document.querySelectorAll("#panel-chat .msg.media img")[1]`)
		var varied string
		if err := browser.Evaluate(ctx, `new URL([...document.querySelectorAll("#panel-chat .msg.media")].at(-1).querySelector(".note a").href).searchParams.get("id")`, &varied); err != nil || varied == "" || varied == generated {
			t.Fatalf("varied image id = %q (parent %q), %v", varied, generated, err)
		}
		assertBrowserPredicate(t, ctx, browser, `(() => { const again = [...document.querySelector("#panel-chat .msg.media").querySelectorAll("button")].find((button) => button.textContent === "regenerate"); if (!again) return false; again.click(); return true; })()`)
		settle("regenerate says the store memoized the unchanged request", mediaCards+` === 3 && !document.querySelector(".composer .btn").disabled &&
      [...document.querySelectorAll("#panel-chat .msg.media")].at(-1).querySelector(".note").textContent.includes("the store memoized")`)
		t.Log("vary leg: vary made a new image from the record and regenerate memoized to the same output")
		// 10d. Lineage and next steps: the image card's lineage names the run
		// that made it with its request, and offers as a next step every
		// active capability whose declared slot takes an image; the step opens
		// that mode with the image in the slot.
		assertBrowserPredicate(t, ctx, browser, `(() => { const open = [...document.querySelector("#panel-chat .msg.media").querySelectorAll("button")].find((button) => button.textContent === "lineage"); if (!open) return false; open.click(); return true; })()`)
		settle("the image card shows the run that made it", `(() => { const block = document.querySelector("#panel-chat .msg.media .lineage");
      return !!block && block.textContent.includes("made by run") && block.textContent.includes("request ") && block.textContent.includes("prompt "); })()`)
		var steps []string
		if err := browser.Evaluate(ctx, `[...document.querySelectorAll("#panel-chat .msg.media .lineage [aria-label='next steps'] .chip")].map((chip) => chip.textContent)`, &steps); err != nil {
			t.Fatal(err)
		}
		t.Logf("lineage leg: the image's next steps are %q", steps)
		if len(steps) > 0 {
			assertBrowserPredicate(t, ctx, browser, `(() => { document.querySelector("#panel-chat .msg.media .lineage [aria-label='next steps'] .chip").click(); return true; })()`)
			settle("the next step opened its mode with the image in the slot", `(() => {
      const slot = [...document.querySelectorAll(".mode-controls label.control")].find((label) => label.querySelector("input") && label.querySelector("input").value === `+strconv.Quote(generated)+`);
      return !!slot && document.querySelector('.composer select[aria-label="mode"]').value !== "image-gen"; })()`)
			t.Log("lineage leg: the next step opened its mode with the image in its declared slot")
		}
	} else {
		t.Log("media-out leg not taken: the store declares no host oscillator image model")
	}
	if generationMode("speech", "") {
		say(t, ctx, browser, "hello from the lane")
		settle("speech out as an artifact card", `!!document.querySelector("#panel-chat .msg.media audio") && !document.querySelector(".composer .btn").disabled`)
		t.Log("speech-out leg: a clip landed as an artifact card")
	} else {
		t.Log("speech-out leg not taken: the store declares no speech model")
	}
	// 11. Clip back in: a clip from the host oscillator video model lands as
	// a card, and with LiveEdit chosen in the same mode the card's "use as
	// input" fills the request's source clip with the card's stored id
	// instead of attaching bytes.
	if generationMode("video-gen", "model.oscillator-video-prepare") {
		say(t, ctx, browser, "a test clip")
		settle("clip out as an artifact card", `!document.querySelector(".composer .btn").disabled &&
      !!document.querySelector("#panel-chat .msg.media video") && !!document.querySelector("#panel-chat .msg.media .note a")`)
		t.Log("clip-out leg: a clip landed as an artifact card")
		// Presets from declared bounds: the Wan prompt form prefills the profile's
		// geometry, and an aspect chip moves width and height by the stride
		// while keeping the pixel area; the leg never runs the generation.
		if generationMode("video-gen", "model.latent-video-prepare") {
			settle("the Wan form prefills the declared geometry with preset chips", `(() => {
      const field = (name) => [...document.querySelectorAll(".mode-controls label.control")].find((label) => label.textContent.trim().startsWith(name));
      const width = field("width") && field("width").querySelector("input");
      return !!width && Number(width.value) > 0 && document.querySelectorAll(".mode-controls .preset-chips .chip").length > 0; })()`)
			assertBrowserPredicate(t, ctx, browser, `(() => {
      const field = (name) => [...document.querySelectorAll(".mode-controls label.control")].find((label) => label.textContent.trim().startsWith(name)).querySelector("input");
      const before = Number(field("width").value) * Number(field("height").value), step = Number(field("width").step);
      [...document.querySelectorAll(".mode-controls .preset-chips .chip")].find((chip) => chip.textContent === "9:16").click();
      const width = Number(field("width").value), height = Number(field("height").value);
      return width % step === 0 && height % step === 0 && height > width && Math.abs(width * height - before) < before / 10; })()`)
			t.Log("presets leg: the Wan form's aspect chip moved the declared geometry by its stride")
			captureStates("mode-video")
		}
		if generationMode("video-gen", "model.reference-video-prepare") {
			assertBrowserPredicate(t, ctx, browser, `(() => {
      const card = [...document.querySelectorAll("#panel-chat .msg.media")].find((card) => card.querySelector("video"));
      const again = card && [...card.querySelectorAll("button")].find((button) => button.textContent === "use as input");
      if (!again) return false; again.click(); return true; })()`)
			settle("clip fills the source artifact", `(() => {
      const field = [...document.querySelectorAll(".mode-controls label.control")].find((label) => label.textContent.trim().startsWith("source clip"));
      const input = field && field.querySelector("input");
      return !!input && input.value.includes(":sha256:") && document.querySelectorAll(".composer .attachment-row").length === 0; })()`)
			t.Log("clip-in leg: the clip's stored id filled LiveEdit's source clip")
		} else {
			t.Log("clip-in leg not taken: the store declares no LiveEdit model")
		}
	} else {
		t.Log("clip-out leg not taken: the store declares no host oscillator video model")
	}
	// 12. Ask about an image: with the declared VQA model chosen, the image
	// card's "use as input" fills the request's image with the card's stored
	// id, the message body is the question, and the answer lands as the
	// assistant's text.
	if generationMode("vqa", "model.vqa-prepare") {
		assertBrowserPredicate(t, ctx, browser, `(() => {
      const card = [...document.querySelectorAll("#panel-chat .msg.media")].find((card) => card.querySelector("img"));
      const again = card && [...card.querySelectorAll("button")].find((button) => button.textContent === "use as input");
      if (!again) return false; again.click(); return true; })()`)
		settle("image fills the vqa image control", `(() => {
      const field = [...document.querySelectorAll(".mode-controls label.control")].find((label) => label.textContent.trim().startsWith("image"));
      const input = field && field.querySelector("input");
      return !!input && input.value.includes(":sha256:") && document.querySelectorAll(".composer .attachment-row").length === 0; })()`)
		var beforeVQA int
		if err := browser.Evaluate(ctx, `document.querySelectorAll('#panel-chat .msg.assistant').length`, &beforeVQA); err != nil {
			t.Fatal(err)
		}
		say(t, ctx, browser, "What does this image show?")
		settle("the answer lands as the assistant's text", `document.querySelectorAll('#panel-chat .msg.assistant').length===`+strconv.Itoa(beforeVQA+1)+` && !document.querySelector(".send-button").disabled &&
      (([...document.querySelectorAll("#panel-chat .msg.assistant .body")].at(-1) || {}).textContent || "").trim().length > 0 &&
      document.querySelectorAll("#panel-chat .msg.error").length === 0`)
		var answer string
		if err := browser.Evaluate(ctx, `([...document.querySelectorAll("#panel-chat .msg.assistant .body")].at(-1) || {}).textContent`, &answer); err != nil {
			t.Fatal(err)
		}
		t.Logf("vqa leg: the VQA model answered %q", answer)
		// 12b. Fresh attachment in vqa mode (the served chat model refuses
		// images): stored through the intake route, the card names the
		// artifact, the image control takes its id, the answer is about it.
		const vqaImage = `[...document.querySelectorAll(".mode-controls label.control")].find((label) => label.textContent.trim().startsWith("image"))`
		var filled string
		if err := browser.Evaluate(ctx, `((field) => field ? field.querySelector("input").value : "")(`+vqaImage+`)`, &filled); err != nil {
			t.Fatal(err)
		}
		attachTinyPNG()
		settle("the upload fills the vqa image control", `(() => {
      const field = `+vqaImage+`, row = document.querySelector(".composer .attachment-row[data-state=ready]");
      const input = field && field.querySelector("input");
      return !!input && input.value.includes(":sha256:") && input.value !== `+strconv.Quote(filled)+` && !!row && [...row.querySelectorAll('a')].some(link => link.textContent==='Stored file' && link.getAttribute('href').includes(encodeURIComponent(input.value))); })()`)
		// The slot is declared (label "image", media image) and its strip lists the stored
		// attachment; choosing it from the strip fills the slot with the same id.
		settle("the image slot lists the stored attachment", `(() => {
      const field = `+vqaImage+`, input = field && field.querySelector("input");
      const thumb = field && [...field.querySelectorAll(".intake-thumb")].find((button) => button.dataset.id === input.value);
      if (!thumb) return false;
      input.value = ""; thumb.click();
      return input.value === thumb.dataset.id; })()`)
		t.Log("slot leg: the declared image slot listed the stored attachment and filled from it")
		var before int
		if err := browser.Evaluate(ctx, `document.querySelectorAll("#panel-chat .msg.assistant").length`, &before); err != nil {
			t.Fatal(err)
		}
		say(t, ctx, browser, "What colour is this image?")
		settle("the answer about the upload lands as the assistant's text", `document.querySelectorAll("#panel-chat .msg.assistant").length === `+strconv.Itoa(before+1)+` &&
      !document.querySelector(".composer .btn").disabled && document.querySelectorAll("#panel-chat .msg.error").length === 0 &&
      (([...document.querySelectorAll("#panel-chat .msg.assistant .body")].at(-1) || {}).textContent || "").trim().length > 0`)
		if err := browser.Evaluate(ctx, `([...document.querySelectorAll("#panel-chat .msg.assistant .body")].at(-1) || {}).textContent`, &answer); err != nil {
			t.Fatal(err)
		}
		t.Logf("vqa upload leg: the VQA model answered %q about the stored attachment", answer)
	} else {
		t.Log("vqa leg not taken: the store declares no VQA model")
	}
	// 12c. Transcribe: an attached clip stores through the intake route and
	// fills the audio control; the transcript lands as the assistant's text.
	// Taken only when the lane store holds an active transcription recipe.
	if generationMode("transcription", "model.transcribe-audio") {
		assertBrowserPredicate(t, ctx, browser, `(() => {
      const picker = document.querySelector(".composer input[type=file]");
      const transfer = new DataTransfer();
      transfer.items.add(new File([Uint8Array.from(atob(`+strconv.Quote(toneWAVBase64())+`), (char) => char.charCodeAt(0))], "tone.wav", { type: "audio/wav" }));
      picker.files = transfer.files;
      picker.dispatchEvent(new Event("change"));
      return true;
    })()`)
		settle("the clip fills the audio control", `(() => {
      const field = [...document.querySelectorAll(".mode-controls label.control")].find((label) => label.textContent.trim().startsWith("audio clip"));
      const input = field && field.querySelector("input"), row = document.querySelector(".composer .attachment-row[data-state=ready]");
      return !!input && input.value.includes(":sha256:") && !!row && [...row.querySelectorAll('a')].some(link => link.textContent==='Stored file' && link.getAttribute('href').includes(encodeURIComponent(input.value))); })()`)
		var before int
		if err := browser.Evaluate(ctx, `document.querySelectorAll("#panel-chat .msg.assistant").length`, &before); err != nil {
			t.Fatal(err)
		}
		say(t, ctx, browser, "transcribe")
		settle("the transcript lands as the assistant's text", `document.querySelectorAll("#panel-chat .msg.assistant").length === `+strconv.Itoa(before+1)+` &&
      !document.querySelector(".composer .btn").disabled && document.querySelectorAll("#panel-chat .msg.error").length === 0 &&
      (([...document.querySelectorAll("#panel-chat .msg.assistant .body")].at(-1) || {}).textContent || "").trim().length > 0`)
		t.Log("transcription leg: the transcript of the attached clip landed as the assistant's text")
	} else {
		t.Log("transcription leg not taken: the store declares no active transcription recipe")
	}
	// 13. Remote turn: picker marks the remote model; serving it swaps the
	// child to the relay; welcome + assistant turns carry the marker; answer
	// from the fake provider, no input-token count.
	if remoteName != "" {
		openPicker()
		settle("the picker marks the remote model", `[...document.querySelectorAll(".topbar .card .row")].some((row) =>
      row.querySelector(".mono") && row.querySelector(".mono").textContent === `+strconv.Quote(remoteName)+` && [...row.querySelectorAll(".tag")].some((tag) => tag.textContent === "remote"))`)
		switchTo(remoteName)
		say(t, ctx, browser, "hello relay")
		// The provider's usage fills the facts the refused count route left empty.
		settle("the remote turn answers with its marker and the provider's counts", `!document.querySelector(".composer .btn").disabled &&
      (([...document.querySelectorAll("#panel-chat .msg.assistant .body")].at(-1) || {}).textContent || "").includes("Hello from the relay") &&
      [...document.querySelectorAll("#panel-chat .msg.assistant .role .tag")].some((tag) => tag.textContent === "remote") &&
      [...document.querySelectorAll(".statgrid .stat")].some((stat) => stat.textContent.includes("Input")) &&
      document.querySelectorAll("#panel-chat .msg.error").length === 0`)
		t.Log("remote leg: the declared remote model answered through the relay with its marker")
	}
	// 14. Key entry: the keyless model lists refused with a key field; the
	// entered key lifts the refusal (proxy and child hold it in memory),
	// the model serves and answers.
	if entryName != "" {
		openPicker()
		assertBrowserPredicate(t, ctx, browser, `(() => {
      const row = [...document.querySelectorAll(".topbar .card .row")].find((node) => node.querySelector(".mono") && node.querySelector(".mono").textContent === `+strconv.Quote(entryName)+`);
      const key = row && row.querySelector("input[type=password]"), use = row && [...row.querySelectorAll("button")].find((button) => button.textContent === "use key");
      if (!key || !use || !row.textContent.includes("is not set")) return false;
      key.value = "lane-key"; use.click(); return true; })()`)
		settle("the entered key lifts the refusal", `[...document.querySelectorAll(".topbar .card .row")].some((row) =>
      row.querySelector(".mono") && row.querySelector(".mono").textContent === `+strconv.Quote(entryName)+` && [...row.querySelectorAll("button")].some((button) => button.textContent === "serve"))`)
		switchTo(entryName)
		say(t, ctx, browser, "hello again")
		settle("the keyed model answers", `!document.querySelector(".composer .btn").disabled &&
      (([...document.querySelectorAll("#panel-chat .msg.assistant .body")].at(-1) || {}).textContent || "").includes("Hello after the key") &&
      document.querySelectorAll("#panel-chat .msg.error").length === 0`)
		t.Log("key-entry leg: the keyless declaration served after the key was entered on the page")
	}
	// 15. The provider declared on the cold page (leg 0b) is listed by the
	// picker over the served child; it serves and answers. A declaration
	// is a store claim bound to the serving binary's source commit, so a
	// modified tree saw the refusal and proves nothing here.
	if entryName != "" {
		if !declaredOnPage {
			t.Log("declare-from-page leg: the declaration was refused on a modified tree; the serve is proven on a clean tree")
		} else {
			switchTo("page-model")
			say(t, ctx, browser, "hello from the page")
			settle("the page-declared model answers", `!document.querySelector(".composer .btn").disabled &&
      (([...document.querySelectorAll("#panel-chat .msg.assistant .body")].at(-1) || {}).textContent || "").includes("from the page's provider") &&
      document.querySelectorAll("#panel-chat .msg.error").length === 0`)
			t.Log("declare-from-page leg: a provider declared on the Library tab served a turn through the relay")
		}
		// 16. Retire from the page: the key-entry model's Library row retires
		// it with the section's reason; the catalog relists without it. The
		// retirement is a store claim like the declaration, so a modified
		// tree sees the refusal and a clean tree sees the row leave.
		assertBrowserPredicate(t, ctx, browser, `(() => { location.hash = "#library"; return true; })()`)
		settle("the library lists the key-entry model with its retire control", `[...document.querySelectorAll("#panel-library tr")].some((row) =>
      row.textContent.includes(`+strconv.Quote(entryName)+`) && [...row.querySelectorAll("button")].some((button) => button.textContent === "retire"))`)
		assertBrowserPredicate(t, ctx, browser, `(() => {
      document.querySelector("input[aria-label='retirement reason']").value = "the browser lane's fake provider closed";
      const catalogNote = [...document.querySelectorAll("#panel-library .section-title")].find((node) => node.textContent === "Local models").parentElement.nextElementSibling;
      catalogNote.textContent = ""; catalogNote.dataset.laneRetirement = "1";
      const row = [...document.querySelectorAll("#panel-library tr")].find((row) => row.textContent.includes(`+strconv.Quote(entryName)+`));
      [...row.querySelectorAll("button")].find((button) => button.textContent === "retire").click(); return true; })()`)
		retirement := `![...document.querySelectorAll("#panel-library tr")].some((row) => row.textContent.includes(` + strconv.Quote(entryName) + `))`
		build, err := buildinfo.ReadFile(binary)
		if err != nil {
			t.Fatal(err)
		}
		// The child uses its embedded revision when present. Development
		// binaries without one use the worktree, as ExecutableCodeCommit does.
		stamped := false
		modified := false
		for _, setting := range build.Settings {
			stamped = stamped || setting.Key == "vcs.revision" && setting.Value != ""
			modified = modified || setting.Key == "vcs.modified" && setting.Value == "true"
		}
		if stamped && modified {
			retirement = `document.querySelector("[data-lane-retirement]")?.textContent === "executable was built from modified source; exact code revision unavailable"`
		}
		if !stamped {
			if _, err := runrecord.VerifyingCommit(testutil.RepoRoot(t)); err != nil {
				retirement = `document.querySelector("[data-lane-retirement]")?.textContent === ` + strconv.Quote(err.Error())
			}
		}
		settle("the retirement follows the served binary's source identity", retirement)
		var retiredOnPage bool
		if err := browser.Evaluate(ctx, `![...document.querySelectorAll("#panel-library tr")].some((row) => row.textContent.includes(`+strconv.Quote(entryName)+`))`, &retiredOnPage); err != nil {
			t.Fatal(err)
		}
		t.Logf("retire-from-page leg: retired on the page %v", retiredOnPage)
		assertBrowserPredicate(t, ctx, browser, `(() => { location.hash = "#chat"; return true; })()`)
		settle("back on the front page after the retirement", `!!document.querySelector("#panel-chat.active .composer textarea")`)
	}
}

// laneIdleIntake: the cold page's library intake for the journey, the
// launcher's own steps (internal/providerintake binds them for the swap
// command; this package's tests cannot import it without a cycle).
func laneIdleIntake() LibraryIntake {
	return LibraryIntake{
		DeclareProvider: func(ctx context.Context, store *overgodb.Store, declaration ProviderDeclaration) ([]DeclaredProvider, error) {
			commit, err := runrecord.ExecutableCodeCommit(".")
			if err != nil {
				return nil, err
			}
			declarations, err := remoteprovider.DeclareDocument(ctx, store, remoteprovider.Document{Name: declaration.Name, Endpoint: declaration.Endpoint, KeyEnvironment: declaration.KeyEnvironment, Models: declaration.Models, ContextLength: declaration.ContextLength}, commit)
			declared := make([]DeclaredProvider, 0, len(declarations))
			for _, item := range declarations {
				declared = append(declared, DeclaredProvider{Location: item.Location, Model: item.Model.String(), Recipe: item.Recipe.ID.String(), Refusal: remoteprovider.Refusal(item.Provider)})
			}
			return declared, err
		},
		ListProviderModels: func(ctx context.Context, endpoint, keyEnvironment string) ([]ProviderModel, error) {
			listed, err := remoterelay.ListModels(ctx, remoteprovider.Provider{Endpoint: strings.TrimRight(endpoint, "/"), KeyEnvironment: keyEnvironment}, nil)
			models := make([]ProviderModel, 0, len(listed))
			for _, model := range listed {
				models = append(models, ProviderModel{ID: model.ID, Name: model.Name, ContextLength: model.ContextLength})
			}
			return models, err
		},
	}
}

// retireLaneRemote retires every lane-store declaration at a location
// once the journey ends (an earlier run that failed mid-leg may have left
// one under the same location), so the catalog stops offering the fakes;
// a location with no declaration is nothing to retire.
func retireLaneRemote(t *testing.T, storePath, location string) {
	t.Helper()
	commit, err := runrecord.HeadCommit(testutil.RepoRoot(t))
	if err != nil {
		t.Error(err)
		return
	}
	laneStore, err := overgodb.Open(storePath)
	if err != nil {
		t.Errorf("retire %s: %v", location, err)
		return
	}
	defer laneStore.Close()
	for {
		err := remoteprovider.Retire(context.WithoutCancel(t.Context()), laneStore, 256, location, commit, "the browser lane's fake provider closed with the journey")
		if err == nil {
			continue
		}
		if !strings.Contains(err.Error(), "no declared model") {
			t.Errorf("retire %s: %v", location, err)
		}
		return
	}
}

// declareLaneRemote: declare a remote model (fake loopback provider
// answering pieces under the key "lane-key") in the lane store, retire on
// cleanup; the variable is set to value when given; returns the picker
// name, "" when the store refuses the declaration.
func declareLaneRemote(t *testing.T, storePath, name, variable, value string, pieces []string) string {
	t.Helper()
	fake, _ := relaytest.Serve(t, "", "lane-key", pieces)
	if value != "" {
		t.Setenv(variable, value)
	}
	provider := remoteprovider.Provider{Name: name, Endpoint: fake.URL, KeyEnvironment: variable, Model: "lane/" + name + "-model"}
	// A killed prior journey cannot run cleanup. Its dead endpoint must not
	// compete with this fixture under the same picker name.
	retireLaneRemote(t, storePath, remoteprovider.Location(provider))
	commit, err := runrecord.HeadCommit(testutil.RepoRoot(t))
	if err != nil {
		t.Fatal(err)
	}
	laneStore, err := overgodb.Open(storePath)
	if err != nil {
		t.Fatal(err)
	}
	declaration, err := remoteprovider.Declare(t.Context(), laneStore, provider, commit)
	if closeErr := laneStore.Close(); closeErr != nil {
		t.Fatal(closeErr)
	}
	if err != nil {
		t.Logf("remote leg not taken: the store cannot take the declaration: %v", err)
		return ""
	}
	t.Cleanup(func() { retireLaneRemote(t, storePath, declaration.Location) })
	return path.Base(declaration.Location)
}

func TestLaneRemoteFixtureReplacesPriorEndpoint(t *testing.T) {
	storePath := t.TempDir()
	first := declareLaneRemote(t, storePath, "webui-lane-cleanup", "OVERGO_WEBUI_LANE_CLEANUP_KEY", "lane-key", []string{"old"})
	second := declareLaneRemote(t, storePath, "webui-lane-cleanup", "OVERGO_WEBUI_LANE_CLEANUP_KEY", "lane-key", []string{"current"})
	if first == "" || second != first {
		t.Fatalf("fixture picker names: %q, %q", first, second)
	}
	store, err := overgodb.Open(storePath)
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	declared, err := remoteprovider.List(t.Context(), store, 256)
	if err != nil || len(declared) != 1 {
		t.Fatalf("fixture declarations = %d, error = %v; want one current endpoint", len(declared), err)
	}
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

// tinyPNGBase64 encodes a small red square as PNG bytes for the browser to
// attach: a real image a projector can encode, not a bare signature.
// toneWAVBase64: one second of a 16 kHz mono sine, the clip the
// transcription leg attaches.
func toneWAVBase64() string {
	samples := make([]int16, 16000)
	for i := range samples {
		samples[i] = int16(8192 * math.Sin(2*math.Pi*float64(i)/16))
	}
	return base64.StdEncoding.EncodeToString(testutil.MonoPCM16WAV(16000, samples))
}

func tinyPNGBase64(t *testing.T) string {
	t.Helper()
	canvas := image.NewRGBA(image.Rect(0, 0, 16, 16))
	for y := range 16 {
		for x := range 16 {
			canvas.Set(x, y, color.RGBA{R: 200, G: 30, B: 30, A: 255})
		}
	}
	var buffer bytes.Buffer
	if err := png.Encode(&buffer, canvas); err != nil {
		t.Fatal(err)
	}
	return base64.StdEncoding.EncodeToString(buffer.Bytes())
}
