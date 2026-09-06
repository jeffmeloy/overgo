package server

import (
	"bytes"
	"context"
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
	resolver := &modelswap.CatalogResolver{Store: store, Limit: 256}
	// No default: the journey opens on the cold proxy, as overgo_gui.bat
	// without a model does, and chooses the first model from the picker.
	proxy := &modelswap.Proxy{
		Supervisor: supervisor, Resolver: resolver, Keys: resolver,
		Idle: &IdleShell{Catalog: resolver.Catalog, OpenStore: func(context.Context) (*overgodb.Store, error) { return overgodb.Open(store) }, Intake: laneIdleIntake()},
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
      document.querySelector("#proxy-dot").classList.contains("ok") && window.overgo.errors.length === 0`)
	t.Logf("cold-start leg: the picker served %s from the cold page and the front page followed", modelName)

	// 2. The first message streams a reply and fills the context meter.
	say(t, ctx, browser, "Reply with the single word hello.")
	settle("streamed first reply", `document.querySelectorAll("#panel-chat .msg.assistant").length === 1 &&
      document.querySelector("#panel-chat .msg.assistant .body").textContent.trim().length > 0 &&
      document.querySelector('[aria-label="context meter"]').textContent.includes("Input") && !document.querySelector(".composer .btn").disabled`)

	// 2b. The conversation renames in place: the rail's rename control turns the
	// title into a field, Enter saves through the label route, the rail relists.
	settle("the rail lists the conversation", `[...document.querySelectorAll("#conversation-list button")].some((button) => button.textContent === "rename")`)
	// The rail relists once the reply lands, which can replace a field opened
	// during the relist; the step clicks rename until the field stands.
	settle("the title is a field", `(() => {
      if (document.querySelector("#conversation-list input[aria-label='conversation title']")) return true;
      const rename = [...document.querySelectorAll("#conversation-list button")].find((button) => button.textContent === "rename");
      if (rename) rename.click();
      return false; })()`)
	assertBrowserPredicate(t, ctx, browser, `(() => { const field = document.querySelector("#conversation-list input[aria-label='conversation title']"); field.value = "renamed by the lane"; field.dispatchEvent(new KeyboardEvent("keydown", { key: "Enter", bubbles: true })); return true; })()`)
	settle("the rail shows the new title", `[...document.querySelectorAll("#conversation-list button")].some((button) => button.textContent === "renamed by the lane")`)
	t.Log("rename leg: the conversation renamed in place from the rail")

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
			settle("image attached", `!!document.querySelector(".composer .card:not(.refused)")`)
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
        const card = document.querySelector(".composer .card.refused");
        const reason = ((window.overgo.capabilities().media || {}).refusals || {}).image || "";
        return !!card && reason !== "" && card.textContent.includes(reason);
      })()`)
			assertBrowserPredicate(t, ctx, browser, `(() => { document.querySelector(".composer .card.refused button").click(); return !document.querySelector(".composer .card.refused"); })()`)
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
		say(t, ctx, browser, "a test image")
		settle("image out as an artifact card", mediaCards+` === 1 && !!document.querySelector("#panel-chat .msg.media img") &&
      !!document.querySelector("#panel-chat .msg.media .note a") && !document.querySelector(".composer .btn").disabled`)
		t.Log("media-out leg: an image landed as an artifact card")
		assertBrowserPredicate(t, ctx, browser, `(() => { const again = [...document.querySelectorAll("#panel-chat .msg.media button")].find((button) => button.textContent === "use as input"); if (!again) return false; again.click(); return true; })()`)
		settle("image back in as an attachment", `document.querySelectorAll(".composer .card").length === 1`)
		var refused bool
		if err := browser.Evaluate(ctx, `!!document.querySelector(".composer .card.refused")`, &refused); err != nil {
			t.Fatal(err)
		}
		t.Logf("media-in leg: the generated image re-entered the composer, refused=%v", refused)
		assertBrowserPredicate(t, ctx, browser, `(() => { document.querySelector(".composer .card button").click(); return !document.querySelector(".composer .card"); })()`)
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
		if generationMode("video-gen", "model.reference-video-prepare") {
			assertBrowserPredicate(t, ctx, browser, `(() => {
      const card = [...document.querySelectorAll("#panel-chat .msg.media")].find((card) => card.querySelector("video"));
      const again = card && [...card.querySelectorAll("button")].find((button) => button.textContent === "use as input");
      if (!again) return false; again.click(); return true; })()`)
			settle("clip fills the source artifact", `(() => {
      const field = [...document.querySelectorAll(".mode-controls label.control")].find((label) => label.textContent.trim().startsWith("source clip"));
      const input = field && field.querySelector("input");
      return !!input && input.value.includes(":sha256:") && document.querySelectorAll(".composer .card").length === 0; })()`)
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
      return !!input && input.value.includes(":sha256:") && document.querySelectorAll(".composer .card").length === 0; })()`)
		say(t, ctx, browser, "What does this image show?")
		settle("the answer lands as the assistant's text", `!document.querySelector(".composer .btn").disabled &&
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
      const field = `+vqaImage+`, card = document.querySelector(".composer .card");
      const input = field && field.querySelector("input");
      return !!input && input.value.includes(":sha256:") && input.value !== `+strconv.Quote(filled)+` && !!card && !card.classList.contains("refused") && card.textContent.includes("stored as"); })()`)
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
      const input = field && field.querySelector("input"), card = document.querySelector(".composer .card");
      return !!input && input.value.includes(":sha256:") && !!card && !card.classList.contains("refused") && card.textContent.includes("stored as"); })()`)
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
      const row = [...document.querySelectorAll("#panel-library tr")].find((row) => row.textContent.includes(`+strconv.Quote(entryName)+`));
      [...row.querySelectorAll("button")].find((button) => button.textContent === "retire").click(); return true; })()`)
		settle("the retired model leaves the library or the modified tree refuses", `![...document.querySelectorAll("#panel-library tr")].some((row) => row.textContent.includes(`+strconv.Quote(entryName)+`)) ||
      [...document.querySelectorAll("#panel-library .note")].some((note) => note.textContent.includes("worktree is dirty"))`)
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
	commit, err := runrecord.HeadCommit(testutil.RepoRoot(t))
	if err != nil {
		t.Fatal(err)
	}
	laneStore, err := overgodb.Open(storePath)
	if err != nil {
		t.Fatal(err)
	}
	declaration, err := remoteprovider.Declare(t.Context(), laneStore, remoteprovider.Provider{
		Name: name, Endpoint: fake.URL, KeyEnvironment: variable, Model: "lane/" + name + "-model",
	}, commit)
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
