package server

import (
	"encoding/json"
	"fmt"
	"net/http/httptest"
	"os"
	"slices"
	"strconv"
	"strings"
	"testing"

	"overgo/internal/overgodb"
	"overgo/internal/testskip"
	"overgo/internal/webuilane"
)

// hardenedStatesFilter holds, fails or inflates the page's data reads
// (never the shell's manifest, health and assets) by window.laneMode:
// "loading" holds every read until laneRelease(); "failure" answers each
// with a server fault carrying internal text; "oversized" lengthens every
// name, title, label and description in the real answer to one long word.
const hardenedStatesFilter = `(() => {
  const original = window.fetch;
  window.laneMode = null; window.laneHeld = [];
  window.laneRelease = () => { const held = window.laneHeld; window.laneHeld = []; for (const release of held) release(); };
  const inflate = (value) => Array.isArray(value) ? value.map(inflate) : value && typeof value === 'object' ?
    Object.fromEntries(Object.entries(value).map(([key, item]) => [key, typeof item === 'string' && ['name', 'title', 'label', 'description'].includes(key) ? item + '-' + 'x'.repeat(160) : inflate(item)])) : value;
  window.fetch = function (path, init) {
    const url = typeof path === 'string' ? path : path.url, method = (init && init.method) || 'GET';
    const shell = /^\/(workspace\/|health|mod\/|catalog\/)|\.(js|css|html)(\?|$)|\/stream(\?|$)/.test(url);
    if (!window.laneMode || method !== 'GET' || shell) return original.apply(this, arguments);
    const args = arguments;
    if (window.laneMode === 'loading') return new Promise((resolve, reject) => window.laneHeld.push(() => original.apply(this, args).then(resolve, reject)));
    if (window.laneMode === 'failure') return Promise.resolve(new Response(JSON.stringify({ error: { message: 'open C:\\overgo\\hidden\\raw.go: internal fault', type: 'server_error' } }), { status: 500, headers: { 'Content-Type': 'application/json' } }));
    return original.apply(this, args).then(async (response) => {
      if (!response.ok || !(response.headers.get('content-type') || '').includes('json')) return response;
      return new Response(JSON.stringify(inflate(await response.json())), { status: response.status, headers: response.headers });
    });
  };
  return true;
})()`

// TestWebUIBrowserHardenedStates walks every workspace tab through four
// states and records each fault: the empty store the fixture serves (a
// panel must say something), reads held open (a panel must say it is
// loading or show its controls, never stand blank), reads that fail with a
// server fault carrying internal text (the page must show its own words,
// keep the server's under Details, and offer the reload that recovers), and
// answers whose names are one long word (the layout audit must hold).
func TestWebUIBrowserHardenedStates(t *testing.T) {
	t.Parallel()
	if os.Getenv("OVERGO_WEBUI_LANE") != "1" {
		testskip.NotApplicable(t, "hardened states run through cmd/webui-lane")
	}
	browserPath, err := webuilane.FindBrowser(os.Getenv("OVERGO_BROWSER"))
	if err != nil {
		t.Fatal(err)
	}
	repository, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = repository.Close() })
	handler := newTestHandlerForRepository(t, repository, screensGenerator{recipeInspectorGenerator: responseRecipeGenerator(t, &fakeGenerator{}), path: writeTensorFixture(t)})
	defer handler.Close()
	httpServer := httptest.NewServer(handler)
	defer httpServer.Close()
	ctx := t.Context()
	browser, err := webuilane.Open(ctx, browserPath, httpServer.URL+"/")
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
	settle(`!!document.querySelector("#panel-chat.active .composer textarea") && overgo.errors.length === 0`)
	assertBrowserPredicate(t, ctx, browser, hardenedStatesFilter)
	var tabs []string
	if err := browser.Evaluate(ctx, `[...document.querySelectorAll("#panels .panel")].map((panel) => panel.id.slice("panel-".length))`, &tabs); err != nil {
		t.Fatal(err)
	}
	var findings []string
	// visit opens a tab, waits for it (on the page's clock) and answers what it shows.
	visit := func(tab, wait string) (state struct {
		Text       string `json:"text"`
		Blank      bool   `json:"blank"`
		Banners    int    `json:"banners"`
		Recoveries int    `json:"recoveries"`
	}) {
		t.Helper()
		assertBrowserPredicate(t, ctx, browser, `(() => { location.hash = `+strconv.Quote(tab)+`; return true; })()`)
		settle(`!!document.querySelector("#panel-` + tab + `.active") && (` + wait + `)`)
		if err := browser.Evaluate(ctx, `(() => {
  const panel = document.querySelector("#panel-`+tab+`");
  const text = panel.innerText.trim();
  return { text, blank: !text && !panel.querySelector("input, select, textarea, button, canvas, svg, img, video, audio"),
    banners: panel.querySelectorAll(".err-banner").length,
    recoveries: [...panel.querySelectorAll(".err-banner")].filter((banner) => banner.querySelector("button")).length };
})()`, &state); err != nil {
			t.Fatal(err)
		}
		return state
	}
	// remount releases every tab under a new key, so the next visit mounts it afresh under the mode.
	remount := func(mode string) {
		t.Helper()
		assertBrowserPredicate(t, ctx, browser, `(() => { window.laneMode = `+mode+`; const key = document.getElementById("api-key"); key.value = key.value ? "" : "hardened"; key.dispatchEvent(new Event("change")); return true; })()`)
	}
	for _, tab := range tabs {
		if state := visit(tab, "overgo.api.inFlight() === 0"); state.Blank {
			findings = append(findings, "empty: "+tab+" stands blank")
		}
	}
	remount(`"loading"`)
	for _, tab := range tabs {
		visible := `!!(() => { const panel = document.querySelector("#panel-` + tab + `"); return panel && (panel.innerText.trim() || panel.querySelector("input, select, textarea, button, canvas, svg, img, video, audio")); })()`
		if state := visit(tab, visible); state.Blank {
			findings = append(findings, "loading: "+tab+" stands blank while its reads are held")
		}
	}
	assertBrowserPredicate(t, ctx, browser, `(() => { window.laneMode = null; window.laneRelease(); return true; })()`)
	settle(`overgo.api.inFlight() === 0`)
	remount(`"failure"`)
	failed := 0
	for _, tab := range tabs {
		state := visit(tab, "overgo.api.inFlight() === 0")
		if state.Banners > 0 {
			failed++
			if !strings.Contains(state.Text, "The server could not complete this request (HTTP 500).") {
				findings = append(findings, "failure: "+tab+" does not say the server could not complete the request")
			}
		}
		switch {
		case state.Blank:
			findings = append(findings, "failure: "+tab+" stands blank after its reads failed")
		case slices.ContainsFunc([]string{"raw.go", "internal fault", `C:\overgo`}, func(raw string) bool { return strings.Contains(state.Text, raw) }):
			findings = append(findings, "failure: "+tab+" shows the server's internal text")
		case state.Banners > state.Recoveries:
			findings = append(findings, fmt.Sprintf("failure: %s shows %d failure banners, %d with a control that recovers", tab, state.Banners, state.Recoveries))
		}
	}
	if failed == 0 {
		findings = append(findings, "failure: no tab showed a failed read")
	}
	// With the fault gone, the banner's reload recovers the tab it stands in.
	assertBrowserPredicate(t, ctx, browser, `(() => { window.laneMode = null; location.hash = "datasets"; return true; })()`)
	settle(`!!document.querySelector("#panel-datasets.active .err-banner button")`)
	assertBrowserPredicate(t, ctx, browser, `(() => { document.querySelector("#panel-datasets .err-banner button").click(); return true; })()`)
	settle(`!document.querySelector("#panel-datasets .err-banner") && document.querySelector("#panel-datasets").innerText.trim().length > 0 && overgo.api.inFlight() === 0`)
	remount(`"oversized"`)
	for _, viewport := range webuilane.ScreenViewports {
		for _, tab := range tabs {
			visit(tab, "overgo.api.inFlight() === 0")
			// CaptureState sizes the viewport and awaits the page's transitions before it
			// audits; the screens leg keeps the pictures, so this leg writes none.
			measured, err := webuilane.CaptureState(ctx, browser, "", viewport, "oversized-"+tab)
			if err != nil {
				t.Fatal(err)
			}
			for _, finding := range measured {
				findings = append(findings, "oversized: "+finding.String())
			}
		}
	}
	assertBrowserPredicate(t, ctx, browser, `(() => { window.laneMode = null; return true; })()`)
	for _, finding := range findings {
		t.Error(finding)
	}
	encoded, _ := json.Marshal(tabs)
	webuilane.Leg(t, "hardened states leg", "%d tabs %s through empty, loading, failure (%d showed a failed read, and its reload recovered) and oversized states at desktop and phone with %d findings", len(tabs), encoded, failed, len(findings))
}
