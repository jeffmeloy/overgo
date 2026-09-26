package server

import (
	"encoding/json"
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

// TestWebUIBrowserTypeset measures the workbench's type against the scale the
// stylesheet declares: every visible piece of text, on every tab at desktop
// and phone, renders at one of the type tokens (the stylesheet once set
// eleven sizes beside its six tokens), and a long reply keeps to the reading
// measure token rather than running the conversation's full width.
func TestWebUIBrowserTypeset(t *testing.T) {
	t.Parallel()
	if os.Getenv("OVERGO_WEBUI_LANE") != "1" {
		testskip.NotApplicable(t, "the typeset leg runs through cmd/webui-lane")
	}
	stylesheet, err := os.ReadFile("webui/style.css")
	if err != nil {
		t.Fatal(err)
	}
	design, err := webuilane.ParseDesign(string(stylesheet))
	if err != nil {
		t.Fatal(err)
	}
	var scale []float64
	for _, value := range design.Type {
		size, err := strconv.ParseFloat(strings.TrimSuffix(value, "px"), 64)
		if err != nil {
			t.Fatalf("type token %q is not a pixel size", value)
		}
		scale = append(scale, size)
	}
	slices.Sort(scale)
	encodedScale, _ := json.Marshal(scale)
	browserPath, err := webuilane.FindBrowser(os.Getenv("OVERGO_BROWSER"))
	if err != nil {
		t.Fatal(err)
	}
	repository, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = repository.Close() })
	reply := strings.Repeat("A reply long enough to fill several lines of the conversation, so the reading measure decides its width. ", 8)
	handler := newTestHandlerForRepository(t, repository, screensGenerator{recipeInspectorGenerator: responseRecipeGenerator(t, &fakeGenerator{pieces: []string{reply}}), path: writeTensorFixture(t)})
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
	say(t, ctx, browser, "Write a long reply.")
	settle(`(document.querySelector("#panel-chat .msg.assistant .body") || {}).textContent?.length > 400 && overgo.api.inFlight() === 0`)
	var tabs []string
	if err := browser.Evaluate(ctx, `[...document.querySelectorAll("#panels .panel")].map((panel) => panel.id.slice("panel-".length))`, &tabs); err != nil {
		t.Fatal(err)
	}
	offScale := map[string]bool{}
	states := 0
	var measure struct {
		Characters float64 `json:"characters"`
		Measure    float64 `json:"measure"`
	}
	for _, viewport := range webuilane.ScreenViewports {
		for _, tab := range tabs {
			assertBrowserPredicate(t, ctx, browser, `(() => { location.hash = `+strconv.Quote(tab)+`; return true; })()`)
			settle(`!!document.querySelector("#panel-` + tab + `.active") && overgo.api.inFlight() === 0`)
			// The screens leg captures and audits every tab; this leg only measures type.
			if err := webuilane.SettleViewport(ctx, browser, viewport); err != nil {
				t.Fatal(err)
			}
			states++
			var sizes []string
			if err := browser.Evaluate(ctx, `(() => {
  const scale = `+string(encodedScale)+`, found = new Set();
  for (const node of document.querySelectorAll("body *")) {
    if (![...node.childNodes].some((child) => child.nodeType === 3 && child.textContent.trim()) || !node.checkVisibility({ checkVisibilityCSS: true })) continue;
    const rect = node.getBoundingClientRect();
    if (!rect.width || !rect.height) continue;
    const size = parseFloat(getComputedStyle(node).fontSize);
    if (!scale.includes(size)) found.add(node.tagName.toLowerCase() + (node.className && typeof node.className === "string" ? "." + node.className.trim().split(/\s+/).join(".") : "") + " " + size + "px");
  }
  return [...found];
})()`, &sizes); err != nil {
				t.Fatal(err)
			}
			for _, size := range sizes {
				offScale[viewport.Name+" "+tab+": "+size] = true
			}
			if tab == "chat" && viewport.Name == webuilane.ScreenViewports[0].Name {
				if err := browser.Evaluate(ctx, `(() => {
  const body = document.querySelector("#panel-chat .msg.assistant .body"), style = getComputedStyle(body);
  const context = document.createElement("canvas").getContext("2d");
  context.font = style.fontWeight + " " + style.fontSize + " " + style.fontFamily;
  return { characters: body.getBoundingClientRect().width / context.measureText("0").width,
    measure: parseFloat(getComputedStyle(document.documentElement).getPropertyValue("--measure")) };
})()`, &measure); err != nil {
					t.Fatal(err)
				}
			}
		}
	}
	findings := slices.Sorted(func(yield func(string) bool) {
		for finding := range offScale {
			if !yield(finding) {
				return
			}
		}
	})
	for _, finding := range findings {
		t.Error("off the type scale: " + finding)
	}
	if !(measure.Measure > 0) || measure.Characters > measure.Measure+0.5 {
		t.Errorf("a long reply runs %.1f characters wide against a measure of %.1f", measure.Characters, measure.Measure)
	}
	t.Logf("typeset leg: %d states on the %v px scale with %d sizes off it; a long reply runs %.1f of %.0f characters", states, scale, len(findings), measure.Characters, measure.Measure)
}
