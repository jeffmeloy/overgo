package webuilane

import (
	"os"
	"overgo/internal/testskip"
	"strings"
	"testing"
)

func TestWebUIBrowserCaptureWaitsForTransition(t *testing.T) {
	if os.Getenv("OVERGO_WEBUI_LANE") != "1" {
		testskip.NotApplicable(t, "capture settling runs through cmd/webui-lane")
	}
	path, err := FindBrowser(os.Getenv("OVERGO_BROWSER"))
	if err != nil {
		t.Fatal(err)
	}
	ctx := t.Context()
	html := `<style>body{margin:0;background:rgb(0,0,0);color:rgb(255,255,255);transition:background-color 1s linear}body.light{background:rgb(255,255,255);color:rgb(0,0,0)}@keyframes pulse{to{opacity:.5}}.spinner{animation:pulse 1s infinite}.paused{animation:pulse 1s paused}</style><body><span class="spinner"></span><span class="paused"></span><p>Readable after the theme transition.</p></body>`
	browser, err := Open(ctx, path, "data:text/html,"+strings.ReplaceAll(html, " ", "%20"))
	if err != nil {
		t.Fatal(err)
	}
	defer browser.Close()
	if err := browser.Evaluate(ctx, `(() => { void document.body.offsetWidth; document.body.className = "light"; return getComputedStyle(document.body).backgroundColor; })()`, nil); err != nil {
		t.Fatal(err)
	}
	findings, err := CaptureState(ctx, browser, "", ScreenViewports[0], "theme-transition")
	if err != nil || len(findings) != 0 {
		t.Fatalf("transient layout captured: %v, %v", findings, err)
	}
}

// TestWebUIBrowserLayoutAudit pins the audit over a page built to fault in
// every measured way: a block wider than the viewport, a control past its
// edge and one too small, two fixed surfaces overlapping, a line cut
// without an ellipsis, text too faint to read, a control named only by
// its placeholder and a label on a role-less element; the same page with
// those faults absent audits clean, and the screenshot is a PNG.
func TestWebUIBrowserLayoutAudit(t *testing.T) {
	if os.Getenv("OVERGO_WEBUI_LANE") != "1" {
		testskip.NotApplicable(t, "the layout audit runs through cmd/webui-lane")
	}
	browserPath, err := FindBrowser(os.Getenv("OVERGO_BROWSER"))
	if err != nil {
		t.Fatal(err)
	}
	// A data URL ends at a "#", so the colours are written as rgb().
	faulty := `<body style="margin:0;background:rgb(255,255,255);color:rgb(0,0,0)">` +
		`<div style="width:3000px;height:10px"></div>` +
		`<button style="position:absolute;left:2000px;width:40px;height:40px">far</button>` +
		`<button style="width:10px;height:10px;padding:0">tiny</button>` +
		`<div style="position:fixed;top:0;left:0;width:100px;height:100px"></div>` +
		`<div style="position:fixed;top:50px;left:50px;width:100px;height:100px"></div>` +
		`<p style="white-space:nowrap;overflow:hidden;width:40px">a line of text far longer than its box</p>` +
		`<input placeholder="only a placeholder" style="width:200px;height:30px"><div aria-label="ignored">dot</div>` +
		`<p style="color:rgb(187,187,187)">faint words</p>` +
		`<button style="width:40px;height:40px">▸</button><h2 style="font-size:17px">A flat heading</h2>` +
		`<div style="height:10px;background-image:linear-gradient(90deg,rgb(0,0,255),rgb(0,255,255))"></div>` +
		`<p style="opacity:.5">faded words</p></body>`
	clean := `<body style="margin:0;background:rgb(255,255,255);color:rgb(0,0,0)"><button style="width:40px;height:40px">fine</button>` +
		`<span id="size-label">Size</span><input aria-labelledby="size-label" style="width:200px;height:30px">` +
		`<label style="display:block">Named <input style="width:200px;height:30px"></label><select aria-label="Chosen" style="height:30px"><option>one</option></select><div role="img" aria-label="status">dot</div>` +
		`<div role="status" style="position:absolute;width:1px;height:1px;overflow:hidden;white-space:nowrap;clip-path:inset(50%)"><span>Response ready.</span></div>` +
		`<p style="white-space:nowrap;overflow:hidden;text-overflow:ellipsis;width:40px">a line of text cut by design</p>` +
		`<h2 style="font-size:20px">A heading a step above</h2><button disabled style="opacity:.5;width:40px;height:40px">off</button>` +
		`<button aria-label="Open" style="width:40px;height:40px"><svg viewBox="0 0 24 24" width="20" height="20"><path d="M4 6h16"/></svg></button>` +
		`<details id="collapsed" open><summary>Details</summary><p style="color:rgb(187,187,187)">hidden faint words</p><button style="width:10px;height:10px;padding:0">hidden tiny</button></details>` +
		`<script>const details=document.getElementById('collapsed');details.querySelector('p').getBoundingClientRect();details.open=false;</script></body>`
	audit := func(html string) []LayoutFinding {
		t.Helper()
		ctx := t.Context()
		// A data URL keeps a "+" literal, so spaces travel percent-encoded; a glyph needs the charset.
		browser, err := Open(ctx, browserPath, "data:text/html;charset=utf-8,"+strings.ReplaceAll(html, " ", "%20"))
		if err != nil {
			t.Fatal(err)
		}
		defer browser.Close()
		if err := browser.SetViewport(ctx, 800, 600); err != nil {
			t.Fatal(err)
		}
		findings, err := browser.LayoutAudit(ctx)
		if err != nil {
			t.Fatal(err)
		}
		image, err := browser.Screenshot(ctx)
		if err != nil || len(image) < 8 || string(image[1:4]) != "PNG" {
			t.Fatalf("screenshot = %d bytes, %v", len(image), err)
		}
		return findings
	}
	kinds := map[string]bool{}
	for _, finding := range audit(faulty) {
		kinds[finding.Kind] = true
	}
	for _, kind := range []string{layoutOverflow, layoutOutside, layoutSmall, layoutOverlap, layoutClipped, layoutContrast, layoutUnlabelled, layoutIgnoredLabel,
		layoutGlyph, layoutFlatHeading, layoutGradient, layoutFaded} {
		if !kinds[kind] {
			t.Errorf("the faulty page audited without %s", kind)
		}
	}
	if findings := audit(clean); len(findings) != 0 {
		t.Errorf("the clean page audited with findings: %v", findings)
	}
	t.Log("layout audit leg: the design floor finds a glyph standing in for an icon, a flat heading, a gradient fill and faded live text, and passes drawn icons, a stepped heading and a disabled dimmed control")
	modal := `<body style="margin:0;background:rgb(255,255,255);color:rgb(0,0,0)"><p style="color:rgb(187,187,187)">inactive words</p><button style="width:10px;height:10px;padding:0">inactive tiny</button><dialog id="modal" style="background:rgb(255,255,255);color:rgb(0,0,0)"><p>Active dialog</p><button style="width:40px;height:40px">fine</button></dialog><script>document.getElementById('modal').showModal()</script></body>`
	if findings := audit(modal); len(findings) != 0 {
		t.Errorf("the modal page audited inactive content: %v", findings)
	}
	faintModal := strings.Replace(modal, "<p>Active dialog</p>", `<p style="color:rgb(187,187,187)">Active dialog</p>`, 1)
	if findings := audit(faintModal); len(findings) != 1 || findings[0].Kind != layoutContrast {
		t.Errorf("the faulty modal audited as %v", findings)
	}
	// A phone-wide page whose header pushes the content below the first screen's upper part.
	tall := `<body style="margin:0;background:rgb(255,255,255);color:rgb(0,0,0)"><div style="height:500px"></div><div id="panels">content</div></body>`
	ctx := t.Context()
	browser, err := Open(ctx, browserPath, "data:text/html,"+strings.ReplaceAll(tall, " ", "%20"))
	if err != nil {
		t.Fatal(err)
	}
	defer browser.Close()
	findings, err := CaptureState(ctx, browser, "", ScreenViewports[1], "tall")
	if err != nil || len(findings) != 1 || findings[0].Finding.Kind != layoutHeader {
		t.Errorf("the tall header audited as %v, %v", findings, err)
	}
	if summary := CaptureSummary(3, nil); !strings.Contains(summary, "tab mount errors: 0; unlabelled controls: 0;") || !strings.Contains(summary, "captured 3 states") || !strings.HasSuffix(summary, "0 layout findings") {
		t.Errorf("summary = %q", summary)
	}
	// A tab that fails at every viewport counts once; its failures are not layout findings.
	failed := LayoutFinding{Kind: mountError, Selector: "#panel-runs", Detail: "banner: refused"}
	repeated := []StateFinding{{Viewport: "desktop", State: "runs", Finding: failed}, {Viewport: "phone", State: "runs", Finding: failed}}
	if summary := CaptureSummary(2, repeated); !strings.Contains(summary, "tab mount errors: 1;") || !strings.HasSuffix(summary, "0 layout findings") {
		t.Errorf("summary with one failing tab = %q", summary)
	}
}

// TestWebUIBrowserPromisePredicateHolds holds a wait on a predicate that
// answers a promise (a fetch of a server count) to what the promise settles
// to: a pending promise is not an answer, one that settles false or rejects
// keeps the wait going, and the wait ends only once one settles true.
func TestWebUIBrowserPromisePredicateHolds(t *testing.T) {
	t.Parallel()
	if os.Getenv("OVERGO_WEBUI_LANE") != "1" {
		testskip.NotApplicable(t, "predicate waits run through cmd/webui-lane")
	}
	path, err := FindBrowser(os.Getenv("OVERGO_BROWSER"))
	if err != nil {
		t.Fatal(err)
	}
	ctx := t.Context()
	browser, err := Open(ctx, path, "data:text/html,<body></body>")
	if err != nil {
		t.Fatal(err)
	}
	defer browser.Close()
	if err := browser.Evaluate(ctx, `(() => { window.checks = 0; window.done = false; setTimeout(() => { window.done = true; }, 300); return true; })()`, nil); err != nil {
		t.Fatal(err)
	}
	if err := browser.Eventually(ctx, `new Promise((resolve, reject) => { window.checks++; setTimeout(() => window.checks % 2 ? resolve(window.done) : reject(new Error("not yet")), 5); })`); err != nil {
		t.Fatal(err)
	}
	var state struct {
		Done   bool
		Checks int
	}
	if err := browser.Evaluate(ctx, `({Done: window.done, Checks: window.checks})`, &state); err != nil {
		t.Fatal(err)
	}
	if !state.Done || state.Checks < 2 {
		t.Fatalf("a promise predicate held before it settled true: %+v", state)
	}
	t.Logf("promise predicate leg: a wait held only once its promise settled true, after %d checks", state.Checks)
}
