package webuilane

import (
	"context"
	"errors"
	"os"
	"strings"
	"testing"
	"time"

	"overgo/internal/testevidence"
)

// TestWebUIBrowserLayoutAudit pins the audit over a page built to fault in
// every measured way: a block wider than the viewport, a control past its
// edge and one too small, two fixed surfaces overlapping, a line cut
// without an ellipsis, and text too faint to read; the same page with
// those faults absent audits clean, and the screenshot is a PNG.
func TestWebUIBrowserLayoutAudit(t *testing.T) {
	if os.Getenv("OVERGO_WEBUI_LANE") != "1" {
		t.Skip(testevidence.ShortIntegrationSkip + ": the layout audit runs through cmd/webui-lane")
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
		`<p style="color:rgb(187,187,187)">faint words</p></body>`
	clean := `<body style="margin:0;background:rgb(255,255,255);color:rgb(0,0,0)"><button style="width:40px;height:40px">fine</button>` +
		`<div role="status" style="position:absolute;width:1px;height:1px;overflow:hidden;white-space:nowrap;clip-path:inset(50%)"><span>Response ready.</span></div>` +
		`<p style="white-space:nowrap;overflow:hidden;text-overflow:ellipsis;width:40px">a line of text cut by design</p>` +
		`<details id="collapsed" open><summary>Details</summary><p style="color:rgb(187,187,187)">hidden faint words</p><button style="width:10px;height:10px;padding:0">hidden tiny</button></details>` +
		`<script>const details=document.getElementById('collapsed');details.querySelector('p').getBoundingClientRect();details.open=false;</script></body>`
	audit := func(html string) []LayoutFinding {
		t.Helper()
		ctx, cancel := context.WithTimeoutCause(t.Context(), time.Minute, errors.New("webui lane: the synthetic page did not audit"))
		defer cancel()
		// A data URL keeps a "+" literal, so spaces travel percent-encoded.
		browser, err := Open(ctx, browserPath, "data:text/html,"+strings.ReplaceAll(html, " ", "%20"))
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
	for _, kind := range []string{layoutOverflow, layoutOutside, layoutSmall, layoutOverlap, layoutClipped, layoutContrast} {
		if !kinds[kind] {
			t.Errorf("the faulty page audited without %s", kind)
		}
	}
	if findings := audit(clean); len(findings) != 0 {
		t.Errorf("the clean page audited with findings: %v", findings)
	}
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
	ctx, cancel := context.WithTimeoutCause(t.Context(), time.Minute, errors.New("webui lane: the tall header did not audit"))
	defer cancel()
	browser, err := Open(ctx, browserPath, "data:text/html,"+strings.ReplaceAll(tall, " ", "%20"))
	if err != nil {
		t.Fatal(err)
	}
	defer browser.Close()
	findings, err := CaptureState(ctx, browser, "", ScreenViewports[1], "tall")
	if err != nil || len(findings) != 1 || findings[0].Finding.Kind != layoutHeader {
		t.Errorf("the tall header audited as %v, %v", findings, err)
	}
	if summary := CaptureSummary(3, nil); !strings.Contains(summary, "captured 3 states") || !strings.HasSuffix(summary, "0 layout findings") {
		t.Errorf("summary = %q", summary)
	}
}
