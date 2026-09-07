package webuilane

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"path/filepath"
	"strconv"
	"strings"
	"time"

	"overgo/internal/clioptions"
)

// Screenshot captures the viewport as PNG bytes through the browser's own
// screenshot command: what a reader sees, never a guess at it.
func (browser *Browser) Screenshot(ctx context.Context) ([]byte, error) {
	var response struct {
		Data string `json:"data"`
	}
	if err := browser.Call(ctx, "Page.captureScreenshot", map[string]any{"format": "png"}, &response); err != nil {
		return nil, err
	}
	if response.Data == "" {
		return nil, errors.New("webui lane: the browser returned no screenshot")
	}
	return base64.StdEncoding.DecodeString(response.Data)
}

// LayoutFinding is one measured layout fault on the page: what kind, the
// element it sits on, and the measurement behind it.
type LayoutFinding struct {
	Kind     string `json:"kind"`
	Selector string `json:"selector"`
	Detail   string `json:"detail"`
}

// Layout fault kinds the audit measures; the script names them from here.
const (
	// layoutOverflow: the document is wider than the viewport.
	layoutOverflow = "horizontal-overflow"
	// layoutOutside: a control or card lies outside the viewport's width.
	layoutOutside = "outside-viewport"
	// layoutOverlap: two fixed or sticky surfaces overlap.
	layoutOverlap = "overlapping-surfaces"
	// layoutClipped: single-line text is cut without an ellipsis.
	layoutClipped = "clipped-text"
	// layoutSmall: a control is narrower or shorter than a finger's target.
	layoutSmall = "small-control"
	// layoutContrast: text sits on its background below the readable ratio.
	layoutContrast = "low-contrast"
)

// String renders a finding on one line for a log.
func (finding LayoutFinding) String() string {
	return finding.Kind + " " + finding.Selector + ": " + finding.Detail
}

// LayoutAudit measures the page's layout as it stands: the document's
// width against the viewport, every control and card inside it, fixed and
// sticky surfaces apart, single-line text uncut, controls at least a
// finger wide and tall, and text against its background at the readable
// contrast ratio (the level-AA thresholds). The measurements come from
// the browser's own geometry and computed styles; the script is owned
// here so the criteria never drift between pages.
func (browser *Browser) LayoutAudit(ctx context.Context) ([]LayoutFinding, error) {
	var encoded string
	if err := browser.Evaluate(ctx, layoutAuditScript(), &encoded); err != nil {
		return nil, err
	}
	var findings []LayoutFinding
	if err := json.Unmarshal([]byte(encoded), &findings); err != nil {
		return nil, fmt.Errorf("webui lane: layout audit answered %q: %w", encoded, err)
	}
	return findings, nil
}

// Viewport is one size a page state is captured and audited at.
type Viewport struct {
	Name          string
	Width, Height int
}

// ScreenViewports are the sizes every state is captured at: a desktop
// window and a phone.
var ScreenViewports = []Viewport{{"desktop", 1280, 800}, {"phone", 390, 844}}

// StateFinding is a layout finding located at a captured state.
type StateFinding struct {
	Viewport, State string
	Finding         LayoutFinding
}

// String renders the finding with its state for a log.
func (finding StateFinding) String() string {
	return finding.Viewport + " " + finding.State + ": " + finding.Finding.String()
}

// ColourSchemes are the reader's schemes every state is captured under.
var ColourSchemes = []string{"dark", "light"}

// SetColorScheme emulates the reader's colour scheme ("dark" or "light")
// so a capture and its audit measure the page under each.
func (browser *Browser) SetColorScheme(ctx context.Context, scheme string) error {
	return browser.Call(ctx, "Emulation.setEmulatedMedia", map[string]any{
		"features": []map[string]string{{"name": "prefers-color-scheme", "value": scheme}},
	}, nil)
}

// CaptureState sets the viewport, captures the page as it stands (into dir
// as <viewport>-<state>.png when dir is set) and audits its layout; the
// findings carry the viewport and state.
func CaptureState(ctx context.Context, browser *Browser, dir string, viewport Viewport, state string) ([]StateFinding, error) {
	if err := browser.SetViewport(ctx, viewport.Width, viewport.Height); err != nil {
		return nil, err
	}
	// The page reports the new size and paints twice before the measurement, so a
	// layout mid-transition from the previous viewport is never captured.
	if err := browser.Eventually(ctx, fmt.Sprintf("window.innerWidth === %d && window.innerHeight === %d", viewport.Width, viewport.Height)); err != nil {
		return nil, err
	}
	if err := browser.Evaluate(ctx, `new Promise((settled) => requestAnimationFrame(() => requestAnimationFrame(() => settled(true))))`, nil); err != nil {
		return nil, err
	}
	if dir != "" {
		if err := clioptions.EnsureOutputDirectory(dir); err != nil {
			return nil, err
		}
		image, err := browser.Screenshot(ctx)
		if err != nil {
			return nil, err
		}
		if err := clioptions.WriteOutputFile(filepath.Join(dir, viewport.Name+"-"+state+".png"), image); err != nil {
			return nil, err
		}
	}
	measured, err := browser.LayoutAudit(ctx)
	if err != nil {
		return nil, err
	}
	findings := make([]StateFinding, 0, len(measured))
	for _, finding := range measured {
		findings = append(findings, StateFinding{Viewport: viewport.Name, State: state, Finding: finding})
	}
	return findings, nil
}

// CaptureStates walks every workbench tab and the model picker at each
// viewport under each colour scheme (a light state's name carries the
// "-light" suffix), audits each state's layout, and, with dir set, writes
// each state's capture there. The page is ready once the chat composer
// stands with no page error; a tab's own request settles within settle,
// and a slow tab is captured as it stands. It answers the states captured
// and every finding, and leaves the dark scheme emulated.
func CaptureStates(ctx context.Context, browser *Browser, dir string, settle time.Duration) (int, []StateFinding, error) {
	if err := browser.Eventually(ctx, `!!document.querySelector("#panel-chat.active .composer textarea") && window.overgo.errors.length === 0`); err != nil {
		return 0, nil, err
	}
	var tabs []string
	if err := browser.Evaluate(ctx, `[...document.querySelectorAll("#panels .panel")].map((panel) => panel.id.slice("panel-".length))`, &tabs); err != nil {
		return 0, nil, err
	}
	if len(tabs) == 0 {
		return 0, nil, errors.New("webui lane: the page mounted no tab")
	}
	states := 0
	var findings []StateFinding
	capture := func(viewport Viewport, state string) error {
		measured, err := CaptureState(ctx, browser, dir, viewport, state)
		if err != nil {
			return err
		}
		states++
		findings = append(findings, measured...)
		return nil
	}
	for _, scheme := range ColourSchemes {
		if err := browser.SetColorScheme(ctx, scheme); err != nil {
			return states, findings, err
		}
		suffix := ""
		if scheme != ColourSchemes[0] {
			suffix = "-" + scheme
		}
		for _, viewport := range ScreenViewports {
			for _, tab := range tabs {
				if err := browser.Evaluate(ctx, `location.hash = `+strconv.Quote(tab), nil); err != nil {
					return states, findings, err
				}
				if err := browser.Eventually(ctx, `!!document.querySelector("#panel-`+tab+`.active")`); err != nil {
					return states, findings, fmt.Errorf("%s %s: %w", viewport.Name, tab, err)
				}
				loading, done := context.WithTimeoutCause(ctx, settle, errors.New("webui lane: the tab kept loading"))
				_ = browser.Eventually(loading, `!document.querySelector("#panel-`+tab+` .note")?.textContent.startsWith("loading")`)
				done()
				if err := capture(viewport, tab+suffix); err != nil {
					return states, findings, err
				}
			}
			// The picker over the chat tab, closed again with Escape.
			if err := browser.Evaluate(ctx, `(() => { location.hash = "chat"; if (!document.querySelector(".topbar .card")) document.getElementById("model-pill").click(); return true; })()`, nil); err != nil {
				return states, findings, err
			}
			if err := browser.Eventually(ctx, `!!document.querySelector(".topbar .card .row .mono")`); err != nil {
				return states, findings, fmt.Errorf("%s picker: %w", viewport.Name, err)
			}
			if err := capture(viewport, "picker"+suffix); err != nil {
				return states, findings, err
			}
			if err := browser.Evaluate(ctx, `document.dispatchEvent(new KeyboardEvent("keydown", { key: "Escape" }))`, nil); err != nil {
				return states, findings, err
			}
		}
	}
	return states, findings, browser.SetColorScheme(ctx, ColourSchemes[0])
}

// CaptureSummary renders a capture walk's verdict on one line.
func CaptureSummary(states int, findings []StateFinding) string {
	return fmt.Sprintf("screens leg: captured %d states at %d viewports under %d colour schemes with %d layout findings", states, len(ScreenViewports), len(ColourSchemes), len(findings))
}

// layoutAuditScript is the audit with its fault kinds named from the
// constants above. Controls are the native focusable elements; a
// scrollable strip's own overflow is by design and its children are not
// outside the viewport; an input inside its label is reached through the
// label; an ellipsis is a designed cut; disabled controls carry a
// designed dimming; the readable ratio is 4.5, or 3 for large text; the
// canvas paints white under a transparent body.
func layoutAuditScript() string {
	return strings.NewReplacer(
		"KIND_OVERFLOW", layoutOverflow, "KIND_OUTSIDE", layoutOutside, "KIND_OVERLAP", layoutOverlap,
		"KIND_CLIPPED", layoutClipped, "KIND_SMALL", layoutSmall, "KIND_CONTRAST", layoutContrast,
	).Replace(layoutAuditTemplate)
}

const layoutAuditTemplate = `(() => {
  const findings = [];
  const limit = 40;
  const minimumControl = 24;
  const readable = 4.5, readableLarge = 3;
  const note = (kind, node, detail) => { if (findings.length < limit) findings.push({ kind, selector: describe(node), detail }); };
  const describe = (node) => {
    const parts = [];
    for (let current = node, depth = 0; current && current.nodeType === 1 && depth < 3; current = current.parentElement, depth++) {
      let part = current.tagName.toLowerCase();
      if (current.id) part += "#" + current.id;
      else if (current.classList.length) part += "." + [...current.classList].slice(0, 2).join(".");
      parts.unshift(part);
    }
    return parts.join(" > ");
  };
  const visible = (node) => {
    const rect = node.getBoundingClientRect();
    if (rect.width === 0 && rect.height === 0) return false;
    const style = getComputedStyle(node);
    return style.visibility !== "hidden" && style.display !== "none";
  };
  const scrollable = (node) => {
    for (let current = node.parentElement; current; current = current.parentElement) {
      const style = getComputedStyle(current);
      if (["auto", "scroll"].includes(style.overflowX)) return true;
    }
    return false;
  };
  const width = window.innerWidth;
  if (document.documentElement.scrollWidth > width + 1) note("KIND_OVERFLOW", document.documentElement, "document " + document.documentElement.scrollWidth + "px wide in a " + width + "px viewport");
  const controls = [...document.querySelectorAll("button, a[href], input, select, textarea, [role=button]")].filter((node) => visible(node) && node.type !== "hidden");
  for (const node of controls) {
    const rect = node.getBoundingClientRect();
    if (!scrollable(node) && (rect.right > width + 1 || rect.left < -1)) note("KIND_OUTSIDE", node, "spans " + Math.round(rect.left) + ".." + Math.round(rect.right) + "px of " + width);
    const target = (node.closest("label") || node).getBoundingClientRect();
    if (!node.disabled && (target.width < minimumControl || target.height < minimumControl)) note("KIND_SMALL", node, Math.round(target.width) + "x" + Math.round(target.height) + "px");
  }
  for (const node of document.querySelectorAll(".card, .pill")) {
    const rect = node.getBoundingClientRect();
    if (visible(node) && !scrollable(node) && (rect.right > width + 1 || rect.left < -1)) note("KIND_OUTSIDE", node, "spans " + Math.round(rect.left) + ".." + Math.round(rect.right) + "px of " + width);
  }
  const surfaces = [...document.querySelectorAll("*")].filter((node) => visible(node) && ["fixed", "sticky"].includes(getComputedStyle(node).position));
  for (let i = 0; i < surfaces.length; i++) for (let j = i + 1; j < surfaces.length; j++) {
    if (surfaces[i].contains(surfaces[j]) || surfaces[j].contains(surfaces[i])) continue;
    const a = surfaces[i].getBoundingClientRect(), b = surfaces[j].getBoundingClientRect();
    const overlap = Math.min(a.right, b.right) - Math.max(a.left, b.left), overlapY = Math.min(a.bottom, b.bottom) - Math.max(a.top, b.top);
    if (overlap > 1 && overlapY > 1) note("KIND_OVERLAP", surfaces[i], "overlaps " + describe(surfaces[j]) + " by " + Math.round(overlap) + "x" + Math.round(overlapY) + "px");
  }
  for (const node of document.querySelectorAll("*")) {
    if (!visible(node) || ["INPUT", "TEXTAREA", "SELECT"].includes(node.tagName)) continue;
    const style = getComputedStyle(node);
    if (style.whiteSpace === "nowrap" && ["hidden", "clip"].includes(style.overflowX) && style.textOverflow !== "ellipsis" && node.scrollWidth > node.clientWidth + 1) {
      note("KIND_CLIPPED", node, node.scrollWidth + "px of text in " + node.clientWidth + "px");
    }
  }
  const channel = (value) => { const c = value / 255; return c <= 0.03928 ? c / 12.92 : Math.pow((c + 0.055) / 1.055, 2.4); };
  const parse = (color) => { const m = /rgba?\(([^)]+)\)/.exec(color); if (!m) return null; const p = m[1].split(",").map(Number); return { r: p[0], g: p[1], b: p[2], a: p.length > 3 ? p[3] : 1 }; };
  const luminance = (c) => 0.2126 * channel(c.r) + 0.7152 * channel(c.g) + 0.0722 * channel(c.b);
  const backgroundOf = (node) => {
    for (let current = node; current; current = current.parentElement) {
      const style = getComputedStyle(current);
      const color = parse(style.backgroundColor);
      if (color && color.a > 0.99) return color;
      // A gradient paints its first stop behind the text nearest it.
      const stop = style.backgroundImage !== "none" ? parse(style.backgroundImage) : null;
      if (stop && stop.a > 0.99) return stop;
    }
    const ground = parse(getComputedStyle(document.body).backgroundColor);
    return ground && ground.a > 0.99 ? ground : { r: 255, g: 255, b: 255, a: 1 };
  };
  const hasText = (node) => [...node.childNodes].some((child) => child.nodeType === 3 && child.textContent.trim().length > 0);
  for (const node of document.querySelectorAll("body *")) {
    if (!hasText(node) || !visible(node)) continue;
    const style = getComputedStyle(node);
    if (node.closest("[disabled]") || Number(style.opacity) < 1) continue;
    const color = parse(style.color);
    if (!color) continue;
    const background = backgroundOf(node);
    const l1 = luminance(color), l2 = luminance(background);
    const ratio = (Math.max(l1, l2) + 0.05) / (Math.min(l1, l2) + 0.05);
    const size = parseFloat(style.fontSize), weight = parseInt(style.fontWeight, 10) || 400;
    const threshold = size >= 24 || (size >= 18.66 && weight >= 700) ? readableLarge : readable;
    if (ratio < threshold) note("KIND_CONTRAST", node, ratio.toFixed(2) + ":1 at " + size + "px (" + style.color + " on " + style.backgroundColor + ")");
  }
  return JSON.stringify(findings);
})()`
