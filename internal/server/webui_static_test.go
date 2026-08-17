package server

import (
	"net/http"
	"net/http/httptest"
	"regexp"
	"strings"
	"testing"
)

func TestWebUIServesEmbeddedAssets(t *testing.T) {
	handler := newTestHandler(t, &fakeGenerator{})
	cases := []struct {
		path        string
		contentType string
		needle      string
	}{
		{"/", "text/html; charset=utf-8", "overgo"},
		{"/index.html", "text/html; charset=utf-8", "probing /health"},
		{"/probe.js", "text/javascript; charset=utf-8", "window.probe"},
		{"/app.html", "text/html; charset=utf-8", "workbench"},
		{"/style.css", "text/css; charset=utf-8", "--acc"},
		{"/boot.js", "text/javascript; charset=utf-8", "window.overgo"},
		{"/viz.js", "text/javascript; charset=utf-8", "sparkline"},
		{"/md.js", "text/javascript; charset=utf-8", "overgo.md"},
		{"/mod/chat.js", "text/javascript; charset=utf-8", "/v1/chat/completions"},
		{"/mod/runtime.js", "text/javascript; charset=utf-8", "/slots"},
		{"/mod/datasets.js", "text/javascript; charset=utf-8", "/datasets"},
		{"/mod/training.js", "text/javascript; charset=utf-8", "/runs"},
		{"/mod/analyze_model.js", "text/javascript; charset=utf-8", "/analyze/model"},
		{"/mod/analyze_vocab.js", "text/javascript; charset=utf-8", "/analyze/vocab"},
		{"/mod/analyze_logits.js", "text/javascript; charset=utf-8", "completion_probabilities"},
		{"/mod/analyze_states.js", "text/javascript; charset=utf-8", "/analyze/states"},
		{"/mod/analyze_attention.js", "text/javascript; charset=utf-8", "/analyze/attention"},
		{"/mod/analyze_tensors.js", "text/javascript; charset=utf-8", "/analyze/tensors"},
	}
	for _, testCase := range cases {
		response := serveTestRequest(handler, http.MethodGet, testCase.path, "")
		if response.Code != http.StatusOK {
			t.Fatalf("GET %s status = %d", testCase.path, response.Code)
		}
		if got := response.Header().Get("Content-Type"); got != testCase.contentType {
			t.Fatalf("GET %s content-type = %q, want %q", testCase.path, got, testCase.contentType)
		}
		if !strings.Contains(response.Body.String(), testCase.needle) {
			t.Fatalf("GET %s body missing %q", testCase.path, testCase.needle)
		}
	}
}

func TestWebUIRuntimeMonitor(t *testing.T) {
	handler := newTestHandler(t, &fakeGenerator{})
	get := func(path string) string {
		return serveTestRequest(handler, http.MethodGet, path, "").Body.String()
	}
	runtime := get("/mod/runtime.js")
	for _, token := range []string{"overgo.poller", "onActivate", "onDeactivate", `api.get("/slots"`} {
		if !strings.Contains(runtime, token) {
			t.Errorf("runtime module missing %q", token)
		}
	}
	if strings.Contains(runtime, "include_text") {
		t.Error("runtime monitor requests retained text")
	}
	boot := get("/boot.js")
	for _, token := range []string{"function poller", "document.hidden", "t.onActivate", "t.onDeactivate"} {
		if !strings.Contains(boot, token) {
			t.Errorf("boot lifecycle missing %q", token)
		}
	}
	if !strings.Contains(get("/app.html"), "/mod/runtime.js") {
		t.Error("app shell does not load runtime module")
	}
}

func TestWebUIUnknownPathReturns404(t *testing.T) {
	handler := newTestHandler(t, &fakeGenerator{})
	response := serveTestRequest(handler, http.MethodGet, "/no/such/asset.js", "")
	if response.Code != http.StatusNotFound {
		t.Fatalf("status = %d, want 404", response.Code)
	}
	if !strings.Contains(response.Body.String(), "route not found") {
		t.Fatalf("body = %s", response.Body.String())
	}
}

// The GUI shell and its assets must load without a bearer token even when the
// server is key-protected — otherwise the page could never present the key
// field. API routes stay protected; only the static assets are public.
func TestWebUIAssetsPublicWhenAPIKeyConfigured(t *testing.T) {
	handler, err := New(Config{ModelID: testModelID, MaxTokens: testMaxTokens, APIKey: "secret"}, &fakeGenerator{})
	if err != nil {
		t.Fatal(err)
	}
	asset := serveTestRequest(handler, http.MethodGet, "/", "")
	if asset.Code != http.StatusOK {
		t.Fatalf("GET / with key configured = %d, want 200 (assets public)", asset.Code)
	}
	// A protected API route with no bearer must still be rejected.
	protected := serveTestRequest(handler, http.MethodGet, "/analyze/model", "")
	if protected.Code != http.StatusUnauthorized {
		t.Fatalf("GET /analyze/model without bearer = %d, want 401", protected.Code)
	}
}

// TestWebUIStyleInvariants guards two concrete regressions: (1) client CSS
// custom properties must reference names the palette actually defines — an
// undefined var silently renders nothing (the energy-entropy bar filled with a
// non-existent --accent); (2) the panel must own its horizontal overflow so a
// wide table cannot force the whole page body to scroll sideways.
func TestWebUIStyleInvariants(t *testing.T) {
	handler := newTestHandler(t, &fakeGenerator{})

	// Every var(--name) used in a module must be a name style.css defines.
	css := serveTestRequest(handler, http.MethodGet, "/style.css", "").Body.String()
	defined := map[string]bool{}
	for _, m := range regexp.MustCompile(`--[a-z0-9-]+\s*:`).FindAllString(css, -1) {
		defined[strings.TrimSpace(strings.TrimSuffix(m, ":"))] = true
	}
	for _, asset := range []string{"/mod/analyze_tensors.js", "/viz.js", "/mod/analyze_states.js", "/mod/analyze_attention.js"} {
		body := serveTestRequest(handler, http.MethodGet, asset, "").Body.String()
		for _, ref := range regexp.MustCompile(`var\(\s*(--[a-z0-9-]+)`).FindAllStringSubmatch(body, -1) {
			if !defined[ref[1]] {
				t.Errorf("%s references undefined CSS var %q", asset, ref[1])
			}
		}
	}

	// Panels scroll their own overflow so wide tables never scroll the body.
	if !strings.Contains(css, "overflow-x:auto") {
		t.Error("style.css: expected a panel overflow-x:auto rule so wide tables scroll in-panel")
	}
}

// TestWebUIAuthUX guards the centralized 401 handling: a shared friendlyError
// helper, a global key-required banner, and the previously-inconsistent tabs
// routing their errors through the helper instead of leaking the raw bearer
// error.
func TestWebUIAuthUX(t *testing.T) {
	handler := newTestHandler(t, &fakeGenerator{})
	get := func(p string) string { return serveTestRequest(handler, http.MethodGet, p, "").Body.String() }

	boot := get("/boot.js")
	for _, needle := range []string{"friendlyError", "showAuthNotice", "auth-banner"} {
		if !strings.Contains(boot, needle) {
			t.Errorf("boot.js missing %q", needle)
		}
	}
	css := get("/style.css")
	for _, needle := range []string{".auth-banner", ".keyfield.needs-key"} {
		if !strings.Contains(css, needle) {
			t.Errorf("style.css missing %q", needle)
		}
	}
	for _, asset := range []string{"/mod/analyze_model.js", "/mod/analyze_vocab.js", "/mod/analyze_tensors.js"} {
		if !strings.Contains(get(asset), "friendlyError") {
			t.Errorf("%s does not route errors through friendlyError", asset)
		}
	}
}

// TestWebUIHeatmapLegend guards the heatmap readability additions: a color-scale
// legend built from the same ramp() the cells use, and axis tick labels driven by
// caller-supplied labels (so a reader can map color→value and read the axes
// instead of hovering every cell).
func TestWebUIHeatmapLegend(t *testing.T) {
	handler := newTestHandler(t, &fakeGenerator{})
	viz := serveTestRequest(handler, http.MethodGet, "/viz.js", "").Body.String()
	for _, needle := range []string{"colorScaleLegend", "linearGradient", "rowLabels", "colLabels"} {
		if !strings.Contains(viz, needle) {
			t.Errorf("viz.js heatmap missing %q", needle)
		}
	}
	for _, asset := range []string{"/mod/analyze_attention.js", "/mod/analyze_states.js"} {
		if !strings.Contains(serveTestRequest(handler, http.MethodGet, asset, "").Body.String(), "labels:") {
			t.Errorf("%s does not pass labels to heatmap (axis ticks)", asset)
		}
	}
}

// TestWebUIShellCache guards the shared /analyze/model cache (fetched once for
// capability gating, the Model tab, and the lens vocab size) and the periodic
// health re-probe so a dropped/restored server updates the status pill.
func TestWebUIShellCache(t *testing.T) {
	handler := newTestHandler(t, &fakeGenerator{})
	get := func(p string) string { return serveTestRequest(handler, http.MethodGet, p, "").Body.String() }
	boot := get("/boot.js")
	for _, needle := range []string{"modelInfo", "invalidateModel", "setInterval(refreshStatus"} {
		if !strings.Contains(boot, needle) {
			t.Errorf("boot.js missing %q", needle)
		}
	}
	for _, asset := range []string{"/mod/analyze_model.js", "/mod/analyze_logits.js"} {
		body := get(asset)
		if !strings.Contains(body, "modelInfo") {
			t.Errorf("%s does not use the shared modelInfo cache", asset)
		}
		if strings.Contains(body, `api.get("/analyze/model")`) {
			t.Errorf("%s still fetches /analyze/model directly (bypasses the cache)", asset)
		}
	}
}

// TestWebUICancel guards the cancelable-run path for the tabs that drive a real
// forward pass (lens, hidden-states, attention). The abort semantics live once
// in the shared runner; each tab drives it and forwards the signal.
func TestWebUICancel(t *testing.T) {
	handler := newTestHandler(t, &fakeGenerator{})
	get := func(p string) string { return serveTestRequest(handler, http.MethodGet, p, "").Body.String() }
	boot := get("/boot.js")
	for _, needle := range []string{"AbortController", "controller.abort", "AbortError", "function runner"} {
		if !strings.Contains(boot, needle) {
			t.Errorf("boot.js shared runner missing %q", needle)
		}
	}
	for _, asset := range []string{"/mod/analyze_logits.js", "/mod/analyze_states.js", "/mod/analyze_attention.js"} {
		body := get(asset)
		if !strings.Contains(body, "overgo.runner(") {
			t.Errorf("%s does not use the shared overgo.runner", asset)
		}
		if !strings.Contains(body, "{ signal }") {
			t.Errorf("%s does not forward the abort signal to its request", asset)
		}
	}
}

// TestWebUIClientDedup guards the tightening: displayToken is defined once in the
// shell and the token-showing tabs use it instead of each redefining it.
func TestWebUIClientDedup(t *testing.T) {
	handler := newTestHandler(t, &fakeGenerator{})
	get := func(p string) string { return serveTestRequest(handler, http.MethodGet, p, "").Body.String() }
	if !strings.Contains(get("/boot.js"), "function displayToken") {
		t.Error("boot.js should define the shared displayToken")
	}
	for _, asset := range []string{"/mod/analyze_logits.js", "/mod/analyze_states.js", "/mod/analyze_attention.js"} {
		if strings.Contains(get(asset), "function displayToken") {
			t.Errorf("%s redefines displayToken instead of using the shared overgo.displayToken", asset)
		}
	}
}

// TestWebUIChatMarkdown guards the chat markdown renderer: it must build DOM
// (never innerHTML — model output is untrusted), scheme-check link hrefs, offer
// copy buttons, and be wired into the chat tab and the shell load order.
func TestWebUIChatMarkdown(t *testing.T) {
	handler := newTestHandler(t, &fakeGenerator{})
	get := func(p string) string { return serveTestRequest(handler, http.MethodGet, p, "").Body.String() }

	md := get("/md.js")
	if strings.Contains(md, ".innerHTML") {
		t.Error("md.js assigns .innerHTML — model output must render as DOM text, never markup")
	}
	for _, needle := range []string{"md-codeblock", "clipboard.writeText", "https?:", "createTextNode"} {
		if !strings.Contains(md, needle) {
			t.Errorf("md.js missing %q", needle)
		}
	}
	chat := get("/mod/chat.js")
	for _, needle := range []string{"overgo.md(", "overgo.copyButton"} {
		if !strings.Contains(chat, needle) {
			t.Errorf("chat.js does not use %q", needle)
		}
	}
	// md.js must load before chat.js so overgo.md exists when chat renders.
	app := get("/app.html")
	if strings.Index(app, "/md.js") < 0 || strings.Index(app, "/md.js") > strings.Index(app, "/mod/chat.js") {
		t.Error("app.html must load /md.js before /mod/chat.js")
	}
}

// TestWebUIContentSecurityPolicy guards the strict CSP: same-origin scripts only
// (no inline), nosniff, and no framing. The landing page must therefore carry no
// inline <script> or inline event handler — those moved to probe.js.
func TestWebUIContentSecurityPolicy(t *testing.T) {
	handler := newTestHandler(t, &fakeGenerator{})
	for _, p := range []string{"/", "/index.html", "/app.html", "/boot.js"} {
		resp := serveTestRequest(handler, http.MethodGet, p, "")
		csp := resp.Header().Get("Content-Security-Policy")
		if !strings.Contains(csp, "script-src 'self';") {
			t.Errorf("GET %s CSP script-src not strict-self: %q", p, csp)
		}
		if strings.Contains(csp, "script-src 'self' 'unsafe-inline'") {
			t.Errorf("GET %s CSP allows inline script: %q", p, csp)
		}
		for _, directive := range []string{"object-src 'none'", "frame-ancestors 'none'", "connect-src 'self'"} {
			if !strings.Contains(csp, directive) {
				t.Errorf("GET %s CSP missing %q", p, directive)
			}
		}
		if resp.Header().Get("X-Content-Type-Options") != "nosniff" {
			t.Errorf("GET %s missing X-Content-Type-Options: nosniff", p)
		}
	}
	index := serveTestRequest(handler, http.MethodGet, "/index.html", "").Body.String()
	if strings.Contains(index, "<script>") {
		t.Error("index.html has an inline <script> — blocked by script-src 'self'")
	}
	if strings.Contains(index, "onclick=") {
		t.Error("index.html has an inline onclick — blocked by script-src 'self'")
	}
}

func TestWebUIRejectsNonGet(t *testing.T) {
	handler := newTestHandler(t, &fakeGenerator{})
	response := httptest.NewRecorder()
	handler.ServeHTTP(response, httptest.NewRequest(http.MethodPost, "/", nil))
	if response.Code != http.StatusNotFound {
		t.Fatalf("POST / status = %d, want 404", response.Code)
	}
}
