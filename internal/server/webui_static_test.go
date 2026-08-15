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
		{"/app.html", "text/html; charset=utf-8", "workbench"},
		{"/style.css", "text/css; charset=utf-8", "--acc"},
		{"/boot.js", "text/javascript; charset=utf-8", "window.overgo"},
		{"/viz.js", "text/javascript; charset=utf-8", "sparkline"},
		{"/mod/chat.js", "text/javascript; charset=utf-8", "/v1/chat/completions"},
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

func TestWebUIRejectsNonGet(t *testing.T) {
	handler := newTestHandler(t, &fakeGenerator{})
	response := httptest.NewRecorder()
	handler.ServeHTTP(response, httptest.NewRequest(http.MethodPost, "/", nil))
	if response.Code != http.StatusNotFound {
		t.Fatalf("POST / status = %d, want 404", response.Code)
	}
}
