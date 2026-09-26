package server

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"testing"

	"overgo/internal/testskip"
	"overgo/internal/webuilane"
)

// TestWebUIBrowserRemoteTurns drives a model served at a hosted provider
// through the relay: the picker tags the hosted entry remote, a turn
// proceeds though the count route refuses (the provider tokenizes), the
// assistant turn carries the remote marker, and the turn's facts state the
// provider's own usage.
func TestWebUIBrowserRemoteTurns(t *testing.T) {
	// Serial: the relay fixture sets the process environment.
	if os.Getenv("OVERGO_WEBUI_LANE") != "1" {
		t.Skip(testskip.Inapplicable + ": the remote turns leg uses Chromium through cmd/webui-lane")
	}
	handler, _ := newRelayHandler(t, []string{"Remote ", "answer"})
	var health struct {
		Model string `json:"model"`
	}
	if err := json.Unmarshal(serveTestRequest(handler, http.MethodGet, "/health", "").Body.Bytes(), &health); err != nil || health.Model == "" {
		t.Fatalf("health = %+v, %v", health, err)
	}
	catalog, err := json.Marshal(map[string]any{"truncated": false, "models": []map[string]any{
		{"model": health.Model, "location": "remote://fake/vendor/model", "present": true, "recipe": "recipe:remote",
			"capabilities": []map[string]any{{"task": "inference", "recipe": "recipe:remote", "tier": "contract-tested"}}},
		{"model": "model:local", "location": "C:/models/local.gguf", "present": true, "recipe": "recipe:local",
			"capabilities": []map[string]any{{"task": "inference", "recipe": "recipe:local", "tier": "verified"}}},
	}})
	if err != nil {
		t.Fatal(err)
	}
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/catalog/models" {
			w.Header().Set("Content-Type", "application/json")
			_, _ = w.Write(catalog)
			return
		}
		handler.ServeHTTP(w, r)
	}))
	defer server.Close()
	path, err := webuilane.FindBrowser(os.Getenv("OVERGO_BROWSER"))
	if err != nil {
		t.Fatal(err)
	}
	ctx := t.Context()
	browser, err := webuilane.Open(ctx, path, server.URL)
	if err != nil {
		t.Fatal(err)
	}
	defer browser.Close()
	check := func(expression string) { t.Helper(); assertBrowserPredicate(t, ctx, browser, expression) }
	settle := func(expression string) {
		t.Helper()
		if err := browser.Eventually(ctx, expression); err != nil {
			t.Fatalf("%s: %v", expression, err)
		}
	}
	settle(`!!document.querySelector('.composer textarea') && !document.querySelector('.send-button').disabled`)

	// The picker tags the hosted entry remote and the local one not.
	check(`(() => { document.getElementById('model-pill').click(); return true; })()`)
	settle(`[...document.querySelectorAll('dialog[open] .row[tabindex]')].length === 2`)
	check(`(() => {
  const tags = (name) => [...[...document.querySelectorAll('dialog[open] .row[tabindex]')].find(row => row.textContent.includes(name)).querySelectorAll('.tag')].map(tag => tag.textContent);
  return tags('remote://fake/vendor/model').includes('remote') && !tags('local.gguf').includes('remote');
})()`)
	check(`(() => { document.querySelector('dialog[open]').close(); return true; })()`)

	// A turn proceeds though the count route refuses, marked remote, with the provider's usage.
	check(`(() => {
  const input = document.querySelector('.composer textarea');
  input.value = 'hi'; input.dispatchEvent(new Event('input', { bubbles: true }));
  document.querySelector('.send-button').click();
  return true;
})()`)
	settle(`[...document.querySelectorAll('.msg.assistant')].some(message => message.textContent.includes('Remote answer') && [...message.querySelectorAll('.tag')].some(tag => tag.textContent === 'remote'))`)
	settle(`(() => { const stat = (label) => [...document.querySelectorAll('.stat')].find(node => node.querySelector('.k').textContent === label);
  return !!stat('Input') && stat('Input').querySelector('.v').textContent.startsWith('7') && !!stat('Completion'); })()`)
	webuilane.Leg(t, "remote turns leg", "the picker tagged the hosted entry remote, a turn proceeded though the count route refused, its assistant turn carried the remote marker, and its facts stated the provider's usage")
}
