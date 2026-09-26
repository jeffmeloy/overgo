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

// TestWebUIBrowserModelPicker drives the model picker over a catalog the
// leg's server answers in the catalog's shape: the header states the served
// model's measured evidence, each row carries its evidence and capability
// tags, the served row says so, a stale row gives its reason and offers no
// Serve, the note names the last model this browser served, and a switch
// shows its progress on the row and as an operation chip, then reports that
// the server did not confirm the model with a way to retry.
func TestWebUIBrowserModelPicker(t *testing.T) {
	t.Parallel()
	if os.Getenv("OVERGO_WEBUI_LANE") != "1" {
		t.Skip(testskip.Inapplicable + ": the model picker leg uses Chromium through cmd/webui-lane")
	}
	handler := newTestHandlerWithRepository(t, responseRecipeGenerator(t, &fakeGenerator{}))
	var health struct {
		Model string `json:"model"`
	}
	if err := json.Unmarshal(serveTestRequest(handler, http.MethodGet, "/health", "").Body.Bytes(), &health); err != nil || health.Model == "" {
		t.Fatalf("health = %+v, %v", health, err)
	}
	// The served row's file carries the name the page shows for the served model.
	var manifest workspaceManifestResponse
	if err := json.Unmarshal(serveTestRequest(handler, http.MethodGet, "/workspace/manifest", "").Body.Bytes(), &manifest); err != nil || manifest.Model == nil {
		t.Fatalf("manifest = %+v, %v", manifest, err)
	}
	served := manifest.Model.Name
	catalog, err := json.Marshal(map[string]any{"truncated": false, "models": []map[string]any{
		{
			"model": health.Model, "location": "C:/models/" + served, "present": true, "recipe": "recipe:served",
			"benchmark": map[string]any{"prompt_tokens_per_second_p50": 120.4, "decode_tokens_per_second_p50": 31.6},
			"evals":     []map[string]any{{"suite": "store/arithmetic", "metrics": map[string]any{"accuracy": 0.9}}},
			"capabilities": []map[string]any{
				{"task": "inference", "recipe": "recipe:served", "tier": "verified"},
				{"task": "embedding", "recipe": "recipe:served-embedding", "tier": "contract-tested"},
			},
		},
		{"model": "model:other", "location": "C:/models/other.gguf", "present": true, "recipe": "recipe:other",
			"capabilities": []map[string]any{{"task": "inference", "recipe": "recipe:other", "tier": "verified"}}},
		{"model": "model:stale", "location": "C:/models/stale.gguf", "present": true, "recipe": "recipe:stale",
			"stale": "architecture qwen9 is not supported"},
	}})
	if err != nil {
		t.Fatal(err)
	}
	// swapping: the switch's health probe, held until the leg has seen its progress.
	swapping, release := make(chan struct{}, 1), make(chan struct{})
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch {
		case r.URL.Path == "/catalog/models":
			w.Header().Set("Content-Type", "application/json")
			_, _ = w.Write(catalog)
			return
		case r.URL.Path == "/health" && r.URL.Query().Get("swap") != "":
			swapping <- struct{}{}
			<-release
			// A proxy answers, but the served model is unchanged: the switch is not confirmed.
			w.Header().Set("X-Overgo-Swap-Proxy", "child")
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
	settle(`!!document.querySelector('.composer textarea')`)
	// The header states the served model's measured evidence.
	settle(`document.getElementById('model-evidence').textContent === '120 prompt tok/s · 32 decode tok/s · arithmetic 0.90'`)

	// The picker lists each row with its evidence, tags and control.
	check(`(() => { try { localStorage.setItem('overgo.model', 'remembered.gguf'); } catch (_) {} document.getElementById('model-pill').click(); return true; })()`)
	settle(`[...document.querySelectorAll('dialog[open] .row[tabindex]')].length === 3`)
	check(`(() => {
  window.pickerRow = (name) => [...document.querySelectorAll('dialog[open] .row[tabindex]')].find(row => row.querySelector('.mono').textContent === name);
  window.pickerTags = (name) => [...pickerRow(name).querySelectorAll('.tag')].map(tag => tag.textContent);
  window.pickerButton = (name, label) => [...pickerRow(name).querySelectorAll('button')].find(button => button.textContent.startsWith(label));
  return true;
})()`)
	check(`document.querySelector('dialog[open] .note').textContent.includes('last time you served remembered.gguf')`)
	check(`pickerRow(` + "`" + served + "`" + `).textContent.includes('120 prompt tok/s') && pickerTags(` + "`" + served + "`" + `).join(',') === 'inference,embedding · contract-tested,Serving'`)
	check(`!!pickerButton('other.gguf', 'Serve') && pickerTags('other.gguf').join(',') === 'inference'`)
	check(`pickerTags('stale.gguf').includes('unservable: architecture qwen9 is not supported') && !pickerButton('stale.gguf', 'Serve') && pickerRow('stale.gguf').querySelector('details').open`)

	// A switch shows its progress on the row and as an operation chip.
	check(`(() => { pickerButton('other.gguf', 'Serve').click(); return true; })()`)
	<-swapping
	settle(`(pickerButton('other.gguf', 'loading') || {}).textContent?.match(/^loading… [0-9]+s$/) !== null`)
	settle(`[...document.querySelectorAll('.operation-chip')].some(chip => chip.textContent.startsWith('model switch other.gguf') && chip.textContent.includes('running'))`)
	close(release)
	// Unconfirmed, it says so and offers the retry.
	settle(`document.querySelector('dialog[open]').textContent.includes('The server did not confirm the requested model') && !![...document.querySelectorAll('dialog[open] button')].find(button => button.textContent === 'Retry loading model')`)
	settle(`[...document.querySelectorAll('.operation-chip')].some(chip => chip.textContent.startsWith('model switch other.gguf') && chip.textContent.includes('failed'))`)
	webuilane.Leg(t, "model picker leg", "the header and rows stated measured evidence and capability tags, the served row said so, a stale row gave its reason without Serve, the remembered model was named, and a switch showed its progress and chip before reporting it unconfirmed with a retry")
}
