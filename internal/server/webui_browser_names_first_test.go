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

// TestWebUIBrowserNamesFirst drives how a model is named where a person
// picks it out: a file name no other model shares reads alone; one that
// others share, as every Hugging Face checkpoint's model.safetensors does,
// reads with as many folders above it as tell it apart; a shard reads with
// its folder. The Library and the picker name models the same way, the
// full location in the Library row's title.
func TestWebUIBrowserNamesFirst(t *testing.T) {
	t.Parallel()
	if os.Getenv("OVERGO_WEBUI_LANE") != "1" {
		t.Skip(testskip.Inapplicable + ": the names leg uses Chromium through cmd/webui-lane")
	}
	handler := newTestHandlerWithRepository(t, responseRecipeGenerator(t, &fakeGenerator{}))
	entry := func(model, location string) map[string]any {
		return map[string]any{"model": model, "location": location, "present": true, "recipe": "recipe:" + model,
			"capabilities": []map[string]any{{"task": "inference", "recipe": "recipe:" + model, "tier": "verified"}}}
	}
	catalog, err := json.Marshal(map[string]any{"truncated": false, "models": []map[string]any{
		entry("model:flux", "C:/models/flux-dev/transformer/model.safetensors"),
		entry("model:turbo", "C:/models/sd-turbo/transformer/model.safetensors"),
		entry("model:shard", "C:/models/big-model/model-00001-of-00008.safetensors"),
		entry("model:solo", "C:/models/solo.gguf"),
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
	browser, err := webuilane.Open(ctx, path, server.URL+"/#library")
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
	const named = `'flux-dev/transformer/model.safetensors|sd-turbo/transformer/model.safetensors|big-model/model-00001-of-00008.safetensors|solo.gguf'`

	// The Library names each model by what tells it apart, its location in the title.
	settle(`document.querySelectorAll('#panel-library table.grid tbody tr').length === 4`)
	check(`(() => {
  const cells = [...document.querySelectorAll('#panel-library table.grid tbody tr')].map(row => row.querySelector('td span'));
  return cells.map(cell => cell.textContent).join('|') === ` + named + ` && cells[0].title === 'C:/models/flux-dev/transformer/model.safetensors';
})()`)

	// The picker names them the same way.
	check(`(() => { document.getElementById('model-pill').click(); return true; })()`)
	settle(`document.querySelectorAll('dialog[open] .row[tabindex]').length === 4`)
	check(`[...document.querySelectorAll('dialog[open] .row[tabindex] > .mono')].map(name => name.textContent).join('|') === ` + named)
	for _, viewport := range webuilane.ScreenViewports {
		findings, err := webuilane.CaptureState(ctx, browser, os.Getenv("OVERGO_WEBUI_LANE_SCREENS"), viewport, "names-first")
		if err != nil || len(findings) != 0 {
			t.Fatalf("%s: %v %v", viewport.Name, findings, err)
		}
	}
	webuilane.Leg(t, "names leg", "a file name no other model shares read alone, shared ones read with the folders that tell them apart, a shard with its folder, alike in the Library and the picker")
}
