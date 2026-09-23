package server

import (
	"net/http"
	"net/http/httptest"
	"os"
	"slices"
	"sync"
	"testing"

	"overgo/internal/testskip"
	"overgo/internal/webuilane"
)

// TestWebUIBrowserRouteExplorer sends explorer rows from a real browser and
// reads the requests the server received: a DELETE row sends DELETE with its
// query (it once sent POST, which starts a hub download instead of cancelling
// one) and a GET row sends GET with its query.
func TestWebUIBrowserRouteExplorer(t *testing.T) {
	if os.Getenv("OVERGO_WEBUI_LANE") != "1" {
		t.Skip(testskip.Inapplicable + ": the route explorer runs through cmd/webui-lane")
	}
	handler := newTestHandler(t, &fakeGenerator{})
	defer handler.Close()
	var mu sync.Mutex
	var received []string
	server := httptest.NewServer(http.HandlerFunc(func(response http.ResponseWriter, request *http.Request) {
		if request.URL.Path == "/hub/downloads" {
			mu.Lock()
			received = append(received, request.Method+" "+request.URL.RequestURI())
			mu.Unlock()
		}
		handler.ServeHTTP(response, request)
	}))
	defer server.Close()
	path, err := webuilane.FindBrowser(os.Getenv("OVERGO_BROWSER"))
	if err != nil {
		t.Fatal(err)
	}
	ctx := t.Context()
	browser, err := webuilane.Open(ctx, path, server.URL+"/app.html#api")
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
	settle(`document.querySelectorAll('#panel-api .card').length > 10`)
	// The filter narrows the table to the rows naming the typed path.
	assertBrowserPredicate(t, ctx, browser, `(() => {
  const filter = document.querySelector('#panel-api input[type="search"]');
  filter.value = 'hub/downloads';
  filter.dispatchEvent(new Event('input'));
  const shown = [...document.querySelectorAll('#panel-api .card')].filter((card) => !card.hidden);
  return shown.length === 3 && shown.every((card) => card.dataset.route.endsWith('/hub/downloads'));
})()`)
	for _, row := range []struct{ method, query string }{{http.MethodDelete, "id=explorer-probe"}, {http.MethodGet, "id=explorer-list"}} {
		label := row.method + " /hub/downloads query"
		assertBrowserPredicate(t, ctx, browser, `(() => {
  const editor = document.querySelector('#panel-api [aria-label="`+label+`"]');
  if (!editor) return false;
  editor.value = "`+row.query+`";
  editor.closest('.card').querySelector('button').click();
  return true;
})()`)
		settle(`(() => { const out = document.querySelector('#panel-api [aria-label="` + label + `"]').closest('.card').querySelector('pre'); return !out.hidden && out.textContent.length > 0; })()`)
	}
	mu.Lock()
	defer mu.Unlock()
	for _, want := range []string{"DELETE /hub/downloads?id=explorer-probe", "GET /hub/downloads?id=explorer-list"} {
		if !slices.Contains(received, want) {
			t.Errorf("server received %v; want %q", received, want)
		}
	}
	if slices.ContainsFunc(received, func(request string) bool { return request[:4] == "POST" }) {
		t.Errorf("the explorer sent a POST to /hub/downloads: %v", received)
	}
	t.Log("route explorer leg: the filter narrows the table; DELETE and GET rows send their declared method and query; no row starts a download it was asked to cancel")
}
