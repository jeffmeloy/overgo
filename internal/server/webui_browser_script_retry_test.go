package server

import (
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"testing"

	"overgo/internal/overgodb"
)

// TestWebUIBrowserRetriesAFailedScript fails the first fetch of one tab
// module and holds the shell to fetching it again and starting, as it must
// when a busy host briefly runs out of socket buffers.
func TestWebUIBrowserRetriesAFailedScript(t *testing.T) {
	t.Parallel()
	requireWebUILane(t, "the script retry leg uses Chromium through cmd/webui-lane")
	store, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	handler := newTestHandlerForRepository(t, store, responseRecipeGenerator(t, &fakeGenerator{}))
	const failed = "/mod/agent.js"
	var failures atomic.Int32
	server := httptest.NewServer(http.HandlerFunc(func(response http.ResponseWriter, request *http.Request) {
		if request.URL.Path == failed && failures.CompareAndSwap(0, 1) {
			http.Error(response, "unavailable", http.StatusServiceUnavailable)
			return
		}
		handler.ServeHTTP(response, request)
	}))
	defer server.Close()
	_, browser, _, settle := openBrowserLeg(t, server.URL)
	defer browser.Close()
	settle(`!!document.querySelector('.composer textarea')`)
	if failures.Load() != 1 {
		t.Fatalf("the first fetch of %s was not failed", failed)
	}
}
