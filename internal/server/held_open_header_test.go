package server

import (
	"net/http"
	"net/http/httptest"
	"testing"

	"overgo/internal/apimanifest"
	"overgo/internal/testcheck"
)

// TestActivityStreamNamesItselfHeldOpen holds the server's side of the swap proxy's drain: the workbench's activity
// stream, which a page keeps open for as long as it listens, answers with the held-open header, and an ordinary
// route does not, so a swap waits for the requests that are finishing and not for the page's own subscription.
func TestActivityStreamNamesItselfHeldOpen(t *testing.T) {
	t.Parallel()
	handler := newTestHandler(t, &fakeGenerator{})
	defer handler.Close()
	server := httptest.NewServer(handler)
	defer server.Close()
	for path, held := range map[string]bool{"/runtime/activity/stream": true, "/v1/models": false} {
		request, err := http.NewRequestWithContext(t.Context(), http.MethodGet, server.URL+path, nil)
		testcheck.NoError(t, err)
		response, err := server.Client().Do(request)
		testcheck.NoError(t, err)
		_ = response.Body.Close()
		if got := response.Header.Get(apimanifest.HeldOpenHeader) != ""; got != held {
			t.Errorf("%s names itself held open = %t, want %t", path, got, held)
		}
	}
}
