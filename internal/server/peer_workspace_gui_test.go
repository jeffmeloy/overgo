package server

import (
	"net/http"
	"strings"
	"testing"
)

// TestPeerWorkspaceUsesGlobalOperations keeps the peers tab from owning
// operation state: the peer workspace leg drives the tab, and its source
// holds no poller or operation read of its own (the runtime stream carries
// operations).
func TestPeerWorkspaceUsesGlobalOperations(t *testing.T) {
	t.Parallel()
	module := serveTestRequest(newTestHandler(t, &fakeGenerator{}), http.MethodGet, "/mod/peers.js", "").Body.String()
	for _, forbidden := range []string{"setInterval", "overgo.poller", `api.get("/operations`} {
		if strings.Contains(module, forbidden) {
			t.Errorf("peer workspace duplicates operation authority with %q", forbidden)
		}
	}
}
