package server

import (
	"net/http"
	"strings"
	"testing"
)

// TestRouteTableIsTheOnlyDispatcher holds the agent precondition to one
// answer on every route it guards: without the agent runtime every agent
// route answers alike. internal/gate's TestServerRouteTableIsTheOnlyDispatcher
// holds the handlers' source to no path or method re-dispatch.
func TestRouteTableIsTheOnlyDispatcher(t *testing.T) {
	t.Parallel()
	handler := newTestHandler(t, &fakeGenerator{})
	defer handler.Close()
	guarded := 0
	for _, route := range routeCatalog {
		if segment, _, _ := strings.Cut(strings.TrimPrefix(route.Path, "/"), "/"); segment != "agents" {
			continue
		}
		guarded++
		answer := serveTestRequest(handler, route.Methods[0], route.Path, "{}")
		if answer.Code != http.StatusServiceUnavailable {
			t.Errorf("%s %s without the agent runtime answered %d, want %d: %s", route.Methods[0], route.Path, answer.Code, http.StatusServiceUnavailable, answer.Body)
		}
	}
	if guarded == 0 {
		t.Fatal("no route is guarded by a workspace precondition")
	}
}
