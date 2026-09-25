package server

import (
	"net/http"
	"strings"
	"testing"
)

// TestRouteTableIsTheOnlyDispatcher holds each workspace precondition to one
// answer on every route it guards: without the agent runtime every agent
// route answers alike, and without a workspace every peer or automation route
// answers what is missing. internal/gate's TestServerRouteTableIsTheOnlyDispatcher
// holds the handlers' source to no path or method re-dispatch.
func TestRouteTableIsTheOnlyDispatcher(t *testing.T) {
	t.Parallel()
	handler := newTestHandler(t, &fakeGenerator{})
	defer handler.Close()
	guarded := 0
	for _, route := range routeCatalog {
		segment, _, _ := strings.Cut(strings.TrimPrefix(route.Path, "/"), "/")
		want := map[string]int{"agents": http.StatusServiceUnavailable, "peers": http.StatusNotImplemented, "automations": http.StatusNotImplemented}[segment]
		if want == 0 || route.Path == "/automations/webhook" {
			continue
		}
		guarded++
		answer := serveTestRequest(handler, route.Methods[0], route.Path, "{}")
		if answer.Code != want {
			t.Errorf("%s %s without its workspace answered %d, want %d: %s", route.Methods[0], route.Path, answer.Code, want, answer.Body)
		}
	}
	if guarded == 0 {
		t.Fatal("no route is guarded by a workspace precondition")
	}
}
