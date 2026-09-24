package server

import (
	"net/http"
	"net/http/httptest"
	"slices"
	"testing"
)

// TestMutationsRefreshTheirInventories holds the route table to its inventory
// declarations: every route that changes an inventory accepts a call other
// than a GET, and a successful such call tells every workspace which
// inventory it changed while a refused one or a GET tells none. The one
// event stream leg shows the open tab reading the changed inventory again.
func TestMutationsRefreshTheirInventories(t *testing.T) {
	t.Parallel()
	declared := 0
	for _, route := range routeCatalog {
		if !route.Inventory {
			continue
		}
		declared++
		if !slices.ContainsFunc(route.Methods, func(method string) bool { return method != http.MethodGet }) {
			t.Errorf("inventory route %s changes nothing: it accepts only %v", route.Path, route.Methods)
		}
	}
	if declared == 0 {
		t.Fatal("no route declares the inventory it changes")
	}

	handler := newTestHandler(t, &fakeGenerator{})
	defer handler.Close()
	events, unsubscribe, err := handler.events.subscribe(handler.config.MaxStoredResponses)
	if err != nil {
		t.Fatal(err)
	}
	defer unsubscribe()
	answering := func(status int) routeDescriptor {
		return routeDescriptor{Path: "/peers/state", Inventory: true, Handler: func(_ *Handler, response http.ResponseWriter, _ *http.Request) {
			response.WriteHeader(status)
		}}
	}
	for _, call := range []struct {
		method  string
		status  int
		changes bool
	}{
		{http.MethodPost, http.StatusCreated, true},
		{http.MethodPost, http.StatusConflict, false},
		{http.MethodGet, http.StatusOK, false},
	} {
		handler.serveInventoryChange(answering(call.status), httptest.NewRecorder(), httptest.NewRequest(call.method, "/peers/state", nil))
		select {
		case event := <-events:
			if !call.changes || event.name != "workspace.changed" || event.value.(workspaceChange).Inventory != "/peers" {
				t.Errorf("%s answered %d published %q %+v", call.method, call.status, event.name, event.value)
			}
		default:
			if call.changes {
				t.Errorf("%s answered %d published no change", call.method, call.status)
			}
		}
	}
}
