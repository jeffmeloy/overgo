package server

import (
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"overgo/internal/clioptions"
)

// TestRouteAuthenticationRefusesUnauthenticated holds the serving-surface
// authentication policy: every route that mutates state, executes tools, or
// reaches outward requires the bearer credential, and a listener that would
// expose unauthenticated routes beyond this host refuses to start.
func TestRouteAuthenticationRefusesUnauthenticated(t *testing.T) {
	mutating := map[string]bool{
		http.MethodPost: true, http.MethodPut: true,
		http.MethodPatch: true, http.MethodDelete: true,
	}
	for _, route := range routeCatalog {
		if route.Path == "/automations/webhook" {
			// The webhook verifies its raw transport bytes against the
			// active automation secrets; that is its authentication.
			continue
		}
		for _, method := range route.Methods {
			if mutating[method] && route.Authentication != routeBearer {
				t.Errorf("mutating route %s %s authentication = %q, want bearer", method, route.Path, route.Authentication)
			}
		}
	}
	for _, path := range []string{
		"/hub/search", "/hub/downloads", "/agent/tools", "/agent/step",
		"/agent/approval", "/agent/provenance", "/agent/sessions",
	} {
		route, found := resolveRoute(path)
		if !found || route.Authentication != routeBearer {
			t.Errorf("route %s authentication = %q found=%t, want bearer", path, route.Authentication, found)
		}
	}
	handler := &Handler{config: Config{APIKey: testAPIKey}}
	request := func(path, token string) *httptest.ResponseRecorder {
		recorder := httptest.NewRecorder()
		input := httptest.NewRequest(http.MethodGet, path, nil)
		if token != "" {
			input.Header.Set("Authorization", token)
		}
		handler.ServeHTTP(recorder, input)
		return recorder
	}
	if response := request("/agent/tools", ""); response.Code != http.StatusUnauthorized {
		t.Errorf("unauthenticated agent route status = %d, want %d", response.Code, http.StatusUnauthorized)
	}
	if response := request("/agent/tools", testBearerToken); response.Code == http.StatusUnauthorized {
		t.Errorf("authenticated agent route refused: %s", response.Body.String())
	}
	for _, accepted := range []struct{ address, credential string }{
		{"127.0.0.1:8080", ""},
		{"localhost:8080", ""},
		{"[::1]:8080", ""},
		{"0.0.0.0:8080", "credential"},
	} {
		if err := clioptions.RequireLoopbackWithoutCredential(accepted.address, accepted.credential); err != nil {
			t.Errorf("accepted listen %+v refused: %v", accepted, err)
		}
	}
	for _, refused := range []string{"0.0.0.0:8080", ":8080", "192.168.1.5:8080", "example.internal:8080"} {
		err := clioptions.RequireLoopbackWithoutCredential(refused, "")
		if err == nil || !strings.Contains(err.Error(), refused) {
			t.Errorf("credential-less listen %q error = %v, want refusal naming the address", refused, err)
		}
	}
}
