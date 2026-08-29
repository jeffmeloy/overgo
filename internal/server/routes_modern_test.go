package server

import (
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func TestRouteManifestMatchesRuntimeMux(t *testing.T) {
	manifest := APIManifestRoutes()
	expected := 0
	for _, route := range routeCatalog {
		expected += len(route.Methods)
	}
	if len(manifest) != expected {
		t.Fatalf("manifest routes = %d, runtime method routes = %d", len(manifest), expected)
	}
	remaining := make(map[string]routeAuthentication, expected)
	for _, route := range routeCatalog {
		for _, method := range route.Methods {
			remaining[method+" "+route.Path] = route.Authentication
		}
	}
	for _, route := range manifest {
		key := route.Method + " " + route.Path
		authentication, ok := remaining[key]
		if !ok || route.Authentication != string(authentication) {
			t.Fatalf("manifest route %q authentication = %q", key, route.Authentication)
		}
		delete(remaining, key)
	}
	if len(remaining) != 0 {
		t.Fatalf("runtime routes absent from manifest: %v", remaining)
	}
}

func TestRouteAuthenticationAndMethodSemantics(t *testing.T) {
	handler := &Handler{config: Config{APIKey: testAPIKey}}
	request := func(method, path string, authorized bool) *httptest.ResponseRecorder {
		recorder := httptest.NewRecorder()
		input := httptest.NewRequest(method, path, nil)
		if authorized {
			input.Header.Set("Authorization", testBearerToken)
		}
		handler.ServeHTTP(recorder, input)
		return recorder
	}
	unauthorized := request(http.MethodPost, "/props", false)
	if unauthorized.Code != http.StatusUnauthorized || unauthorized.Header().Get("WWW-Authenticate") != "Bearer" {
		t.Fatalf("authentication precedence = %d %s", unauthorized.Code, unauthorized.Body.String())
	}
	for _, method := range []string{http.MethodPost, http.MethodHead} {
		response := request(method, "/props", true)
		if response.Code != http.StatusMethodNotAllowed || response.Header().Get("Allow") != http.MethodGet ||
			!strings.Contains(response.Body.String(), `"method_not_allowed"`) {
			t.Fatalf("%s method response = %d allow=%q body=%s", method, response.Code, response.Header().Get("Allow"), response.Body.String())
		}
	}
	public := request(http.MethodPost, "/health", false)
	if public.Code != http.StatusMethodNotAllowed || public.Header().Get("Allow") != http.MethodGet {
		t.Fatalf("public method response = %d allow=%q", public.Code, public.Header().Get("Allow"))
	}
}

func TestPathValueCompatibility(t *testing.T) {
	var repository string
	mux := http.NewServeMux()
	mux.HandleFunc("GET /models/{repository}", func(_ http.ResponseWriter, request *http.Request) {
		repository = request.PathValue("repository")
	})
	response := httptest.NewRecorder()
	mux.ServeHTTP(response, httptest.NewRequest(http.MethodGet, "/models/acme%2Ftiny", nil))
	if response.Code != http.StatusOK || repository != "acme/tiny" {
		t.Fatalf("escaped path value = %q status=%d", repository, response.Code)
	}
	if route, found := resolveRoute("/v1/models/acme/tiny"); !found || route.Path != versionedRouteFallback.Path {
		t.Fatalf("versioned fallback = (%+v, %t)", route, found)
	}
}
