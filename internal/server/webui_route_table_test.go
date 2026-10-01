package server

import (
	"encoding/json"
	"net/http"
	"regexp"
	"strings"
	"testing"
)

// TestWebUIRouteTable: every route the client calls is declared in the
// route catalog, and the API explorer answers that table as the API
// manifest projects it, with its bound schemas.
func TestWebUIRouteTable(t *testing.T) {
	t.Parallel()
	// Every path a module or library calls through the API client resolves
	// in the table (query strings stripped); the shell's own assets and the
	// versioned fallback are the only paths outside it.
	called := regexp.MustCompile(`api\.(?:get|post|stream|events)\(\s*"(/[A-Za-z0-9._-][A-Za-z0-9/._-]*)`)
	seen := map[string]bool{}
	for name, source := range webuiJavaScript(t) {
		for _, match := range called.FindAllStringSubmatch(source, -1) {
			path := match[1]
			if seen[path] {
				continue
			}
			seen[path] = true
			if _, routed := routesByPath[path]; !routed && !strings.HasPrefix(path, versionedRouteFallback.Path) {
				t.Errorf("%s calls %s, which the route table does not declare", name, path)
			}
		}
	}
	if len(seen) < 40 {
		t.Errorf("only %d client routes found; the scan did not cover the modules", len(seen))
	}

	handler := newTestHandler(t, &fakeGenerator{})
	response := serveTestRequest(handler, http.MethodGet, "/workspace/routes", "")
	if response.Code != http.StatusOK {
		t.Fatalf("GET /workspace/routes status = %d", response.Code)
	}
	var table workspaceRoutesResponse
	if err := json.Unmarshal(response.Body.Bytes(), &table); err != nil {
		t.Fatal(err)
	}
	if len(table.Routes) != len(APIManifestRoutes()) {
		t.Errorf("explorer table carries %d routes, manifest projection %d", len(table.Routes), len(APIManifestRoutes()))
	}
	if len(table.Schemas) != len(workspaceRouteSchemas) {
		t.Errorf("explorer binds %d schemas, declared %d", len(table.Schemas), len(workspaceRouteSchemas))
	}
}
