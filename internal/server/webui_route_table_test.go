package server

import (
	"encoding/json"
	"net/http"
	"os"
	"regexp"
	"strings"
	"testing"
)

// TestWebUIRouteTable pins the one-route-table ratchet (professional GUI
// campaign, gui-simplify/one-route-table): every route the client calls is
// declared in the route catalog with its method and authentication, the
// static server carries no hand-written authorization block or prefix
// chain, the explorer renders from the same table, and the API manifest
// projects that table (its docs copy is held current by api-manifest -check).
func TestWebUIRouteTable(t *testing.T) {
	static, err := os.ReadFile("webui_static.go")
	if err != nil {
		t.Fatal(err)
	}
	for _, forbidden := range []string{"h.authorized(", "strings.HasPrefix(request.URL.Path", "WWW-Authenticate"} {
		if strings.Contains(string(static), forbidden) {
			t.Errorf("webui_static.go still routes or authorizes by hand (%s); the route table owns both", forbidden)
		}
	}

	// Every path a module or library calls through the API client resolves
	// in the table (query strings stripped); the shell's own assets and the
	// versioned fallback are the only paths outside it.
	called := regexp.MustCompile(`api\.(?:get|post|stream|events)\(\s*"(/[A-Za-z0-9._-][A-Za-z0-9/._-]*)`)
	sources := webuiJavaScript(t)
	seen := map[string]bool{}
	for name, source := range sources {
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
	for _, path := range []string{"/workspace/manifest", "/workspace/schema", "/workspace/routes", "/operations/evidence", "/agents", "/agents/chat", "/automations", "/peers"} {
		if _, routed := routesByPath[path]; !routed {
			t.Errorf("GUI route %s is absent from the route table", path)
		}
	}
	manifest := serveTestRequest(handler, http.MethodGet, "/workspace/manifest", "").Body.String()
	if !strings.Contains(manifest, `"id":"api"`) {
		t.Error("the workspace manifest does not declare the API explorer tab")
	}
	if !strings.Contains(sources["mod/api.js"], `"/workspace/routes"`) || !strings.Contains(sources["mod/api.js"], "overgo.schemaForm(") {
		t.Error("the explorer does not render from the route table with schema forms")
	}
}
