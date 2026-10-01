package server

import (
	"io/fs"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// TestWebUIOneShell: every tab the workspace manifest declares names a
// module file the client carries, the manifest publishes those modules,
// and the historical /app.html address serves the same shell as /.
func TestWebUIOneShell(t *testing.T) {
	t.Parallel()
	declaration, err := parseWorkspaceManifest()
	if err != nil {
		t.Fatal(err)
	}
	for _, tab := range declaration.Tabs {
		module := "mod/" + tab.module() + ".js"
		if _, err := fs.Stat(webuiFS, module); err != nil {
			t.Errorf("tab %s declares module %s: %v", tab.ID, module, err)
		}
	}

	handler := newTestHandler(t, &fakeGenerator{})
	index := serveTestRequest(handler, http.MethodGet, "/index.html", "")
	for _, alias := range []string{"/", "/app.html"} {
		response := serveTestRequest(handler, http.MethodGet, alias, "")
		if response.Code != http.StatusOK || response.Body.String() != index.Body.String() {
			t.Errorf("GET %s does not serve the one shell (status %d)", alias, response.Code)
		}
	}
	manifest := serveTestRequest(handler, http.MethodGet, "/workspace/manifest", "")
	if manifest.Code != http.StatusOK || !strings.Contains(manifest.Body.String(), `"module":"discovery"`) {
		t.Fatalf("manifest does not publish tab modules: %d %s", manifest.Code, manifest.Body.String())
	}
}

// TestWebUIDevelopmentDirectory: with Config.WebUIDir set the server serves
// the client from that directory with caching off, so an edit shows on
// reload without a rebuild; paths that escape the directory stay refused.
func TestWebUIDevelopmentDirectory(t *testing.T) {
	t.Parallel()
	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, "index.html"), []byte("<!DOCTYPE html><title>dev shell</title>"), 0o644); err != nil {
		t.Fatal(err)
	}
	handler, err := New(Config{WebUIDir: dir}, &fakeGenerator{})
	if err != nil {
		t.Fatal(err)
	}
	defer handler.Close()
	response := serveTestRequest(handler, http.MethodGet, "/", "")
	if response.Code != http.StatusOK || !strings.Contains(response.Body.String(), "dev shell") {
		t.Fatalf("development directory not served: %d %s", response.Code, response.Body.String())
	}
	if got := response.Header().Get("Cache-Control"); got != "no-store" {
		t.Errorf("development Cache-Control = %q, want no-store", got)
	}
	if got := response.Header().Get("Content-Security-Policy"); !strings.Contains(got, "script-src 'self';") {
		t.Errorf("development serving dropped the CSP: %q", got)
	}
	if response := serveTestRequest(handler, http.MethodGet, "/../server.go", ""); response.Code != http.StatusNotFound {
		t.Errorf("traversal status = %d, want 404", response.Code)
	}
	if response := serveTestRequest(handler, http.MethodGet, "/boot.js", ""); response.Code != http.StatusNotFound {
		t.Errorf("development directory falls back to the embed: status %d", response.Code)
	}
}
