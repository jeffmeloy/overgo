package server

import (
	"io/fs"
	"net/http"
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"testing"
)

// TestWebUIOneShell pins the one-shell ratchet (professional GUI campaign,
// gui-simplify/one-shell): the embedded client carries exactly one HTML
// document, that document lists no tab module by hand (boot.js loads the
// shared libraries and every manifest-declared module itself), every
// declared tab names a module file that exists, the historical /app.html
// address serves the same shell, and a development directory serves
// assets from disk with caching disabled. The counts only tighten.
func TestWebUIOneShell(t *testing.T) {
	var documents []string
	if err := fs.WalkDir(webuiFS, ".", func(name string, entry fs.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if !entry.IsDir() && strings.HasSuffix(name, ".html") {
			documents = append(documents, name)
		}
		return nil
	}); err != nil {
		t.Fatal(err)
	}
	if len(documents) != 1 || documents[0] != "index.html" {
		t.Fatalf("embedded shells = %v, want exactly index.html", documents)
	}
	shell, err := fs.ReadFile(webuiFS, "index.html")
	if err != nil {
		t.Fatal(err)
	}
	scripts := regexp.MustCompile(`<script[^>]*src="([^"]+)"`).FindAllStringSubmatch(string(shell), -1)
	if len(scripts) != 1 || scripts[0][1] != "/boot.js" {
		t.Fatalf("shell script tags = %v, want only /boot.js", scripts)
	}
	for _, needle := range []string{`id="model-pill"`, `id="global-operation-shell"`, `id="panels"`, `class="shell"`, `nav class="sidebar"`} {
		if !strings.Contains(string(shell), needle) {
			t.Errorf("shell lacks %s", needle)
		}
	}
	if strings.Contains(string(shell), "<script>") || strings.Contains(string(shell), "onclick=") {
		t.Error("shell carries inline script, blocked by script-src 'self'")
	}

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
	boot, err := fs.ReadFile(webuiFS, "boot.js")
	if err != nil {
		t.Fatal(err)
	}
	for _, needle := range []string{"loadWorkspaceModules", `"/md.js"`, `"/operations_shell.js"`, `"/schema_form.js"`, `"/viz.js"`, `"/workflow.js"`, "declaration.module"} {
		if !strings.Contains(string(boot), needle) {
			t.Errorf("boot.js loader lacks %s", needle)
		}
	}
	// md.js must be loaded before any module so overgo.md exists when chat renders.
	if strings.Index(string(boot), `"/md.js"`) > strings.Index(string(boot), "loadWorkspaceModules(") {
		t.Error("boot.js must list /md.js among the libraries loaded before the modules")
	}

	handler := newTestHandler(t, &fakeGenerator{})
	index := serveTestRequest(handler, http.MethodGet, "/index.html", "")
	for _, alias := range []string{"/", "/app.html"} {
		response := serveTestRequest(handler, http.MethodGet, alias, "")
		if response.Code != http.StatusOK || response.Body.String() != index.Body.String() {
			t.Errorf("GET %s does not serve the one shell (status %d)", alias, response.Code)
		}
	}
	if response := serveTestRequest(handler, http.MethodGet, "/probe.js", ""); response.Code != http.StatusNotFound {
		t.Errorf("GET /probe.js status = %d, want 404: the landing probe is gone", response.Code)
	}
	manifest := serveTestRequest(handler, http.MethodGet, "/workspace/manifest", "")
	if manifest.Code != http.StatusOK || !strings.Contains(manifest.Body.String(), `"module":"image"`) {
		t.Fatalf("manifest does not publish tab modules: %d %s", manifest.Code, manifest.Body.String())
	}
}

// TestWebUIDevelopmentDirectory: with Config.WebUIDir set the server serves
// the client from that directory with caching off, so an edit shows on
// reload without a rebuild; paths that escape the directory stay refused.
func TestWebUIDevelopmentDirectory(t *testing.T) {
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
