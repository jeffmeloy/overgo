package server

import (
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"overgo/internal/overgodb"
)

// TestIdleShellDelegatesWorkbenchRoutes: with a workbench behind it the
// idle shell answers downloads, validation and operations over the store
// while no model serves; without one those routes refuse as before.
func TestIdleShellDelegatesWorkbenchRoutes(t *testing.T) {
	store, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	workbench := newTestHandlerForRepository(t, store, &fakeGenerator{})
	defer workbench.Close()
	shell := &IdleShell{Repository: store, Workbench: workbench}
	front := httptest.NewServer(shell)
	defer front.Close()
	serve := func(method, path, body string) *httptest.ResponseRecorder {
		t.Helper()
		request := httptest.NewRequest(method, path, strings.NewReader(body))
		request.Header.Set("Content-Type", "application/json")
		response := httptest.NewRecorder()
		shell.ServeHTTP(response, request)
		return response
	}
	if response := serve(http.MethodGet, "/hub/downloads", ""); response.Code != http.StatusOK || !strings.Contains(response.Body.String(), `"downloads"`) {
		t.Fatalf("downloads through the idle shell = %d %s", response.Code, response.Body)
	}
	if response := serve(http.MethodGet, "/operations/decisions", ""); response.Code != http.StatusOK {
		t.Fatalf("decisions through the idle shell = %d %s", response.Code, response.Body)
	}
	// Validation reaches the workbench's own admission (here: no intake assembled), never the idle refusal.
	if response := serve(http.MethodPost, "/library/validate", `{"path":"model.gguf"}`); response.Code == http.StatusServiceUnavailable || strings.Contains(response.Body.String(), idleRefusal) {
		t.Fatalf("validation through the idle shell = %d %s", response.Code, response.Body)
	}
	var health struct {
		Model string `json:"model"`
	}
	decode(t, front.URL+"/health", &health)
	if health.Model != "" {
		t.Fatalf("the shell's own health names a model: %+v", health)
	}
	bare := &IdleShell{Repository: store}
	request := httptest.NewRequest(http.MethodGet, "/hub/downloads", nil)
	response := httptest.NewRecorder()
	bare.ServeHTTP(response, request)
	if response.Code != http.StatusServiceUnavailable || !strings.Contains(response.Body.String(), idleRefusal) {
		t.Fatalf("downloads without a workbench = %d %s", response.Code, response.Body)
	}
}
