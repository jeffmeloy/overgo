package server

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/discovery"
	"overgo/internal/recipe"
)

// TestIdleShellAnswersWhileNothingServes pins the idle shell: the client and
// its boot routes answer, the catalog lists the store's rows, and an API
// route refuses with the reason.
func TestIdleShellAnswersWhileNothingServes(t *testing.T) {
	model, err := artifact.IdentifyBytes(artifact.KindModel, []byte("idle-model"))
	if err != nil {
		t.Fatal(err)
	}
	served, err := artifact.IdentifyBytes(artifact.KindRecipe, []byte("idle-recipe"))
	if err != nil {
		t.Fatal(err)
	}
	shell := &IdleShell{Catalog: func(context.Context) ([]discovery.CatalogEntry, bool, error) {
		return []discovery.CatalogEntry{{Model: model, Location: "C:/models/idle.gguf", Present: true,
			Capabilities: []discovery.Capability{{Task: recipe.TaskInference, Recipe: served, Tier: "verified"}}}}, true, nil
	}}
	front := httptest.NewServer(shell)
	defer front.Close()

	page := get(t, front.URL+"/app.html")
	if page.StatusCode != http.StatusOK || !strings.Contains(page.Header.Get("Content-Type"), "text/html") || page.Header.Get("Content-Security-Policy") == "" {
		t.Fatalf("shell = %d %q", page.StatusCode, page.Header)
	}
	var health struct {
		Status string `json:"status"`
		Model  string `json:"model"`
	}
	decode(t, front.URL+"/health", &health)
	if health.Status != "ok" || health.Model != "" {
		t.Fatalf("health = %+v", health)
	}
	var manifest workspaceManifestResponse
	decode(t, front.URL+"/workspace/manifest", &manifest)
	if len(manifest.Tabs) == 0 || manifest.Model != nil {
		t.Fatalf("manifest = %+v", manifest)
	}
	for _, tab := range manifest.Tabs {
		if tab.Enabled || tab.Refusal != idleRefusal {
			t.Fatalf("tab %s = %+v", tab.ID, tab)
		}
	}
	var catalog struct {
		Models    []catalogModel `json:"models"`
		Truncated bool           `json:"truncated"`
	}
	decode(t, front.URL+"/catalog/models", &catalog)
	if len(catalog.Models) != 1 || !catalog.Truncated || catalog.Models[0].Recipe != served.String() || !catalog.Models[0].Present {
		t.Fatalf("catalog = %+v", catalog)
	}
	refused := get(t, front.URL+"/v1/chat/completions")
	if refused.StatusCode != http.StatusServiceUnavailable {
		t.Fatalf("api route while nothing serves = %d", refused.StatusCode)
	}
	var envelope struct {
		Error struct {
			Code    string `json:"type"`
			Message string `json:"message"`
		} `json:"error"`
	}
	if err := json.NewDecoder(refused.Body).Decode(&envelope); err != nil {
		t.Fatal(err)
	}
	if envelope.Error.Code != "no_model_serves" || envelope.Error.Message != idleRefusal {
		t.Fatalf("refusal = %+v", envelope)
	}
	missing := get(t, front.URL+"/absent.js")
	if missing.StatusCode != http.StatusNotFound {
		t.Fatalf("absent asset = %d", missing.StatusCode)
	}

	broken := &IdleShell{Catalog: func(context.Context) ([]discovery.CatalogEntry, bool, error) {
		return nil, false, errors.New("store gone")
	}}
	brokenFront := httptest.NewServer(broken)
	defer brokenFront.Close()
	if failed := get(t, brokenFront.URL+"/catalog/models"); failed.StatusCode != http.StatusInternalServerError {
		t.Fatalf("catalog failure = %d", failed.StatusCode)
	}
}

func get(t *testing.T, url string) *http.Response {
	t.Helper()
	response, err := http.Get(url)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { response.Body.Close() })
	return response
}

func decode(t *testing.T, url string, into any) {
	t.Helper()
	response := get(t, url)
	if response.StatusCode != http.StatusOK {
		t.Fatalf("%s = %d", url, response.StatusCode)
	}
	if err := json.NewDecoder(response.Body).Decode(into); err != nil {
		t.Fatal(err)
	}
}
