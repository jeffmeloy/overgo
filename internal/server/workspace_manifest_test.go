package server

import (
	"encoding/json"
	"net/http"
	"strings"
	"testing"
)

func TestWorkspaceManifest(t *testing.T) {
	handler := newTestHandlerWithRepository(t, &fakeGenerator{})
	response := serveTestRequest(handler, http.MethodGet, "/workspace/manifest", "")
	if response.Code != http.StatusOK {
		t.Fatalf("workspace manifest status=%d body=%s", response.Code, response.Body.String())
	}
	var manifest workspaceManifestResponse
	if err := json.Unmarshal(response.Body.Bytes(), &manifest); err != nil {
		t.Fatal(err)
	}
	if manifest.Version != 1 || len(manifest.Sections) == 0 || len(manifest.Tabs) == 0 {
		t.Fatalf("workspace manifest = %+v", manifest)
	}
	sections := make(map[string]bool, len(manifest.Sections))
	for _, section := range manifest.Sections {
		sections[section.ID] = true
	}
	seen := make(map[string]bool, len(manifest.Tabs))
	for _, tab := range manifest.Tabs {
		if seen[tab.ID] || !sections[tab.Section] || tab.Label == "" {
			t.Fatalf("invalid workspace tab = %+v", tab)
		}
		seen[tab.ID] = true
	}
	for _, required := range []string{"chat", "datasets", "training-jobs", "compositions", "attention"} {
		if !seen[required] {
			t.Fatalf("workspace manifest lacks %q", required)
		}
	}
}

func TestWorkspaceManifestCapabilityRefusal(t *testing.T) {
	response := serveTestRequest(newTestHandler(t, &fakeGenerator{}), http.MethodGet, "/workspace/manifest", "")
	var manifest workspaceManifestResponse
	if response.Code != http.StatusOK || json.Unmarshal(response.Body.Bytes(), &manifest) != nil {
		t.Fatalf("workspace manifest status=%d body=%s", response.Code, response.Body.String())
	}
	byID := make(map[string]workspaceTab, len(manifest.Tabs))
	for _, tab := range manifest.Tabs {
		byID[tab.ID] = tab
	}
	if !byID["chat"].Enabled || byID["attention"].Enabled || byID["attention"].Refusal == "" ||
		byID["datasets"].Enabled || byID["datasets"].Refusal == "" {
		t.Fatalf("workspace capability projection = %+v", byID)
	}
}

func TestWebUIWorkspaceRoutes(t *testing.T) {
	handler := newTestHandler(t, &fakeGenerator{})
	boot := serveTestRequest(handler, http.MethodGet, "/boot.js", "").Body.String()
	for _, expected := range []string{"/workspace/manifest", "bindWorkspaceManifest", "tab.enabled", "tab.refusal"} {
		if !strings.Contains(boot, expected) {
			t.Errorf("workspace boot lacks %q", expected)
		}
	}
	for _, duplicate := range []string{"SECTION_ORDER", "tab.requires", "refreshCapabilities"} {
		if strings.Contains(boot, duplicate) {
			t.Errorf("workspace boot retains client authority %q", duplicate)
		}
	}
	for _, path := range []string{"/mod/chat.js", "/mod/analyze_attention.js", "/mod/datasets.js"} {
		module := serveTestRequest(handler, http.MethodGet, path, "").Body.String()
		if strings.Contains(module, "section:") || strings.Contains(module, "requires:") || strings.Contains(module, "label:") {
			t.Errorf("%s retains navigation or capability authority", path)
		}
	}
}
