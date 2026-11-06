package main

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"overgo/internal/overgodbtest"
	"overgo/internal/server"
	"overgo/internal/testcheck"
	"testing"
)

func TestIdleAudioWorkspace(t *testing.T) {
	store := overgodbtest.Open(t, t.TempDir())
	// Exercise the actual launcher composition across the server package boundary.
	// One entry is enough to prove an empty catalog does not invent a runnable task.
	handler, err := idleWorkbench(store, server.LibraryIntake{}, 1)
	testcheck.NoError(t, err)
	defer handler.Close()
	response := httptest.NewRecorder()
	handler.ServeHTTP(response, httptest.NewRequest(http.MethodGet, "/workspace/manifest", nil))
	var manifest struct {
		Tabs []struct {
			ID      string `json:"id"`
			Enabled bool   `json:"enabled"`
		}
	}
	if response.Code != http.StatusOK || json.Unmarshal(response.Body.Bytes(), &manifest) != nil {
		t.Fatalf("manifest: %d %s", response.Code, response.Body)
	}
	expected := map[string]bool{"chat": false, "generation": false, "model": false, "lens": false}
	for _, tab := range manifest.Tabs {
		if _, wanted := expected[tab.ID]; wanted {
			if tab.Enabled {
				t.Fatalf("empty launcher invents capability: %+v", tab)
			}
			expected[tab.ID] = true
		}
	}
	for id, found := range expected {
		if !found {
			t.Fatalf("manifest omitted expected tab %s", id)
		}
	}
	response = httptest.NewRecorder()
	handler.ServeHTTP(response, httptest.NewRequest(http.MethodGet, "/generation/capabilities", nil))
	var refusal struct {
		Error struct {
			Type string `json:"type"`
		} `json:"error"`
	}
	if response.Code != http.StatusNotImplemented || json.Unmarshal(response.Body.Bytes(), &refusal) != nil || refusal.Error.Type != "unsupported_operation" {
		t.Fatalf("empty native audio catalog must retain its explicit refusal: %d %s", response.Code, response.Body)
	}
}
