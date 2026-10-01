package server

import (
	"encoding/json"
	"net/http"
	"strings"
	"testing"
)

// TestFrontPageOperations pins what the front page's header reads: the
// health probe names the served model for the banner pill on every page and
// reports the device's current and peak memory when the generator runs on
// one, and the catalog endpoint the servable-model picker reads answers with
// the store's models.
func TestFrontPageOperations(t *testing.T) {
	t.Parallel()
	handler := newTestHandlerWithRepository(t, &fakeGenerator{})
	health := serveTestRequest(handler, http.MethodGet, "/health", "")
	var probe struct {
		Status string `json:"status"`
		Model  string `json:"model"`
		Device struct {
			CurrentBytes uint64 `json:"current_bytes"`
			PeakBytes    uint64 `json:"peak_bytes"`
		} `json:"device"`
	}
	if err := json.Unmarshal(health.Body.Bytes(), &probe); err != nil {
		t.Fatal(err)
	}
	if probe.Status != "ok" || probe.Model != testModelID || probe.Device.PeakBytes == 0 || probe.Device.CurrentBytes == 0 {
		t.Fatalf("health = %s", health.Body.String())
	}
	catalog := serveTestRequest(handler, http.MethodGet, "/catalog/models", "")
	if catalog.Code != http.StatusOK || !strings.Contains(catalog.Body.String(), `"models"`) {
		t.Fatalf("catalog status=%d body=%s", catalog.Code, catalog.Body.String())
	}
}
