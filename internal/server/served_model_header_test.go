package server

import (
	"net/http"
	"strings"
	"testing"
)

// TestServedModelHeader pins the always-visible model identity: the
// health endpoint names the served model for the banner pill on every
// page, the shell wires the pill into the servable-model picker, and
// the catalog endpoint the picker reads answers with the store's
// models. One process serves one model, so the picker hands out the
// relaunch command instead of pretending to hot-swap.
func TestServedModelHeader(t *testing.T) {
	handler := newTestHandlerWithRepository(t, &fakeGenerator{})
	health := serveTestRequest(handler, http.MethodGet, "/health", "")
	if health.Code != http.StatusOK || !strings.Contains(health.Body.String(), testModelID) {
		t.Fatalf("health status=%d body=%s", health.Code, health.Body.String())
	}
	boot := serveTestRequest(handler, http.MethodGet, "/boot.js", "")
	if boot.Code != http.StatusOK ||
		!strings.Contains(boot.Body.String(), "wireModelPicker") ||
		!strings.Contains(boot.Body.String(), "overgo_gui.bat") {
		t.Fatalf("boot.js lacks the model picker wiring")
	}
	shell := serveTestRequest(handler, http.MethodGet, "/app.html", "")
	if shell.Code != http.StatusOK || !strings.Contains(shell.Body.String(), `id="model-pill"`) {
		t.Fatalf("shell lacks the model pill")
	}
	catalog := serveTestRequest(handler, http.MethodGet, "/catalog/models", "")
	if catalog.Code != http.StatusOK || !strings.Contains(catalog.Body.String(), `"models"`) {
		t.Fatalf("catalog status=%d body=%s", catalog.Code, catalog.Body.String())
	}
}
