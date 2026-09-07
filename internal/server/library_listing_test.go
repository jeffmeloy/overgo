package server

import (
	"context"
	"errors"
	"net/http"
	"strings"
	"testing"
)

// TestLibraryProviderModelsListsThroughTheIntake: the listing route needs
// endpoint and key variable, answers that it needs the launcher's intake,
// then lists through it; the intake's refusal is the route's.
func TestLibraryProviderModelsListsThroughTheIntake(t *testing.T) {
	handler := newTestHandler(t, &fakeGenerator{})
	path := "/library/providers/models?endpoint=https://fake.example/api/v1&key_environment=OVERGO_LISTING_ROUTE_KEY"
	if incomplete := serveTestRequest(handler, http.MethodGet, "/library/providers/models?endpoint=https://fake.example", ""); incomplete.Code != http.StatusBadRequest {
		t.Fatalf("listing without a key variable status=%d body=%s", incomplete.Code, incomplete.Body.String())
	}
	if absent := serveTestRequest(handler, http.MethodGet, path, ""); absent.Code != http.StatusNotImplemented {
		t.Fatalf("listing without the intake status=%d body=%s", absent.Code, absent.Body.String())
	}
	handler.config.LibraryIntake.ListProviderModels = func(_ context.Context, endpoint, keyEnvironment string) ([]ProviderModel, error) {
		if endpoint != "https://fake.example/api/v1" || keyEnvironment != "OVERGO_LISTING_ROUTE_KEY" {
			return nil, errors.New("the listing lost a field")
		}
		return []ProviderModel{{ID: "vendor/one", Name: "One", ContextLength: 4096}, {ID: "vendor/two"}}, nil
	}
	listed := serveTestRequest(handler, http.MethodGet, path, "")
	if listed.Code != http.StatusOK || !strings.Contains(listed.Body.String(), `{"id":"vendor/one","name":"One","context_length":4096}`) || !strings.Contains(listed.Body.String(), `{"id":"vendor/two"}`) {
		t.Fatalf("listed status=%d body=%s", listed.Code, listed.Body.String())
	}
	handler.config.LibraryIntake.ListProviderModels = func(context.Context, string, string) ([]ProviderModel, error) {
		return nil, errors.New("OVERGO_LISTING_ROUTE_KEY is not set")
	}
	if refused := serveTestRequest(handler, http.MethodGet, path, ""); refused.Code != http.StatusUnprocessableEntity || !strings.Contains(refused.Body.String(), "is not set") {
		t.Fatalf("refused status=%d body=%s", refused.Code, refused.Body.String())
	}
}
