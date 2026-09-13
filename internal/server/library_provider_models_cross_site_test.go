package server

import (
	"context"
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"testing"
)

// TestProviderListingRefusesCrossSiteGET: the provider listing sends the
// named environment variable as a bearer token to the endpoint the query
// names, so a foreign page must never reach it; the page itself and a
// loopback client still list.
func TestProviderListingRefusesCrossSiteGET(t *testing.T) {
	var calls atomic.Int32
	list := func(context.Context, string, string) ([]ProviderModel, error) {
		calls.Add(1)
		return []ProviderModel{{ID: "listed"}}, nil
	}
	handler := newTestHandler(t, &fakeGenerator{})
	handler.config.LibraryIntake = LibraryIntake{ListProviderModels: list}
	shell := &IdleShell{Intake: LibraryIntake{ListProviderModels: list}}
	for name, target := range map[string]http.Handler{"server": handler, "idle shell": shell} {
		t.Run(name, func(t *testing.T) {
			before := calls.Load()
			request := httptest.NewRequest(http.MethodGet, "/library/providers/models?endpoint=https://provider.invalid/v1&key_environment=PROVIDER_KEY", nil)
			request.Header.Set("Sec-Fetch-Site", "cross-site")
			response := httptest.NewRecorder()
			target.ServeHTTP(response, request)
			if response.Code != http.StatusForbidden || calls.Load() != before {
				t.Fatalf("cross-site listing = %d %s after %d call(s)", response.Code, response.Body, calls.Load()-before)
			}
			request = httptest.NewRequest(http.MethodGet, "/library/providers/models?endpoint=https://provider.invalid/v1&key_environment=PROVIDER_KEY", nil)
			request.Header.Set("Sec-Fetch-Site", "same-origin")
			response = httptest.NewRecorder()
			target.ServeHTTP(response, request)
			if response.Code != http.StatusOK || calls.Load() != before+1 {
				t.Fatalf("same-origin listing = %d %s after %d call(s)", response.Code, response.Body, calls.Load()-before)
			}
		})
	}
}
