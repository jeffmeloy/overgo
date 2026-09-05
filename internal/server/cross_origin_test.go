package server

import (
	"context"
	"net"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"overgo/internal/inference"
)

func TestCrossOriginMutationProtection(t *testing.T) {
	for _, test := range []struct {
		name, host, origin, fetchSite, contentType, key, bearer string
		want                                                    int
	}{
		{name: "cross-site form", origin: "https://attacker.invalid", fetchSite: "cross-site", contentType: "text/plain", want: http.StatusForbidden},
		{name: "old browser origin", origin: "https://attacker.invalid", contentType: "application/json", want: http.StatusForbidden},
		{name: "opaque origin", origin: "null", contentType: "application/json", want: http.StatusForbidden},
		{name: "same-site is not same-origin", fetchSite: "same-site", contentType: "application/json", want: http.StatusForbidden},
		{name: "different local port", origin: "http://localhost:8081", contentType: "application/json", want: http.StatusForbidden},
		{name: "fake bearer in credential-less mode", origin: "https://attacker.invalid", bearer: "invented", contentType: "application/json", want: http.StatusForbidden},
		{name: "rebinding matching origin", host: "attacker.invalid:8080", origin: "http://attacker.invalid:8080", contentType: "application/json", want: http.StatusForbidden},
		{name: "rebinding matching fetch metadata", host: "attacker.invalid:8080", fetchSite: "same-origin", contentType: "application/json", want: http.StatusForbidden},
		{name: "loopback suffix", host: "localhost.attacker.invalid:8080", contentType: "application/json", want: http.StatusForbidden},
		{name: "remote IP host", host: "192.0.2.1:8080", contentType: "application/json", want: http.StatusForbidden},
		{name: "same-origin text", origin: "http://localhost:8080", contentType: "text/plain", want: http.StatusUnsupportedMediaType},
		{name: "same-origin form", fetchSite: "same-origin", contentType: "application/x-www-form-urlencoded", want: http.StatusUnsupportedMediaType},
		{name: "browser missing media type", fetchSite: "same-origin", want: http.StatusUnsupportedMediaType},
		{name: "malformed media type", contentType: "application/json; charset", want: http.StatusUnsupportedMediaType},
		{name: "CLI explicit text", contentType: "text/plain", want: http.StatusUnsupportedMediaType},
		{name: "same-origin JSON", origin: "http://localhost:8080", contentType: "application/json", want: http.StatusOK},
		{name: "fetch metadata JSON", fetchSite: "same-origin", contentType: "application/json; charset=utf-8", want: http.StatusOK},
		{name: "CLI JSON", contentType: "application/json", want: http.StatusOK},
		{name: "CLI legacy implicit JSON", want: http.StatusOK},
		{name: "IPv4 loopback", host: "127.0.0.1:8080", contentType: "application/json", want: http.StatusOK},
		{name: "IPv6 loopback", host: "[::1]:8080", contentType: "application/json", want: http.StatusOK},
		{name: "case insensitive localhost", host: "LOCALHOST:8080", contentType: "application/json", want: http.StatusOK},
		{name: "authenticated API", host: "api.example:8080", origin: "https://client.example", fetchSite: "cross-site", contentType: "application/json", key: "secret", bearer: "secret", want: http.StatusOK},
		{name: "missing credential", contentType: "application/json", key: "secret", want: http.StatusUnauthorized},
		{name: "invalid credential", origin: "http://localhost:8080", contentType: "application/json", key: "secret", bearer: "wrong", want: http.StatusUnauthorized},
	} {
		t.Run(test.name, func(t *testing.T) {
			generator := &fakeLoRAGenerator{fakeGenerator: &fakeGenerator{}, adapters: []inference.LoRAAdapterInfo{{ID: 0, Path: "adapter.gguf", Scale: 1}}}
			handler := newTestHandler(t, generator)
			handler.config.APIKey = test.key
			request := httptest.NewRequest(http.MethodPost, "/lora-adapters", strings.NewReader(`[{"id":0,"scale":0.25}]`))
			request.Host = test.host
			if request.Host == "" {
				request.Host = "localhost:8080"
			}
			request = request.WithContext(context.WithValue(request.Context(), http.LocalAddrContextKey, &net.TCPAddr{IP: net.IPv4(127, 0, 0, 1), Port: 8080}))
			if test.origin != "" {
				request.Header.Set("Origin", test.origin)
			}
			if test.fetchSite != "" {
				request.Header.Set("Sec-Fetch-Site", test.fetchSite)
			}
			if test.contentType != "" {
				request.Header.Set("Content-Type", test.contentType)
			}
			if test.bearer != "" {
				request.Header.Set("Authorization", "Bearer "+test.bearer)
			}
			// Forwarded values cannot override the actual request authority.
			request.Header.Set("X-Forwarded-Host", "localhost:8080")
			request.Header.Set("X-Forwarded-For", "127.0.0.1")
			response := httptest.NewRecorder()
			handler.ServeHTTP(response, request)
			if response.Code != test.want {
				t.Fatalf("status = %d, want %d; body = %s", response.Code, test.want, response.Body.String())
			}
			if test.want == http.StatusOK {
				if len(generator.requested) != 1 || generator.requested[0].Scale != 0.25 {
					t.Fatalf("allowed mutation = %v", generator.requested)
				}
			} else if len(generator.requested) != 0 {
				t.Fatalf("rejected request mutated adapters: %v", generator.requested)
			}
		})
	}
	// Exercise the actual HTTP server transport, including its LocalAddr context.
	t.Run("transport rejects rebinding reads", func(t *testing.T) {
		server := httptest.NewServer(newTestHandler(t, &fakeGenerator{}))
		defer server.Close()
		for _, host := range []string{"attacker.invalid", "localhost"} {
			request, err := http.NewRequest(http.MethodGet, server.URL+"/v1/models", nil)
			if err != nil {
				t.Fatal(err)
			}
			request.Host = host
			response, err := server.Client().Do(request)
			if err != nil {
				t.Fatal(err)
			}
			response.Body.Close()
			want := http.StatusForbidden
			if host == "localhost" {
				want = http.StatusOK
			}
			if response.StatusCode != want {
				t.Fatalf("host %q status = %d, want %d", host, response.StatusCode, want)
			}
		}
	})
}
