package apimanifest

import (
	"context"
	"net"
	"net/http"
	"net/http/httptest"
	"testing"
)

// TestAdmitCredentialless pins the shared admission: a foreign Host on a
// real listener and a foreign origin are refused with their types, an
// embedded handler without listener authority skips the Host rule, and a
// same-origin or header-less request is admitted.
func TestAdmitCredentialless(t *testing.T) {
	for _, test := range []struct {
		name, method, host, origin, fetchSite string
		listener                              bool
		want                                  string
	}{
		{name: "cross-site post", method: http.MethodPost, origin: "https://attacker.invalid", fetchSite: "cross-site", listener: true, want: RefusalOrigin},
		{name: "opaque origin", method: http.MethodPost, origin: "null", listener: true, want: RefusalOrigin},
		{name: "different local port", method: http.MethodPost, origin: "http://localhost:8081", listener: true, want: RefusalOrigin},
		{name: "rebinding host", method: http.MethodGet, host: "attacker.invalid:8080", listener: true, want: RefusalHost},
		{name: "loopback suffix", method: http.MethodGet, host: "localhost.attacker.invalid:8080", listener: true, want: RefusalHost},
		{name: "remote ip", method: http.MethodGet, host: "192.0.2.1:8080", listener: true, want: RefusalHost},
		{name: "embedded handler skips the host rule", method: http.MethodGet, host: "attacker.invalid:8080", want: ""},
		{name: "same-origin post", method: http.MethodPost, origin: "http://localhost:8080", listener: true, want: ""},
		{name: "fetch metadata same-origin", method: http.MethodPost, fetchSite: "same-origin", listener: true, want: ""},
		{name: "cli post", method: http.MethodPost, listener: true, want: ""},
		{name: "ipv6 loopback", method: http.MethodGet, host: "[::1]:8080", listener: true, want: ""},
	} {
		t.Run(test.name, func(t *testing.T) {
			request := httptest.NewRequest(test.method, "/providers/key", nil)
			request.Host = test.host
			if request.Host == "" {
				request.Host = "localhost:8080"
			}
			if test.listener {
				request = request.WithContext(context.WithValue(request.Context(), http.LocalAddrContextKey, &net.TCPAddr{IP: net.IPv4(127, 0, 0, 1), Port: 8080}))
			}
			if test.origin != "" {
				request.Header.Set("Origin", test.origin)
			}
			if test.fetchSite != "" {
				request.Header.Set("Sec-Fetch-Site", test.fetchSite)
			}
			refusal := AdmitCredentialless(request)
			got := ""
			if refusal != nil {
				got = refusal.Type
				if refusal.Error() == "" {
					t.Fatal("a refusal without a message")
				}
			}
			if got != test.want {
				t.Fatalf("refusal = %q, want %q", got, test.want)
			}
		})
	}
}
