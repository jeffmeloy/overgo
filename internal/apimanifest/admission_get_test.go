package apimanifest

import (
	"context"
	"net"
	"net/http"
	"net/http/httptest"
	"testing"
)

// TestCredentiallessGETSideEffects: a GET whose handling has a side effect
// is admitted as a mutation, so cross-site fetch metadata or a foreign
// Origin refuses it while same-origin pages and loopback clients pass; the
// plain admission keeps letting the same GET through, which is why the
// side-effecting routes must ask for the stricter one.
func TestCredentiallessGETSideEffects(t *testing.T) {
	for _, test := range []struct {
		name, origin, fetchSite string
		want                    string
	}{
		{name: "cross-site fetch", fetchSite: "cross-site", want: RefusalOrigin},
		{name: "same-site fetch", fetchSite: "same-site", want: RefusalOrigin},
		{name: "foreign origin", origin: "https://attacker.invalid", want: RefusalOrigin},
		{name: "opaque origin", origin: "null", want: RefusalOrigin},
		{name: "different local port", origin: "http://localhost:8081", want: RefusalOrigin},
		{name: "same-origin fetch", fetchSite: "same-origin", want: ""},
		{name: "matching origin", origin: "http://localhost:8080", want: ""},
		{name: "loopback client without metadata", want: ""},
	} {
		t.Run(test.name, func(t *testing.T) {
			request := httptest.NewRequest(http.MethodGet, "/health?swap=alpha", nil)
			request.Host = "localhost:8080"
			request = request.WithContext(context.WithValue(request.Context(), http.LocalAddrContextKey, &net.TCPAddr{IP: net.IPv4(127, 0, 0, 1), Port: 8080}))
			if test.origin != "" {
				request.Header.Set("Origin", test.origin)
			}
			if test.fetchSite != "" {
				request.Header.Set("Sec-Fetch-Site", test.fetchSite)
			}
			if refusal := AdmitCredentialless(request); refusal != nil {
				t.Fatalf("plain admission refused a GET: %v", refusal)
			}
			got := ""
			if refusal := AdmitCredentiallessEffect(request); refusal != nil {
				got = refusal.Type
				if refusal.Error() == "" {
					t.Fatal("a refusal without a message")
				}
			}
			if got != test.want {
				t.Fatalf("effect refusal = %q, want %q", got, test.want)
			}
			if request.Method != http.MethodGet {
				t.Fatal("the effect admission changed the request's method")
			}
		})
	}
	post := httptest.NewRequest(http.MethodPost, "/providers/key", nil)
	post.Header.Set("Sec-Fetch-Site", "cross-site")
	if refusal := AdmitCredentiallessEffect(post); refusal == nil || refusal.Type != RefusalOrigin {
		t.Fatalf("effect admission on a cross-site POST = %v", refusal)
	}
}
