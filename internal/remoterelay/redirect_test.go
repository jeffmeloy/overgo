package remoterelay

import (
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"

	"overgo/internal/inference"
	"overgo/internal/testcheck"
)

// TestAProvidersKeyFollowsNoRedirectOffItsEndpoint holds a provider's key to the endpoint it was admitted to (SQA-363):
// an https provider that answers the model listing and the generation with a redirect to plain http on its own host, at
// another port, has both refused, naming the redirect, and its key never reaches the plain endpoint. Go's client had
// followed it with the Authorization header, since the host name is the same.
// TestAProvidersKeyedClientStopsAfterTenRedirects holds a provider's keyed client, given no redirect policy, to Go's
// own bound (relay-redirect/bound, SQA-363): a models route that redirects to itself is refused at its tenth redirect,
// its key sent ten times as Go's default policy sends it, where a policy that only refused redirects off the endpoint
// followed it without end. The
// server ends the loop itself past a hundred requests, so the test ends either way.
func TestAProvidersKeyedClientStopsAfterTenRedirects(t *testing.T) {
	var requests atomic.Int32
	looping := httptest.NewServer(http.HandlerFunc(func(response http.ResponseWriter, request *http.Request) {
		if requests.Add(1) > 100 {
			http.Error(response, "the loop ran past a hundred requests", http.StatusInternalServerError)
			return
		}
		http.Redirect(response, request, request.URL.Path, http.StatusTemporaryRedirect)
	}))
	defer looping.Close()
	if _, err := listModels(t.Context(), looping.URL, "test-key", nil); err == nil || requests.Load() != 10 {
		t.Fatalf("a models route redirecting to itself ended %v after %d requests, want it refused after the tenth", err, requests.Load())
	}
}

func TestAProvidersKeyFollowsNoRedirectOffItsEndpoint(t *testing.T) {
	var leaked atomic.Bool
	plain := httptest.NewServer(http.HandlerFunc(func(response http.ResponseWriter, request *http.Request) {
		leaked.Store(leaked.Load() || request.Header.Get("Authorization") != "")
	}))
	defer plain.Close()
	admitted := httptest.NewTLSServer(http.HandlerFunc(func(response http.ResponseWriter, request *http.Request) {
		http.Redirect(response, request, plain.URL+request.URL.Path, http.StatusTemporaryRedirect)
	}))
	defer admitted.Close()
	t.Setenv("OVERGO_REMOTE_RELAY_TEST_KEY", "test-key")
	provider, definition := testProvider(t, admitted.URL)
	if _, err := listModels(t.Context(), provider.Endpoint, "test-key", admitted.Client()); err == nil || !strings.Contains(err.Error(), "redirected") {
		t.Fatalf("a listing redirected to plain http ended %v, want it refused naming the redirect", err)
	}
	generator := testcheck.Must(New(provider, definition, admitted.Client()))(t)
	if _, _, err := generator.Generate(t.Context(), "hello", inference.GenerateOptions{MaxNewTokens: 4}); err == nil || !strings.Contains(err.Error(), "redirected") {
		t.Fatalf("a generation redirected to plain http ended %v, want it refused naming the redirect", err)
	}
	if leaked.Load() {
		t.Fatal("the provider's key reached the plain http endpoint it was redirected to")
	}
}
