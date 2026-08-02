package server

import (
	"bytes"
	"context"
	"image"
	"image/png"
	"net/http"
	"net/http/httptest"
	"net/netip"
	"net/url"
	"path/filepath"
	"strconv"
	"testing"
	"time"
)

func TestRepositoryRemoteMediaPolicy(t *testing.T) {
	policy, err := LoadRemoteMediaPolicy(filepath.Join("..", "..", "media_policy.yaml"))
	if err != nil {
		t.Fatal(err)
	}
	if policy.Enabled || len(policy.AllowedSchemes) != 1 || policy.AllowedSchemes[0] != "https" ||
		policy.AllowPrivateNetworks || policy.MaxRedirects != 2 || policy.MaxConcurrentFetches != 4 {
		t.Fatalf("default policy = %+v", policy)
	}
}

func testRemoteMediaPolicy(t *testing.T, target string) *RemoteMediaPolicy {
	t.Helper()
	parsed, err := url.Parse(target)
	if err != nil {
		t.Fatal(err)
	}
	port, err := strconv.Atoi(parsed.Port())
	if err != nil {
		t.Fatal(err)
	}
	return &RemoteMediaPolicy{
		Enabled: true, AllowedSchemes: []string{parsed.Scheme},
		AllowedHosts: []string{parsed.Hostname()}, AllowedPorts: []int{port},
		AllowPrivateNetworks: true, MaxRedirects: 1, MaxConcurrentFetches: 2,
		ConnectTimeout: time.Second, ResponseHeaderTimeout: time.Second,
		TotalTimeout: 2 * time.Second, MaxResponseBytes: 1 << 20,
	}
}

func TestRemoteMediaFetcherAllowlistRedirectAndContentType(t *testing.T) {
	var encoded bytes.Buffer
	if err := png.Encode(&encoded, image.NewRGBA(image.Rect(0, 0, 1, 1))); err != nil {
		t.Fatal(err)
	}
	server := httptest.NewServer(http.HandlerFunc(func(response http.ResponseWriter, request *http.Request) {
		switch request.URL.Path {
		case "/redirect":
			http.Redirect(response, request, "/image", http.StatusFound)
		case "/image":
			response.Header().Set("Content-Type", "image/png")
			_, _ = response.Write(encoded.Bytes())
		default:
			response.Header().Set("Content-Type", "text/plain")
			_, _ = response.Write([]byte("not media"))
		}
	}))
	defer server.Close()
	fetcher, err := newRemoteMediaFetcher(testRemoteMediaPolicy(t, server.URL))
	if err != nil {
		t.Fatal(err)
	}
	data, err := fetcher.fetch(context.Background(), server.URL+"/redirect", "image")
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(data, encoded.Bytes()) {
		t.Fatal("remote image bytes changed")
	}
	if _, err := fetcher.fetch(context.Background(), server.URL+"/text", "image"); err == nil {
		t.Fatal("non-image content type accepted")
	}
}

func TestRemoteMediaFetcherRejectsPrivateResolutionByDefault(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(http.ResponseWriter, *http.Request) {}))
	defer server.Close()
	policy := testRemoteMediaPolicy(t, server.URL)
	policy.AllowPrivateNetworks = false
	fetcher, err := newRemoteMediaFetcher(policy)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := fetcher.fetch(context.Background(), server.URL, "image"); err == nil {
		t.Fatal("private DNS resolution accepted")
	}
}

func TestRemoteMediaFetcherRejectsURLPolicyViolations(t *testing.T) {
	policy := testRemoteMediaPolicy(t, "https://media.example.com:443")
	fetcher, err := newRemoteMediaFetcher(policy)
	if err != nil {
		t.Fatal(err)
	}
	for _, source := range []string{
		"http://media.example.com:443/image.png",
		"https://other.example.com/image.png",
		"https://media.example.com:444/image.png",
		"https://user@media.example.com/image.png",
		"https://media.example.com/image.png#fragment",
	} {
		target, parseErr := url.Parse(source)
		if parseErr != nil {
			t.Fatal(parseErr)
		}
		if err := fetcher.validateURL(target); err == nil {
			t.Fatalf("URL policy accepted %q", source)
		}
	}
}

func TestRemoteMediaFetcherEnforcesResponseByteLimit(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(response http.ResponseWriter, _ *http.Request) {
		response.Header().Set("Content-Type", "image/png")
		_, _ = response.Write(bytes.Repeat([]byte{1}, 32))
	}))
	defer server.Close()
	policy := testRemoteMediaPolicy(t, server.URL)
	policy.MaxResponseBytes = 16
	fetcher, err := newRemoteMediaFetcher(policy)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := fetcher.fetch(context.Background(), server.URL, "image"); err == nil {
		t.Fatal("oversized remote media response accepted")
	}
}

func TestRemoteMediaPolicyRequiresExplicitAllowlist(t *testing.T) {
	policy := testRemoteMediaPolicy(t, "https://example.com:443")
	policy.AllowedHosts = nil
	if _, err := newRemoteMediaFetcher(policy); err == nil {
		t.Fatal("enabled policy without host allowlist accepted")
	}
	if publicMediaIP(netip.MustParseAddr("127.0.0.1")) ||
		publicMediaIP(netip.MustParseAddr("192.0.2.1")) ||
		!publicMediaIP(netip.MustParseAddr("8.8.8.8")) {
		t.Fatal("public media IP classification is incorrect")
	}
}
