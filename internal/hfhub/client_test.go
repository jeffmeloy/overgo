package hfhub

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// hubFixture serves a minimal hub: one model with an LFS weights file and a
// small config, plus a dataset namespace, recording every authorization
// header it sees.
func hubFixture(t *testing.T, weights []byte) (*httptest.Server, *[]string) {
	t.Helper()
	digest := sha256.Sum256(weights)
	var authorizations []string
	mux := http.NewServeMux()
	mux.HandleFunc("/api/models", func(response http.ResponseWriter, request *http.Request) {
		authorizations = append(authorizations, request.Header.Get("Authorization"))
		if request.URL.Query().Get("limit") == "" {
			http.Error(response, "limit required", http.StatusBadRequest)
			return
		}
		_ = json.NewEncoder(response).Encode([]Listing{{ID: "acme/tiny", Downloads: 7}})
	})
	mux.HandleFunc("/api/models/acme/tiny", func(response http.ResponseWriter, request *http.Request) {
		authorizations = append(authorizations, request.Header.Get("Authorization"))
		_ = json.NewEncoder(response).Encode(map[string]any{
			"id": "acme/tiny", "sha": "rev0",
			"siblings": []map[string]any{
				{"rfilename": "config.json", "size": 2},
				{"rfilename": "model.safetensors", "lfs": map[string]any{
					"sha256": hex.EncodeToString(digest[:]), "size": len(weights),
				}},
			},
		})
	})
	mux.HandleFunc("/acme/tiny/resolve/rev0/config.json", func(response http.ResponseWriter, _ *http.Request) {
		_, _ = response.Write([]byte("{}"))
	})
	mux.HandleFunc("/acme/tiny/resolve/rev0/model.safetensors", func(response http.ResponseWriter, _ *http.Request) {
		_, _ = response.Write(weights)
	})
	server := httptest.NewServer(mux)
	t.Cleanup(server.Close)
	return server, &authorizations
}

func TestHubSearchAuthorizesAndBounds(t *testing.T) {
	server, authorizations := hubFixture(t, []byte("weights"))
	client, err := New(server.URL, "secret-token")
	if err != nil {
		t.Fatal(err)
	}
	listings, err := client.Search(context.Background(), SearchQuery{Kind: KindModel, Search: "tiny", Limit: 5})
	if err != nil || len(listings) != 1 || listings[0].ID != "acme/tiny" {
		t.Fatalf("listings=%v err=%v", listings, err)
	}
	if len(*authorizations) == 0 || (*authorizations)[0] != "Bearer secret-token" {
		t.Fatalf("authorization not sent: %v", *authorizations)
	}
	if _, err := client.Search(context.Background(), SearchQuery{Kind: KindModel}); err == nil {
		t.Fatal("unbounded search accepted")
	}
	if _, err := client.Search(context.Background(), SearchQuery{Kind: "spaces", Limit: 1}); err == nil {
		t.Fatal("unknown namespace accepted")
	}
}

func TestHubDownloadVerifiesDeclaredDigest(t *testing.T) {
	weights := []byte("exact weight bytes")
	server, _ := hubFixture(t, weights)
	client, err := New(server.URL, "")
	if err != nil {
		t.Fatal(err)
	}
	destination := t.TempDir()
	var observed []Progress
	resolved, err := client.Download(context.Background(), DownloadRequest{
		Kind: KindModel, Repository: "acme/tiny", Destination: destination,
		Observe: func(progress Progress) { observed = append(observed, progress) },
	})
	if err != nil {
		t.Fatal(err)
	}
	if resolved.Revision != "rev0" || len(resolved.Files) != 2 {
		t.Fatalf("resolved=%+v", resolved)
	}
	stored, err := os.ReadFile(filepath.Join(destination, "model.safetensors"))
	if err != nil || string(stored) != string(weights) {
		t.Fatalf("stored=%q err=%v", stored, err)
	}
	if len(observed) == 0 || observed[len(observed)-1].Received != int64(len(weights)) {
		t.Fatalf("progress=%v", observed)
	}
	entries, err := os.ReadDir(destination)
	if err != nil {
		t.Fatal(err)
	}
	for _, entry := range entries {
		if strings.Contains(entry.Name(), ".partial-") {
			t.Fatalf("partial file survived: %s", entry.Name())
		}
	}
}

func TestHubDownloadRefusesTamperedBytes(t *testing.T) {
	weights := []byte("declared bytes")
	destination := t.TempDir()
	// A lying hub declares the digest of one content and serves another.
	mux := http.NewServeMux()
	digest := sha256.Sum256(weights)
	mux.HandleFunc("/api/models/acme/tiny", func(response http.ResponseWriter, _ *http.Request) {
		_ = json.NewEncoder(response).Encode(map[string]any{
			"id": "acme/tiny", "sha": "rev0",
			"siblings": []map[string]any{{"rfilename": "model.safetensors", "lfs": map[string]any{
				"sha256": hex.EncodeToString(digest[:]), "size": len(weights),
			}}},
		})
	})
	mux.HandleFunc("/acme/tiny/resolve/rev0/model.safetensors", func(response http.ResponseWriter, _ *http.Request) {
		_, _ = response.Write([]byte("differently sized"))
	})
	lying := httptest.NewServer(mux)
	defer lying.Close()
	liar, err := New(lying.URL, "")
	if err != nil {
		t.Fatal(err)
	}
	if _, err := liar.Download(context.Background(), DownloadRequest{
		Kind: KindModel, Repository: "acme/tiny", Destination: destination,
	}); err == nil {
		t.Fatal("tampered download accepted")
	}
	if _, statErr := os.Stat(filepath.Join(destination, "model.safetensors")); statErr == nil {
		t.Fatal("tampered bytes landed under the final name")
	}
}

func TestHubRefusesEscapingPaths(t *testing.T) {
	if _, err := secureJoin(t.TempDir(), "../outside.bin"); err == nil {
		t.Fatal("path escape accepted")
	}
	if !validRepositoryID("acme/tiny") || validRepositoryID("a/b/c") || validRepositoryID("..") {
		t.Fatal("repository id validation is wrong")
	}
}
