package server

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
)

// hubTestServer serves one model repository with a digest-declared file, the
// same shape the workbench's discovery surface consumes.
func hubTestServer(t *testing.T, weights []byte) *httptest.Server {
	t.Helper()
	digest := sha256.Sum256(weights)
	mux := http.NewServeMux()
	mux.HandleFunc("GET /api/models", func(response http.ResponseWriter, request *http.Request) {
		_ = json.NewEncoder(response).Encode([]map[string]any{{"id": "acme/tiny", "downloads": 3}})
	})
	mux.HandleFunc("GET /api/models/acme/tiny", func(response http.ResponseWriter, _ *http.Request) {
		_ = json.NewEncoder(response).Encode(map[string]any{
			"id": "acme/tiny", "sha": "rev0",
			"siblings": []map[string]any{{"rfilename": "weights.bin", "lfs": map[string]any{
				"sha256": hex.EncodeToString(digest[:]), "size": len(weights),
			}}},
		})
	})
	mux.HandleFunc("GET /acme/tiny/resolve/rev0/weights.bin", func(response http.ResponseWriter, _ *http.Request) {
		_, _ = response.Write(weights)
	})
	server := httptest.NewServer(mux)
	t.Cleanup(server.Close)
	return server
}

func hubTestHandler(t *testing.T, endpoint, downloadRoot string) *Handler {
	t.Helper()
	handler, err := New(Config{
		ModelID:            testModelID,
		MaxTokens:          testMaxTokens,
		DefaultTemperature: testNeutralTemperature,
		DefaultTopP:        testFullTopP,
		Analysis:           testAnalysisPolicy,
		HubEndpoint:        endpoint,
		HubDownloadRoot:    downloadRoot,
	}, &fakeGenerator{})
	if err != nil {
		t.Fatal(err)
	}
	return handler
}

// TestWorkbenchAPIHubSearch proxies one bounded query and refuses an
// unbounded or oversized one.
func TestWorkbenchAPIHubSearch(t *testing.T) {
	hub := hubTestServer(t, []byte("weights"))
	handler := hubTestHandler(t, hub.URL, t.TempDir())
	recorder := httptest.NewRecorder()
	handler.ServeHTTP(recorder, httptest.NewRequest(http.MethodGet, "/hub/search?q=tiny", nil))
	if recorder.Code != http.StatusOK || !strings.Contains(recorder.Body.String(), "acme/tiny") {
		t.Fatalf("search status=%d body=%s", recorder.Code, recorder.Body.String())
	}
	recorder = httptest.NewRecorder()
	handler.ServeHTTP(recorder, httptest.NewRequest(http.MethodGet, "/hub/search?limit=100000", nil))
	if recorder.Code != http.StatusBadRequest {
		t.Fatalf("oversized limit status=%d", recorder.Code)
	}
	recorder = httptest.NewRecorder()
	handler.ServeHTTP(recorder, httptest.NewRequest(http.MethodPost, "/hub/search", nil))
	if recorder.Code != http.StatusMethodNotAllowed {
		t.Fatalf("post status=%d", recorder.Code)
	}
}

// TestWorkbenchAPIDownloadJob runs one verified download to completion and
// observes it through the job listing.
func TestWorkbenchAPIDownloadJob(t *testing.T) {
	weights := []byte("downloadable weight bytes")
	hub := hubTestServer(t, weights)
	root := t.TempDir()
	handler := hubTestHandler(t, hub.URL, root)
	recorder := httptest.NewRecorder()
	handler.ServeHTTP(recorder, httptest.NewRequest(http.MethodPost, "/hub/downloads",
		strings.NewReader(`{"kind":"models","repository":"acme/tiny","revision":"","directory":""}`)))
	if recorder.Code != http.StatusAccepted {
		t.Fatalf("start status=%d body=%s", recorder.Code, recorder.Body.String())
	}
	for {
		recorder = httptest.NewRecorder()
		handler.ServeHTTP(recorder, httptest.NewRequest(http.MethodGet, "/hub/downloads", nil))
		var listed struct {
			Downloads []DownloadJob `json:"downloads"`
		}
		if err := json.Unmarshal(recorder.Body.Bytes(), &listed); err != nil {
			t.Fatal(err)
		}
		if len(listed.Downloads) != 1 {
			t.Fatalf("downloads=%v", listed.Downloads)
		}
		job := listed.Downloads[0]
		if job.State == downloadStateFailed {
			t.Fatalf("download failed: %s", job.Error)
		}
		if job.State == downloadStateSucceeded {
			if job.Files != 1 || job.Revision != "rev0" {
				t.Fatalf("job=%+v", job)
			}
			break
		}
		// The job finishes once its goroutine has run; the scheduler paces the look.
		runtime.Gosched()
	}
	stored, err := os.ReadFile(filepath.Join(root, "tiny", "weights.bin"))
	if err != nil || string(stored) != string(weights) {
		t.Fatalf("stored=%q err=%v", stored, err)
	}
}

// TestWorkbenchAPIDownloadRefusals pins the refusal surface: no configured
// root, and a directory trying to escape it.
func TestWorkbenchAPIDownloadRefusals(t *testing.T) {
	hub := hubTestServer(t, []byte("w"))
	handler := hubTestHandler(t, hub.URL, "")
	recorder := httptest.NewRecorder()
	handler.ServeHTTP(recorder, httptest.NewRequest(http.MethodPost, "/hub/downloads",
		strings.NewReader(`{"kind":"models","repository":"acme/tiny","revision":"","directory":""}`)))
	if recorder.Code != http.StatusServiceUnavailable {
		t.Fatalf("no-root status=%d", recorder.Code)
	}
	handler = hubTestHandler(t, hub.URL, t.TempDir())
	recorder = httptest.NewRecorder()
	handler.ServeHTTP(recorder, httptest.NewRequest(http.MethodPost, "/hub/downloads",
		strings.NewReader(`{"kind":"models","repository":"acme/tiny","revision":"","directory":"../escape"}`)))
	if recorder.Code != http.StatusBadRequest {
		t.Fatalf("escape status=%d body=%s", recorder.Code, recorder.Body.String())
	}
}

// TestWorkbenchAPICatalogWithoutStore reports the absent repository as a
// typed unavailability rather than an empty catalog.
func TestWorkbenchAPICatalogWithoutStore(t *testing.T) {
	handler := hubTestHandler(t, "", t.TempDir())
	recorder := httptest.NewRecorder()
	handler.ServeHTTP(recorder, httptest.NewRequest(http.MethodGet, "/catalog/models", nil))
	if recorder.Code != http.StatusServiceUnavailable {
		t.Fatalf("catalog status=%d body=%s", recorder.Code, recorder.Body.String())
	}
}
