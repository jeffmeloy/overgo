package server

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"overgo/internal/hfhub"
)

// TestDownloadShutdownOwnership pins the one download shutdown owner:
// closing the handler cancels the in-flight transfer, records the
// interruption on the job, returns only after the transfer goroutine has
// unwound, refuses new admissions, and retains verified-source progress that
// a new client can resume after shutdown.
func TestDownloadShutdownOwnership(t *testing.T) {
	transferStarted := make(chan struct{})
	weights := []byte("weights")
	digest := sha256.Sum256(weights)
	var requests atomic.Int64
	mux := http.NewServeMux()
	mux.HandleFunc("GET /api/models/acme/tiny", func(response http.ResponseWriter, _ *http.Request) {
		_ = json.NewEncoder(response).Encode(map[string]any{
			"id": "acme/tiny", "sha": "rev0",
			"siblings": []map[string]any{{"rfilename": "weights.bin", "lfs": map[string]any{
				"sha256": hex.EncodeToString(digest[:]), "size": len(weights),
			}}},
		})
	})
	mux.HandleFunc("GET /acme/tiny/resolve/rev0/weights.bin", func(response http.ResponseWriter, request *http.Request) {
		response.Header().Set("ETag", `"weights-v1"`)
		if requests.Add(1) == 1 {
			response.Header().Set("Content-Length", strconv.Itoa(len(weights)))
			_, _ = response.Write(weights[:1])
			response.(http.Flusher).Flush()
			close(transferStarted)
			<-request.Context().Done()
			return
		}
		if request.Header.Get("Range") != "bytes=1-" || request.Header.Get("If-Range") != `"weights-v1"` {
			t.Errorf("resume headers = %v", request.Header)
		}
		http.ServeContent(response, request, "weights.bin", time.Time{}, bytes.NewReader(weights))
	})
	hub := httptest.NewServer(mux)
	defer hub.Close()
	root := t.TempDir()
	handler := hubTestHandler(t, hub.URL, root)
	started := httptest.NewRecorder()
	handler.ServeHTTP(started, httptest.NewRequest(http.MethodPost, "/hub/downloads",
		strings.NewReader(`{"kind":"models","repository":"acme/tiny","revision":"","directory":"tiny"}`)))
	if started.Code != http.StatusAccepted {
		t.Fatalf("admission status=%d body=%s", started.Code, started.Body.String())
	}
	<-transferStarted
	// Wait for the registry's observed prefix, not just a server-side write.
	for handler.downloads.snapshot()[0].Received == 0 {
		if t.Context().Err() != nil {
			t.Fatal(t.Context().Err())
		}
		runtime.Gosched()
	}
	// Close owns shutdown: it cancels the transfer, records the
	// interruption, and returns only after the transfer goroutine has
	// unwound -- so every assertion below reads settled state, and any
	// still-running transfer would race the partial-file walk into flaking.
	if err := handler.Close(); err != nil {
		t.Fatal(err)
	}
	jobs := handler.downloads.snapshot()
	if len(jobs) != 1 || jobs[0].State != downloadStateCancelled ||
		!strings.Contains(jobs[0].Error, "shutdown interrupted") {
		t.Fatalf("shutdown job record = %+v", jobs)
	}
	var partials []string
	if err := filepath.WalkDir(root, func(path string, entry os.DirEntry, err error) error {
		if err == nil && !entry.IsDir() && strings.Contains(entry.Name(), ".partial-") && !strings.HasSuffix(entry.Name(), ".etag") {
			partials = append(partials, path)
		}
		return err
	}); err != nil {
		t.Fatal(err)
	}
	if len(partials) != 1 {
		t.Fatalf("shutdown did not retain one partial: %v", partials)
	}
	if got, err := os.ReadFile(partials[0]); err != nil || string(got) != "w" {
		t.Fatalf("shutdown prefix=%q: %v", got, err)
	}
	final := filepath.Join(root, "tiny", "weights.bin")
	if _, err := os.Stat(final); !errors.Is(err, os.ErrNotExist) {
		t.Fatalf("shutdown published partial: %v", err)
	}
	client, err := hfhub.New(hub.URL, "")
	if err != nil {
		t.Fatal(err)
	}
	if _, err := client.Download(t.Context(), hfhub.DownloadRequest{Kind: hfhub.KindModel, Repository: "acme/tiny", Destination: filepath.Dir(final)}); err != nil {
		t.Fatal(err)
	}
	if got, err := os.ReadFile(final); err != nil || !bytes.Equal(got, weights) {
		t.Fatalf("resumed bytes=%q: %v", got, err)
	}
	refused := httptest.NewRecorder()
	handler.ServeHTTP(refused, httptest.NewRequest(http.MethodPost, "/hub/downloads",
		strings.NewReader(`{"kind":"models","repository":"acme/tiny","revision":"","directory":"again"}`)))
	if refused.Code != http.StatusServiceUnavailable || !strings.Contains(refused.Body.String(), "shut down") {
		t.Fatalf("post-shutdown admission status=%d body=%s", refused.Code, refused.Body.String())
	}
}

func TestDownloadJobCallerBudget(t *testing.T) {
	hub := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if strings.HasPrefix(r.URL.Path, "/api/") {
			_, _ = w.Write([]byte(`{"id":"acme/tiny","sha":"commit1","siblings":[{"rfilename":"weights.bin","size":8}]}`))
			return
		}
		w.Header().Set("Content-Length", "8")
		w.Header().Set("ETag", `"weights-v1"`)
		_, _ = w.Write([]byte("abcd"))
		w.(http.Flusher).Flush()
		<-r.Context().Done()
	}))
	defer hub.Close()
	root := t.TempDir()
	handler := hubTestHandler(t, hub.URL, root)
	t.Cleanup(func() { _ = handler.Close() })
	for _, timeout := range []string{"invalid", "-1s"} {
		refused := httptest.NewRecorder()
		handler.ServeHTTP(refused, httptest.NewRequest(http.MethodPost, "/hub/downloads", strings.NewReader(`{"repository":"acme/tiny","timeout":"`+timeout+`"}`)))
		if refused.Code != http.StatusBadRequest {
			t.Fatalf("invalid timeout accepted: %d %s", refused.Code, refused.Body.String())
		}
	}
	started := httptest.NewRecorder()
	handler.ServeHTTP(started, httptest.NewRequest(http.MethodPost, "/hub/downloads", strings.NewReader(`{"repository":"acme/tiny","timeout":"1s"}`)))
	if started.Code != http.StatusAccepted {
		t.Fatalf("admission: %d %s", started.Code, started.Body.String())
	}
	handler.downloads.transfers.Wait()
	jobs := handler.downloads.snapshot()
	if len(jobs) != 1 || jobs[0].State != downloadStateFailed || !strings.Contains(jobs[0].Error, "deadline exceeded") || jobs[0].Received != 4 {
		t.Fatalf("budget result=%+v", jobs)
	}
	final := filepath.Join(root, "tiny", "weights.bin")
	if _, err := os.Stat(final); !errors.Is(err, os.ErrNotExist) {
		t.Fatalf("budget published incomplete bytes: %v", err)
	}
	partials, err := filepath.Glob(final + ".partial-*")
	if err != nil {
		t.Fatal(err)
	}
	retained := false
	for _, path := range partials {
		if strings.HasSuffix(path, ".etag") {
			continue
		}
		got, err := os.ReadFile(path)
		if err != nil || string(got) != "abcd" {
			t.Fatalf("retained bytes=%q: %v", got, err)
		}
		retained = true
	}
	if !retained {
		t.Fatal("budget lost received bytes")
	}
}
