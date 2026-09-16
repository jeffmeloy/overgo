package server

import (
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

// TestDownloadShutdownOwnership pins the one download shutdown owner:
// closing the handler cancels the in-flight transfer, records the
// interruption on the job, returns only after the transfer goroutine has
// unwound, refuses new admissions, and leaves no partial file under the
// download root.
func TestDownloadShutdownOwnership(t *testing.T) {
	transferStarted := make(chan struct{})
	digest := sha256.Sum256([]byte("w"))
	mux := http.NewServeMux()
	mux.HandleFunc("GET /api/models/acme/tiny", func(response http.ResponseWriter, _ *http.Request) {
		_ = json.NewEncoder(response).Encode(map[string]any{
			"id": "acme/tiny", "sha": "rev0",
			"siblings": []map[string]any{{"rfilename": "weights.bin", "lfs": map[string]any{
				"sha256": hex.EncodeToString(digest[:]), "size": 1,
			}}},
		})
	})
	mux.HandleFunc("GET /acme/tiny/resolve/rev0/weights.bin", func(response http.ResponseWriter, request *http.Request) {
		close(transferStarted)
		<-request.Context().Done()
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
		if err == nil && !entry.IsDir() {
			partials = append(partials, path)
		}
		return err
	}); err != nil {
		t.Fatal(err)
	}
	if len(partials) != 0 {
		t.Fatalf("shutdown left partial files: %v", partials)
	}
	refused := httptest.NewRecorder()
	handler.ServeHTTP(refused, httptest.NewRequest(http.MethodPost, "/hub/downloads",
		strings.NewReader(`{"kind":"models","repository":"acme/tiny","revision":"","directory":"again"}`)))
	if refused.Code != http.StatusServiceUnavailable || !strings.Contains(refused.Body.String(), "shut down") {
		t.Fatalf("post-shutdown admission status=%d body=%s", refused.Code, refused.Body.String())
	}
}
