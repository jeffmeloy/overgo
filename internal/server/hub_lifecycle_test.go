package server

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"runtime"
	"strconv"
	"strings"
	"testing"
)

// TestWorkbenchAPIDownloadAdmissionAndCancellation pins the transfer
// lifecycle bounds: a full running set refuses admission, DELETE
// cancels a running job, and a cancelled job stays cancelled when its
// transfer unwinds.
func TestWorkbenchAPIDownloadAdmissionAndCancellation(t *testing.T) {
	release := make(chan struct{})
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
		select {
		case <-release:
		case <-request.Context().Done():
		}
		_, _ = response.Write([]byte("w"))
	})
	hub := httptest.NewServer(mux)
	defer hub.Close()
	defer close(release)
	handler := hubTestHandler(t, hub.URL, t.TempDir())
	start := func(directory string) *httptest.ResponseRecorder {
		recorder := httptest.NewRecorder()
		handler.ServeHTTP(recorder, httptest.NewRequest(http.MethodPost, "/hub/downloads",
			strings.NewReader(`{"kind":"models","repository":"acme/tiny","revision":"","directory":"`+directory+`"}`)))
		return recorder
	}
	accepted := make([]*httptest.ResponseRecorder, handler.config.MaxConcurrent)
	for index := range accepted {
		accepted[index] = start(strconv.Itoa(index))
		if accepted[index].Code != http.StatusAccepted {
			t.Fatalf("admission %d status = %d", index, accepted[index].Code)
		}
	}
	if refused := start(strconv.Itoa(len(accepted))); refused.Code != http.StatusTooManyRequests {
		t.Fatalf("backlog status = %d body=%s", refused.Code, refused.Body.String())
	}
	var job DownloadJob
	if err := json.Unmarshal(accepted[0].Body.Bytes(), &job); err != nil {
		t.Fatal(err)
	}
	cancel := httptest.NewRecorder()
	handler.ServeHTTP(cancel, httptest.NewRequest(http.MethodDelete,
		"/hub/downloads?id="+strconv.FormatUint(job.ID, identifierRadix), nil))
	if cancel.Code != http.StatusOK || !strings.Contains(cancel.Body.String(), downloadStateCancelled) {
		t.Fatalf("cancel status=%d body=%s", cancel.Code, cancel.Body.String())
	}
	// The cancelled job settles once its goroutine has recorded the
	// interruption; the scheduler, not a clock, paces the look.
	for {
		listed := httptest.NewRecorder()
		handler.ServeHTTP(listed, httptest.NewRequest(http.MethodGet, "/hub/downloads", nil))
		if strings.Count(listed.Body.String(), downloadStateCancelled) == 1 {
			break
		}
		runtime.Gosched()
	}
	if admitted := start(strconv.Itoa(len(accepted) + 1)); admitted.Code != http.StatusAccepted {
		t.Fatalf("post-cancel admission status = %d body=%s", admitted.Code, admitted.Body.String())
	}
}
