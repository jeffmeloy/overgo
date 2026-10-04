package server

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"sync"
	"testing"

	"overgo/internal/overgodb"
)

// fakeLoop is a loop whose status the test sets; a stop holds it with the
// detail the stop gave, unless the test makes stopping fail.
type fakeLoop struct {
	mu      sync.Mutex
	status  LoopStatus
	refusal error
	details []string
}

func (loop *fakeLoop) LoopStatus(context.Context) (LoopStatus, error) {
	loop.mu.Lock()
	defer loop.mu.Unlock()
	return loop.status, nil
}

func (loop *fakeLoop) StopLoop(_ context.Context, detail string) error {
	loop.mu.Lock()
	defer loop.mu.Unlock()
	if loop.refusal != nil {
		return loop.refusal
	}
	loop.details = append(loop.details, detail)
	loop.status.Stopped, loop.status.StopReason = true, "user-stop: "+detail
	return nil
}

// runningLoop is a supervised loop working a row after a landing.
func runningLoop() LoopStatus {
	return LoopStatus{Running: true, Row: "gui-next/do", RowTitle: "Build the next view",
		Landing: LoopLanding{Commit: "0f8480fcacd5b99b856b797d89a0ee6048751473", Row: "gui-phase-profile/do", Subject: "Show a turn's tokens per forward", At: "2026-09-27T21:04:00-04:00"}}
}

// TestLoopStatusAndStop: the loop routes answer that no loop is here on a
// server the launcher gave none; with one, the status states whether it
// runs, the row it works and its last landing, and a stop records a user
// stop from the workbench and answers the stopped status, or the reason it
// could not.
func TestLoopStatusAndStop(t *testing.T) {
	t.Parallel()
	status := func(handler *Handler, method, path string) (int, LoopStatus) {
		t.Helper()
		response := serveTestRequest(handler, method, path, "")
		var decoded LoopStatus
		_ = json.Unmarshal(response.Body.Bytes(), &decoded)
		return response.Code, decoded
	}
	if code, _ := status(newTestHandler(t, &fakeGenerator{}), http.MethodGet, "/loop/status"); code != http.StatusNotFound {
		t.Fatalf("a server without a loop answered %d", code)
	}
	store, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	loop := &fakeLoop{status: runningLoop()}
	handler := newTestHandlerForRepository(t, store, &fakeGenerator{}, func(config *Config) { config.Loop = loop })
	if code, read := status(handler, http.MethodGet, "/loop/status"); code != http.StatusOK || read != runningLoop() {
		t.Fatalf("status %d %+v", code, read)
	}
	code, stopped := status(handler, http.MethodPost, "/loop/stop")
	if code != http.StatusOK || !stopped.Stopped || stopped.StopReason != "user-stop: "+loopStopDetail || len(loop.details) != 1 || loop.details[0] != loopStopDetail {
		t.Fatalf("stop %d %+v, recorded %v", code, stopped, loop.details)
	}
	loop.refusal = errors.New("plan: stop changed; review its current identity before retrying")
	response := serveTestRequest(handler, http.MethodPost, "/loop/stop", "")
	if response.Code != http.StatusConflict || !json.Valid(response.Body.Bytes()) {
		t.Fatalf("a refused stop answered %d %s", response.Code, response.Body.String())
	}
}
