package server

import (
	"bufio"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"runtime"
	"strings"
	"sync"
	"testing"
	"time"

	"overgo/internal/tokenizer"
)

// monitorLog reads the monitor's ring through the route the Requests tab reads.
func monitorLog(t *testing.T, handler http.Handler) requestLogResponse {
	t.Helper()
	response := serveTestRequest(handler, http.MethodGet, "/requests", "")
	var log requestLogResponse
	if response.Code != http.StatusOK || json.Unmarshal(response.Body.Bytes(), &log) != nil {
		t.Fatalf("requests status=%d body=%s", response.Code, response.Body.String())
	}
	return log
}

// TestRequestMonitorRecordsWhatTheServerAnswered drives routes and reads them back newest first,
// each with its method, route, status and duration, a generation with the tokens it made; the
// static page and the monitor's own reads are left out, and the ring holds a bounded number.
func TestRequestMonitorRecordsWhatTheServerAnswered(t *testing.T) {
	t.Parallel()
	handler := newTestHandler(t, &fakeGenerator{pieces: []string{"A", "B", "C"}})
	defer handler.Close()
	for _, call := range []struct {
		method, path, body string
		status             int
	}{
		{http.MethodGet, "/v1/models", "", http.StatusOK},
		{http.MethodPost, "/completion", `{"prompt":"hi","n_predict":3}`, http.StatusOK},
		{http.MethodPost, "/completion", `{`, http.StatusBadRequest},
		{http.MethodDelete, "/v1/models", "", http.StatusMethodNotAllowed},
		{http.MethodGet, "/", "", http.StatusOK},
	} {
		if response := serveTestRequest(handler, call.method, call.path, call.body); response.Code != call.status {
			t.Fatalf("%s %s answered %d, want %d: %s", call.method, call.path, response.Code, call.status, response.Body.String())
		}
	}
	log := monitorLog(t, handler)
	want := []struct {
		method, route string
		status        int
	}{
		{http.MethodDelete, "/v1/models", http.StatusMethodNotAllowed},
		{http.MethodPost, "/completion", http.StatusBadRequest},
		{http.MethodPost, "/completion", http.StatusOK},
		{http.MethodGet, "/v1/models", http.StatusOK},
	}
	if len(log.Requests) != len(want) || log.Total != uint64(len(want)) {
		t.Fatalf("the monitor holds %d of %d requests, want %d: %+v", len(log.Requests), log.Total, len(want), log.Requests)
	}
	for index, entry := range log.Requests {
		if entry.Method != want[index].method || entry.Route != want[index].route || entry.Status != want[index].status || entry.InFlight || entry.DurationMS < 0 || entry.StartedUnixMS == 0 {
			t.Errorf("request %d = %+v, want %s %s answered %d", index, entry, want[index].method, want[index].route, want[index].status)
		}
	}
	if completion := log.Requests[2]; completion.CompletionTokens != 3 || completion.PromptTokens == 0 {
		t.Errorf("the generation made %d prompt and %d completion tokens, want some and three", completion.PromptTokens, completion.CompletionTokens)
	}
	if other := log.Requests[3]; other.PromptTokens != 0 || other.CompletionTokens != 0 {
		t.Errorf("a request that generated nothing carries tokens: %+v", other)
	}
	if again := monitorLog(t, handler); len(again.Requests) != len(want) {
		t.Fatalf("reading the monitor recorded itself: %d requests, want %d", len(again.Requests), len(want))
	}

	for range requestMonitorCapacity + 8 {
		serveTestRequest(handler, http.MethodGet, "/v1/models", "")
	}
	if full := monitorLog(t, handler); len(full.Requests) != requestMonitorCapacity || full.Limit != requestMonitorCapacity || full.Requests[0].Route != "/v1/models" {
		t.Fatalf("the monitor holds %d requests, want the newest %d", len(full.Requests), requestMonitorCapacity)
	}
}

// floatSlackMS is the rounding of a sum of two durations held as milliseconds, far below a clock tick.
const floatSlackMS = 1e-9

// clockMoves returns once the clock the monitor times with has moved, which on a platform whose
// monotonic clock ticks coarsely is the least a measured wait can show; the scheduler paces it.
func clockMoves(t *testing.T) {
	t.Helper()
	begin := time.Now()
	for time.Since(begin) <= 0 && t.Context().Err() == nil {
		runtime.Gosched()
	}
}

// slowTokenizer is a generator whose prompts take a measurable time to tokenize.
type slowTokenizer struct{ *fakeGenerator }

func (generator slowTokenizer) TokenizeText(text string, addSpecial, parseSpecial bool) ([]tokenizer.TokenID, error) {
	begin := time.Now()
	for time.Since(begin) <= 0 {
		runtime.Gosched()
	}
	return generator.fakeGenerator.TokenizeText(text, addSpecial, parseSpecial)
}

// TestRequestMonitorShowsAdmissionWait holds one place and sends a second request that has to
// wait for it: the monitor shows the second one's wait for the place, which the first, that took
// the free place at once, did not have, and the time each spent preparing its prompt, both
// beside the duration that includes them, so a request that is slow to start says whether it
// was waiting or preparing.
func TestRequestMonitorShowsAdmissionWait(t *testing.T) {
	t.Parallel()
	generator := &fakeGenerator{pieces: []string{"A"}, started: make(chan struct{}), release: make(chan struct{})}
	handler, err := New(Config{ModelID: testModelID, MaxTokens: testMaxTokens, MaxConcurrent: 1}, slowTokenizer{generator})
	if err != nil {
		t.Fatal(err)
	}
	defer handler.Close()
	var served sync.WaitGroup
	send := func() {
		served.Go(func() {
			if response := serveTestRequest(handler, http.MethodPost, "/v1/completions", `{"prompt":"hi","max_tokens":1}`); response.Code != http.StatusOK {
				t.Errorf("completion answered %d: %s", response.Code, response.Body.String())
			}
		})
	}
	send()
	<-generator.started
	send()
	// The second request is waiting for the place once the director counts it; the scheduler, not a clock, paces the look.
	for handler.sessions.Snapshot().Waiting == 0 && t.Context().Err() == nil {
		runtime.Gosched()
	}
	// The place is released once the clock the monitor reads has moved, so the wait it measured is not a tick shorter than any.
	clockMoves(t)
	close(generator.release)
	served.Wait()
	log := monitorLog(t, handler).Requests
	if len(log) != 2 {
		t.Fatalf("the monitor holds %d requests, want the two completions: %+v", len(log), log)
	}
	first, second := log[1], log[0]
	if second.AdmissionMS <= 0 || second.AdmissionMS <= first.AdmissionMS {
		t.Errorf("the request that waited for the place shows %v ms of admission and the one that did not %v, want the waiter's the greater", second.AdmissionMS, first.AdmissionMS)
	}
	for index, entry := range []requestEntry{first, second} {
		// Whole clock ticks, as milliseconds: the two waits lie inside the duration, up to the floating point sum.
		if entry.PreparationMS <= 0 || entry.AdmissionMS+entry.PreparationMS > entry.DurationMS+floatSlackMS {
			t.Errorf("request %d shows %v ms of admission and %v of preparation inside a duration of %v", index, entry.AdmissionMS, entry.PreparationMS, entry.DurationMS)
		}
	}
}

// TestRequestMonitorRecordsARefusedRequest checks a request the credential refuses is recorded
// with the status it was answered, since a refusal is part of an error rate.
func TestRequestMonitorRecordsARefusedRequest(t *testing.T) {
	t.Parallel()
	handler, err := New(Config{ModelID: testModelID, APIKey: "secret", MaxTokens: testMaxTokens}, &fakeGenerator{})
	if err != nil {
		t.Fatal(err)
	}
	defer handler.Close()
	if response := serveTestRequest(handler, http.MethodGet, "/props", ""); response.Code != http.StatusUnauthorized {
		t.Fatalf("an unauthorized call answered %d", response.Code)
	}
	snapshot := handler.requests.snapshot()
	if len(snapshot.Requests) != 1 || snapshot.Requests[0].Status != http.StatusUnauthorized || snapshot.Requests[0].Route != "/props" {
		t.Fatalf("the refused call was recorded as %+v", snapshot.Requests)
	}
}

// TestRequestMonitorStreamsThroughTheRecorder serves a streamed completion over a real
// connection: the first event reaches the client while the request is still in flight, which the
// monitor then shows without a status, and the finished request holds the tokens it streamed.
func TestRequestMonitorStreamsThroughTheRecorder(t *testing.T) {
	t.Parallel()
	const tokenDelay = 150 * time.Millisecond
	handler := newTestHandler(t, &fakeGenerator{pieces: []string{"A", "B", "C", "D"}, tokenDelay: tokenDelay})
	defer handler.Close()
	server := httptest.NewServer(handler)
	defer server.Close()
	response, err := http.Post(server.URL+"/completion", "application/json", strings.NewReader(`{"prompt":"hi","n_predict":4,"stream":true}`))
	if err != nil {
		t.Fatal(err)
	}
	defer response.Body.Close()
	if response.StatusCode != http.StatusOK || !strings.HasPrefix(response.Header.Get("Content-Type"), "text/event-stream") {
		t.Fatalf("stream answered %d as %q", response.StatusCode, response.Header.Get("Content-Type"))
	}
	events := bufio.NewScanner(response.Body)
	for events.Scan() {
		if strings.HasPrefix(events.Text(), "data:") {
			break
		}
	}
	inFlight := monitorLog(t, handler).Requests
	if len(inFlight) != 1 || !inFlight[0].InFlight || inFlight[0].Status != 0 || inFlight[0].Route != "/completion" {
		t.Fatalf("with the first event read, the monitor shows %+v, want the stream in flight", inFlight)
	}
	for events.Scan() {
	}
	// Closing the server returns once its handlers have, which is when the monitor has the request's end.
	server.Close()
	finished := handler.requests.snapshot().Requests
	if len(finished) != 1 || finished[0].InFlight || finished[0].Status != http.StatusOK || finished[0].CompletionTokens != 4 || finished[0].DurationMS < float64(tokenDelay/time.Millisecond) {
		t.Fatalf("the finished stream = %+v, want 200 with four tokens that took at least a token's delay", finished)
	}
}

// TestRequestMonitorBoundsWhatItStores holds the ring to bytes as well as entries: the route is
// recorded before the caller is authenticated and a method may be any token up to the header
// limit, so a method the monitor does not know is stored as OTHER, not as sent; and the health
// probes a page repeats every few seconds are not recorded, while a call to any other route is.
func TestRequestMonitorBoundsWhatItStores(t *testing.T) {
	t.Parallel()
	handler := newTestHandler(t, &fakeGenerator{})
	defer handler.Close()
	long := strings.Repeat("X", 1<<20)
	for _, call := range []struct{ method, path string }{
		{long, "/v1/models"},
		{http.MethodGet, "/health"},
		{http.MethodGet, "/healthz"},
		{http.MethodGet, "/v1/health"},
		{http.MethodGet, "/v1/models"},
	} {
		serveTestRequest(handler, call.method, call.path, "")
	}
	log := monitorLog(t, handler)
	if log.Total != 2 || len(log.Requests) != 2 {
		t.Fatalf("the monitor holds %d of %d requests, want the two to /v1/models: %+v", len(log.Requests), log.Total, log.Requests)
	}
	for _, entry := range log.Requests {
		if entry.Route != "/v1/models" || len(entry.Method) > len(http.MethodOptions) {
			t.Errorf("the monitor kept %+v", entry)
		}
	}
	if log.Requests[1].Method != otherMethod || log.Requests[0].Method != http.MethodGet {
		t.Errorf("methods = %q then %q, want %q then %q", log.Requests[1].Method, log.Requests[0].Method, otherMethod, http.MethodGet)
	}
}
