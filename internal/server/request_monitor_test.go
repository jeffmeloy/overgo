package server

import (
	"bufio"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"
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
		serveTestRequest(handler, http.MethodGet, "/health", "")
	}
	if full := monitorLog(t, handler); len(full.Requests) != requestMonitorCapacity || full.Limit != requestMonitorCapacity || full.Requests[0].Route != "/health" {
		t.Fatalf("the monitor holds %d requests, want the newest %d", len(full.Requests), requestMonitorCapacity)
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
