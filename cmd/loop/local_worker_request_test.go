package main

import (
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"overgo/internal/testcheck"
)

// TestTheLocalWorkerDrawsGreedilyWithoutPenalties: the local worker asks the chat route for a greedy draw with no
// penalty, whatever the model's recipe adds. Qwen3.8's recipe adds a presence penalty of 1.5, which turns the model away
// from names code repeats and kept the server's sampler off its greedy draw: every token of a tool call asked the
// grammar about the whole vocabulary, 1.03 s a token with the GPU idle (2026-11-09, the server's /slots timings).
func TestTheLocalWorkerDrawsGreedilyWithoutPenalties(t *testing.T) {
	t.Parallel()
	var asked map[string]any
	server := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {
		body, err := io.ReadAll(request.Body)
		if err == nil {
			err = json.Unmarshal(body, &asked)
		}
		if err != nil {
			http.Error(writer, err.Error(), http.StatusBadRequest)
			return
		}
		_, _ = io.WriteString(writer, `{"choices": [{"message": {"role": "assistant", "content": "done"}}]}`)
	}))
	defer server.Close()
	var log strings.Builder
	work := testcheck.Must(workLocally(t.Context(), server.URL, "model", "the step", nil, &log))(t)
	if work.Answer != "done" {
		t.Fatalf("the worker answered %q, want the route's reply", work.Answer)
	}
	for field, want := range map[string]float64{"temperature": 0, "presence_penalty": 0, "frequency_penalty": 0, "repeat_penalty": 1} {
		if got, given := asked[field].(float64); !given || got != want {
			t.Errorf("the worker asked %s %v, want %v", field, asked[field], want)
		}
	}
}
