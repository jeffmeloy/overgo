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

// TestTheLocalWorkerThinksWhenAsked holds the worker's thinking to its setting (loop-proof/worker-thinking): asked to
// think, the worker asks the chat route for the model's thinking and for room for it, and otherwise for neither, as
// every request did before. The owner asked for the six replayed tasks to be run once with thinking on, where the model's
// ability is what failed them (2026-11-10).
func TestTheLocalWorkerThinksWhenAsked(t *testing.T) {
	t.Parallel()
	type asked struct {
		thinking bool
		tokens   int
	}
	for _, thinking := range []bool{false, true} {
		var got asked
		route := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			var request struct {
				MaxTokens int `json:"max_tokens"`
				Kwargs    struct {
					EnableThinking *bool `json:"enable_thinking"`
				} `json:"chat_template_kwargs"`
			}
			testcheck.NoError(t, json.NewDecoder(r.Body).Decode(&request))
			if request.Kwargs.EnableThinking == nil {
				t.Error("the worker's request names no enable_thinking")
			} else {
				got = asked{thinking: *request.Kwargs.EnableThinking, tokens: request.MaxTokens}
			}
			testcheck.Must(io.WriteString(w, `{"choices":[{"message":{"role":"assistant","content":"done"}}]}`))(t)
		}))
		var log strings.Builder
		testcheck.Must(workLocally(t.Context(), route.URL, workerModel{name: "local", thinking: thinking}, "the step", nil, &log))(t)
		route.Close()
		want := asked{thinking: false, tokens: localReplyTokens}
		if thinking {
			want = asked{thinking: true, tokens: localThinkingTokens}
		}
		if got != want {
			t.Errorf("a worker asked to think %t asked the route for %+v, want %+v", thinking, got, want)
		}
	}
}
