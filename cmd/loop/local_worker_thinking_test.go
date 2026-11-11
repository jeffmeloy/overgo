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

// TestAThinkingWorkerSendsADepthAndNoSamplingOfItsOwn holds the worker's settings to its mode (loop-proof/thinking-depth):
// thinking, it sends no temperature or penalty of its own, so the server serves the model's set for thinking, and asks
// for the shallowest depth, low; with thinking off it draws greedily with no penalty, as measured on 2026-11-09, and asks
// for no depth.
func TestAThinkingWorkerSendsADepthAndNoSamplingOfItsOwn(t *testing.T) {
	t.Parallel()
	for _, thinking := range []bool{false, true} {
		var asked map[string]any
		route := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			testcheck.NoError(t, json.NewDecoder(r.Body).Decode(&asked))
			testcheck.Must(io.WriteString(w, `{"choices":[{"message":{"role":"assistant","content":"done"}}]}`))(t)
		}))
		var log strings.Builder
		testcheck.Must(workLocally(t.Context(), route.URL, workerModel{name: "local", thinking: thinking}, "the step", nil, &log))(t)
		route.Close()
		kwargs, _ := asked["chat_template_kwargs"].(map[string]any)
		_, sampled := asked["temperature"]
		_, penalised := asked["presence_penalty"]
		depth, deep := kwargs["reasoning_effort"]
		if thinking && (sampled || penalised || depth != "low") {
			t.Errorf("a thinking worker asked for %v; want no temperature or penalty of its own and reasoning_effort low", asked)
		}
		if !thinking && (asked["temperature"] != float64(0) || asked["presence_penalty"] != float64(0) || deep) {
			t.Errorf("a worker with thinking off asked for %v; want temperature 0, no presence penalty and no depth", asked)
		}
	}
}

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
