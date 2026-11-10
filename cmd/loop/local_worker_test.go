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

// TestTheLocalWorkerRunsEachToolCallItsModelMakes holds the local worker's loop to the chat route's protocol against a
// scripted route: each tool call a reply makes is run and its result sent back under the call's id, a call with
// arguments that are not an object is answered with why, and the worker ends at the first reply with no call. The
// tools are offered as functions with their arguments' schema. Every request asks for a greedy draw with no penalty,
// whatever the model's recipe adds: Qwen3.8's recipe adds a presence penalty of 1.5, which kept the server's sampler off
// its greedy draw, every token of a tool call asking the grammar about the whole vocabulary, 1.03 s a token with the
// GPU idle (2026-11-09, the server's /slots timings).
func TestTheLocalWorkerRunsEachToolCallItsModelMakes(t *testing.T) {
	t.Parallel()
	replies := []string{
		`{"role":"assistant","content":"","tool_calls":[{"id":"a","type":"function","function":{"name":"echo","arguments":"{\"text\":\"one\"}"}},{"id":"b","type":"function","function":{"name":"echo","arguments":"not json"}}]}`,
		`{"role":"assistant","content":"done: echoed one"}`,
	}
	var asked [][]chatMessage
	route := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var request struct {
			Model    string           `json:"model"`
			Messages []chatMessage    `json:"messages"`
			Tools    []map[string]any `json:"tools"`
			// Pointers, so a field the request leaves out fails as a wrong value does.
			Temperature      *float64 `json:"temperature"`
			PresencePenalty  *float64 `json:"presence_penalty"`
			FrequencyPenalty *float64 `json:"frequency_penalty"`
			RepeatPenalty    *float64 `json:"repeat_penalty"`
		}
		testcheck.NoError(t, json.NewDecoder(r.Body).Decode(&request))
		if r.URL.Path != "/v1/chat/completions" || request.Model != "local" || len(request.Tools) != 1 || request.Tools[0]["type"] != "function" {
			t.Errorf("the worker asked %s for %q with tools %v", r.URL.Path, request.Model, request.Tools)
		}
		for field, asked := range map[string][2]*float64{
			"temperature": {request.Temperature, new(float64)}, "presence_penalty": {request.PresencePenalty, new(float64)},
			"frequency_penalty": {request.FrequencyPenalty, new(float64)}, "repeat_penalty": {request.RepeatPenalty, func() *float64 { one := 1.0; return &one }()},
		} {
			if asked[0] == nil || *asked[0] != *asked[1] {
				t.Errorf("the worker asked %s %v, want %v", field, asked[0], *asked[1])
			}
		}
		asked = append(asked, request.Messages)
		testcheck.Must(io.WriteString(w, `{"choices":[{"message":`+replies[len(asked)-1]+`}]}`))(t)
	}))
	t.Cleanup(route.Close)
	echo := workerTool{name: "echo", description: "Echo the text.", arguments: []string{"text"}, run: func(arguments map[string]string) (string, error) {
		return "echoed " + arguments["text"], nil
	}}
	var log strings.Builder
	work := testcheck.Must(workLocally(t.Context(), route.URL, "local", "the step", []workerTool{echo}, &log))(t)
	if work.Turns != len(replies) || work.Calls != 2 || work.Answer != "done: echoed one" || len(asked) != len(replies) {
		t.Fatalf("the worker worked %+v over %d requests, want two replies, two calls and the last reply's answer", work, len(asked))
	}
	sent := asked[1][len(asked[1])-2:]
	if sent[0].Role != "tool" || sent[0].ToolCallID != "a" || sent[0].Content != "echoed one" ||
		sent[1].ToolCallID != "b" || !strings.Contains(sent[1].Content, "not one JSON object") {
		t.Fatalf("the worker sent back %+v, want each call's result under its id", sent)
	}
	t.Logf("the worker's log:\n%s", log.String())
}
