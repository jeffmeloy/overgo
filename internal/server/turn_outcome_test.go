package server

import (
	"encoding/json"
	"net/http"
	"strings"
	"testing"
)

// TestProtocolsEncodeOneOutcome holds every text protocol to one outcome for
// one generation: the same turn, ended by a requested stop sequence or by
// the output limit, reports the same output count on every protocol (the
// tokens kept through the stop, never those generated past it), and each
// protocol names the same ending its own way. Anthropic names a matched
// stop sequence stop_sequence, whole or streamed.
func TestProtocolsEncodeOneOutcome(t *testing.T) {
	t.Parallel()
	type outcome struct {
		ending string
		output int
	}
	read := func(t *testing.T, path, body string) outcome {
		t.Helper()
		answer := serveTestRequest(newTestHandler(t, &fakeGenerator{pieces: []string{"A", "B", "C"}}), http.MethodPost, path, body)
		if answer.Code != http.StatusOK {
			t.Fatalf("%s status=%d body=%s", path, answer.Code, answer.Body)
		}
		if strings.Contains(body, `"stream":true`) {
			for line := range strings.SplitSeq(answer.Body.String(), "\n") {
				payload, found := strings.CutPrefix(line, "data: ")
				if !found || !strings.Contains(payload, `"message_delta"`) {
					continue
				}
				var delta struct {
					Delta struct {
						StopReason   string  `json:"stop_reason"`
						StopSequence *string `json:"stop_sequence"`
					} `json:"delta"`
					Usage struct {
						OutputTokens int `json:"output_tokens"`
					} `json:"usage"`
				}
				if err := json.Unmarshal([]byte(payload), &delta); err != nil {
					t.Fatal(err)
				}
				ending := delta.Delta.StopReason
				if delta.Delta.StopSequence != nil {
					ending += ":" + *delta.Delta.StopSequence
				}
				return outcome{ending, delta.Usage.OutputTokens}
			}
			t.Fatalf("no message_delta in %s", answer.Body)
		}
		var decoded struct {
			Choices []struct {
				FinishReason string `json:"finish_reason"`
			} `json:"choices"`
			Status       string  `json:"status"`
			StopReason   string  `json:"stop_reason"`
			StopSequence *string `json:"stop_sequence"`
			Usage        struct {
				CompletionTokens int `json:"completion_tokens"`
				OutputTokens     int `json:"output_tokens"`
			} `json:"usage"`
		}
		if err := json.Unmarshal(answer.Body.Bytes(), &decoded); err != nil {
			t.Fatal(err)
		}
		switch {
		case len(decoded.Choices) != 0:
			return outcome{decoded.Choices[0].FinishReason, decoded.Usage.CompletionTokens}
		case decoded.Status != "":
			return outcome{decoded.Status, decoded.Usage.OutputTokens}
		}
		ending := decoded.StopReason
		if decoded.StopSequence != nil {
			ending += ":" + *decoded.StopSequence
		}
		return outcome{ending, decoded.Usage.OutputTokens}
	}
	for _, turn := range []struct {
		name, chatLimit, responseLimit string
		want                           map[string]outcome
	}{
		// The fake generates past the stop, as a runner may within one
		// token; the count keeps only what the stop kept.
		{"stop sequence", `"max_tokens":8,"stop":"B"`, `"max_output_tokens":8,"stop":"B"`, map[string]outcome{
			"chat": {"stop", 2}, "completions": {"stop", 2}, "responses": {"completed", 2},
			"anthropic": {"stop_sequence:B", 2}, "anthropic stream": {"stop_sequence:B", 2},
		}},
		{"output limit", `"max_tokens":2`, `"max_output_tokens":2`, map[string]outcome{
			"chat": {"length", 2}, "completions": {"length", 2}, "responses": {"incomplete", 2},
			"anthropic": {"max_tokens", 2}, "anthropic stream": {"max_tokens", 2},
		}},
	} {
		anthropicLimit := strings.Replace(turn.chatLimit, `"stop":"B"`, `"stop_sequences":["B"]`, 1)
		requests := map[string][2]string{
			"chat":             {"/v1/chat/completions", `{` + turn.chatLimit + `,"messages":[{"role":"user","content":"hi"}]}`},
			"completions":      {"/v1/completions", `{` + turn.chatLimit + `,"prompt":"hi"}`},
			"responses":        {"/v1/responses", `{` + turn.responseLimit + `,"store":false,"input":"hi"}`},
			"anthropic":        {"/v1/messages", `{` + anthropicLimit + `,"messages":[{"role":"user","content":"hi"}]}`},
			"anthropic stream": {"/v1/messages", `{` + anthropicLimit + `,"stream":true,"messages":[{"role":"user","content":"hi"}]}`},
		}
		for protocol, want := range turn.want {
			t.Run(turn.name+"/"+protocol, func(t *testing.T) {
				t.Parallel()
				request := requests[protocol]
				if got := read(t, request[0], request[1]); got != want {
					t.Fatalf("outcome = %+v, want %+v", got, want)
				}
			})
		}
	}
}
