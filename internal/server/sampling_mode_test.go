package server

import (
	"net/http"
	"testing"

	"overgo/internal/sampling"
)

// TestAModelIsServedTheSamplingOfItsMode holds the server's half of loop-proof/sampling-by-mode: a chat request with
// thinking off (chat_template_kwargs.enable_thinking false) starts from the model's thinking-off set and one thinking
// from its set to think with, a request's own temperature standing over either; a model with no thinking-off set is
// served its one set in both modes. The record's half is the test of this name in internal/modelrecipe.
func TestAModelIsServedTheSamplingOfItsMode(t *testing.T) {
	t.Parallel()
	generator := &fakeGenerator{}
	handler := newTestHandler(t, generator)
	thinking := handler.defaultSampling
	chat := func(fields string) sampling.Config {
		t.Helper()
		response := serveTestRequest(handler, http.MethodPost, "/v1/chat/completions", `{"messages":[{"role":"user","content":"hi"}],"max_tokens":1`+fields+`}`)
		if response.Code != http.StatusOK {
			t.Fatalf("chat status=%d body=%s", response.Code, response.Body.String())
		}
		generator.mu.Lock()
		defer generator.mu.Unlock()
		return generator.sampling
	}
	const thinkingOff = `,"chat_template_kwargs":{"enable_thinking":false}`
	if got := chat(thinkingOff); got.Temperature != thinking.Temperature || got.TopP != thinking.TopP || got.PresencePenalty != thinking.PresencePenalty {
		t.Fatalf("a model of one set served thinking off %+v; want its one set", got)
	}

	off := thinking
	off.Temperature, off.TopP, off.PresencePenalty, off.RepeatLastN = 0.7, 0.8, 1.5, -1
	handler.thinkingOffSampling = &off
	if got := chat(thinkingOff); got.Temperature != 0.7 || got.TopP != 0.8 || got.PresencePenalty != 1.5 {
		t.Fatalf("a request with thinking off was served %+v; want the thinking-off set", got)
	}
	if got := chat(""); got.Temperature != thinking.Temperature || got.TopP != thinking.TopP || got.PresencePenalty != thinking.PresencePenalty {
		t.Fatalf("a thinking request was served %+v; want the set to think with", got)
	}
	if got := chat(`,"temperature":0.25` + thinkingOff); got.Temperature != 0.25 || got.TopP != 0.8 {
		t.Fatalf("a request's own temperature with thinking off was served %+v; want 0.25 over the thinking-off set", got)
	}
}
