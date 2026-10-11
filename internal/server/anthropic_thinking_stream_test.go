package server

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"net/http/httptest"
	"runtime"
	"strconv"
	"strings"
	"testing"

	"overgo/internal/inference"
)

// parseMeter is a reasoning generator that notes how many bytes the process
// had allocated at the moment a turn's output reached its parser.
type parseMeter struct {
	*reasoningGenerator
	allocated uint64
}

// ParseChatOutput reads the allocation counter before it parses, so the
// reading leaves out whatever a route allocates after its output is whole.
func (meter *parseMeter) ParseChatOutput(
	output string,
	tools []inference.ChatTool,
) (inference.ChatMessage, error) {
	var stats runtime.MemStats
	runtime.ReadMemStats(&stats)
	meter.allocated = stats.TotalAlloc
	return meter.reasoningGenerator.ParseChatOutput(output, tools)
}

// cancellableThinker is a generator a test can cancel part-way whose output
// still parses as a thinking turn, so a route that went on after the
// cancellation would have a turn to publish.
type cancellableThinker struct{ *declaredReasoningGenerator }

// ParseChatOutput splits thinking from the answer as reasoningGenerator does;
// the parser the embedded generator inherits never yields thinking.
func (thinker cancellableThinker) ParseChatOutput(
	output string,
	tools []inference.ChatTool,
) (inference.ChatMessage, error) {
	return (&reasoningGenerator{fakeGenerator: thinker.fakeGenerator}).ParseChatOutput(output, tools)
}

// newThinkingHandler builds a handler whose token limit leaves room for the
// smallest thinking budget the Anthropic route accepts.
func newThinkingHandler(t testing.TB, generator Generator) *Handler {
	t.Helper()
	handler, err := New(Config{
		ModelID: testModelID, MaxTokens: 2048,
		DefaultTemperature: testNeutralTemperature, DefaultTopP: testFullTopP,
	}, generator)
	if err != nil {
		t.Fatal(err)
	}
	return handler
}

// thinkingRequest is a thinking turn's body, streamed or whole, with any
// further fields the caller appends.
func thinkingRequest(stream bool, fields string) string {
	return fmt.Sprintf(
		`{"model":"test-model","max_tokens":1025,"stream":%v,"thinking":{"type":"enabled","budget_tokens":1024},"messages":[{"role":"user","content":"inspect"}]%s}`,
		stream, fields,
	)
}

// TestStreamedThinkingHoldsItsOutputOnce holds a streamed thinking turn to the
// storage a whole one uses. The generation pump already keeps every piece it
// releases, so a route that also collects the pieces itself has allocated one
// more copy of the output by the time it parses. The two routes are compared
// at that moment, so the bound needs no measured constant: a second copy shows
// as about one output's length, and the margin is half of that.
func TestStreamedThinkingHoldsItsOutputOnce(t *testing.T) {
	// Serial: it reads the process's allocation counter, which a parallel neighbour would add to.
	payload := strings.Repeat("m", 1<<20)
	moved := func(stream bool) uint64 {
		best := ^uint64(0)
		for range 3 {
			meter := &parseMeter{reasoningGenerator: &reasoningGenerator{fakeGenerator: &fakeGenerator{
				pieces: []string{"check facts</think>", payload},
			}}}
			handler := newThinkingHandler(t, meter)
			body := thinkingRequest(stream, "")
			var before runtime.MemStats
			runtime.GC()
			runtime.ReadMemStats(&before)
			response := serveTestRequest(handler, http.MethodPost, "/v1/messages", body)
			// A turn that never reached the parser, or lost its output, earns no reading.
			if response.Code != http.StatusOK || meter.allocated == 0 || response.Body.Len() <= len(payload) {
				t.Fatalf("stream=%v status=%d parsed at %d, body of %d bytes", stream, response.Code, meter.allocated, response.Body.Len())
			}
			if allocated := meter.allocated - before.TotalAlloc; allocated < best {
				best = allocated
			}
		}
		return best
	}
	streamed, whole := moved(true), moved(false)
	t.Logf("allocated when the output reached the parser: streamed %d bytes, buffered %d", streamed, whole)
	if streamed >= whole+uint64(len(payload)/2) {
		t.Fatalf("a streamed thinking turn had allocated %d bytes when its output reached the parser, a buffered one %d: a second copy of a %d-byte output", streamed, whole, len(payload))
	}
}

// TestStreamedThinkingStopsAndFlushesAsABufferedTurnDoes holds a thinking turn
// to one answer whether it is streamed or sent whole: a matched stop sequence
// hides itself and everything after it, a held prefix that never completed a
// stop is published at the end, and the counts and stop reason agree. A
// streamed turn cancelled part-way publishes no turn at all.
func TestStreamedThinkingStopsAndFlushesAsABufferedTurnDoes(t *testing.T) {
	t.Parallel()
	for _, test := range []struct {
		name     string
		pieces   []string
		text     string
		reason   string
		sequence string
		tokens   int
	}{
		{
			name:   "matched stop stays hidden",
			pieces: []string{"check facts</think>", "final answerHA", "LTleak", "more"},
			text:   "final answer", reason: "stop_sequence", sequence: "HALT", tokens: 3,
		},
		{
			name:   "unmatched tail is kept",
			pieces: []string{"check facts</think>", "final answerHA"},
			text:   "final answerHA", reason: "end_turn", tokens: 2,
		},
	} {
		for _, stream := range []bool{false, true} {
			t.Run(fmt.Sprintf("%s/stream=%v", test.name, stream), func(t *testing.T) {
				t.Parallel()
				handler := newThinkingHandler(t, &reasoningGenerator{fakeGenerator: &fakeGenerator{pieces: test.pieces}})
				response := serveTestRequest(
					handler, http.MethodPost, "/v1/messages", thinkingRequest(stream, `,"stop_sequences":["HALT"]`),
				)
				body := response.Body.String()
				if response.Code != http.StatusOK {
					t.Fatalf("status = %d body=%s", response.Code, body)
				}
				if !stream {
					var result anthropicResponse
					if err := json.Unmarshal(response.Body.Bytes(), &result); err != nil {
						t.Fatal(err)
					}
					sequence := ""
					if result.StopSequence != nil {
						sequence = *result.StopSequence
					}
					if len(result.Content) != 2 || result.Content[0].Thinking != "check facts" ||
						result.Content[1].Text != test.text || result.StopReason != test.reason ||
						sequence != test.sequence || result.Usage.OutputTokens != test.tokens {
						t.Fatalf("whole turn = %s", body)
					}
					return
				}
				sequence := "null"
				if test.sequence != "" {
					sequence = strconv.Quote(test.sequence)
				}
				for _, want := range []string{
					`"delta":{"thinking":"check facts","type":"thinking_delta"}`,
					`"delta":{"text":"` + test.text + `","type":"text_delta"}`,
					`"delta":{"stop_reason":"` + test.reason + `","stop_sequence":` + sequence + `}`,
					fmt.Sprintf(`"usage":{"output_tokens":%d}`, test.tokens),
				} {
					if !strings.Contains(body, want) {
						t.Fatalf("streamed turn lacks %s: %s", want, body)
					}
				}
				// One delta of each kind, each matched whole above, is all the text the
				// turn published. The body is not searched for the hidden words: the
				// signature is random and could spell one.
				if strings.Count(body, `"type":"text_delta"`) != 1 || strings.Count(body, `"type":"thinking_delta"`) != 1 {
					t.Fatalf("streamed turn published more than its two blocks: %s", body)
				}
			})
		}
	}
	t.Run("a cancelled generation publishes no turn", func(t *testing.T) {
		t.Parallel()
		// The pieces parse as a whole thinking turn, so only the route stopping
		// at the failed generation keeps that turn's blocks out of the body.
		generator := &declaredReasoningGenerator{
			fakeGenerator: &fakeGenerator{pieces: []string{"check facts</think>", "final answer"}}, prompt: "assistant",
		}
		ctx, cancel := context.WithCancelCause(t.Context())
		defer cancel(nil)
		generator.afterToken = func() { cancel(context.Canceled) }
		response := httptest.NewRecorder()
		newThinkingHandler(t, cancellableThinker{generator}).ServeHTTP(response, httptest.NewRequest(
			http.MethodPost, "/v1/messages", strings.NewReader(thinkingRequest(true, "")),
		).WithContext(ctx))
		body := response.Body.String()
		if !errors.Is(generator.lastError, context.Canceled) {
			t.Fatalf("generation ended with %v, not the cancellation: %s", generator.lastError, body)
		}
		for _, event := range []string{"event: content_block_start", "event: message_delta", "event: message_stop"} {
			if strings.Contains(body, event) {
				t.Fatalf("a cancelled turn published %q: %s", event, body)
			}
		}
	})
}
