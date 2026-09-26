package server

import (
	"context"
	"encoding/json"
	"maps"
	"net/http"
	"slices"
	"strings"
	"sync/atomic"
	"testing"

	"overgo/internal/gguf"
	"overgo/internal/inference"
	"overgo/internal/modelrecipe"
	"overgo/internal/overgodb"
	"overgo/internal/recipe"
	"overgo/internal/runrecord"
	"overgo/internal/servingtest"
	"overgo/internal/testutil"
	"overgo/internal/tokenizer"
)

// stoppingRunner: a real runner whose next armed generation blocks after its
// stopAt-th token until the request is cancelled, as a reader's Stop does.
type stoppingRunner struct {
	*inference.Runner
	stopAt  int
	armed   atomic.Bool
	reached chan struct{}
}

func (runner *stoppingRunner) Generate(ctx context.Context, prompt string, options inference.GenerateOptions) ([]tokenizer.TokenID, string, error) {
	if !runner.armed.CompareAndSwap(true, false) {
		return runner.Runner.Generate(ctx, prompt, options)
	}
	delivered, onToken := 0, options.OnToken
	options.OnToken = func(event inference.TokenEvent) error {
		if onToken != nil {
			if err := onToken(event); err != nil {
				return err
			}
		}
		if delivered++; delivered == runner.stopAt {
			close(runner.reached)
			<-ctx.Done()
			return context.Cause(ctx)
		}
		return nil
	}
	return runner.Runner.Generate(ctx, prompt, options)
}

// openSessionRunner serves the hermetic model on the host path with a chat
// template that joins the messages' text, with any adapters loaded.
func openSessionRunner(t *testing.T, contextLength uint32, adapters ...inference.LoRAConfig) *inference.Runner {
	t.Helper()
	path := testutil.HermeticLlamaGGUFWith(t, contextLength,
		testutil.GGUFScalar("tokenizer.chat_template", gguf.ValueTypeString, "{% for m in messages %}{{ m.content }}{% endfor %}"))
	loaded, err := servingtest.ResolveActiveGGUFWithPolicy(path, recipe.PlacementHybrid, modelrecipe.DecodeSessionCapacity, recipe.ResidencyHostReference)
	if err != nil {
		t.Fatal(err)
	}
	runner, err := inference.OpenWithProgram(t.Context(), &loaded, inference.OpenOptions{LoRAAdapters: adapters})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = runner.Close() })
	return runner
}

// TestResponsesContinueSavedSession: a turn stopped after three tokens keeps
// its token context, and continuing it produces the tokens and text of the
// uninterrupted greedy turn while recomputing one token; a continuation that
// asks another sampler, or that another model serves, is refused with why.
func TestResponsesContinueSavedSession(t *testing.T) {
	t.Parallel()
	store, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	runner := &stoppingRunner{Runner: openSessionRunner(t, 64), stopAt: 3, reached: make(chan struct{})}
	handler := newTestHandlerForRepository(t, store, runner)
	post := func(target *Handler, body map[string]any) (int, map[string]any) {
		t.Helper()
		body["model"] = testModelID
		data, err := json.Marshal(body)
		if err != nil {
			t.Fatal(err)
		}
		response := serveTestRequest(target, http.MethodPost, "/v1/responses", string(data))
		var decoded map[string]any
		_ = json.Unmarshal(response.Body.Bytes(), &decoded)
		return response.Code, decoded
	}
	stored := func(id string) (runrecord.InteractionContext, string) {
		t.Helper()
		interaction, found, err := runrecord.ResolveInteraction(t.Context(), store, id)
		if err != nil || !found || !interaction.Context.Valid() {
			t.Fatalf("interaction %s = %+v found=%v err=%v", id, interaction, found, err)
		}
		tokens, err := runrecord.RequireInteractionContext(t.Context(), store, interaction.Context)
		if err != nil {
			t.Fatal(err)
		}
		transcript, err := runrecord.RequireInteractionTranscript(t.Context(), store, interaction.Message)
		if err != nil {
			t.Fatal(err)
		}
		return tokens, transcript.Messages[len(transcript.Messages)-1].Content
	}
	turn := map[string]any{"input": "ab", "temperature": 0, "ignore_eos": true, "max_output_tokens": 8, "store": true}

	// The uninterrupted greedy turn reaches its output limit and keeps its context.
	code, reference := post(handler, maps.Clone(turn))
	if code != http.StatusOK || reference["status"] != "incomplete" {
		t.Fatalf("reference status=%d %v", code, reference)
	}
	referenceTokens, referenceText := stored(reference["id"].(string))

	// The same turn, stopped after three tokens.
	runner.armed.Store(true)
	streamed := make(chan struct{})
	go func() {
		defer close(streamed)
		stopped := maps.Clone(turn)
		stopped["stream"], stopped["model"] = true, testModelID
		data, _ := json.Marshal(stopped)
		serveTestRequest(handler, http.MethodPost, "/v1/responses", string(data))
	}()
	<-runner.reached
	stoppedID := runningResponse(t, handler)
	if cancelled := serveTestRequest(handler, http.MethodPost, "/interactions/cancel",
		`{"response":"`+stoppedID+`","model":"`+testModelID+`"}`); cancelled.Code != http.StatusOK {
		t.Fatalf("cancel status=%d body=%s", cancelled.Code, cancelled.Body.String())
	}
	<-streamed
	stoppedTokens, stoppedText := stored(stoppedID)
	if len(stoppedTokens.Tokens) != len(referenceTokens.Tokens)-5 || !strings.HasPrefix(referenceText, stoppedText) {
		t.Fatalf("stopped %v %q of reference %v %q", stoppedTokens.Tokens, stoppedText, referenceTokens.Tokens, referenceText)
	}

	// A continuation asking another sampler is refused with both.
	code, refused := post(handler, map[string]any{"previous_response_id": stoppedID, "continue": true, "temperature": 0.7})
	if code != http.StatusConflict || !strings.Contains(refused["error"].(map[string]any)["message"].(string), "sampled with") {
		t.Fatalf("sampler mismatch status=%d %v", code, refused)
	}
	// A server serving another model refuses the session, naming both models.
	other := newTestHandlerForRepository(t, store, openSessionRunner(t, 128))
	code, refused = post(other, map[string]any{"previous_response_id": stoppedID, "continue": true})
	if code != http.StatusConflict || !strings.Contains(refused["error"].(map[string]any)["message"].(string), "was generated by model") {
		t.Fatalf("model mismatch status=%d %v", code, refused)
	}

	// Continued, it is the uninterrupted turn, with one token recomputed.
	code, continued := post(handler, map[string]any{"previous_response_id": stoppedID, "continue": true, "max_output_tokens": 5})
	if code != http.StatusOK {
		t.Fatalf("continue status=%d %v", code, continued)
	}
	usage := continued["usage"].(map[string]any)
	cached := usage["input_tokens_details"].(map[string]any)["cached_tokens"].(float64)
	if input := usage["input_tokens"].(float64); cached != input-1 || int(input) != len(stoppedTokens.Tokens) {
		t.Fatalf("continuation usage %v: re-evaluated its context of %d tokens", usage, len(stoppedTokens.Tokens))
	}
	continuedTokens, continuedText := stored(continued["id"].(string))
	if !slices.Equal(continuedTokens.Tokens, referenceTokens.Tokens) || continuedText != referenceText {
		t.Fatalf("continued %v %q, uninterrupted %v %q", continuedTokens.Tokens, continuedText, referenceTokens.Tokens, referenceText)
	}
}

// runningResponse names the one response the handler is still generating.
func runningResponse(t *testing.T, handler *Handler) string {
	t.Helper()
	handler.inflight.mu.Lock()
	defer handler.inflight.mu.Unlock()
	for id, turn := range handler.inflight.turns {
		if _, done, _, _, _ := turn.snapshot(); !done {
			return id
		}
	}
	t.Fatal("no response is generating")
	return ""
}
