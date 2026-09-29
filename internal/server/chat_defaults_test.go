package server

import (
	"context"
	"encoding/json"
	"maps"
	"net/http"
	"slices"
	"strconv"
	"testing"

	"overgo/internal/inference"
	"overgo/internal/overgodb"
	"overgo/internal/runrecord"
	"overgo/internal/tokenizer"
)

// TestChatDefaults: the capability document offers the chat the output the
// server accepts, and a greedy turn cut at its output limit, continued with
// the rest of the budget, is the uninterrupted turn token for token.
func TestChatDefaults(t *testing.T) {
	t.Parallel()
	store, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	handler := newTestHandlerForRepository(t, store, openSessionRunner(t, 64))
	var manifest workspaceManifestResponse
	if err := json.Unmarshal(serveTestRequest(handler, http.MethodGet, "/workspace/manifest", "").Body.Bytes(), &manifest); err != nil || manifest.Model == nil {
		t.Fatalf("manifest = %+v, %v", manifest, err)
	}
	if manifest.Model.MaxOutputTokens != handler.config.MaxTokens {
		t.Fatalf("the chat is offered %d output tokens; the server accepts %d", manifest.Model.MaxOutputTokens, handler.config.MaxTokens)
	}

	post := func(body map[string]any) map[string]any {
		t.Helper()
		body["model"] = testModelID
		data, err := json.Marshal(body)
		if err != nil {
			t.Fatal(err)
		}
		response := serveTestRequest(handler, http.MethodPost, "/v1/responses", string(data))
		var decoded map[string]any
		if err := json.Unmarshal(response.Body.Bytes(), &decoded); err != nil || response.Code != http.StatusOK {
			t.Fatalf("status %d %s", response.Code, response.Body.String())
		}
		return decoded
	}
	stored := func(id string) ([]uint32, string) {
		t.Helper()
		interaction, found, err := runrecord.ResolveInteraction(t.Context(), store, id)
		if err != nil || !found {
			t.Fatalf("interaction %s found=%v err=%v", id, found, err)
		}
		transcript, err := runrecord.RequireInteractionTranscript(t.Context(), store, interaction.Message)
		if err != nil {
			t.Fatal(err)
		}
		ids := []uint32{}
		if interaction.Context.Valid() {
			tokens, err := runrecord.RequireInteractionContext(t.Context(), store, interaction.Context)
			if err != nil {
				t.Fatal(err)
			}
			for _, token := range tokens.Tokens {
				ids = append(ids, uint32(token))
			}
		}
		return ids, transcript.Messages[len(transcript.Messages)-1].Content
	}
	turn := map[string]any{"input": "ab", "temperature": 0, "ignore_eos": true, "store": true}

	whole := maps.Clone(turn)
	whole["max_output_tokens"] = 8
	uninterrupted := post(whole)
	wholeTokens, wholeText := stored(uninterrupted["id"].(string))

	cut := maps.Clone(turn)
	cut["max_output_tokens"] = 3
	limited := post(cut)
	if limited["status"] != "incomplete" {
		t.Fatalf("a turn at its output limit ended %v", limited["status"])
	}
	continued := post(map[string]any{"previous_response_id": limited["id"], "continue": true, "max_output_tokens": 5})
	continuedTokens, continuedText := stored(continued["id"].(string))
	if !slices.Equal(continuedTokens, wholeTokens) || continuedText != wholeText {
		t.Fatalf("continued %v %q, uninterrupted %v %q", continuedTokens, continuedText, wholeTokens, wholeText)
	}

	// A turn the model ends on the last token its limit allows ended itself:
	// nothing was cut off, and there is nothing to continue.
	ending := newTestHandler(t, endingGenerator{&fakeGenerator{pieces: []string{"A", "B"}}})
	for limit, want := range map[int]string{2: "completed", 1: "incomplete"} {
		response := serveTestRequest(ending, http.MethodPost, "/v1/responses", `{"model":"`+testModelID+`","input":"ab","max_output_tokens":`+strconv.Itoa(limit)+`}`)
		var decoded map[string]any
		if err := json.Unmarshal(response.Body.Bytes(), &decoded); err != nil || decoded["status"] != want {
			t.Fatalf("a turn ending on its second token under a %d-token limit = %d %s, want %s", limit, response.Code, response.Body.String(), want)
		}
	}
}

// endingGenerator is a fake whose last generated token ends its turn.
type endingGenerator struct{ *fakeGenerator }

func (generator endingGenerator) Generate(ctx context.Context, prompt string, options inference.GenerateOptions) ([]tokenizer.TokenID, string, error) {
	ids, text, err := generator.fakeGenerator.Generate(ctx, prompt, options)
	if err == nil && len(ids) != 0 && options.MaxNewTokens >= len(generator.pieces) {
		ids[len(ids)-1] = generator.SamplingEOGTokens()[0]
	}
	return ids, text, err
}
