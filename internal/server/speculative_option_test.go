package server

import (
	"context"
	"encoding/json"
	"net/http"
	"slices"
	"strings"
	"testing"

	"overgo/internal/inference"
	"overgo/internal/tokenizer"
)

// draftGenerator is a served model with a draft head of three tokens: a
// speculative turn reports two verify spans, one accepting two drafts and
// one accepting all three, and records the options it ran under.
type draftGenerator struct {
	*recipeInspectorGenerator
	ran []inference.GenerateOptions
}

func (*draftGenerator) SpeculativeDraftTokens() int { return 3 }

func (generator *draftGenerator) Generate(_ context.Context, _ string, options inference.GenerateOptions) ([]tokenizer.TokenID, string, error) {
	generator.ran = append(generator.ran, options)
	if options.SpeculativeDecode && options.OnDraftSpan != nil {
		options.OnDraftSpan(3, 2)
		options.OnDraftSpan(3, 3)
	}
	ids := slices.Clone(options.PromptTokenIDs)
	for index, word := range []string{"drafted ", "reply"} {
		ids = append(ids, tokenizer.TokenID(100+index))
		if err := options.OnToken(inference.TokenEvent{ID: tokenizer.TokenID(100 + index), Piece: word, Index: index}); err != nil {
			return ids, "", err
		}
	}
	return ids, "", nil
}

// TestSpeculativeDecodingDeclaredOption: the capability document offers
// speculative decoding, with its draft span, only when the served model
// declares a draft head and context shifting is off; a speculative turn runs
// greedily on the device from an uncached prompt and reports its drafted
// and accepted tokens; a request without a draft head or under non-greedy
// sampling is refused with why.
func TestSpeculativeDecodingDeclaredOption(t *testing.T) {
	t.Parallel()
	offered := func(handler *Handler) *workspaceSpeculative {
		t.Helper()
		var manifest workspaceManifestResponse
		if err := json.Unmarshal(serveTestRequest(handler, http.MethodGet, "/workspace/manifest", "").Body.Bytes(), &manifest); err != nil || manifest.Model == nil {
			t.Fatalf("manifest = %+v, %v", manifest, err)
		}
		return manifest.Model.Speculative
	}
	turn := func(handler *Handler, temperature string) (int, string) {
		t.Helper()
		response := serveTestRequest(handler, http.MethodPost, "/v1/responses",
			`{"model":"`+testModelID+`","input":"hello","temperature":`+temperature+`,"max_output_tokens":4,"speculative":true}`)
		return response.Code, response.Body.String()
	}

	generator := &draftGenerator{recipeInspectorGenerator: responseRecipeGenerator(t, &fakeGenerator{})}
	handler := newTestHandlerWithRepository(t, generator)
	if declared := offered(handler); declared == nil || declared.DraftTokens != 3 {
		t.Fatalf("offered speculation = %+v", declared)
	}
	// The stored turn's inspection reports its drafted and accepted tokens.
	streamed := serveTestRequest(handler, http.MethodPost, "/v1/responses",
		`{"model":"`+testModelID+`","input":"hello","temperature":0,"max_output_tokens":4,"speculative":true,"stream":true,"store":true}`)
	if streamed.Code != http.StatusOK || !strings.Contains(streamed.Body.String(), `"draft_n":6,"draft_n_accepted":5`) {
		t.Fatalf("speculative turn = %d %s", streamed.Code, streamed.Body.String())
	}
	var inspected struct {
		Timings slotStatusTimings `json:"timings"`
	}
	inspection := serveTestRequest(handler, http.MethodGet, "/interactions/inspect?response=resp_1", "")
	if err := json.Unmarshal(inspection.Body.Bytes(), &inspected); inspection.Code != http.StatusOK || err != nil || inspected.Timings.DraftN != 6 || inspected.Timings.DraftNAccepted != 5 {
		t.Fatalf("inspection = %d %s", inspection.Code, inspection.Body.String())
	}
	ran := generator.ran[len(generator.ran)-1]
	if !ran.SpeculativeDecode || !ran.DeviceGreedy || ran.CachePrompt {
		t.Fatalf("speculative options = speculative %v greedy %v cached %v", ran.SpeculativeDecode, ran.DeviceGreedy, ran.CachePrompt)
	}
	if code, body := turn(handler, "0.8"); code != http.StatusBadRequest || !strings.Contains(body, "set temperature 0") {
		t.Fatalf("a sampled speculative turn = %d %s", code, body)
	}

	plain := newTestHandler(t, &fakeGenerator{})
	if declared := offered(plain); declared != nil {
		t.Fatalf("a model without a draft head offered %+v", declared)
	}
	if code, body := turn(plain, "0"); code != http.StatusBadRequest || !strings.Contains(body, "declares a draft head") {
		t.Fatalf("a speculative turn without a draft head = %d %s", code, body)
	}
	shifting := newTestHandler(t, &draftGenerator{recipeInspectorGenerator: responseRecipeGenerator(t, &fakeGenerator{})})
	shifting.config.ContextShift = true
	if declared := offered(shifting); declared != nil {
		t.Fatalf("context shifting offered %+v", declared)
	}
}
