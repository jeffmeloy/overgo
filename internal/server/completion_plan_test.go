package server

import (
	"encoding/json"
	"fmt"
	"net/http"
	"strings"
	"testing"
)

// TestCompletionsShareThePlan holds /v1/completions to the shared generation
// plan: every prompt of a list is admitted against the model context before
// any generates, and an oversized one refuses the whole request as the
// client's error without starting a stream, reserving an identifier or
// holding the session. Fitting prompts generate from their prepared ids.
func TestCompletionsShareThePlan(t *testing.T) {
	t.Parallel()
	for _, stream := range []bool{false, true} {
		t.Run(fmt.Sprintf("stream=%t", stream), func(t *testing.T) {
			base := &fakeGenerator{}
			generator := &protocolAdmissionGenerator{recipeInspectorGenerator: responseRecipeGenerator(t, base)}
			handler := newTestHandlerWithRepository(t, generator)
			prepared, err := handler.preparePrompt(t.Context(), nativePrompt{Text: "inspect"}, true)
			if err != nil {
				t.Fatal(err)
			}
			short, err := json.Marshal(prepared.TokenIDs)
			if err != nil {
				t.Fatal(err)
			}
			long, err := json.Marshal(append(prepared.TokenIDs, prepared.TokenIDs...))
			if err != nil {
				t.Fatal(err)
			}
			fits, over := uint32(len(prepared.TokenIDs)), uint32(2*len(prepared.TokenIDs))
			generator.limit = fits
			body := fmt.Sprintf(`{"prompt":[%s,%s],"max_tokens":1,"stream":%t}`, short, long, stream)
			response := serveTestRequest(handler, http.MethodPost, "/v1/completions", body)
			var failure apiErrorEnvelope
			if err := json.Unmarshal(response.Body.Bytes(), &failure); err != nil {
				t.Fatalf("refusal is not JSON: %s (%v)", response.Body, err)
			}
			if response.Code != http.StatusBadRequest || failure.Error.Code != "context_length_exceeded" ||
				!strings.Contains(failure.Error.Message, fmt.Sprintf("prompt has %d tokens", over)) {
				t.Fatalf("status=%d refusal=%s", response.Code, response.Body)
			}
			if response.Flushed || generator.calls.Load() != 0 || handler.nextID.Load() != 0 {
				t.Fatalf("refusal started execution: flushed=%t calls=%d", response.Flushed, generator.calls.Load())
			}
			lease, available := handler.acquireSession(-1)
			if !available {
				t.Fatal("refusal leaked the session lease")
			}
			handler.releaseSession(lease)

			generator.limit = over
			response = serveTestRequest(handler, http.MethodPost, "/v1/completions", body)
			if response.Code != http.StatusOK || generator.calls.Load() != 2 {
				t.Fatalf("fitting list: status=%d calls=%d body=%s", response.Code, generator.calls.Load(), response.Body)
			}
			if len(base.promptIDs) != int(over) {
				t.Fatalf("last prompt generated from %d ids, want its %d prepared ids", len(base.promptIDs), over)
			}
		})
	}
}
