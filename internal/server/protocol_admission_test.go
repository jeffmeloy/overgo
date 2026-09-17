package server

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"image"
	"image/png"
	"net/http"
	"strings"
	"sync/atomic"
	"testing"

	"overgo/internal/inference"
	"overgo/internal/tokenizer"
)

type protocolAdmissionGenerator struct {
	*recipeInspectorGenerator
	limit uint32
	calls atomic.Int32
}

func (g *protocolAdmissionGenerator) ModelProperties() inference.ModelProperties {
	properties := g.recipeInspectorGenerator.ModelProperties()
	properties.ContextLength = g.limit
	return properties
}

func (g *protocolAdmissionGenerator) Generate(ctx context.Context, prompt string, options inference.GenerateOptions) ([]tokenizer.TokenID, string, error) {
	g.calls.Add(1)
	return g.recipeInspectorGenerator.Generate(ctx, prompt, options)
}

func TestProtocolContextOverflow(t *testing.T) {
	var encoded bytes.Buffer
	if err := png.Encode(&encoded, image.NewRGBA(image.Rect(0, 0, 1, 1))); err != nil {
		t.Fatal(err)
	}
	imageData := base64.StdEncoding.EncodeToString(encoded.Bytes())
	dataURI := "data:image/png;base64," + imageData
	for _, media := range []bool{false, true} {
		chatContent, responseContent, anthropicContent := `"inspect"`, `"inspect"`, `"inspect"`
		if media {
			chatContent = `[{"type":"text","text":"inspect"},{"type":"image_url","image_url":{"url":"` + dataURI + `"}}]`
			responseContent = `[{"type":"input_text","text":"inspect"},{"type":"input_image","image_url":"` + dataURI + `"}]`
			anthropicContent = `[{"type":"text","text":"inspect"},{"type":"image","source":{"type":"base64","media_type":"image/png","data":"` + imageData + `"}}]`
		}
		for _, protocol := range []struct {
			path, input, budget string
		}{
			{"/v1/chat/completions", `"messages":[{"role":"user","content":` + chatContent + `}]`, "max_tokens"},
			{"/v1/responses", `"input":[{"role":"user","content":` + responseContent + `}]`, "max_output_tokens"},
			{"/v1/messages", `"messages":[{"role":"user","content":` + anthropicContent + `}]`, "max_tokens"},
		} {
			for _, stream := range []bool{false, true} {
				for _, shift := range []bool{false, true} {
					t.Run(fmt.Sprintf("%s/media=%t/stream=%t/shift=%t", protocol.path, media, stream, shift), func(t *testing.T) {
						base := &fakeGenerator{}
						generator := &protocolAdmissionGenerator{recipeInspectorGenerator: responseRecipeGenerator(t, base)}
						handler := newTestHandlerWithRepository(t, generator)
						handler.config.ContextShift = shift
						handler.config.ImageProjector = &fakeQwen3VLProjector{}
						// The text fixture tokenizes to BOS, text, EOS; the image
						// projector adds two embedding positions between BOS/EOS.
						promptTokens := uint32(3)
						if media {
							promptTokens = 4
						}
						generator.limit = promptTokens - 1
						body := fmt.Sprintf(`{%s,"%s":1,"stream":%t}`, protocol.input, protocol.budget, stream)
						response := serveTestRequest(handler, http.MethodPost, protocol.path, body)
						var failure apiErrorEnvelope
						if err := json.Unmarshal(response.Body.Bytes(), &failure); err != nil {
							t.Fatalf("refusal is not JSON: %s (%v)", response.Body, err)
						}
						if response.Code != http.StatusBadRequest || failure.Error.Type != "invalid_request_error" ||
							failure.Error.Code != "context_length_exceeded" ||
							!strings.Contains(failure.Error.Message, fmt.Sprintf("prompt has %d tokens", promptTokens)) ||
							!strings.Contains(failure.Error.Message, fmt.Sprintf("context length of %d tokens", generator.limit)) {
							t.Fatalf("status=%d refusal=%s", response.Code, response.Body)
						}
						if response.Flushed || response.Header().Get("Content-Type") != "application/json" || generator.calls.Load() != 0 {
							t.Fatalf("refusal started execution: flushed=%t headers=%v calls=%d", response.Flushed, response.Header(), generator.calls.Load())
						}
						if handler.nextID.Load() != 0 {
							t.Fatal("refused prompt reserved a response identifier")
						}
						listed := serveTestRequest(handler, http.MethodGet, "/interactions", "")
						var conversations conversationListResponse
						if err := json.Unmarshal(listed.Body.Bytes(), &conversations); err != nil || listed.Code != http.StatusOK || len(conversations.Conversations) != 0 {
							t.Fatalf("refused prompt changed durable history: %s (%v)", listed.Body, err)
						}
						lease, available := handler.acquireSession(-1)
						if !available {
							t.Fatal("refusal leaked the session lease")
						}
						handler.releaseSession(lease)

						// Exact-capacity prompts still fit, including cached media
						// prefixes. Admission must not count those tokens twice.
						generator.limit = promptTokens
						base.promptCached = int(promptTokens)
						response = serveTestRequest(handler, http.MethodPost, protocol.path, body)
						if response.Code != http.StatusOK || generator.calls.Load() != 1 {
							t.Fatalf("fitting request after refusal: status=%d calls=%d body=%s", response.Code, generator.calls.Load(), response.Body)
						}
						if base.contextShift != shift || len(base.promptIDs) != int(promptTokens) || (media && (base.projectedInputs == nil || !base.cachePrompt)) {
							t.Fatalf("generation options changed: shift=%t tokens=%v projected=%v cache=%t", base.contextShift, base.promptIDs, base.projectedInputs, base.cachePrompt)
						}
					})
				}
			}
		}
	}
	t.Run("unknown-context", func(t *testing.T) {
		generator := &protocolAdmissionGenerator{recipeInspectorGenerator: responseRecipeGenerator(t, &fakeGenerator{})}
		handler := newTestHandlerWithRepository(t, generator)
		response := serveTestRequest(handler, http.MethodPost, "/v1/responses", `{"input":"inspect","max_output_tokens":1}`)
		if response.Code != http.StatusOK || generator.calls.Load() != 1 {
			t.Fatalf("unknown context was treated as zero capacity: status=%d body=%s", response.Code, response.Body)
		}
	})
}
