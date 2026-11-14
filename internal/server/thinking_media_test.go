package server

import (
	"bytes"
	"encoding/base64"
	"encoding/json"
	"image"
	"image/png"
	"net/http"
	"net/http/httptest"
	"testing"

	"overgo/internal/inference"
	"overgo/internal/testcheck"
)

// historyOptionsGenerator formats a chat asked with template options as historyGenerator formats any chat, its media
// markers kept, so a history asked with thinking off reaches the projector.
type historyOptionsGenerator struct{ *historyGenerator }

func (g historyOptionsGenerator) FormatChatWithOptions(messages []inference.ChatMessage, _ inference.ChatFormatOptions) (string, error) {
	return g.FormatChat(messages)
}

// TestAMediaHistoryKeepsTheThinkingModeAsked holds a chat's sampling to the mode its caller chose whatever its history
// holds (thinking-mode/media, SQA-383): a history of three messages whose first carries an image, asked with thinking
// off, is sampled from the model's thinking-off set, as the same history without the image is. The media history's
// prompt carried no mode, and a prompt with none was sampled as thinking.
func TestAMediaHistoryKeepsTheThinkingModeAsked(t *testing.T) {
	t.Parallel()
	generator := &fakeGenerator{}
	handler := testcheck.Must(New(Config{
		ModelID: testModelID, MaxTokens: testMaxTokens, DefaultTemperature: testNeutralTemperature, DefaultTopP: testFullTopP,
		ImageProjector: &fakeHistoryProjector{},
	}, historyOptionsGenerator{&historyGenerator{fakeGenerator: generator}}))(t)
	off := handler.defaultSampling
	off.Temperature, off.TopP = 0.7, 0.8
	handler.thinkingOffSampling = &off
	var encoded bytes.Buffer
	testcheck.NoError(t, png.Encode(&encoded, image.NewRGBA(image.Rect(0, 0, 1, 1))))
	picture := map[string]any{"type": "image_url", "image_url": map[string]any{"url": "data:image/png;base64," + base64.StdEncoding.EncodeToString(encoded.Bytes())}}
	for name, first := range map[string]any{
		"with an image":    []any{map[string]any{"type": "text", "text": "Look "}, picture},
		"without an image": "Look",
	} {
		body := testcheck.Must(json.Marshal(map[string]any{
			"messages": []any{
				map[string]any{"role": "user", "content": first},
				map[string]any{"role": "assistant", "content": "A square."},
				map[string]any{"role": "user", "content": "What colour is it?"},
			},
			"chat_template_kwargs": map[string]any{"enable_thinking": false},
			"max_tokens":           1,
		}))(t)
		response := httptest.NewRecorder()
		handler.ServeHTTP(response, httptest.NewRequest(http.MethodPost, "/v1/chat/completions", bytes.NewReader(body)))
		if response.Code != http.StatusOK {
			t.Fatalf("a history %s: status %d body %s", name, response.Code, response.Body.String())
		}
		generator.mu.Lock()
		got := generator.sampling
		generator.mu.Unlock()
		if got.Temperature != off.Temperature || got.TopP != off.TopP {
			t.Errorf("a history %s asked with thinking off was sampled %+v; want the thinking-off set", name, got)
		}
	}
}
