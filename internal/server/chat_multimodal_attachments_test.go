package server

import (
	"net/http"
	"strings"
	"testing"
)

// TestChatMultimodalAttachments pins the exact content-part shapes the
// workbench composer sends: a text part beside an image_url data URL,
// an input_audio base64 WAV, and an input_video data URL each decode
// into chat media and reach the projector gate -- the typed
// no-projector refusal is the proof of successful decoding on a
// text-only serving fixture -- while an unsupported part type is a
// typed invalid request, never silently dropped.
func TestChatMultimodalAttachments(t *testing.T) {
	handler := newTestHandler(t, &fakeGenerator{})
	cases := []struct {
		name, part, refusal string
	}{
		{"image", `{"type":"image_url","image_url":{"url":"data:image/png;base64,aGk="}}`, "image projector"},
		{"audio", `{"type":"input_audio","input_audio":{"data":"aGk=","format":"wav"}}`, "audio projector"},
		{"video", `{"type":"input_video","input_video":{"data":"data:video/mp4;base64,aGk="}}`, "video projector"},
	}
	for _, current := range cases {
		body := `{"model":"` + testModelID + `","messages":[{"role":"user","content":[{"type":"text","text":"describe"},` + current.part + `]}]}`
		response := serveTestRequest(handler, http.MethodPost, "/v1/chat/completions", body)
		if response.Code != http.StatusBadRequest || !strings.Contains(response.Body.String(), current.refusal) {
			t.Fatalf("%s part status=%d body=%s", current.name, response.Code, response.Body.String())
		}
	}
	unsupported := `{"model":"` + testModelID + `","messages":[{"role":"user","content":[{"type":"widget","widget":{}}]}]}`
	refused := serveTestRequest(handler, http.MethodPost, "/v1/chat/completions", unsupported)
	if refused.Code != http.StatusBadRequest || !strings.Contains(refused.Body.String(), "unsupported") {
		t.Fatalf("unsupported part status=%d body=%s", refused.Code, refused.Body.String())
	}
}
