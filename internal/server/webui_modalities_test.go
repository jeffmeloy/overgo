package server

import (
	"encoding/json"
	"net/http"
	"strings"
	"testing"

	"overgo/internal/recipe"
)

// TestFrontPageModalities holds the server's side of modality-adaptive
// inference: declared generation capabilities are the capability
// document's enabled modes, an undeclared one is a refused mode and a typed
// error, and the native speech route names its artifact in a header for API
// clients. The generated media leg drives the modes on the page.
func TestFrontPageModalities(t *testing.T) {
	t.Parallel()
	handler, workspace, _, _ := nativeMediaProtocolFixture(t)

	speech := serveTestRequest(handler, http.MethodPost, "/v1/audio/speech",
		`{"model":"`+workspace.capabilities[1].Recipe.String()+`","input":"test"}`)
	if speech.Code != http.StatusOK || speech.Header().Get("X-Overgo-Artifact") != workspace.completion[recipe.TaskSpeech].Outputs[0].String() {
		t.Fatalf("speech status=%d artifact=%q", speech.Code, speech.Header().Get("X-Overgo-Artifact"))
	}

	manifest := serveTestRequest(handler, http.MethodGet, "/workspace/manifest", "")
	var document workspaceManifestResponse
	if err := json.Unmarshal(manifest.Body.Bytes(), &document); err != nil {
		t.Fatal(err)
	}
	if document.Model == nil {
		t.Fatalf("manifest carries no capability document: %s", manifest.Body.String())
	}
	modes := map[string]workspaceMode{}
	for _, mode := range document.Model.Modes {
		modes[mode.ID] = mode
	}
	if !modes["image-gen"].Enabled || !modes["speech"].Enabled {
		t.Errorf("declared generation capabilities are not enabled modes: %+v", document.Model.Modes)
	}
	if modes["video-gen"].Enabled || modes["video-gen"].Refusal == "" {
		t.Errorf("an undeclared capability is not a refused mode: %+v", modes["video-gen"])
	}

	video := serveTestRequest(handler, http.MethodPost, "/v1/videos/generations", `{"prompt":"test"}`)
	if video.Code == http.StatusOK || !strings.Contains(video.Body.String(), `"error"`) {
		t.Fatalf("undeclared video generation status=%d body=%s", video.Code, video.Body.String())
	}
}
