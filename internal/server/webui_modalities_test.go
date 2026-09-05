package server

import (
	"encoding/json"
	"net/http"
	"strings"
	"testing"

	"overgo/internal/recipe"
)

// TestFrontPageModalities pins modality-adaptive inference (professional
// GUI campaign, gui-multimodal/dynamic-inference): the front page's modes
// are the capability document's enabled modes and nothing else, every
// generation mode rides one shared dispatch in composer.js that the
// generation tabs are thin wrappers over, generated media names its
// artifact for provenance (speech through a response header, images and
// video through their artifact URLs), a mode the served recipe lacks is
// refused by the server with a typed error, and a model switch re-derives
// the page from the refreshed capability document without a reload.
func TestFrontPageModalities(t *testing.T) {
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

	get := func(path string) string { return serveTestRequest(handler, http.MethodGet, path, "").Body.String() }
	composer := get("/composer.js")
	for _, needle := range []string{
		"async function* generate(", "function generationTab(", `"/v1/images/generations"`, `"/v1/videos/generations"`,
		`"/v1/videos/edits"`, `"/v1/audio/speech"`, `"/v1/embeddings"`, `"/v1/rerank"`, `"X-Overgo-Artifact"`,
		"artifactOf(item.url)", "overgo.artifactLink(event.artifact)", "options.modes.length > 1",
	} {
		if !strings.Contains(composer, needle) {
			t.Errorf("composer missing %q", needle)
		}
	}
	chat := get("/mod/chat.js")
	for _, needle := range []string{"modes: (capabilities.modes || []).filter((mode) => mode.enabled)", "overgo.generate(mode, text, parts, controller.signal)"} {
		if !strings.Contains(chat, needle) {
			t.Errorf("chat missing %q", needle)
		}
	}
	for _, module := range []string{"/mod/image.js", "/mod/speech.js", "/mod/video.js"} {
		source := get(module)
		if !strings.Contains(source, "window.overgo.generationTab(") || strings.Contains(source, "api.post(") || strings.Contains(source, "/v1/") {
			t.Errorf("%s is not a thin wrapper over the shared dispatch", module)
		}
	}
	boot := get("/boot.js")
	derive := strings.Index(boot, "capabilityDocument = workspaceManifest.model")
	if derive < 0 || !strings.Contains(boot[derive:], "remountActive();") {
		t.Error("boot.js does not re-derive the page from the refreshed capability document after a model switch")
	}
}
