package server

import (
	"encoding/json"
	"net/http"
	"strings"
	"testing"
)

// TestWorkspaceCapabilityDocument pins the one-capability-source ratchet
// (professional GUI campaign): the workspace manifest carries the served
// model's capability document (identity, context length, generation
// defaults, modalities, accepted media with the server's limits, composer
// modes with refusals), the shell exposes it once, and no client module
// decides a capability on its own: zero hard-coded accept lists, zero
// reads of /props or /analyze/model for gating outside the shell.
func TestWorkspaceCapabilityDocument(t *testing.T) {
	handler := newTestHandler(t, &fakeGenerator{})
	response := serveTestRequest(handler, http.MethodGet, "/workspace/manifest", "")
	if response.Code != http.StatusOK {
		t.Fatalf("manifest status = %d", response.Code)
	}
	var manifest struct {
		Model *workspaceModelCapabilities `json:"model"`
	}
	if err := json.Unmarshal(response.Body.Bytes(), &manifest); err != nil {
		t.Fatal(err)
	}
	if manifest.Model == nil {
		t.Fatalf("manifest carries no capability document: %s", response.Body.String())
	}
	document := manifest.Model
	if document.ContextLength == 0 || document.Generation.MaxTokens == 0 {
		t.Errorf("capability document lacks context or generation defaults: %+v", document)
	}
	if !document.Modalities["text"] ||
		document.Media.MaxImageBytes != maxImageBytes || document.Media.MaxImagePixels != maxImagePixels {
		t.Errorf("capability document limits are not the server's: %+v", document.Media)
	}
	modes := map[string]workspaceMode{}
	for _, mode := range document.Modes {
		modes[mode.ID] = mode
	}
	for _, id := range []string{"chat", "image-gen", "video-gen", "speech", "embeddings", "rerank"} {
		mode, found := modes[id]
		if !found {
			t.Errorf("capability document lacks mode %s", id)
			continue
		}
		if !mode.Enabled && mode.Refusal == "" {
			t.Errorf("mode %s is refused without a reason", id)
		}
	}

	sources := webuiJavaScript(t)
	boot := sources["boot.js"]
	for _, needle := range []string{"function capabilities()", "overgo.capabilities", "workspaceManifest.model"} {
		if !strings.Contains(boot, needle) {
			t.Errorf("boot.js does not expose the capability document (%s)", needle)
		}
	}
	if !strings.Contains(sources["composer.js"], "overgo.capabilities()") {
		t.Error("composer.js does not take its accepted media from the capability document")
	}
	for name, source := range sources {
		if !strings.HasPrefix(name, "mod/") {
			continue
		}
		for _, forbidden := range []string{"accept: [", `api.get("/props")`, `api.get("/analyze/model")`, "modalities."} {
			if strings.Contains(source, forbidden) {
				t.Errorf("%s decides a capability itself (%s); read overgo.capabilities()", name, forbidden)
			}
		}
	}
}
