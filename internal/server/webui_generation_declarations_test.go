package server

import (
	"net/http"
	"strings"
	"testing"
)

// TestFrontPageGenerationDeclarations holds the front page's generation
// modes to the generic run route: the generated media leg drives the modes
// over declared capabilities; this bans a per-task request payload in the
// client and checks the routes the modes run through.
func TestFrontPageGenerationDeclarations(t *testing.T) {
	t.Parallel()
	handler := newTestHandlerWithRepository(t, responseRecipeGenerator(t, &fakeGenerator{}))
	defer handler.Close()
	get := func(path string) string { return serveTestRequest(handler, http.MethodGet, path, "").Body.String() }

	composer := get("/composer.js")
	for _, gone := range []string{`"/v1/images/generations"`, `"/v1/videos/generations"`, `"/v1/videos/edits"`, `"/v1/audio/speech"`, `function* blob(`} {
		if strings.Contains(composer, gone) {
			t.Errorf("composer still carries a per-task payload: %q", gone)
		}
	}

	routes := get("/workspace/routes")
	for _, path := range []string{"/generation/capabilities", "/generation/run", "/operations/wait"} {
		if !strings.Contains(routes, path) {
			t.Errorf("route table lacks %s", path)
		}
	}
}
