package server

import (
	"net/http"
	"strings"
	"testing"
)

// TestFrontPageGenerationDeclarations pins the front page's generation
// modes to the server's declarations (professional GUI campaign,
// gui-generation-declarations): the page reads the generation capabilities
// and renders, per mode, the models activated for the task with a refused
// one's reason and the controls the request declares; the composer runs
// the selection through the generic run route, waits on the operation and
// renders its outputs as artifacts; the workflow tabs render controls from
// the same helper; and no per-task request payload remains in the client.
func TestFrontPageGenerationDeclarations(t *testing.T) {
	handler := newTestHandlerWithRepository(t, responseRecipeGenerator(t, &fakeGenerator{}))
	defer handler.Close()
	get := func(path string) string { return serveTestRequest(handler, http.MethodGet, path, "").Body.String() }

	composer := get("/composer.js")
	for _, needle := range []string{`"/generation/run"`, `overgo.waitOperation(`, `overgo.controlValues(`, `const outputKind = (task) =>`, `"/artifacts/content?id="`} {
		if !strings.Contains(composer, needle) {
			t.Errorf("composer missing %q", needle)
		}
	}
	for _, gone := range []string{`"/v1/images/generations"`, `"/v1/videos/generations"`, `"/v1/videos/edits"`, `"/v1/audio/speech"`, `function* blob(`} {
		if strings.Contains(composer, gone) {
			t.Errorf("composer still carries a per-task payload: %q", gone)
		}
	}

	chat := get("/mod/chat.js")
	for _, needle := range []string{`"/generation/capabilities"`, `capability.refusal`, `overgo.controlInputs(`, `"generation model"`, `renderMode(mode)`} {
		if !strings.Contains(chat, needle) {
			t.Errorf("chat module missing %q", needle)
		}
	}

	workflow := get("/workflow.js")
	for _, needle := range []string{`window.overgo.controlInputs = function`, `window.overgo.controlValues = function`, `control.choices`, `value: capability.recipe`, `capability.name || fmt.shortID(capability.recipe)`} {
		if !strings.Contains(workflow, needle) {
			t.Errorf("workflow module missing %q", needle)
		}
	}

	routes := get("/workspace/routes")
	for _, path := range []string{"/generation/capabilities", "/generation/run", "/operations/wait"} {
		if !strings.Contains(routes, path) {
			t.Errorf("route table lacks %s", path)
		}
	}
}
