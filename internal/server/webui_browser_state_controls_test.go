package server

import (
	"net/http/httptest"
	"os"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/overgodb"
	"overgo/internal/recipe"
	"overgo/internal/testutil"
	"overgo/internal/webuilane"
)

// TestWebUIBrowserStateControls drives controls that once disagreed with
// the page: Generate's saved presets read as one row across its form (the
// list, a name and its save) where they stacked in one field's column; the
// reproducing commands are a named group; and Activity says nothing is
// running rather than that there were no operations above a table of them.
func TestWebUIBrowserStateControls(t *testing.T) {
	t.Parallel()
	requireWebUILane(t, "the state controls leg uses Chromium through cmd/webui-lane")
	store, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	generator := &generationWorkspaceGenerator{
		fakeGenerator: &fakeGenerator{},
		run:           testutil.ArtifactID(t, artifact.KindRun, "state-controls-run"),
		repository:    store,
		capability: WorkflowCapability{
			Task: recipe.TaskSpeech, Recipe: testutil.ArtifactID(t, artifact.KindRecipe, "state-controls-recipe"), Location: "C:/models/voice.gguf", Name: "voice",
			Stages: []recipe.Stage{{Node: recipe.Node{ID: "generate", Module: "test.generate"}}},
			Controls: []workflowControl{
				{Name: "text", Type: workflowControlText, Required: true},
				{Name: "seed", Type: workflowControlInteger},
			},
		},
	}
	server := httptest.NewServer(newTestHandlerForRepository(t, store, generator, func(config *Config) { config.Workflows = generator }))
	defer server.Close()
	ctx, browser, check, settle := openBrowserLeg(t, server.URL+"/#generation")
	defer browser.Close()

	// Generate: the presets on one row across the form; the commands a named group.
	settle(`!!document.querySelector('#panel-generation.active .presets')`)
	check(`(() => {
  const row = document.querySelector('#panel-generation .presets'), parts = [...row.children].map(part => part.getBoundingClientRect());
  const grid = row.parentElement.getBoundingClientRect();
  return parts.length === 3 && parts.every(part => Math.abs(part.top - parts[0].top) < 8) && row.getBoundingClientRect().width > grid.width * 0.9
    && document.querySelector('#panel-generation .workflow-commands').getAttribute('role') === 'group';
})()`)
	for _, viewport := range webuilane.ScreenViewports {
		findings, err := webuilane.CaptureState(ctx, browser, os.Getenv("OVERGO_WEBUI_LANE_SCREENS"), viewport, "state-controls")
		if err != nil || len(findings) != 0 {
			t.Fatalf("%s: %v %v", viewport.Name, findings, err)
		}
	}

	// Activity says nothing is running, pointing at the finished work below.
	check(`(() => { location.hash = 'activity'; return true; })()`)
	settle(`!!document.querySelector('#panel-activity.active .empty-table')`)
	check(`document.querySelector('#panel-activity .empty-table').textContent.startsWith('Nothing is running now.')`)
	webuilane.Leg(t, "state controls leg", "Generate's presets read as one row across the form and its reproducing commands as a named group; Activity said nothing was running")
}
