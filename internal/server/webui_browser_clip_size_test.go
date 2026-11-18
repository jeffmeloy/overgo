package server

import (
	"net/http/httptest"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/overgodbtest"
	"overgo/internal/recipe"
	"overgo/internal/testutil"
	"overgo/internal/webuilane"
)

// TestWebUIBrowserAClipIsAskedAtItsDeclaredSize holds a clip's page to asking the clip's size (clip-scale/do): an
// oscillator clip's frames and scale size it, so a fresh form fills both with their declared defaults and a request the
// person sets nothing else of carries them, where a sampling field's default stays a hint. The model-journey lane found
// an oscillator clip asked at scale 0 and refused on 2026-11-17, once a form's sampling defaults became hints
// (preset-form/do) and the scale was counted among them.
func TestWebUIBrowserAClipIsAskedAtItsDeclaredSize(t *testing.T) {
	t.Parallel()
	requireWebUILane(t, "the clip size leg uses Chromium through cmd/webui-lane")
	store := overgodbtest.Open(t, t.TempDir())
	generator := &generationWorkspaceGenerator{
		fakeGenerator: &fakeGenerator{},
		run:           testutil.ArtifactID(t, artifact.KindRun, "clip-size-run"),
		repository:    store,
		capability: WorkflowCapability{
			Task: recipe.TaskVideoGen, Recipe: testutil.ArtifactID(t, artifact.KindRecipe, "clip-size-recipe"), Location: "C:/models/oscillator", Name: "oscillator",
			Stages: []recipe.Stage{{Node: recipe.Node{ID: "generate", Module: "test.generate"}}},
			Controls: []workflowControl{
				{Name: "class", Type: workflowControlInteger, Role: "class"},
				{Name: "frames", Type: workflowControlInteger, Role: "frames", Bounds: &workflowControlBounds{Default: 12, Rate: 12}},
				{Name: "scale", Type: workflowControlInteger, Bounds: &workflowControlBounds{Default: 1}},
				{Name: "steps", Type: workflowControlInteger, Bounds: &workflowControlBounds{Default: 50}},
			},
		},
	}
	server := httptest.NewServer(newTestHandlerForRepository(t, store, generator, func(config *Config) { config.Workflows = generator }))
	defer server.Close()
	_, browser, check, settle := openBrowserLeg(t, server.URL+"/#generation")
	defer browser.Close()
	settle(`!!document.querySelector('#panel-generation.active select[aria-label="capability"] option') && !!document.querySelector('#panel-generation .control-grid input')`)
	check(`(() => {
  const panel = document.querySelector('#panel-generation');
  window.clipField = (name) => [...panel.querySelectorAll('label.control')].find(label => label.querySelector('span').textContent === name).querySelector('input');
  window.clipInput = () => { const cli = [...panel.querySelectorAll('.workflow-command')].find(row => row.querySelector('.tag').textContent === 'command line'); return cli ? cli.querySelector('pre').textContent : ''; };
  [...panel.querySelectorAll('button')].find(button => button.textContent === 'Show command').click();
  return clipField('frames').value === '12' && clipField('scale').value === '1' && clipField('steps').value === '' && clipField('steps').placeholder === '50';
})()`)
	settle(`clipInput() !== ''`)
	check(`clipInput().includes('"frames":12') && clipInput().includes('"scale":1') && !clipInput().includes('"steps"')`)
	webuilane.Leg(t, "clip size leg", "a fresh clip form asked the clip's declared frames and scale, its steps left a hint")
}
