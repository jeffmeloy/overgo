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

// TestWebUIBrowserPresetChoiceSubmitsOnlyWhatThePersonSet holds the generation form to sending what the person set (preset-form/do,
// SQA-340): the form filled its bounded sampling controls with the reference's values, so a preset chosen on a fresh
// form went out beside the reference's steps, which a request's stated value keeps over the preset's, and the draft
// preset ran at the reference's steps. On a fresh form a bounded control shows its default as a hint and leaves its
// value empty; the request a prompt and the draft preset compose carries those two alone, as its command line shows it;
// and one where the person typed the steps the default names and a guidance of 0 carries both beside the preset.
func TestWebUIBrowserPresetChoiceSubmitsOnlyWhatThePersonSet(t *testing.T) {
	t.Parallel()
	requireWebUILane(t, "the preset leg uses Chromium through cmd/webui-lane")
	store := overgodbtest.Open(t, t.TempDir())
	generator := &generationWorkspaceGenerator{
		fakeGenerator: &fakeGenerator{},
		run:           testutil.ArtifactID(t, artifact.KindRun, "preset-form-run"),
		repository:    store,
		capability: WorkflowCapability{
			Task: recipe.TaskVideoGen, Recipe: testutil.ArtifactID(t, artifact.KindRecipe, "preset-form-recipe"), Location: "C:/models/clip", Name: "clip",
			Stages: []recipe.Stage{{Node: recipe.Node{ID: "generate", Module: "test.generate"}}},
			Controls: []workflowControl{
				{Name: "text", Type: workflowControlText, Required: true},
				{Name: "preset", Type: workflowControlText, Choices: []string{"draft"}},
				{Name: "steps", Type: workflowControlInteger, Bounds: &workflowControlBounds{Default: 50}},
				{Name: "guidance", Type: workflowControlNumber, Bounds: &workflowControlBounds{Default: 5}},
			},
		},
	}
	server := httptest.NewServer(newTestHandlerForRepository(t, store, generator, func(config *Config) { config.Workflows = generator }))
	defer server.Close()
	_, browser, check, settle := openBrowserLeg(t, server.URL+"/#generation")
	defer browser.Close()
	settle(`!!document.querySelector('#panel-generation.active select[aria-label="capability"] option') && !!document.querySelector('#panel-generation .control-grid input')`)
	check(`(() => {
  window.presetPanel = () => document.querySelector('#panel-generation');
  window.presetField = (name) => [...presetPanel().querySelectorAll('label.control')].find(label => label.querySelector('span').textContent.startsWith(name)).querySelector('input, select, textarea');
  window.presetSet = (name, value) => { const field = presetField(name); field.value = value; field.dispatchEvent(new Event('input', { bubbles: true })); field.dispatchEvent(new Event('change', { bubbles: true })); };
  window.presetCommand = () => { [...presetPanel().querySelectorAll('button')].find(button => button.textContent === 'Show command').click(); return true; };
  window.presetInput = () => { const cli = [...presetPanel().querySelectorAll('.workflow-command')].find(row => row.querySelector('.tag').textContent === 'command line'); return cli ? cli.querySelector('pre').textContent : ''; };
  return true;
})()`)
	check(`presetField('steps').value === '' && presetField('steps').placeholder === '50' && presetField('guidance').value === '' && presetField('guidance').placeholder === '5'`)

	// A fresh form with the draft preset chosen: the prompt and the preset alone.
	check(`(() => { presetSet('text', 'a fox'); presetSet('preset', 'draft'); return presetCommand(); })()`)
	settle(`presetInput() !== ''`)
	check(`presetInput().includes(` + "`-input '{\"text\":\"a fox\",\"preset\":\"draft\"}'`" + `)`)

	// The steps the default names and a guidance of 0, typed: both go beside the preset.
	check(`(() => { presetSet('steps', '50'); presetSet('guidance', '0'); return presetCommand(); })()`)
	settle(`presetInput().includes('"steps"')`)
	check(`presetInput().includes(` + "`-input '{\"text\":\"a fox\",\"preset\":\"draft\",\"steps\":50,\"guidance\":0}'`" + `)`)
	webuilane.Leg(t, "preset leg", "a fresh form composed the prompt and the chosen preset alone, and the steps and guidance the person typed beside them")
}
