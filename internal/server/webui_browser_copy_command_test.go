package server

import (
	"net/http/httptest"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/overgodb"
	"overgo/internal/recipe"
	"overgo/internal/testutil"
	"overgo/internal/webuilane"
)

// TestWebUIBrowserCopyCommand drives Show command on a workflow tab: with
// the entries typed, the page shows the command line and the HTTP request
// that reproduce the submission, each with a copy control, without running
// it; changing an entry clears them, and a missing required entry shows none.
func TestWebUIBrowserCopyCommand(t *testing.T) {
	t.Parallel()
	requireWebUILane(t, "the copy command leg uses Chromium through cmd/webui-lane")
	store, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	recipeID := testutil.ArtifactID(t, artifact.KindRecipe, "copy-command-recipe")
	generator := &generationWorkspaceGenerator{
		fakeGenerator: &fakeGenerator{},
		run:           testutil.ArtifactID(t, artifact.KindRun, "copy-command-run"),
		repository:    store,
		capability: WorkflowCapability{
			Task: recipe.TaskSpeech, Recipe: recipeID, Location: "C:/models/voice.gguf", Name: "voice",
			Stages: []recipe.Stage{{Node: recipe.Node{ID: "generate", Module: "test.generate"}}},
			Controls: []workflowControl{
				{Name: "text", Type: workflowControlText, Required: true},
				{Name: "seed", Type: workflowControlInteger},
			},
		},
	}
	handler := newTestHandlerForRepository(t, store, generator, func(config *Config) { config.Workflows = generator })
	server := httptest.NewServer(handler)
	defer server.Close()
	_, browser, check, settle := openBrowserLeg(t, server.URL)
	defer browser.Close()
	check(`(() => { location.hash = 'generation'; return true; })()`)
	settle(`!!document.querySelector('#panel-generation.active select[aria-label="capability"] option') && !!document.querySelector('#panel-generation .control-grid input, #panel-generation .control-grid textarea')`)
	check(`(() => {
  window.commandPanel = () => document.querySelector('#panel-generation');
  window.commandButton = () => [...commandPanel().querySelectorAll('button')].find(button => button.textContent === 'Show command');
  window.commandEntry = (name) => [...commandPanel().querySelectorAll('.control-grid label.control')].find(label => label.querySelector('span').textContent.startsWith(name)).querySelector('input, textarea');
  window.commandShown = () => [...commandPanel().querySelectorAll('.workflow-command')].map(row => [row.querySelector('.tag').textContent, row.querySelector('pre').textContent, !!row.querySelector('button')]);
  window.commandType = (name, value) => { const entry = commandEntry(name); entry.value = value; entry.dispatchEvent(new Event('input', { bubbles: true })); };
  return true;
})()`)

	// A missing required entry shows no command.
	check(`(() => { commandButton().click(); return true; })()`)
	settle(`commandPanel().querySelector('[aria-label="run status"]').textContent === 'text is required' && commandShown().length === 0`)

	// Typed, the submission's commands show with copy controls and nothing runs.
	check(`(() => { commandType('text', "it's here"); commandType('seed', '17'); commandButton().click(); return true; })()`)
	settle(`commandShown().length === 2`)
	check(`(() => {
  const [cli, http] = commandShown();
  return cli[0] === 'command line' && cli[1] === "go run ./cmd/recipe run -task 'speech' -input '{\"text\":\"it'\\''s here\",\"seed\":17}' 'C:/models/voice.gguf'" && cli[2] &&
    http[0] === 'HTTP' && http[1].startsWith("curl -X POST '" + location.origin + "/generation/run' -H 'Content-Type: application/json' -d '{\"task\":\"speech\"") && http[2];
})()`)
	if generator.executions != 0 {
		t.Fatalf("showing the command ran the workflow %d times", generator.executions)
	}
	// The commands are quoted for a POSIX shell, and the page says which shell they are written for.
	check(`commandPanel().querySelector('.workflow-commands .note').textContent.startsWith('Written for a POSIX shell')`)

	// A changed entry clears the commands, and the note with them.
	check(`(() => { commandType('seed', '18'); return true; })()`)
	settle(`commandShown().length === 0 && !commandPanel().querySelector('.workflow-commands .note')`)
	webuilane.Leg(t, "copy command leg", "with the entries typed the workflow tab showed the command line and HTTP request reproducing the submission with copy controls and ran nothing, a changed entry cleared them, and a missing required entry showed none")
}
