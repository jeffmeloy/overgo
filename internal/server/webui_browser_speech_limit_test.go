package server

import (
	"net/http/httptest"
	"strconv"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/overgodb"
	"overgo/internal/recipe"
	"overgo/internal/testutil"
	"overgo/internal/webuilane"
)

// TestWebUIBrowserSpeechSendsNoFrameLimit holds the page's Speech mode to
// reading a text whole. A speech request with no frame limit is read to its
// end, and one with a limit is a preview cut at it; the page offered the
// limit among its speech options and remembered the last one used, so a
// browser that had ever set one kept sending it and kept getting the cut
// preview. With a limit remembered for the model, Speech mode offers no
// frame limit, a send posts none, and the voice remembered beside it is
// still the one chosen.
func TestWebUIBrowserSpeechSendsNoFrameLimit(t *testing.T) {
	t.Parallel()
	requireWebUILane(t, "the speech limit leg uses Chromium through cmd/webui-lane")
	store, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	capability := WorkflowCapability{
		Task: recipe.TaskSpeech, Recipe: testutil.ArtifactID(t, artifact.KindRecipe, "speech-limit"), Name: "Speech (UI fixture)", Outputs: []recipe.Output{{Name: "audio", Data: recipe.DataAudio}},
		Stages: []recipe.Stage{{Node: recipe.Node{ID: "generate", Module: "test.generate"}}},
		// As the served speech capability declares them: the limit is optional.
		Controls: []workflowControl{
			{Name: "text", Type: workflowControlText, Required: true},
			{Name: "max_frames", Type: workflowControlInteger},
			{Name: "voice", Type: workflowControlText, Required: true, Choices: []string{"alba", "marius"}},
		},
	}
	// One model, the fixture's second one retired, so the remembered values are the chosen model's.
	generator := &speechSurfaceGenerator{retired: true, generationWorkspaceGenerator: &generationWorkspaceGenerator{
		fakeGenerator: &fakeGenerator{}, repository: store, run: testutil.ArtifactID(t, artifact.KindRun, "speech-limit"), capability: capability,
	}}
	handler, err := New(Config{Repository: store, Workflows: generator}, generator)
	if err != nil {
		t.Fatal(err)
	}
	defer handler.Close()
	server := httptest.NewServer(handler)
	defer server.Close()
	ctx, browser, check, settle := openBrowserLeg(t, server.URL)
	defer browser.Close()
	settle(`!!document.querySelector('.composer select[aria-label="mode"] option[value="speech"]')`)

	// A browser that used a limit before: the model's last values hold one, beside a voice.
	check(`(() => {
  overgo.rememberedControls.remember(` + strconv.Quote(capability.Recipe.String()) + `, { max_frames: '2', voice: 'marius' });
  window.posted = [];
  const real = window.fetch.bind(window);
  window.fetch = async (url, options) => {
    if (String(url).endsWith('/generation/run') && options && options.method === 'POST') window.posted.push(JSON.parse(options.body));
    return real(url, options);
  };
  const mode = document.querySelector('.composer select[aria-label="mode"]');
  mode.value = 'speech';
  mode.dispatchEvent(new Event('change'));
  return true;
})()`)
	settle(`document.querySelector('.mode-controls select[aria-label="voice"]')?.value === 'marius'`)
	check(`(() => {
  const offered = [...document.querySelectorAll('[aria-label="Speech options"] .control span, .mode-controls .control span')].map((label) => label.textContent);
  if (offered.includes('max frames')) throw new Error('Speech mode offers a frame limit among ' + JSON.stringify(offered));
  return true;
})()`)
	say(t, ctx, browser, "Read this whole text aloud.")
	settle(`window.posted.length === 1`)
	check(`(() => {
  const input = window.posted[0].input;
  if (Object.hasOwn(input, 'max_frames')) throw new Error('the send posted the remembered frame limit: ' + JSON.stringify(input));
  if (input.voice !== 'marius' || input.text !== 'Read this whole text aloud.') throw new Error('the send posted ' + JSON.stringify(input) + ', want the remembered voice and the typed text');
  return true;
})()`)
	webuilane.Leg(t, "speech limit leg", "with a frame limit remembered for the model, Speech mode offered none and a send posted none, with the remembered voice")
}
