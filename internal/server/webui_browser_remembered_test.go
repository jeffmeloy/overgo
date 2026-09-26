package server

import (
	"context"
	"net/http/httptest"
	"os"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/capabilityruntime"
	"overgo/internal/mediacapability"
	"overgo/internal/modelrecipe"
	"overgo/internal/overgodb"
	"overgo/internal/recipe"
	"overgo/internal/testskip"
	"overgo/internal/webuilane"
)

// TestWebUIBrowserRememberedSettings drives an image capability's controls
// across visits: the entries a run used return after a reload, a named
// preset fills them again, and a past output's record starts a new run from
// the settings that made it.
func TestWebUIBrowserRememberedSettings(t *testing.T) {
	t.Parallel()
	if os.Getenv("OVERGO_WEBUI_LANE") != "1" {
		t.Skip(testskip.Inapplicable + ": the remembered settings leg uses Chromium through cmd/webui-lane")
	}
	store, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	activatedImageModel(t, store)
	// The executor records the request it decoded, as the runtime does: the record a past output reopens.
	requestContract := artifact.JSONContract(artifact.KindFile, "overgo.test-image-gen-input.v1")
	catalog := map[recipe.Task]mediacapability.Capability{recipe.TaskImageGen: {
		Execute: func(_ context.Context, _ artifact.Repository, _ string, _ modelrecipe.CapabilityEvidenceSelection, raw string) (any, error) {
			request, err := requestContract.ContentBytes([]byte(raw))
			if err != nil {
				return nil, err
			}
			return capabilityruntime.Measured{Output: tinyPNG(t), Input: request}, nil
		},
	}}
	workspace := NewStoreGenerationWorkspace(store, BindGenerationCatalog(catalog, mediacapability.Controls, mediacapability.OutputContent), 16)
	handler := newTestHandlerForRepository(t, store, &workspaceTestRuntime{fakeGenerator: &fakeGenerator{}, WorkflowWorkspaceAPI: workspace})
	server := httptest.NewServer(handler)
	defer server.Close()
	path, err := webuilane.FindBrowser(os.Getenv("OVERGO_BROWSER"))
	if err != nil {
		t.Fatal(err)
	}
	ctx := t.Context()
	browser, err := webuilane.Open(ctx, path, server.URL)
	if err != nil {
		t.Fatal(err)
	}
	defer browser.Close()
	check := func(expression string) { t.Helper(); assertBrowserPredicate(t, ctx, browser, expression) }
	settle := func(expression string) {
		t.Helper()
		if err := browser.Eventually(ctx, expression); err != nil {
			t.Fatalf("%s: %v", expression, err)
		}
	}
	const helpers = `(() => {
  window.rememberField = (label) => ([...document.querySelectorAll('.composer label.control')].find(control => control.firstChild.textContent === label) || { querySelector: () => null }).querySelector('input, select');
  window.rememberSet = (label, value) => { const field = rememberField(label); field.value = value; field.dispatchEvent(new Event('input', { bubbles: true })); };
  window.rememberButton = (label) => [...document.querySelectorAll('button')].find(button => button.textContent === label);
  const mode = document.querySelector('.composer select[aria-label="mode"]'); mode.value = 'image-gen'; mode.dispatchEvent(new Event('change', { bubbles: true }));
  return true;
})()`
	open := func() {
		t.Helper()
		settle(`!!document.querySelector('.composer select[aria-label="mode"] option[value="image-gen"]')`)
		check(helpers)
		settle(`!!rememberField('class') && !!rememberField('seed') && !(document.querySelector('select[aria-label="generation model"]') || { disabled: true }).disabled`)
	}
	open()

	// A run with chosen entries.
	check(`(() => {
  rememberSet('class', '3'); rememberSet('seed', '11');
  const input = document.querySelector('.composer textarea'); input.value = 'a picture'; input.dispatchEvent(new Event('input', { bubbles: true }));
  document.querySelector('.send-button').click();
  return true;
})()`)
	settle(`[...document.querySelectorAll('.artifact.msg.media')].some(card => !!card.querySelector('img'))`)

	// Reloaded, the capability's entries return.
	if err := browser.Evaluate(ctx, `location.reload()`, nil); err != nil {
		t.Fatal(err)
	}
	open()
	check(`rememberField('class').value === '3' && rememberField('seed').value === '11'`)

	// A named preset fills the entries again.
	check(`(() => {
  rememberSet('class', '5');
  const name = document.querySelector('.composer [aria-label="Preset name"]'); name.value = 'five';
  rememberButton('Save preset').click();
  rememberSet('class', '2');
  const preset = document.querySelector('.composer select[aria-label="preset"]'); preset.value = 'five'; preset.dispatchEvent(new Event('change', { bubbles: true }));
  return true;
})()`)
	check(`rememberField('class').value === '5' && [...document.querySelector('.composer select[aria-label="preset"]').options].some(option => option.value === 'five')`)

	// A past output's record starts a new run from its settings.
	settle(`!!document.querySelector('.intake-thumb')`)
	check(`(() => { document.querySelector('.intake-thumb').click(); return true; })()`)
	settle(`!!rememberButton('Use these settings')`)
	check(`(() => { rememberButton('Use these settings').click(); return true; })()`)
	settle(`rememberField('class').value === '3' && rememberField('seed').value === '11'`)
	webuilane.Leg(t, "remembered settings leg", "the entries a run used returned after a reload, a named preset filled them again, and a past output's record started a new run from the settings that made it")
}
