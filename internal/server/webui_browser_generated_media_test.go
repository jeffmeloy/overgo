package server

import (
	"net/http/httptest"
	"os"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/recipe"
	"overgo/internal/testskip"
	"overgo/internal/testutil"
	"overgo/internal/webuilane"
)

// TestWebUIBrowserGeneratedMedia drives the front page's generation modes
// over declared image and speech workflows: a speech run lands as a media
// card stored as its artifact, whose operation's live events the activity
// dialog lists; an image run's output, used as input while a capability
// with an image slot is selected, fills that slot with the artifact; and an
// embeddings and a rerank turn answer in the thread.
func TestWebUIBrowserGeneratedMedia(t *testing.T) {
	t.Parallel()
	if os.Getenv("OVERGO_WEBUI_LANE") != "1" {
		t.Skip(testskip.Inapplicable + ": the generated media leg uses Chromium through cmd/webui-lane")
	}
	handler, workspace, _, _ := nativeMediaProtocolFixture(t)
	// The speech capability reads its text from the composer, as a synthesizer declares it.
	workspace.capabilities[1].Controls = []WorkflowControl{{Name: "text", Type: WorkflowControlText, Required: true}}
	// An image capability with an image slot, which a reused output fills.
	editRecipe := testutil.ArtifactID(t, artifact.KindRecipe, "native-image-edit-recipe")
	workspace.capabilities = append(workspace.capabilities, WorkflowCapability{
		Task: recipe.TaskImageGen, Recipe: editRecipe, Name: "image edit", Outputs: []recipe.Output{{Name: "image", Data: recipe.DataImage}},
		Stages: []recipe.Stage{{Node: recipe.Node{ID: "edit", Module: "test.edit"}}},
		Controls: []WorkflowControl{
			{Name: "prompt", Type: WorkflowControlText, Required: true},
			{Name: "image", Type: WorkflowControlArtifact, Label: "source image", Media: "image"},
		},
	})
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
	settle(`!!document.querySelector('.composer textarea') && !!document.querySelector('.composer select[aria-label="mode"]')`)
	check(`(() => {
  window.mediaMode = (mode) => { const select = document.querySelector('.composer select[aria-label="mode"]'); select.value = mode; select.dispatchEvent(new Event('change', { bubbles: true })); };
  window.mediaSend = (text) => { const input = document.querySelector('.composer textarea'); input.value = text; input.dispatchEvent(new Event('input', { bubbles: true })); document.querySelector('.send-button').click(); };
  window.mediaCards = (kind) => [...document.querySelectorAll('.artifact.msg.media')].filter(card => card.querySelector(kind));
  window.mediaSlot = (label) => ([...document.querySelectorAll('.composer label.control')].find(control => control.firstChild.textContent === label) || { querySelector: () => null }).querySelector('input');
  return true;
})()`)

	// A speech run lands as a media card linking its stored artifact and lineage.
	check(`(() => { mediaMode('speech'); return true; })()`)
	settle(`!(document.querySelector('select[aria-label="generation model"]') || { disabled: true }).disabled && !document.querySelector('.send-button').disabled`)
	check(`(() => { mediaSend('say this'); return true; })()`)
	settle(`mediaCards('audio').length === 1 && (mediaCards('audio')[0].querySelector('.audio-details a.mono') || {}).textContent?.startsWith('output:') && ['use as input', 'lineage'].every(label => [...mediaCards('audio')[0].querySelectorAll('button')].some(button => button.textContent === label))`)

	// Its operation's live events list in the activity dialog.
	check(`(() => { document.querySelector('.activity-summary').click(); return true; })()`)
	settle(`[...document.querySelectorAll('.activity-dialog .operation-chip')].some(chip => chip.textContent.startsWith('speech'))`)
	check(`(() => { [...document.querySelectorAll('.activity-dialog .operation-chip')].find(chip => chip.textContent.startsWith('speech')).click(); return true; })()`)
	settle(`(() => { const title = [...document.querySelectorAll('.activity-dialog .section-title')].find(node => node.textContent === 'Live events');
  const table = title && title.nextElementSibling; return !!table && [...table.querySelectorAll('tr')].slice(1).some(row => row.textContent.includes('completed')); })()`)
	check(`(() => { document.querySelector('[aria-label="Close activity"]').click(); return true; })()`)

	// An image output, used as input with the edit capability selected, fills its image slot.
	check(`(() => { mediaMode('image-gen'); return true; })()`)
	settle(`!!document.querySelector('select[aria-label="generation model"]') && [...document.querySelector('select[aria-label="generation model"]').options].length === 2`)
	check(`(() => { mediaSend('a picture'); return true; })()`)
	settle(`mediaCards('img').length === 1`)
	check(`(() => {
  const picker = document.querySelector('select[aria-label="generation model"]');
  picker.value = ` + "`" + editRecipe.String() + "`" + `; picker.dispatchEvent(new Event('change', { bubbles: true }));
  return true;
})()`)
	settle(`!!mediaSlot('source image')`)
	check(`(() => { [...mediaCards('img')[0].querySelectorAll('button')].find(button => button.textContent === 'use as input').click(); return true; })()`)
	settle(`(mediaSlot('source image').value || '').startsWith('output:')`)

	// Embeddings and rerank turns answer in the thread.
	check(`(() => { mediaMode('embeddings'); mediaSend('embed me'); return true; })()`)
	settle(`[...document.querySelectorAll('.msg')].some(message => /embedding · [0-9]+ dimensions · norm /.test(message.textContent))`)
	check(`(() => { mediaMode('rerank'); mediaSend('query\nfirst document\nsecond document'); return true; })()`)
	settle(`[...document.querySelectorAll('.msg table')].some(table => table.textContent.includes('first document') && table.textContent.includes('second document'))`)
	webuilane.Leg(t, "generated media leg", "a speech run landed as a stored media card with its live events, an image output filled an image slot as input, and embeddings and rerank turns answered in the thread")
}
