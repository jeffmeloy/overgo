package server

import (
	"net/http/httptest"
	"os"
	"path/filepath"
	"slices"
	"testing"

	"overgo/internal/overgodb"
	"overgo/internal/webuilane"
)

// TestWebUIBrowserDatasetDrafting drives the Datasets tab's drafting action
// in a real browser, on a store that has catalogued nothing and a served
// model that answers one pair. The empty tab offers the action with the
// server's templates. Pasted text drafts a dataset: the tab says how many
// conversations it drafted, the registry lists the new dataset with no
// reload and keeps what was typed below it, and the dataset previews as the
// passage and its question answered. The same name again is refused in
// place, with what was typed kept and a control to send it again. A chosen
// document is stored and drafted from in place of the pasted text. The tab
// with the drafted datasets has no layout finding at desktop and phone
// widths in either colour scheme.
func TestWebUIBrowserDatasetDrafting(t *testing.T) {
	t.Parallel()
	requireWebUILane(t, "the dataset drafting leg uses Chromium through cmd/webui-lane")
	store, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	datasets := t.TempDir()
	const question, answer = "What is wide?", "The sea."
	generator := &fakeGenerator{pieces: []string{`{"question": `, `"` + question + `", `, `"answer": `, `"` + answer + `"}`}}
	handler := newTestHandlerForRepository(t, store, responseRecipeGenerator(t, generator), func(config *Config) {
		config.LibraryIntake.DatasetRoot = datasets
	})
	server := httptest.NewServer(handler)
	defer server.Close()
	ctx, browser, check, settle := openBrowserLeg(t, server.URL)
	defer browser.Close()
	check(`(() => { location.hash = 'datasets'; return true; })()`)
	settle(`!!document.querySelector('#panel-datasets.active [aria-label="Drafting template"]') && /No datasets are catalogued yet\./.test(document.querySelector('#panel-datasets').textContent)`)
	check(`(() => {
  window.datasetsPanel = () => document.querySelector('#panel-datasets');
  window.field = (label) => datasetsPanel().querySelector('[aria-label="' + label + '"]');
  window.press = (text) => { [...datasetsPanel().querySelectorAll('button')].find((button) => button.textContent === text).click(); return true; };
  window.drafted = () => field('drafting result').textContent;
  window.listed = (name) => [...datasetsPanel().querySelectorAll('table button')].find((button) => button.textContent === name);
  return true;
})()`)

	// The empty tab offers the action, with the server's templates.
	check(`[...field('Drafting template').options].map((option) => option.value).join('|') === 'grounded-qa|text-to-sql'`)

	// Pasted text drafts a dataset; the registry lists it with no reload, and what was typed is still there.
	check(`(() => { field('Drafted dataset name').value = 'sea-notes'; field('Text to draft from').value = 'The sea is wide.'; return press('Draft dataset'); })()`)
	settle(`/Drafted 1 conversation into sea-notes\./.test(drafted()) && !!listed('sea-notes')`)
	check(`field('Text to draft from').value === 'The sea is wide.' && field('Drafted dataset name').value === 'sea-notes' && !!field('drafting result').querySelector('a')`)
	check(`(() => { listed('sea-notes').click(); return true; })()`)
	settle(`(() => {
  const turns = [...datasetsPanel().querySelectorAll('.dataset-conversation .dataset-turn')];
  return turns.length === 2 && turns[0].querySelector('.preview-text').textContent === 'The sea is wide.\n\n` + question + `' && turns[1].querySelector('.preview-text').textContent === '` + answer + `';
})()`)

	// The same name again is refused in place: what was typed is kept, and one control sends it again.
	check(`press('Draft dataset')`)
	settle(`/already exists/.test(drafted())`)
	check(`field('Text to draft from').value === 'The sea is wide.' && [...field('drafting result').querySelectorAll('button')].some((button) => button.textContent === 'Draft again')`)

	// A chosen document is stored and drafted from in place of the pasted text.
	check(`(() => {
  const file = new File(['The tide comes in twice a day.'], 'tide.txt', { type: 'text/plain' }), transfer = new DataTransfer();
  transfer.items.add(file);
  const picker = datasetsPanel().querySelector('input[type=file]');
  picker.files = transfer.files;
  picker.dispatchEvent(new Event('change'));
  field('Drafted dataset name').value = 'tide';
  if (!field('Text to draft from').disabled || !/Drafting from tide\.txt\./.test(datasetsPanel().textContent)) throw new Error('the chosen document is not shown as the source: ' + datasetsPanel().textContent);
  return press('Draft dataset');
})()`)
	settle(`/Drafted 1 conversation into tide\./.test(drafted()) && !!listed('tide')`)
	if passages := draftedPassages(t, filepath.Join(datasets, "tide"), question, answer); !slices.Equal(passages, []string{"The tide comes in twice a day."}) {
		t.Fatalf("the dataset drafted from the document holds the passages %q, want the document's text and not the pasted text", passages)
	}

	for _, scheme := range webuilane.ColourSchemes {
		if err := browser.SetColorScheme(ctx, scheme); err != nil {
			t.Fatal(err)
		}
		for _, viewport := range webuilane.ScreenViewports {
			findings, err := webuilane.CaptureState(ctx, browser, os.Getenv("OVERGO_WEBUI_LANE_SCREENS"), viewport, "dataset-drafting-"+scheme)
			if err != nil || len(findings) != 0 {
				t.Fatalf("the Datasets tab with drafted datasets, at %s in the %s scheme: %v %v", viewport.Name, scheme, findings, err)
			}
		}
	}
	webuilane.Leg(t, "dataset drafting leg", "the Datasets tab drafted a conversation dataset from pasted text and from a chosen document with the served model, listed each with no reload, and refused a taken name in place")
}
