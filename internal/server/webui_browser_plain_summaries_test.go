package server

import (
	"net/http/httptest"
	"os"
	"testing"

	"overgo/internal/overgodb"
	"overgo/internal/testskip"
	"overgo/internal/webuilane"
)

// TestWebUIBrowserPlainSummaries drives the pages that led with
// identifiers: Recipe opens with a sentence saying what it serves and how,
// its facts in plain labels and its identifiers in a closed fold; the
// Library's note speaks in sentences and its rows give the model and what it
// does before its identity; the runs read as how long each took and what it
// used and made, the commit last.
func TestWebUIBrowserPlainSummaries(t *testing.T) {
	t.Parallel()
	if os.Getenv("OVERGO_WEBUI_LANE") != "1" {
		t.Skip(testskip.Inapplicable + ": the plain summaries leg uses Chromium through cmd/webui-lane")
	}
	store, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	server := httptest.NewServer(newTestHandlerForRepository(t, store, responseRecipeGenerator(t, &fakeGenerator{})))
	defer server.Close()
	path, err := webuilane.FindBrowser(os.Getenv("OVERGO_BROWSER"))
	if err != nil {
		t.Fatal(err)
	}
	ctx := t.Context()
	browser, err := webuilane.Open(ctx, path, server.URL+"/#recipe")
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

	// Recipe: a sentence first, plain labels, identifiers folded away.
	settle(`!!document.querySelector('#panel-recipe .recipe-summary')`)
	check(`(() => {
  const panel = document.querySelector('#panel-recipe');
  const labels = [...panel.querySelectorAll(':scope > .statgrid .stat .k')].map(label => label.textContent);
  const fold = [...panel.querySelectorAll('details.fold')].find(details => details.querySelector('summary').textContent === 'Identifiers');
  return panel.querySelector('.recipe-summary').textContent.startsWith('Serves ') && labels.join(',') === 'Model,Task,Runs on,Weights held,Runtime,Version'
    && !!fold && !fold.open && fold.textContent.includes('recipe:') && !panel.querySelector(':scope > .statgrid').textContent.includes('recipe:');
})()`)

	// Library: the note in sentences, the identity after what the model does.
	check(`(() => { location.hash = 'library'; return true; })()`)
	settle(`!!document.querySelector('#panel-library.active table.grid.stack th')`)
	check(`(() => {
  const headers = [...document.querySelector('#panel-library table.grid.stack').querySelectorAll('th')].map(cell => cell.textContent);
  const note = document.querySelector('#panel-library .note').textContent;
  return headers.slice(0, 3).join(',') === 'Model,Capabilities,Identity' && !/profiles [0-9]+\/[0-9]+|cmd\//.test(note);
})()`)

	// Runs: how long and what it used and made, the commit last.
	check(`(() => { location.hash = 'runs'; return true; })()`)
	settle(`!!document.querySelector('#panel-runs.active table.grid th')`)
	check(`[...document.querySelectorAll('#panel-runs table.grid th')].map(cell => cell.textContent.toLowerCase()).join(',') === 'outcome,recipe,took,slowest phases,inputs and outputs,commit'`)
	for _, viewport := range webuilane.ScreenViewports {
		findings, err := webuilane.CaptureState(ctx, browser, os.Getenv("OVERGO_WEBUI_LANE_SCREENS"), viewport, "plain-runs")
		if err != nil || len(findings) != 0 {
			t.Fatalf("%s: %v %v", viewport.Name, findings, err)
		}
	}
	webuilane.Leg(t, "plain summaries leg", "Recipe opened with a sentence saying what it serves and how, plain labels and its identifiers in a closed fold; the Library's note spoke in sentences with each model's identity after what it does; runs read as how long each took and what it used and made, the commit last")
}
