package server

import (
	"net/http/httptest"
	"os"
	"testing"

	"overgo/internal/overgodb"
	"overgo/internal/webuilane"
)

// TestWebUIBrowserDemoDatasets holds the Datasets tab's demo sets in a real
// browser, on a store that has catalogued nothing. The empty tab offers the
// two sets beside the import; one press registers the identity set, which
// then lists as a dataset, is no longer offered, and previews as a
// conversation that says who the demo assistant is; the other set is still
// offered above the list. The tab has no layout finding in either state at
// desktop and phone widths in either colour scheme.
func TestWebUIBrowserDemoDatasets(t *testing.T) {
	t.Parallel()
	requireWebUILane(t, "the demo datasets leg uses Chromium through cmd/webui-lane")
	store, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	datasets := t.TempDir()
	handler := newTestHandlerForRepository(t, store, responseRecipeGenerator(t, &fakeGenerator{}), func(config *Config) {
		config.LibraryIntake.DatasetRoot = datasets
	})
	server := httptest.NewServer(handler)
	defer server.Close()
	ctx, browser, check, settle := openBrowserLeg(t, server.URL)
	defer browser.Close()
	const offered = `[...document.querySelectorAll('#panel-datasets button')].filter((button) => button.textContent.startsWith('Add the demo')).map((button) => button.textContent).join('|')`
	// audited captures the tab in each colour scheme at each width and wants no layout finding.
	audited := func(state string) {
		t.Helper()
		for _, scheme := range webuilane.ColourSchemes {
			if err := browser.SetColorScheme(ctx, scheme); err != nil {
				t.Fatal(err)
			}
			for _, viewport := range webuilane.ScreenViewports {
				findings, err := webuilane.CaptureState(ctx, browser, os.Getenv("OVERGO_WEBUI_LANE_SCREENS"), viewport, state+"-"+scheme)
				if err != nil || len(findings) != 0 {
					t.Fatalf("the Datasets tab, %s, at %s in the %s scheme: %v %v", state, viewport.Name, scheme, findings, err)
				}
			}
		}
	}

	// Nothing catalogued: the import, and the two demo sets beside it.
	check(`(() => { location.hash = 'datasets'; return true; })()`)
	settle(`!!document.querySelector('#panel-datasets.active') && /No datasets are catalogued yet\./.test(document.querySelector('#panel-datasets').textContent)`)
	check(offered + ` === 'Add the demo instruction set|Add the demo identity set'`)
	audited("demo-datasets-empty")

	// One press registers the identity set: it lists, and is offered no more.
	check(`(() => { [...document.querySelectorAll('#panel-datasets button')].find((button) => button.textContent === 'Add the demo identity set').click(); return true; })()`)
	settle(`[...document.querySelectorAll('#panel-datasets table button')].some((button) => button.textContent === 'demo-identity') || !!document.querySelector('#panel-datasets [aria-label="demo dataset status"] *')`)
	check(`(() => {
  const refusal = document.querySelector('#panel-datasets [aria-label="demo dataset status"]');
  if (refusal && refusal.textContent) throw new Error('the demo set was not registered: ' + refusal.textContent);
  const still = ` + offered + `;
  if (still !== 'Add the demo instruction set') throw new Error('after the identity set was added the tab offers ' + JSON.stringify(still));
  return true;
})()`)

	// It previews as the conversation it is.
	check(`(() => { [...document.querySelectorAll('#panel-datasets table button')].find((button) => button.textContent === 'demo-identity').click(); return true; })()`)
	settle(`[...document.querySelectorAll('#panel-datasets .dataset-conversation .dataset-turn')].some((turn) => /Pebble/.test(turn.textContent))`)
	audited("demo-datasets-added")
	webuilane.Leg(t, "demo datasets leg", "the Datasets tab of an empty store offered the two demo sets, one press registered the identity set, which then listed, was offered no more and previewed as its conversation")
}
