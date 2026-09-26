package server

import (
	"net/http/httptest"
	"os"
	"testing"

	"overgo/internal/composition/compositiontest"
	"overgo/internal/testskip"
	"overgo/internal/webuilane"
)

// TestWebUIBrowserCompositions drives the compositions tab over a published
// composition authority: the card states compatibility, the recipe graph,
// bridge training and the evaluation and promotion history with its
// completion stages; activating it makes it the active composition, whose
// compiled plan shows its runtime memory and latency evidence.
func TestWebUIBrowserCompositions(t *testing.T) {
	t.Parallel()
	if os.Getenv("OVERGO_WEBUI_LANE") != "1" {
		t.Skip(testskip.Inapplicable + ": the compositions leg uses Chromium through cmd/webui-lane")
	}
	store, authority := compositiontest.Authority(t)
	handler := newTestHandlerForRepository(t, store, responseRecipeGenerator(t, &fakeGenerator{}))
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
	settle(`!!document.querySelector('.composer textarea')`)
	check(`(() => { location.hash = 'compositions'; return true; })()`)
	settle(`document.querySelectorAll('#panel-compositions .composition-card').length === 1`)
	check(`(() => {
  window.compositionCard = () => document.querySelector('#panel-compositions .composition-card');
  window.compositionTags = () => [...compositionCard().querySelectorAll('.tag')].map(tag => tag.textContent);
  window.compositionSections = () => [...compositionCard().querySelectorAll('.section-title')].map(title => title.textContent);
  return true;
})()`)

	// The card states compatibility, the graph, training and the promotion history.
	check(`compositionTags().includes('compatible') && !compositionTags().includes('active') && compositionTags().includes('` + string(authority.Bridge.Graph.Operator) + `')`)
	check(`['contract-tested passed', 'evidence-promoted passed', 'runtime-wired pending', 'production-active pending'].every(tag => compositionTags().includes(tag))`)
	check(`compositionSections().join('|') === 'Bridge training|Evaluation and promotion history' && compositionCard().textContent.includes('bridge_loss, bridge_update_l2') && compositionCard().textContent.includes('2 trials')`)

	// Activated, it is the active composition with its runtime evidence.
	check(`(() => { [...compositionCard().querySelectorAll('button')].find(button => button.textContent === 'Activate').click(); return true; })()`)
	settle(`compositionTags().includes('active') && ['runtime-wired passed', 'production-active passed'].every(tag => compositionTags().includes(tag)) && ![...compositionCard().querySelectorAll('button')].some(button => button.textContent === 'Activate')`)
	check(`compositionSections().includes('Runtime memory and latency evidence') && [...compositionCard().querySelectorAll('.stat .k')].some(label => label.textContent === 'Session bytes')`)
	webuilane.Leg(t, "compositions leg", "the card stated compatibility, the recipe graph, bridge training and the promotion history with its completion stages, and activating it made it active with its runtime memory and latency evidence")
}
