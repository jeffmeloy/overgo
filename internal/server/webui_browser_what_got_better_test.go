package server

import (
	"context"
	"net/http/httptest"
	"os"
	"testing"

	"overgo/internal/overgodb"
	"overgo/internal/testskip"
	"overgo/internal/webuilane"
)

// TestWebUIBrowserWhatGotBetter drives what got better in the Library: with
// two published scorecards it says how many cells got better and worse and
// lists every move under its kind -- better and worse changes with both
// values in their units, newly measured and no longer measured cells, new and
// gone models -- whether or not work aimed at them; before two are published
// it says when they will show.
func TestWebUIBrowserWhatGotBetter(t *testing.T) {
	t.Parallel()
	if os.Getenv("OVERGO_WEBUI_LANE") != "1" {
		t.Skip(testskip.Inapplicable + ": the what got better leg uses Chromium through cmd/webui-lane")
	}
	store, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	rows := testScorecard(t)
	serve := func(history *ScorecardHistory) *httptest.Server {
		return httptest.NewServer(newTestHandlerForRepository(t, store, &fakeGenerator{}, func(config *Config) {
			config.Scorecard = func(context.Context, *overgodb.Store) ([]ScorecardRow, error) { return rows, nil }
			config.ScorecardHistory = func() (*ScorecardHistory, error) { return history, nil }
		}))
	}
	moved := serve(testScorecardHistory())
	defer moved.Close()
	first := serve(nil)
	defer first.Close()
	path, err := webuilane.FindBrowser(os.Getenv("OVERGO_BROWSER"))
	if err != nil {
		t.Fatal(err)
	}
	ctx := t.Context()
	browser, err := webuilane.Open(ctx, path, moved.URL+"/#library")
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
	const group = `(kind) => [...document.querySelectorAll('#panel-library .scorecard-moves[data-kind="' + kind + '"] li')].map(item => item.textContent)`

	// Two published scorecards: how many got better and worse, every move under its kind.
	settle(`!!document.querySelector('#panel-library .scorecard-moves')`)
	check(`(() => {
  const moves = ` + group + `;
  const summary = document.querySelector('#panel-library .scorecard-history .note').textContent;
  return summary.includes('0123456789ab') && summary.includes('fedcba987654') && summary.includes('1 better, 1 worse') && summary.includes('whether or not work aimed at it')
    && moves('better').join('|') === 'gemma-e4b inference, Throughput: 40.0 tok/s to 45.0 tok/s'
    && moves('worse').join('|') === 'gemma-e4b inference, First output: 100.0 ms to 130.0 ms'
    && moves('measured').join('|') === 'gemma-e4b inference, Quality: 0.700'
    && moves('unmeasured').join('|') === 'gemma-e4b inference, Peak memory: no benchmark record'
    && moves('added').join('|') === 'qwen-0.5b inference' && moves('removed').join('|') === 'retired inference';
})()`)
	check(`[...document.querySelectorAll('#panel-library .scorecard-moves h4')].map(title => title.textContent).join(',') === 'Better,Worse,Newly measured,No longer measured,New,Gone'`)
	for _, viewport := range webuilane.ScreenViewports {
		findings, err := webuilane.CaptureState(ctx, browser, os.Getenv("OVERGO_WEBUI_LANE_SCREENS"), viewport, "what-got-better")
		if err != nil || len(findings) != 0 {
			t.Fatalf("%s: %v %v", viewport.Name, findings, err)
		}
	}

	// Before two are published, it says when they will show.
	if err := browser.Evaluate(ctx, `location.href = `+"`"+first.URL+"/#library`", nil); err != nil {
		t.Fatal(err)
	}
	settle(`location.origin === ` + "`" + first.URL + "`" + ` && !!document.querySelector('#panel-library .scorecard-history .note')`)
	check(`document.querySelector('#panel-library .scorecard-history .note').textContent.includes('once two are published') && !document.querySelector('#panel-library .scorecard-moves')`)
	webuilane.Leg(t, "what got better leg", "with two published scorecards the Library said how many cells got better and worse and listed every move under its kind with both values in their units, whether or not work aimed at it; before two were published it said when they would show")
}
