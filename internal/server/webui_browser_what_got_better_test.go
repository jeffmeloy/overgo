package server

import (
	"context"
	"errors"
	"net/http/httptest"
	"os"
	"testing"

	"overgo/internal/overgodb"
	"overgo/internal/webuilane"
)

// TestWebUIBrowserWhatGotBetter drives what got better in the Library's Scorecard view: with
// two published scorecards it says how many cells got better and worse and
// lists every move under its kind -- better and worse changes with both
// values in their units, newly measured and no longer measured cells, new and
// gone models -- whether or not work aimed at them; before two are published
// it says when they will show.
func TestWebUIBrowserWhatGotBetter(t *testing.T) {
	t.Parallel()
	requireWebUILane(t, "the what got better leg uses Chromium through cmd/webui-lane")
	store, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	rows := testScorecard(t)
	serve := func(read ScorecardHistoryReader) *httptest.Server {
		return httptest.NewServer(newTestHandlerForRepository(t, store, &fakeGenerator{}, func(config *Config) {
			config.Scorecard = func(context.Context, *overgodb.Store) ([]ScorecardRow, error) { return rows, nil }
			config.ScorecardHistory = read
		}))
	}
	moved := serve(func() (*ScorecardHistory, error) { return testScorecardHistory(), nil })
	defer moved.Close()
	first := serve(func() (*ScorecardHistory, error) { return nil, nil })
	defer first.Close()
	unreadable := serve(func() (*ScorecardHistory, error) { return nil, errors.New("scorecard-x.json: unknown field") })
	defer unreadable.Close()
	ctx, browser, check, settle := openBrowserLeg(t, moved.URL+"/#library")
	defer browser.Close()
	// A view opens once the page has: a hash names it after the Library has mounted.
	openScorecard := func() {
		t.Helper()
		settle(`!!document.querySelector('#panel-library.active .view-chooser')`)
		check(`(() => { location.hash = 'scorecard'; return true; })()`)
	}
	const group = `(kind) => [...document.querySelectorAll('#panel-library .view-host .scorecard-moves[data-kind="' + kind + '"] li')].map(item => item.textContent)`

	// Two published scorecards: how many got better and worse, every move under its kind.
	openScorecard()
	settle(`!!document.querySelector('#panel-library .view-host .scorecard-moves')`)
	check(`(() => {
  const moves = ` + group + `;
  const summary = document.querySelector('#panel-library .view-host .scorecard-history .note').textContent;
  return summary.includes('0123456789ab') && summary.includes('fedcba987654') && summary.includes('1 better, 1 worse') && summary.includes('whether or not work aimed at it')
    && moves('better').join('|') === 'gemma-e4b inference, Throughput: 40.0 tok/s to 45.0 tok/s'
    && moves('worse').join('|') === 'gemma-e4b inference, First output: 100.0 ms to 130.0 ms'
    && moves('changed').join('|') === 'gemma-e4b inference, Setup steps: 2 to 3'
    && moves('measured').join('|') === 'gemma-e4b inference, Quality: 0.700'
    && moves('unmeasured').join('|') === 'gemma-e4b inference, Peak memory: no benchmark record'
    && moves('unknown').join('|') === 'gemma-e4b inference, Load'
    && document.querySelector('#panel-library .view-host .scorecard-moves[data-kind="unknown"] li').title === 'measured another way than the baseline'
    && moves('added').join('|') === 'qwen-0.5b inference' && moves('removed').join('|') === 'retired inference';
})()`)
	check(`[...document.querySelectorAll('#panel-library .view-host .scorecard-moves h4')].map(title => title.textContent).join('|') === 'Better|Worse|Changed, with no better or worse|Newly measured|No longer measured|Measured differently, so not compared|New|Gone'`)
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
	settle(`location.origin === ` + "`" + first.URL + "`")
	openScorecard()
	settle(`!!document.querySelector('#panel-library .view-host .scorecard-history .note')`)
	check(`document.querySelector('#panel-library .view-host .scorecard-history .note').textContent.includes('once two are published') && !document.querySelector('#panel-library .view-host .scorecard-moves')`)

	// A published record that cannot be read costs the history its answer; the scorecard below it still shows.
	if err := browser.Evaluate(ctx, `location.href = `+"`"+unreadable.URL+"/#library`", nil); err != nil {
		t.Fatal(err)
	}
	settle(`location.origin === ` + "`" + unreadable.URL + "`")
	openScorecard()
	settle(`!!document.querySelector('#panel-library .view-host .scorecard-history .note')`)
	check(`document.querySelector('#panel-library .view-host .scorecard-history .note').textContent === 'What moved between scorecards could not be read: scorecard-x.json: unknown field' && !!document.querySelector('#panel-library .view-host table.grid')`)
	webuilane.Leg(t, "what got better leg", "with two published scorecards the Scorecard said how many cells got better and worse and listed every move under its kind, including changes with no direction and cells measured differently, whether or not work aimed at it; before two were published it said when they would show; an unreadable record named itself while the scorecard still showed")
}
