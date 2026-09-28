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

// TestWebUIBrowserScorecardView drives the scorecard in the Library tab:
// each model and task is a row whose measured cells read in their units --
// milliseconds, tokens a second, a share, bytes, steps -- with what measured
// them in their titles, and whose unmeasured cells say so with the reason;
// a server without the scorecard reader shows no scorecard.
func TestWebUIBrowserScorecardView(t *testing.T) {
	t.Parallel()
	if os.Getenv("OVERGO_WEBUI_LANE") != "1" {
		t.Skip(testskip.Inapplicable + ": the scorecard leg uses Chromium through cmd/webui-lane")
	}
	store, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	rows := testScorecard(t)
	scored := httptest.NewServer(newTestHandlerForRepository(t, store, &fakeGenerator{}, func(config *Config) {
		config.Scorecard = func(context.Context, *overgodb.Store) ([]ScorecardRow, error) { return rows, nil }
	}))
	defer scored.Close()
	plain := httptest.NewServer(newTestHandlerWithRepository(t, &fakeGenerator{}))
	defer plain.Close()
	path, err := webuilane.FindBrowser(os.Getenv("OVERGO_BROWSER"))
	if err != nil {
		t.Fatal(err)
	}
	ctx := t.Context()
	browser, err := webuilane.Open(ctx, path, scored.URL+"/#library")
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
	const cells = `(row) => [...document.querySelectorAll('#panels .scorecard tbody tr')][row].querySelectorAll('td')`

	// The measured row reads in its units, with what measured each cell in its title.
	settle(`!!document.querySelector('#panels .scorecard') && !document.querySelector('#panels .scorecard').hidden && document.querySelectorAll('#panels .scorecard tbody tr').length === 2`)
	check(`(() => {
  const measured = [...(` + cells + `)(0)].map(cell => cell.textContent);
  return measured.join('|') === 'gemma-e4b|inference|12.3 ms|45.7 tok/s|0.813|2.5%|1520.0 ms|1.50 GB|3'
    && (` + cells + `)(0)[2].querySelector('span').title === 'benchmark time to first token, median';
})()`)

	// Each header stands over its column.
	check(`(() => { const headers = [...document.querySelectorAll('#panels .scorecard th')]; return [...(` + cells + `)(0)].every((cell, index) => Math.abs(cell.getBoundingClientRect().left - headers[index].getBoundingClientRect().left) < 1); })()`)

	// The unmeasured cells say so, with the reason.
	check(`(() => {
  const training = (` + cells + `)(1);
  return training[1].textContent === 'training' && training[2].textContent === 'not measured' && training[2].querySelector('span').title === 'a training session has no first output'
    && [...training].slice(3).every(cell => cell.textContent === 'not measured');
})()`)
	for _, viewport := range webuilane.ScreenViewports {
		findings, err := webuilane.CaptureState(ctx, browser, os.Getenv("OVERGO_WEBUI_LANE_SCREENS"), viewport, "scorecard")
		if err != nil || len(findings) != 0 {
			t.Fatalf("%s: %v %v", viewport.Name, findings, err)
		}
	}

	// A server without the scorecard reader shows no scorecard.
	if err := browser.Evaluate(ctx, `location.href = `+"`"+plain.URL+"/#library`", nil); err != nil {
		t.Fatal(err)
	}
	settle(`location.origin === ` + "`" + plain.URL + "`" + ` && !!document.querySelector('#panels .scorecard') && !!document.querySelector('#panels .catalog-note, #panels .note')`)
	settle(`document.querySelector('#panels .scorecard').hidden`)
	webuilane.Leg(t, "scorecard leg", "each model and task was a scorecard row whose measured cells read in their units with what measured them in their titles and whose unmeasured cells said so with the reason; a server without the scorecard reader showed no scorecard")
}
