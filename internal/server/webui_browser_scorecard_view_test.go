package server

import (
	"context"
	"net/http/httptest"
	"os"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/overgodb"
	"overgo/internal/testutil"
	"overgo/internal/webuilane"
)

// TestWebUIBrowserScorecardView drives the Scorecard, a view of the Library
// the way Inspect's analyses are views of Inspect: one choice away, with no
// navigation entry of its own, holding what a user feels of each model while
// the Library's models view keeps inventory and intake. Each model and task is a row whose measured cells read in their units --
// milliseconds, tokens a second, a share, bytes, steps -- with what measured
// them in their titles; a cell no record measures is one quiet dash with its
// reason in its title and for assistive technology; a held-out quality names
// what it measures and which way it improves, and a trained model's reads
// against its base as beating it or falling below it; a server without the
// scorecard reader says the view is not available there.
func TestWebUIBrowserScorecardView(t *testing.T) {
	t.Parallel()
	requireWebUILane(t, "the scorecard leg uses Chromium through cmd/webui-lane")
	store, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	// Two trained models: a perplexity under its base's beats it (lower is better), an accuracy under its base's falls below it (higher is better).
	perplexityBase, accuracyBase := 10.4, 0.4
	trained := func(name string, cell ScorecardCell) ScorecardRow {
		return ScorecardRow{Model: testutil.ArtifactBytesID(t, artifact.KindModel, []byte(name)), Name: name, Task: "training", Cells: []ScorecardCell{cell}}
	}
	rows := append(testScorecard(t),
		trained("tuned-a", ScorecardCell{Metric: "held-out-quality", Value: 9.1, Label: "perplexity, wikitext (raw)", Measure: "perplexity", Direction: "minimize", Base: &perplexityBase}),
		trained("tuned-b", ScorecardCell{Metric: "held-out-quality", Value: 0.3, Label: "accuracy, arc (zero-shot)", Measure: "accuracy", Direction: "maximize", Base: &accuracyBase}))
	scored := httptest.NewServer(newTestHandlerForRepository(t, store, &fakeGenerator{}, func(config *Config) {
		config.Scorecard = func(context.Context, *overgodb.Store) ([]ScorecardRow, error) { return rows, nil }
	}))
	defer scored.Close()
	plain := httptest.NewServer(newTestHandlerWithRepository(t, &fakeGenerator{}))
	defer plain.Close()
	ctx, browser, check, settle := openBrowserLeg(t, scored.URL+"/#library")
	defer browser.Close()
	// A view opens once the page has: a hash names it after the Library has mounted.
	openScorecard := func() {
		t.Helper()
		settle(`!!document.querySelector('#panel-library.active .view-chooser')`)
		check(`(() => { location.hash = 'scorecard'; return true; })()`)
	}
	const cells = `(row) => [...document.querySelectorAll('#panel-library .view-host tbody tr')][row].querySelectorAll('td')`

	// The measured row reads in its units, with what measured each cell in its title.
	openScorecard()
	settle(`document.querySelectorAll('#panel-library.active .view-host tbody tr').length === 4`)
	check(`(() => {
  const measured = [...(` + cells + `)(0)].map(cell => cell.textContent);
  return measured.join('|') === 'gemma-e4b|inference|12.3 ms|45.7 tok/s|0.813|2.5%|1.52 s|1.50 GB|3'
    && (` + cells + `)(0)[2].querySelector('span').title === 'benchmark time to first token, median';
})()`)

	// Each header stands over its column.
	check(`(() => { const headers = [...document.querySelectorAll('#panel-library .view-host th')]; return [...(` + cells + `)(0)].every((cell, index) => Math.abs(cell.getBoundingClientRect().left - headers[index].getBoundingClientRect().left) < 1); })()`)

	// A cell no record measures is one dash, its reason in its title and for assistive technology.
	check(`(() => {
  const training = (` + cells + `)(1);
  const gap = training[2].querySelector('.scorecard-gap');
  return training[1].textContent === 'training' && gap.firstChild.textContent === '—' && gap.title === 'a training session has no first output'
    && gap.querySelector('.sr-only').textContent === 'not measured: a training session has no first output'
    && [...training].slice(2).every(cell => !!cell.querySelector('.scorecard-gap') && cell.classList.contains('gap'));
})()`)

	// A trained model's quality reads against its base, whichever way its metric improves.
	check(`(() => {
  const beats = (` + cells + `)(2)[4], below = (` + cells + `)(3)[4], value = beats.firstChild;
  return value.firstChild.textContent === '9.100' && beats.querySelector('.note').textContent.trim() === 'perplexity' && value.title === 'perplexity, wikitext (raw) · lower is better'
    && beats.querySelector('.tag.ok').textContent === 'beats base' && beats.querySelector('.tag.ok').title === 'the base model scored 10.400'
    && below.querySelector('.tag.caution').textContent === 'below base' && below.querySelector('.tag.caution').title === 'the base model scored 0.400';
})()`)
	for _, viewport := range webuilane.ScreenViewports {
		findings, err := webuilane.CaptureState(ctx, browser, os.Getenv("OVERGO_WEBUI_LANE_SCREENS"), viewport, "scorecard")
		if err != nil || len(findings) != 0 {
			t.Fatalf("%s: %v %v", viewport.Name, findings, err)
		}
	}

	// The scorecard is a view the Library offers beside its models, with no navigation entry of its own.
	check(`![...document.querySelectorAll('.sidebar .tab')].some(tab => tab.textContent.startsWith('Scorecard'))`)
	check(`[...document.querySelectorAll('#panel-library .view-chooser .chip')].map(chip => chip.textContent).join('|') === 'Models|Scorecard' && document.querySelector('#panel-library .chip[aria-selected="true"]').textContent === 'Scorecard'`)
	check(`(() => { [...document.querySelectorAll('#panel-library .view-chooser .chip')].find(chip => chip.textContent === 'Models').click(); return true; })()`)
	settle(`document.querySelector('#panel-library .chip[aria-selected="true"]').textContent === 'Models' && !!document.querySelector('#panel-library .section-title') && !document.querySelector('#panel-library .view-host .scorecard-history') && location.hash === '#library'`)

	// A server without the scorecard reader says so.
	if err := browser.Evaluate(ctx, `location.href = `+"`"+plain.URL+"/#library`", nil); err != nil {
		t.Fatal(err)
	}
	settle(`location.origin === ` + "`" + plain.URL + "`")
	openScorecard()
	settle(`!!document.querySelector('#panel-library.active .view-host') && document.querySelector('#panel-library .view-host').textContent.includes("Scorecard isn't available on this server")`)
	webuilane.Leg(t, "scorecard leg", "the Scorecard was a view the Library offered beside its models, with no navigation entry; each model and task was a row whose measured cells read in their units with what measured them in their titles, a cell no record measures was one dash with its reason, and a trained model's quality read against its base; a server without the scorecard reader refused the view")
}
