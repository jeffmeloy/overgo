package server

import (
	"net/http/httptest"
	"os"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/overgodb"
	"overgo/internal/runrecord"
	"overgo/internal/testskip"
	"overgo/internal/testutil"
	"overgo/internal/webuilane"
)

// TestWebUIBrowserTablesAlike drives what made tables disagree: a failed
// run's status tag keeps its own size in the caution tone where it once took
// the form-field grid of the tokenizer class it borrowed; an identifier link
// button reads in monospace where a later rule reset its font; a measured
// span reads in the unit that fits it; the Library heads its status column;
// and every disclosure opens on the drawn chevron.
func TestWebUIBrowserTablesAlike(t *testing.T) {
	t.Parallel()
	if os.Getenv("OVERGO_WEBUI_LANE") != "1" {
		t.Skip(testskip.Inapplicable + ": the tables leg uses Chromium through cmd/webui-lane")
	}
	store, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	// A succeeded run and a failed one, so the Runs table shows both tones.
	for _, outcome := range []runrecord.Outcome{runrecord.OutcomeSucceeded, runrecord.OutcomeFailed} {
		recipeID := testutil.ArtifactID(t, artifact.KindRecipe, "tables-recipe-"+string(outcome))
		outputID := testutil.ArtifactID(t, artifact.KindOutput, "tables-output-"+string(outcome))
		if _, err := store.Commit(t.Context(), artifact.Batch{Key: "fixture/facts/" + string(outcome), Artifacts: []artifact.Descriptor{{ID: recipeID}, {ID: outputID}}}); err != nil {
			t.Fatal(err)
		}
		failure := ""
		if outcome == runrecord.OutcomeFailed {
			failure = "fixture-failure"
		}
		run, err := runrecord.NewRun(recipeID, outcome, nil, []artifact.ID{outputID}, failure)
		if err != nil {
			t.Fatal(err)
		}
		batch, err := run.Batch("fixture/run/" + string(outcome))
		if err != nil {
			t.Fatal(err)
		}
		if _, err := store.Commit(t.Context(), batch); err != nil {
			t.Fatal(err)
		}
	}
	server := httptest.NewServer(newTestHandlerForRepository(t, store, responseRecipeGenerator(t, &fakeGenerator{})))
	defer server.Close()
	path, err := webuilane.FindBrowser(os.Getenv("OVERGO_BROWSER"))
	if err != nil {
		t.Fatal(err)
	}
	ctx := t.Context()
	browser, err := webuilane.Open(ctx, path, server.URL+"/#runs")
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

	// Runs: each status tag at its own size and tone; the recipe link in monospace.
	settle(`[...document.querySelectorAll('#panel-runs table.grid .tag')].length === 2`)
	check(`(() => {
  const tags = [...document.querySelectorAll('#panel-runs table.grid .tag')];
  const failed = tags.find(tag => tag.textContent === 'failed'), succeeded = tags.find(tag => tag.textContent === 'succeeded');
  const tone = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  const colour = (node) => { const probe = document.createElement('span'); probe.style.color = node; document.body.append(probe); const value = getComputedStyle(probe).color; probe.remove(); return value; };
  return !!failed && !!succeeded && failed.classList.contains('caution') && succeeded.classList.contains('ok')
    && getComputedStyle(failed).display === 'inline-block' && failed.getBoundingClientRect().width < failed.closest('td').getBoundingClientRect().width * 0.8
    && getComputedStyle(failed).color === colour(tone('--amber')) && getComputedStyle(succeeded).color === colour(tone('--ok'));
})()`)
	check(`(() => { const link = document.querySelector('#panel-runs table.grid .link-button.mono'); return !!link && getComputedStyle(link).fontFamily.includes('monospace'); })()`)

	// A measured span reads in the unit that fits it.
	check(`overgo.fmt.elapsed(189.02) === '189.0 ms' && overgo.fmt.elapsed(11082.79) === '11.08 s' && overgo.fmt.elapsed(90876) === '91 s' && overgo.fmt.elapsed(600000) === '10m'`)

	// Every disclosure, folded section or not, opens on the drawn chevron.
	check(`(() => {
  const details = document.createElement('details'), summary = document.createElement('summary');
  summary.textContent = 'Details'; details.append(summary); document.body.append(details);
  const closed = getComputedStyle(summary, '::before').transform;
  details.open = true;
  const opened = getComputedStyle(summary, '::before').transform;
  const drawn = getComputedStyle(summary).listStyleType === 'none' && getComputedStyle(summary, '::before').content === '""' && closed !== opened;
  details.remove();
  return drawn;
})()`)

	// The Library heads every column, its status column included.
	check(`(() => { location.hash = 'library'; return true; })()`)
	settle(`!!document.querySelector('#panel-library.active')`)
	check(`[...document.querySelectorAll('#panel-library table.grid th')].map(th => th.textContent).join(',').startsWith('Model,Capabilities,Identity,Status')`)
	for _, viewport := range webuilane.ScreenViewports {
		findings, err := webuilane.CaptureState(ctx, browser, os.Getenv("OVERGO_WEBUI_LANE_SCREENS"), viewport, "tables-alike")
		if err != nil || len(findings) != 0 {
			t.Fatalf("%s: %v %v", viewport.Name, findings, err)
		}
	}
	webuilane.Leg(t, "tables leg", "a failed run's tag kept its own size in the caution tone and a succeeded one read ok; the recipe link read in monospace; measured spans read in the unit that fits; every disclosure opened on the drawn chevron; the Library headed its status column")
}
