package server

import (
	"context"
	"errors"
	"net/http/httptest"
	"os"
	"slices"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/runrecord"
	"overgo/internal/testskip"
	"overgo/internal/testutil"
	"overgo/internal/webuilane"
)

// standaloneEvaluationFixture answers the evaluation workspace with two
// finished runs that compare, over the launch workspace's campaign.
type standaloneEvaluationFixture struct {
	*evaluationWorkspaceFixture
	history []EvaluationHistoryEntry
}

func (fixture *standaloneEvaluationFixture) EvaluationHistory(context.Context, artifact.ID) ([]EvaluationHistoryEntry, error) {
	return fixture.history, nil
}

func (fixture *standaloneEvaluationFixture) CompareEvaluations(_ context.Context, left, right artifact.ID) (EvaluationComparison, error) {
	return EvaluationComparison{Left: left, Right: right, Metrics: []EvaluationMetricDelta{{
		Name: "accuracy", Left: 0.5, Right: 0.75, Delta: 0.25, Direction: runrecord.DirectionMaximize, Improved: true,
	}}}, nil
}

// TestWebUIBrowserStandaloneTabs drives two workbench tabs over their own
// workspaces: the recipe tab shows the admitted recipe's compiled stages in
// their compiled order and its required facts, and a refused recipe's
// reason; the evaluations tab runs the selected plan, lists its history and
// compares two runs.
func TestWebUIBrowserStandaloneTabs(t *testing.T) {
	t.Parallel()
	if os.Getenv("OVERGO_WEBUI_LANE") != "1" {
		t.Skip(testskip.Inapplicable + ": the standalone tabs leg uses Chromium through cmd/webui-lane")
	}
	path, err := webuilane.FindBrowser(os.Getenv("OVERGO_BROWSER"))
	if err != nil {
		t.Fatal(err)
	}
	ctx := t.Context()
	// page serves handler, opens it at hash and answers the page's predicates.
	page := func(handler *Handler, hash string) (check, settle func(string)) {
		t.Helper()
		server := httptest.NewServer(handler)
		t.Cleanup(server.Close)
		browser, err := webuilane.Open(ctx, path, server.URL)
		if err != nil {
			t.Fatal(err)
		}
		t.Cleanup(func() { _ = browser.Close() })
		check = func(expression string) { t.Helper(); assertBrowserPredicate(t, ctx, browser, expression) }
		settle = func(expression string) {
			t.Helper()
			if err := browser.Eventually(ctx, expression); err != nil {
				t.Fatalf("%s: %v", expression, err)
			}
		}
		settle(`!!document.querySelector('.composer textarea')`)
		check(`(() => { location.hash = '` + hash + `'; return true; })()`)
		settle(`!!document.querySelector('#panel-` + hash + `.active')`)
		check(`(() => {
  window.tabRows = (first) => { const table = [...document.querySelectorAll('.panel.active table, [id^=panel-].active table')].find(table => table.querySelector('th')?.textContent === first);
    return table ? [...table.rows].slice(1).map(row => [...row.cells].map(cell => cell.textContent)) : []; };
  window.tabButtons = (label) => [...document.querySelectorAll('[id^=panel-].active button')].filter(button => button.textContent === label);
  return true;
})()`)
		return check, settle
	}

	// The admitted recipe's stages in compiled order, and its required fact.
	description := inspectorDescription(t)
	check, settle := page(newTestHandlerWithRepository(t, &recipeInspectorGenerator{fakeGenerator: &fakeGenerator{}, description: description}), "recipe")
	settle(`tabRows('order').map(row => row[1]).join(',') === 'prepare,execute,publish'`)
	check(`tabRows('order').map(row => row[0]).join(',') === '0,1,2' && tabRows('role').some(row => row[0] === 'model')`)
	// A refused recipe shows why.
	const refusal = "compiled module is unavailable"
	_, settle = page(newTestHandlerWithRepository(t, &recipeInspectorGenerator{fakeGenerator: &fakeGenerator{}, err: errors.New(refusal)}), "recipe")
	settle(`document.querySelector('#panel-recipe').textContent.includes('Admission') && document.querySelector('#panel-recipe').textContent.includes('` + refusal + `')`)

	// The evaluations tab runs the selected plan, lists the history and compares two runs.
	campaign := launchEvaluationWorkspace(t)
	id := func(label string) artifact.ID { return testutil.ArtifactID(t, artifact.KindEvaluation, label) }
	fixture := &standaloneEvaluationFixture{evaluationWorkspaceFixture: campaign}
	for _, label := range []string{"standalone baseline", "standalone candidate"} {
		fixture.history = append(fixture.history, EvaluationHistoryEntry{
			Evaluation: id(label), Run: testutil.ArtifactID(t, artifact.KindRun, label), Report: id(label + " report"),
			Dataset: campaign.capability.Suite.Dataset, Recipe: campaign.capability.Recipe,
			Outcome: runrecord.OutcomeSucceeded, CodeCommit: evaluationTestCommit,
			Metrics: []runrecord.Metric{{Name: "accuracy", Value: 0.5, Direction: runrecord.DirectionMaximize}},
		})
	}
	evaluations := newTestHandler(t, &fakeGenerator{})
	evaluations.config.Evaluation = fixture
	check, settle = page(evaluations, "evaluations")
	settle(`!!document.querySelector('#panel-evaluations input[type=checkbox]')`)
	check(`(() => { document.querySelector('#panel-evaluations input[type=checkbox]').click(); tabButtons('Run')[0].click(); return true; })()`)
	settle(`document.querySelector('#panel-evaluations .note').textContent.startsWith('completed')`)
	campaign.mu.Lock()
	plans := slices.Clone(campaign.plans)
	campaign.mu.Unlock()
	if !slices.Equal(plans, []artifact.ID{campaign.capability.Suite.Plan}) {
		t.Fatalf("the run carried plans %v, want the selected %s", plans, campaign.capability.Suite.Plan)
	}
	settle(`tabRows('outcome').length === 2 && tabButtons('Compare').length === 2`)
	check(`(() => { tabButtons('Compare')[0].click(); tabButtons('Compare')[1].click(); return true; })()`)
	settle(`tabRows('metric').some(row => row[0] === 'accuracy' && row[3] === '0.25' && row[4] === 'improved')`)
	webuilane.Leg(t, "standalone tabs leg", "the recipe tab showed compiled stage order, required facts and a refusal, and the evaluations tab ran the selected plan, listed its history and compared two runs")
}
