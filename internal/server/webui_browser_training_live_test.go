package server

import (
	"context"
	"encoding/json"
	"net/http/httptest"
	"os"
	"strconv"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/operation"
	"overgo/internal/overgodb"
	"overgo/internal/recipe"
	"overgo/internal/testskip"
	"overgo/internal/testutil"
	"overgo/internal/webuilane"
)

// steppedTraining: a training capability whose steps the test releases, one
// loss per step, so the page is watched mid-run; a closed channel ends it.
type steppedTraining struct {
	capability WorkflowCapability
	total      uint64
	losses     chan float64
	run        artifact.ID
}

func (training *steppedTraining) WorkflowCapabilities(_ context.Context, kind WorkflowKind) ([]WorkflowCapability, error) {
	if kind != WorkflowTraining {
		return nil, nil
	}
	return []WorkflowCapability{training.capability}, nil
}

func (training *steppedTraining) ExecuteWorkflow(ctx context.Context, _ WorkflowKind, _ recipe.Task, _ artifact.ID, _ json.RawMessage, reporter operation.Reporter) (operation.Completion, error) {
	var completed uint64
	reporter.Progress(completed, &training.total)
	for {
		select {
		case loss, open := <-training.losses:
			if !open {
				return operation.Completion{Run: training.run}, nil
			}
			completed++
			reporter.Progress(completed, &training.total)
			reporter.Metric(operation.Metric{Name: "dpo_loss", Value: loss})
		case <-ctx.Done():
			return operation.Completion{Run: training.run}, ctx.Err()
		}
	}
}

// TestWebUIBrowserTrainingLive drives a training run on the training page:
// while it runs, the page shows the step against the total, the time
// remaining and the loss drawn as it falls; reloaded mid-run, the page
// follows the same run again with the loss so far; and the run ends
// completed.
func TestWebUIBrowserTrainingLive(t *testing.T) {
	t.Parallel()
	if os.Getenv("OVERGO_WEBUI_LANE") != "1" {
		t.Skip(testskip.Inapplicable + ": the training live leg uses Chromium through cmd/webui-lane")
	}
	store, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	training := &steppedTraining{
		capability: WorkflowCapability{
			Task: recipe.TaskTraining, Recipe: testutil.ArtifactID(t, artifact.KindRecipe, "stepped-training-recipe"), Name: "stepped DPO",
			Stages: []recipe.Stage{{Node: recipe.Node{ID: "train", Module: "test.train"}}},
		},
		total: 6, losses: make(chan float64), run: testutil.ArtifactID(t, artifact.KindRun, "stepped-training-run"),
	}
	// The run a completed operation names is in the store.
	if _, err := store.Commit(t.Context(), artifact.Batch{Key: "stepped/training/run", Artifacts: []artifact.Descriptor{{ID: training.run}}}); err != nil {
		t.Fatal(err)
	}
	handler := newTestHandlerForRepository(t, store, &workspaceTestRuntime{fakeGenerator: &fakeGenerator{}, WorkflowWorkspaceAPI: training})
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
	const helpers = `(() => {
  window.trainingPanel = () => document.querySelector('#panel-training-jobs');
  window.trainingStatus = () => trainingPanel().querySelector(':scope > .note').textContent;
  window.trainingRemaining = () => trainingPanel().querySelector('[aria-label="time remaining"]').textContent;
  window.trainingProgress = () => trainingPanel().querySelector('progress.workflow-progress');
  window.trainingTrend = () => trainingPanel().querySelector('[data-trend="dpo_loss"]');
  return true;
})()`
	open := func() {
		t.Helper()
		check(`(() => { location.hash = 'training-jobs'; return true; })()`)
		settle(`!!document.querySelector('#panel-training-jobs.active select[aria-label="capability"] option')`)
		check(helpers)
	}
	// release: one step per loss, each shown on the page before the next, so
	// the run's wall between steps is the page's, as a real step's is longer.
	completed := 0
	release := func(losses ...float64) {
		t.Helper()
		for _, loss := range losses {
			select {
			case training.losses <- loss:
			case <-ctx.Done():
				t.Fatalf("the run did not take a step: %v", ctx.Err())
			}
			completed++
			settle(`trainingProgress().value === ` + strconv.Itoa(completed))
		}
	}
	open()

	// Running, the page shows the step, the time remaining and the loss as it falls.
	check(`(() => { [...trainingPanel().querySelectorAll('button')].find(button => button.textContent === 'Run').click(); return true; })()`)
	release(0.9, 0.7, 0.6)
	settle(`trainingProgress().max === 6 && /^step 3 of 6 · about [0-9]+(s|m) remaining$/.test(trainingRemaining())`)
	settle(`!!trainingTrend() && trainingTrend().textContent.startsWith('dpo_loss · 3 steps') && !!trainingTrend().querySelector('svg')`)

	// Reloaded mid-run, the page follows the same run with the loss so far.
	if err := browser.Evaluate(ctx, `location.reload()`, nil); err != nil {
		t.Fatal(err)
	}
	open()
	settle(`trainingStatus().startsWith('running') && trainingProgress().value === 3 && !!trainingTrend() && trainingTrend().textContent.startsWith('dpo_loss · 3 steps')`)
	check(`[...trainingPanel().querySelectorAll('button')].find(button => button.textContent === 'Run').disabled`)
	release(0.5, 0.45, 0.4)
	settle(`trainingProgress().value === 6 && trainingTrend().textContent.startsWith('dpo_loss · 6 steps')`)

	// The run ends completed.
	close(training.losses)
	settle(`trainingStatus().startsWith('completed') && trainingRemaining() === '' && ![...trainingPanel().querySelectorAll('button')].find(button => button.textContent === 'Run').disabled`)
	webuilane.Leg(t, "training live leg", "a running training showed its step against the total, the time remaining and the loss as it fell, followed the same run again with its loss after a reload, and ended completed")
}
