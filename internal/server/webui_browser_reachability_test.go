package server

import (
	"context"
	"errors"
	"net/http/httptest"
	"os"
	"testing"
	"time"

	"overgo/internal/overgodb"
	"overgo/internal/testevidence"
	"overgo/internal/webuilane"
)

// TestWebUIBrowserWorkspaceReachability opens each workspace the launch
// enabled in a real browser and confirms a workspace it did not enable
// stays out of the navigation with its reason.
func TestWebUIBrowserWorkspaceReachability(t *testing.T) {
	if os.Getenv("OVERGO_WEBUI_LANE") != "1" {
		t.Skip(testevidence.ShortIntegrationSkip + ": workspace reachability runs through cmd/webui-lane")
	}
	store, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	handler := newTestHandlerForRepository(t, store, launchWorkspaceGenerator(t, WorkflowTraining, WorkflowModelBuild))
	handler.config.Evaluation = launchEvaluationWorkspace(t)
	defer handler.Close()
	server := httptest.NewServer(handler)
	defer server.Close()
	path, err := webuilane.FindBrowser(os.Getenv("OVERGO_BROWSER"))
	if err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithTimeoutCause(t.Context(), 60*time.Second, errors.New("workspace reachability did not settle"))
	defer cancel()
	browser, err := webuilane.Open(ctx, path, server.URL+"/app.html#training-jobs")
	if err != nil {
		t.Fatal(err)
	}
	defer browser.Close()
	settle := func(expression string) {
		t.Helper()
		if err := browser.Eventually(ctx, expression); err != nil {
			t.Fatalf("%s: %v", expression, err)
		}
	}
	// Each enabled workspace mounts its own controls: the workflow tabs list
	// the launch's capability, Evaluations lists the evaluable model's plan.
	settle(`!!document.querySelector('#panel-training-jobs.active select[aria-label="capability"] option')`)
	assertBrowserPredicate(t, ctx, browser, `(() => { location.hash = 'model-builder'; return true; })()`)
	settle(`!!document.querySelector('#panel-model-builder.active select[aria-label="capability"] option')`)
	assertBrowserPredicate(t, ctx, browser, `(() => { location.hash = 'evaluations'; return true; })()`)
	settle(`!!document.querySelector('#panel-evaluations.active select[aria-label="model"] option') && !!document.querySelector('#panel-evaluations input[type="checkbox"]')`)
	assertBrowserPredicate(t, ctx, browser, `[...document.querySelectorAll('button.tab')].filter((button) => !button.hidden).map((button) => button.textContent).join(',').includes('Train,Model Builder')`)
	// The workspace the launch did not enable is absent from the navigation and the manifest names why.
	assertBrowserPredicate(t, ctx, browser, `(async () => {
  const manifest = await overgo.api.get('/workspace/manifest');
  const tab = manifest.tabs.find((candidate) => candidate.id === 'export-jobs');
  const button = [...document.querySelectorAll('button.tab')].find((candidate) => candidate.textContent === 'Export');
  return !tab.enabled && tab.refusal.length > 0 && button.hidden && button.title === tab.refusal;
})()`)
	t.Log("workspace reachability leg: Train, Model Builder and Evaluations mount from the launch declaration; Export stays refused with its reason")
}
