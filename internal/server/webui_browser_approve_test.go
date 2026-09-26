package server

import (
	"net/http/httptest"
	"os"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/overgodb"
	"overgo/internal/testskip"
	"overgo/internal/testutil"
	"overgo/internal/webuilane"
)

// TestWebUIBrowserApproveInPlace decides waiting operations where they
// surface: the status line names the first decision and grants it in one
// click, offering to grant every waiting one together; the Inbox decides
// several at once from their checked cards, and nothing is left waiting.
func TestWebUIBrowserApproveInPlace(t *testing.T) {
	t.Parallel()
	if os.Getenv("OVERGO_WEBUI_LANE") != "1" {
		t.Skip(testskip.Inapplicable + ": the approve in place leg uses Chromium through cmd/webui-lane")
	}
	store, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	handler := newTestHandlerForRepository(t, store, responseRecipeGenerator(t, &fakeGenerator{}))
	recipeID := testutil.ArtifactID(t, artifact.KindRecipe, "approve-in-place-recipe")
	for _, approval := range []struct{ code, summary string }{
		{"publish-model", "Publish the model"}, {"export-weights", "Export the weights"}, {"prune-store", "Prune the store"},
	} {
		blockOperation(t, handler, recipeID, testutil.ArtifactID(t, artifact.KindRun, approval.code+"-run"), approval.code, approval.summary)
	}
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
	settle(`!!document.querySelector('.composer textarea') && !!document.querySelector('.operation-decisions button')`)
	check(`(() => {
  window.approveLine = () => document.querySelector('.operation-decisions');
  window.approveButton = (root, label) => [...root.querySelectorAll('button')].find(button => button.textContent === label);
  return true;
})()`)

	// The status line names the first waiting decision and offers every one together.
	check(`approveLine().textContent.startsWith('generation waits on Publish the model') && !!approveButton(approveLine(), 'Grant publish-model') && !!approveButton(approveLine(), 'Grant all 3')`)
	check(`document.getElementById('inbox-count').textContent === '3 waiting'`)

	// One click grants it in place; the line moves to the next.
	check(`(() => { approveButton(approveLine(), 'Grant publish-model').click(); return true; })()`)
	settle(`approveLine().textContent.startsWith('generation waits on Export the weights') && !!approveButton(approveLine(), 'Grant all 2') && document.getElementById('inbox-count').textContent === '2 waiting'`)

	// The Inbox decides the checked ones together.
	check(`(() => { document.getElementById('inbox-count').click(); return true; })()`)
	settle(`document.querySelectorAll('#panel-inbox.active input[data-operation]:checked').length === 2 && !!approveButton(document.querySelector('#panel-inbox'), 'Grant selected') && !!approveButton(document.querySelector('#panel-inbox'), 'Decline selected')`)
	check(`(() => { approveButton(document.querySelector('#panel-inbox'), 'Grant selected').click(); return true; })()`)
	settle(`document.querySelector('#panel-inbox').textContent.includes('grant recorded for 2 operations')`)
	settle(`document.querySelector('#panel-inbox').textContent.includes('Nothing is waiting on an operator decision.') && document.getElementById('inbox-count').hidden && !approveLine().querySelector('button')`)
	webuilane.Leg(t, "approve in place leg", "the status line named the first waiting decision and granted it in one click with an offer to grant all, and the Inbox granted the two checked decisions together, leaving nothing waiting")
}
