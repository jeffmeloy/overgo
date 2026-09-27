package server

import (
	"net/http/httptest"
	"os"
	"path/filepath"
	"testing"

	"overgo/internal/export"
	"overgo/internal/overgodb"
	"overgo/internal/testskip"
	"overgo/internal/webuilane"
)

// TestWebUIBrowserModelMerge drives a merge from Model Builder: the
// catalogued base offers the other model as its fine-tuned choice, the
// adapter fold is listed with its refusal and cannot be chosen, and a run
// writes the merged GGUF under the output and completes with its run.
func TestWebUIBrowserModelMerge(t *testing.T) {
	t.Parallel()
	if os.Getenv("OVERGO_WEBUI_LANE") != "1" {
		t.Skip(testskip.Inapplicable + ": the model merge leg uses Chromium through cmd/webui-lane")
	}
	ctx := t.Context()
	store, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	mergeFixture(t, store)
	root := t.TempDir()
	workspace := NewGGUFWorkspace(store, root, 64, GGUFWriters{Types: export.GGUFTypes(), Export: export.GGUF, Merge: export.MergeGGUF})
	handler, err := New(Config{Repository: store}, &workspaceTestRuntime{fakeGenerator: &fakeGenerator{}, WorkflowWorkspaceAPI: workspace})
	if err != nil {
		t.Fatal(err)
	}
	defer handler.Close()
	server := httptest.NewServer(handler)
	defer server.Close()
	path, err := webuilane.FindBrowser(os.Getenv("OVERGO_BROWSER"))
	if err != nil {
		t.Fatal(err)
	}
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
	check(`(() => { location.hash = 'model-builder'; return true; })()`)
	settle(`[...document.querySelectorAll('#panel-model-builder.active select[aria-label="capability"] option')].some(option => option.textContent === 'merge / hermetic.gguf')`)
	check(`(() => {
  const panel = document.querySelector('#panel-model-builder');
  const capability = panel.querySelector('select[aria-label="capability"]');
  const fold = [...capability.options].find(option => option.textContent.startsWith('merge / an adapter into its base'));
  if (!fold || !fold.disabled || !fold.textContent.includes('no implementation yet')) return false;
  capability.value = [...capability.options].find(option => option.textContent === 'merge / hermetic.gguf').value;
  capability.dispatchEvent(new Event('change', { bubbles: true }));
  return true;
})()`)
	check(`(() => {
  const panel = document.querySelector('#panel-model-builder');
  const tuned = panel.querySelector('select[aria-label="fine tuned"]');
  if ([...tuned.options].filter(option => option.value).map(option => option.value).join() !== 'tuned.gguf') return false;
  tuned.value = 'tuned.gguf'; tuned.dispatchEvent(new Event('change', { bubbles: true }));
  const entry = (name) => [...panel.querySelectorAll('.control-grid label.control')].find(label => label.querySelector('span').textContent.startsWith(name)).querySelector('input, textarea');
  entry('scale').value = '0.5'; entry('scale').dispatchEvent(new Event('input', { bubbles: true }));
  entry('output').value = 'merged'; entry('output').dispatchEvent(new Event('input', { bubbles: true }));
  [...panel.querySelectorAll('button')].find(button => button.textContent === 'Run').click();
  return true;
})()`)
	settle(`document.querySelector('#panel-model-builder [aria-label="run status"]').textContent.startsWith('completed / ')`)
	if _, err := os.Stat(filepath.Join(root, "merged", "hermetic.merged.gguf")); err != nil {
		t.Fatalf("the merge did not write its GGUF: %v", err)
	}
	webuilane.Leg(t, "model merge leg", "Model Builder offered the catalogued base with the other model as its fine-tuned choice, listed the adapter fold with its refusal and no way to run it, and a run wrote the merged GGUF under the output and completed with its run")
}
