package server

import (
	"net/http/httptest"
	"os"
	"path/filepath"
	"testing"

	"overgo/internal/export"
	"overgo/internal/modelrecipe"
	"overgo/internal/overgodb"
	"overgo/internal/recipe"
	"overgo/internal/servingtest"
	"overgo/internal/testskip"
	"overgo/internal/testutil"
	"overgo/internal/webuilane"
)

// TestWebUIBrowserExportGGUF drives an export from the Export tab: the
// catalogued GGUF model is offered with its types, and a run writes the
// quantized GGUF with its Modelfile under the output and completes with its
// run.
func TestWebUIBrowserExportGGUF(t *testing.T) {
	t.Parallel()
	if os.Getenv("OVERGO_WEBUI_LANE") != "1" {
		t.Skip(testskip.Inapplicable + ": the export gguf leg uses Chromium through cmd/webui-lane")
	}
	ctx := t.Context()
	store, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	if err := servingtest.PublishActiveGGUFWithPolicy(ctx, store, testutil.HermeticLlamaGGUF(t, 64), recipe.PlacementHybrid, modelrecipe.DecodeSessionCapacity, recipe.ResidencyHostReference); err != nil {
		t.Fatal(err)
	}
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
	check(`(() => { location.hash = 'export-jobs'; return true; })()`)
	settle(`[...document.querySelectorAll('#panel-export-jobs.active select[aria-label="capability"] option')].some(option => option.textContent === 'export / hermetic.gguf')`)
	check(`(() => {
  const panel = document.querySelector('#panel-export-jobs');
  const type = panel.querySelector('select[aria-label="type"]');
  if (![...type.options].some(option => option.value === 'f16') || [...type.options].some(option => option.value === 'iq2_xxs')) return false;
  type.value = 'f16'; type.dispatchEvent(new Event('change', { bubbles: true }));
  const output = [...panel.querySelectorAll('.control-grid label.control')].find(label => label.querySelector('span').textContent.startsWith('output')).querySelector('input, textarea');
  output.value = 'shared'; output.dispatchEvent(new Event('input', { bubbles: true }));
  [...panel.querySelectorAll('button')].find(button => button.textContent === 'Run').click();
  return true;
})()`)
	settle(`document.querySelector('#panel-export-jobs [aria-label="run status"]').textContent.startsWith('completed / ')`)
	for _, name := range []string{"hermetic.f16.gguf", "Modelfile"} {
		if _, err := os.Stat(filepath.Join(root, "shared", name)); err != nil {
			t.Fatalf("the export did not write %s: %v", name, err)
		}
	}
	webuilane.Leg(t, "export gguf leg", "the Export tab offered the catalogued GGUF model with the types written without an importance matrix, and a run wrote the quantized GGUF with its Modelfile under the output and completed with its run")
}
