package server

import (
	"context"
	"net/http/httptest"
	"os"
	"strconv"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/overgodb"
	"overgo/internal/recipe"
	"overgo/internal/testskip"
	"overgo/internal/testutil"
	"overgo/internal/webuilane"
)

// unofferedGenerator serves two capabilities for one task: one declares an
// image output, the other an embeddings output the workbench cannot show.
type unofferedGenerator struct {
	*generationWorkspaceGenerator
	embedding WorkflowCapability
}

func (generator *unofferedGenerator) WorkflowCapabilities(ctx context.Context, kind WorkflowKind) ([]WorkflowCapability, error) {
	capabilities, err := generator.generationWorkspaceGenerator.WorkflowCapabilities(ctx, kind)
	if err != nil || kind != WorkflowGeneration {
		return capabilities, err
	}
	return append([]WorkflowCapability{generator.embedding}, capabilities...), nil
}

// TestWebUIBrowserUnofferedTask lists a capability whose declared output the
// workbench has no surface for (an embedding, as the catalog's image
// embedding task declares) as refused and named, never selected and run as
// an image: the composer once took every task it did not recognise for an
// image generator. The capability that declares an image stays offered, and
// the mode's gallery shows the kind that capability declares.
func TestWebUIBrowserUnofferedTask(t *testing.T) {
	t.Parallel()
	if os.Getenv("OVERGO_WEBUI_LANE") != "1" {
		t.Skip(testskip.Inapplicable + ": the unoffered task leg runs through cmd/webui-lane")
	}
	store, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	stages := []recipe.Stage{{Node: recipe.Node{ID: "generate", Module: "test.generate"}}}
	controls := []WorkflowControl{{Name: "prompt", Type: WorkflowControlText, Required: true}}
	image := WorkflowCapability{
		Task: recipe.TaskImageGen, Recipe: testutil.ArtifactID(t, artifact.KindRecipe, "unoffered-image"), Name: "Image model (UI fixture)",
		Stages: stages, Controls: controls, Outputs: []recipe.Output{{Name: "image", Data: recipe.DataImage}},
	}
	embedding := image
	embedding.Recipe = testutil.ArtifactID(t, artifact.KindRecipe, "unoffered-embedding")
	embedding.Name = "Embedding model (UI fixture)"
	embedding.Outputs = []recipe.Output{{Name: "embedding", Data: recipe.DataEmbeddings}}
	generator := &unofferedGenerator{generationWorkspaceGenerator: &generationWorkspaceGenerator{
		fakeGenerator: &fakeGenerator{}, repository: store, run: testutil.ArtifactID(t, artifact.KindRun, "unoffered"), capability: image,
	}, embedding: embedding}
	handler, err := New(Config{Repository: store}, generator)
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
	ctx := t.Context()
	browser, err := webuilane.Open(ctx, path, server.URL)
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
	check := func(expression string) { t.Helper(); assertBrowserPredicate(t, ctx, browser, expression) }
	settle(`!!document.querySelector('.composer select[aria-label="mode"] option[value="image-gen"]')`)
	check(`(() => { const mode = document.querySelector('.composer select[aria-label="mode"]'); mode.value = 'image-gen'; mode.dispatchEvent(new Event('change')); return true; })()`)
	picker := `document.querySelector('.task-model-selector select[aria-label="generation model"]:not([disabled])')`
	settle(`!!` + picker + ` && ` + picker + `.options.length === 2`)
	refused, offered := strconv.Quote(embedding.Recipe.String()), strconv.Quote(image.Recipe.String())
	check(`(() => {
  const picker = ` + picker + `;
  const refused = [...picker.options].find((option) => option.value === ` + refused + `);
  const offered = [...picker.options].find((option) => option.value === ` + offered + `);
  return refused.disabled && refused.title === 'The workbench cannot show embeddings output yet' &&
    !offered.disabled && picker.value === ` + offered + ` &&
    document.querySelector('.composer .intake-strip.gallery').getAttribute('aria-label') === 'Recent image outputs';
})()`)
	// Chosen anyway, the refused capability is not selected to run, and the note says why.
	check(`(() => { const picker = ` + picker + `; picker.value = ` + refused + `; picker.dispatchEvent(new Event('change')); return true; })()`)
	settle(`[...document.querySelectorAll('.composer .note[role="status"]')].some((note) => !note.hidden && note.textContent === 'The workbench cannot show embeddings output yet. Choose another model for this task.')`)
	check(`overgo.errors.length === 0`)
	webuilane.Leg(t, "unoffered task leg", "a capability declaring embeddings output is listed refused and named, the image capability stays offered and selected, and the gallery follows the declared image output")
}
