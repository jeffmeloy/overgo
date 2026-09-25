package server

import (
	"net/http/httptest"
	"os"
	"testing"

	"overgo/internal/inference"
	"overgo/internal/overgodb"
	"overgo/internal/remoteprovider"
	"overgo/internal/testskip"
	"overgo/internal/webuilane"
)

// screensGenerator serves the recipe fixture from a model file that exists.
type screensGenerator struct {
	*recipeInspectorGenerator
	path string
}

func (g screensGenerator) ModelProperties() inference.ModelProperties {
	properties := g.recipeInspectorGenerator.ModelProperties()
	properties.Path = g.path
	return properties
}

// TestWebUIBrowserScreens captures every workbench tab and the model
// picker at desktop and phone sizes over the fixture page and audits each
// state's layout; with OVERGO_WEBUI_LANE_SCREENS set the captures are
// written there as PNGs for a reader. A finding fails the test: the
// layout audit holds at zero, and so does every tab's mount (no error
// banner in its panel, no page error).
func TestWebUIBrowserScreens(t *testing.T) {
	if os.Getenv("OVERGO_WEBUI_LANE") != "1" {
		t.Skip(testskip.Inapplicable + ": browser screens run through cmd/webui-lane")
	}
	browserPath, err := webuilane.FindBrowser(os.Getenv("OVERGO_BROWSER"))
	if err != nil {
		t.Fatal(err)
	}
	repository, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = repository.Close() })
	t.Setenv("OVERGO_SCREENS_TEST_KEY", "screens-key")
	if _, err := remoteprovider.Declare(t.Context(), repository, remoteprovider.Provider{
		Name: "screens", Endpoint: "https://screens.example/api/v1", KeyEnvironment: "OVERGO_SCREENS_TEST_KEY", Model: "vendor/screens-model",
	}, transcriptionHTTPCommit); err != nil {
		t.Fatal(err)
	}
	// The served model is a real GGUF file, so the Tensors tab measures it rather than failing to open it.
	handler := newTestHandlerForRepository(t, repository, screensGenerator{recipeInspectorGenerator: responseRecipeGenerator(t, &fakeGenerator{}), path: writeTensorFixture(t)})
	defer handler.Close()
	httpServer := httptest.NewServer(handler)
	defer httpServer.Close()
	ctx := t.Context()
	browser, err := webuilane.Open(ctx, browserPath, httpServer.URL+"/")
	if err != nil {
		t.Fatal(err)
	}
	defer browser.Close()
	states, findings, err := webuilane.CaptureStates(ctx, browser, os.Getenv("OVERGO_WEBUI_LANE_SCREENS"))
	if err != nil {
		t.Fatal(err)
	}
	for _, finding := range findings {
		t.Error(finding)
	}
	webuilane.Leg(t, "screens leg", "%s", webuilane.CaptureSummary(states, findings))
}
