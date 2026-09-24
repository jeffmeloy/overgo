package server

import (
	"context"
	"net/http/httptest"
	"os"
	"path/filepath"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/operation"
	"overgo/internal/overgodb"
	"overgo/internal/testskip"
	"overgo/internal/testutil"
	"overgo/internal/webuilane"
)

// onboardingTask adds a model from the hub in the Library: search, then one
// add, which downloads, registers and validates it (three clicks before the
// chain: download, register, validate).
var onboardingTask = effortTask{"add a model from the hub", []effortStep{
	{"type", `input[aria-label="Search the Hugging Face hub"]`, "lane"},
	{"key", "Enter", ""},
	{"click", "#panel-library button", "add"},
}, `document.querySelector("#panel-library").textContent.includes("validating in operation")`}

// TestWebUIBrowserModelOnboarding takes a model from a hub search to a
// validation operation in a real browser, through the real download,
// registration and validation routes over a fake hub, and holds what the
// task costs at its effort ceiling.
func TestWebUIBrowserModelOnboarding(t *testing.T) {
	t.Parallel()
	if os.Getenv("OVERGO_WEBUI_LANE") != "1" {
		t.Skip(testskip.Inapplicable + ": the onboarding leg runs through cmd/webui-lane")
	}
	weights := filepath.Join(t.TempDir(), "lane.gguf")
	if err := os.WriteFile(weights, []byte("GGUF onboarding fixture"), 0o644); err != nil {
		t.Fatal(err)
	}
	hub := hubFixtureServer(t, weights)
	store, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	recipeID := testutil.ArtifactID(t, artifact.KindRecipe, "onboarding-recipe")
	runID := testutil.ArtifactID(t, artifact.KindRun, "onboarding-run")
	intake := LibraryIntake{
		ModelFiles: func(path, projector string) (string, string, error) { return path, projector, nil },
		Register: func(_ context.Context, _ *overgodb.Store, path, _ string) (map[string]any, error) {
			return map[string]any{"recipe": recipeID.String(), "path": path}, nil
		},
		Validate: func(context.Context, *overgodb.Store, string, string, []string, int) (artifact.ID, operation.Executor, error) {
			return recipeID, func(context.Context, operation.Reporter) (operation.Completion, error) {
				return operation.Completion{Run: runID}, nil
			}, nil
		},
	}
	downloads := filepath.Join(t.TempDir(), "downloads")
	if err := os.MkdirAll(downloads, 0o755); err != nil {
		t.Fatal(err)
	}
	handler, err := New(Config{
		RuntimePolicy: testRuntimePolicy(), Repository: store, LibraryIntake: intake,
		HubEndpoint: hub.URL, HubDownloadRoot: downloads,
	}, GenerationRefused{Reason: idleRefusal})
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
	browser, err := webuilane.Open(ctx, path, server.URL+"/app.html#library")
	if err != nil {
		t.Fatal(err)
	}
	defer browser.Close()
	settle := func(expression string) {
		t.Helper()
		if err := browser.Eventually(ctx, expression); err != nil {
			var page string
			_ = browser.Evaluate(ctx, `JSON.stringify({errors: overgo.errors, library: document.querySelector("#panel-library") && document.querySelector("#panel-library").innerText.slice(0, 800)})`, &page)
			t.Fatalf("%s: %v; %s", expression, err, page)
		}
	}
	if err := webuilane.SettleViewport(ctx, browser, webuilane.ScreenViewports[0]); err != nil {
		t.Fatal(err)
	}
	settle(`!!document.querySelector('#panel-library.active input[aria-label="Search the Hugging Face hub"]') && overgo.api.inFlight() === 0`)
	effort, requests := measureEffort(t, ctx, browser, settle, onboardingTask)
	assertBrowserPredicate(t, ctx, browser, `overgo.errors.length === 0`)
	t.Logf("one action onboarding leg: a hub model reached validation for %+v with %d requests", effort, requests)
}
