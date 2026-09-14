package server

import (
	"context"
	"errors"
	"net/http/httptest"
	"os"
	"testing"
	"time"

	"overgo/internal/overgodb"
	"overgo/internal/remoteprovider"
	"overgo/internal/testskip"
	"overgo/internal/webuilane"
)

// tabSettle bounds the wait for a tab's own request before its capture.
const tabSettle = 8 * time.Second

// TestWebUIBrowserScreens captures every workbench tab and the model
// picker at desktop and phone sizes over the fixture page and audits each
// state's layout; with OVERGO_WEBUI_LANE_SCREENS set the captures are
// written there as PNGs for a reader. A finding fails the test: the
// layout audit holds at zero.
func TestWebUIBrowserScreens(t *testing.T) {
	if os.Getenv("OVERGO_WEBUI_LANE") != "1" {
		t.Skip(testskip.ShortIntegration + ": browser screens run through cmd/webui-lane")
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
	handler := newTestHandlerForRepository(t, repository, responseRecipeGenerator(t, &fakeGenerator{}))
	defer handler.Close()
	httpServer := httptest.NewServer(handler)
	defer httpServer.Close()
	ctx, cancel := context.WithTimeoutCause(t.Context(), 4*time.Minute, errors.New("webui lane: the screens did not capture"))
	defer cancel()
	browser, err := webuilane.Open(ctx, browserPath, httpServer.URL+"/")
	if err != nil {
		t.Fatal(err)
	}
	defer browser.Close()
	states, findings, err := webuilane.CaptureStates(ctx, browser, os.Getenv("OVERGO_WEBUI_LANE_SCREENS"), tabSettle)
	if err != nil {
		t.Fatal(err)
	}
	for _, finding := range findings {
		t.Error(finding)
	}
	t.Log(webuilane.CaptureSummary(states, findings))
}
