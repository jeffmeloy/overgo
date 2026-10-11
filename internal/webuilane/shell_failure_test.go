package webuilane

import (
	"net/http"
	"net/http/httptest"
	"os"
	"strings"
	"testing"

	"overgo/internal/testskip"
)

// TestWebUIBrowserWaitEndsWhenTheShellFailed holds a wait to the page's own terminal
// state: a shell that failed to start offers only a page reload, so a
// predicate that cannot hold ends at once with what the page shows, instead
// of waiting until the lane's timeout with no page state at all. A page that
// is merely slow keeps waiting until its predicate holds.
func TestWebUIBrowserWaitEndsWhenTheShellFailed(t *testing.T) {
	t.Parallel()
	if os.Getenv("OVERGO_WEBUI_LANE") != "1" {
		testskip.NotApplicable(t, "the shell-failure wait runs Chromium through cmd/webui-lane")
	}
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "text/html")
		// As the real shell does, the page fails after it has loaded.
		_, _ = w.Write([]byte(`<!doctype html><div id="panels"></div><script>addEventListener("load", () => setTimeout(() => { document.getElementById("panels").innerHTML = "<p>Could not load the workspace modules.</p><button>Reload the page</button>"; }));</script>`))
	}))
	defer server.Close()
	executable, err := FindBrowser(os.Getenv("OVERGO_BROWSER"))
	if err != nil {
		t.Fatal(err)
	}
	browser, err := Open(t.Context(), executable, server.URL)
	if err != nil {
		t.Fatal(err)
	}
	defer browser.Close()
	if err := browser.Eventually(t.Context(), `!!document.querySelector('.composer textarea')`); err == nil || !strings.Contains(err.Error(), "shell failed to start") || !strings.Contains(err.Error(), "Could not load the workspace modules.") {
		t.Fatalf("a wait on a failed shell = %v, want it to end naming what the page shows", err)
	}
	if err := browser.Evaluate(t.Context(), `document.getElementById("panels").replaceChildren()`, nil); err != nil {
		t.Fatal(err)
	}
	if err := browser.Eventually(t.Context(), `(() => { setTimeout(() => document.body.append(Object.assign(document.createElement('i'), {id: 'late'}))); return !!document.getElementById('late'); })()`); err != nil {
		t.Fatalf("a predicate the page reaches later = %v", err)
	}
	// A wait the page then blocks ends naming what the page showed when the
	// wait began, both ends of a long text, and the error the workspace
	// recorded: the page reports them itself.
	if err := browser.Evaluate(t.Context(), `document.getElementById("panels").innerText = "Choose a model " + "x".repeat(2000) + " streamed tail"; window.overgo = {errors: ["model list failed"]}`, nil); err != nil {
		t.Fatal(err)
	}
	err = browser.Eventually(t.Context(), `(() => { setTimeout(() => alert("blocked")); return false; })()`)
	for _, want := range []string{"JavaScript dialog", `readyState complete, panels "Choose a model x`, "x ... x", `x streamed tail", last error "model list failed"`} {
		if err == nil || !strings.Contains(err.Error(), want) {
			t.Fatalf("a wait the page blocked = %v, want it to name %q", err, want)
		}
	}
}
