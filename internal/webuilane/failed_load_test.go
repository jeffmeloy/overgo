package webuilane

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"slices"
	"strconv"
	"strings"
	"testing"

	"overgo/internal/testskip"
)

// TestFailedLoadsKeepTheNewestFew holds what a socket keeps of the browser's
// log, with no browser: a network error is kept with its address and the
// browser's words, an entry that is no network error is not, and of more
// failed loads than the socket keeps the newest stay.
func TestFailedLoadsKeepTheNewestFew(t *testing.T) {
	t.Parallel()
	socket := &webSocket{}
	logged := func(source, level, address string) {
		params, err := json.Marshal(map[string]any{"entry": map[string]string{"source": source, "level": level, "text": "Failed to load resource: net::ERR_EMPTY_RESPONSE", "url": address}})
		if err != nil {
			t.Fatal(err)
		}
		socket.retainEvent(cdpEvent{Method: "Log.entryAdded", Params: params})
	}
	logged("javascript", "error", "/thrown.js")
	logged("network", "warning", "/slow.js")
	if len(socket.failedLoads) != 0 {
		t.Fatalf("entries that are no network error were kept as failed loads: %v", socket.failedLoads)
	}
	logged("network", "error", "/first.js")
	if want := []string{"/first.js: Failed to load resource: net::ERR_EMPTY_RESPONSE"}; !slices.Equal(socket.failedLoads, want) {
		t.Fatalf("one failed load is kept as %v, want %v", socket.failedLoads, want)
	}
	for extra := range failedLoadsKept {
		logged("network", "error", "/later-"+strconv.Itoa(extra)+".js")
	}
	if kept := socket.failedLoads; len(kept) != failedLoadsKept || !strings.HasPrefix(kept[0], "/later-0.js") || !strings.HasPrefix(kept[len(kept)-1], "/later-"+strconv.Itoa(failedLoadsKept-1)+".js") {
		t.Fatalf("after one more failed load than the socket keeps it holds %v, want the newest %d", kept, failedLoadsKept)
	}
}

// TestWebUIBrowserFailedShellNamesItsFailedLoads holds a wait on a shell that could
// not load its scripts to the browser's own account of those loads. The page
// is told only that a script failed; the browser knows why, and without it a
// failure that comes and goes says nothing of its cause. Two ways a load
// fails are held: a server that ends the connection with no answer, which
// the wait names with the browser's network error, and one that answers an
// error status, which it names with the status. A shell that failed with no
// load failing says nothing of loads.
func TestWebUIBrowserFailedShellNamesItsFailedLoads(t *testing.T) {
	t.Parallel()
	if os.Getenv("OVERGO_WEBUI_LANE") != "1" {
		testskip.NotApplicable(t, "the failed-load wait runs Chromium through cmd/webui-lane")
	}
	// As the real shell does, the page shows its failure and the reload it offers once its loads have failed.
	const page = `<!doctype html><div id="panels"></div><script>
let failed = 0, loaded = false;
const show = () => { if (failed === 2 && loaded) document.getElementById("panels").innerHTML = "<p>failed to load the scripts</p><button>Reload the page</button>"; };
for (const source of ["/cut.js", "/absent.js"]) {
  const script = document.createElement("script");
  script.src = source;
  script.onerror = () => { failed++; show(); };
  document.head.appendChild(script);
}
addEventListener("load", () => setTimeout(() => { loaded = true; show(); }));
</script>`
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/cut.js":
			// The connection ends with no answer.
			if connection, _, err := http.NewResponseController(w).Hijack(); err == nil {
				_ = connection.Close()
			}
		case "/absent.js":
			http.NotFound(w, r)
		case "/quiet":
			w.Header().Set("Content-Type", "text/html")
			_, _ = w.Write([]byte(`<!doctype html><div id="panels"></div><script>addEventListener("load", () => setTimeout(() => { document.getElementById("panels").innerHTML = "<p>no workspace</p><button>Reload the page</button>"; }));</script>`))
		default:
			w.Header().Set("Content-Type", "text/html")
			_, _ = w.Write([]byte(page))
		}
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
	err = browser.Eventually(t.Context(), `!!document.querySelector('.composer textarea')`)
	if err == nil || !strings.Contains(err.Error(), "shell failed to start") {
		t.Fatalf("a wait on a shell whose scripts did not load = %v, want it to end on the failed shell", err)
	}
	for _, want := range []string{server.URL + "/cut.js", "net::ERR_", server.URL + "/absent.js", "404"} {
		if !strings.Contains(err.Error(), want) {
			t.Errorf("the wait ended with %v, which does not name %q: the browser's account of each failed load", err, want)
		}
	}

	// A shell that failed with every load answered names no load.
	quiet, err := Open(t.Context(), executable, server.URL+"/quiet")
	if err != nil {
		t.Fatal(err)
	}
	defer quiet.Close()
	if err := quiet.Eventually(t.Context(), `false`); err == nil || !strings.Contains(err.Error(), "shell failed to start") || strings.Contains(err.Error(), "failed load") {
		t.Fatalf("a wait on a shell that failed with no load failing = %v, want it to end on the failed shell and name no load", err)
	}
}
