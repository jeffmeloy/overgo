package modelswap

import (
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// TestProxyRefusesCrossSiteSwap: a foreign page cannot launch or swap a
// model through the pill's own GET; the refusal comes before any launch,
// and the same GET from the page itself still reaches the launcher.
func TestProxyRefusesCrossSiteSwap(t *testing.T) {
	record := filepath.Join(t.TempDir(), "arguments")
	t.Setenv("OVERGO_TEST_MODEL_STARTUP", "arguments")
	t.Setenv("OVERGO_TEST_ARGUMENT_RECORD", record)
	supervisor, err := New(ServerLauncher{Binary: os.Args[0], Store: t.TempDir()}, 0)
	if err != nil {
		t.Fatal(err)
	}
	defer supervisor.Close()
	proxy := &Proxy{Supervisor: supervisor, Resolver: mapResolver{"fixture": {Name: "fixture", Location: "fixture.gguf"}}}
	swap := func(fetchSite, origin string) *httptest.ResponseRecorder {
		t.Helper()
		request := httptest.NewRequest(http.MethodGet, "http://localhost/health?swap=fixture", nil)
		if fetchSite != "" {
			request.Header.Set("Sec-Fetch-Site", fetchSite)
		}
		if origin != "" {
			request.Header.Set("Origin", origin)
		}
		response := httptest.NewRecorder()
		proxy.ServeHTTP(response, request)
		return response
	}
	for _, foreign := range [][2]string{{"cross-site", ""}, {"", "https://attacker.invalid"}} {
		response := swap(foreign[0], foreign[1])
		if response.Code != http.StatusForbidden || !strings.Contains(response.Body.String(), `"type":"cross_origin_request"`) {
			t.Fatalf("cross-site swap = %d %s", response.Code, response.Body)
		}
		if _, err := os.Stat(record); err == nil {
			t.Fatal("a cross-site swap reached the launcher")
		}
	}
	// The page's own request carries same-origin fetch metadata and reaches the launcher (whose child fails by design here).
	if response := swap("same-origin", ""); response.Code != http.StatusServiceUnavailable {
		t.Fatalf("same-origin swap = %d %s", response.Code, response.Body)
	}
	if _, err := os.Stat(record); err != nil {
		t.Fatalf("same-origin swap did not launch: %v", err)
	}
}
