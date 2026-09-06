package modelswap

import (
	"io"
	"net/http"
	"net/http/httptest"
	"testing"
)

// TestModelSwapProxyIdleHandlerAnswersColdStart pins the cold start: with
// nothing running and no default the idle handler answers under the
// proxy's header with no model named, a named swap launches the child,
// and from then on the child answers.
func TestModelSwapProxyIdleHandlerAnswersColdStart(t *testing.T) {
	launcher := &upstreamLauncher{}
	supervisor, err := New(launcher, 0)
	if err != nil {
		t.Fatal(err)
	}
	defer supervisor.Close()
	proxy := &Proxy{
		Supervisor: supervisor,
		Resolver:   mapResolver{"alpha": {Name: "alpha", Location: "alpha.gguf"}},
		Idle: http.HandlerFunc(func(response http.ResponseWriter, request *http.Request) {
			response.Header().Set("X-Idle", request.URL.Path)
			io.WriteString(response, "idle")
		}),
	}
	front := httptest.NewServer(proxy)
	defer front.Close()

	idle, err := http.Get(front.URL + "/app.html")
	if err != nil {
		t.Fatal(err)
	}
	body, _ := io.ReadAll(idle.Body)
	idle.Body.Close()
	via, present := idle.Header["X-Overgo-Swap-Proxy"]
	if idle.StatusCode != http.StatusOK || string(body) != "idle" || idle.Header.Get("X-Idle") != "/app.html" || !present || via[0] != "" {
		t.Fatalf("cold request = %d %q header=%q", idle.StatusCode, body, idle.Header)
	}
	if _, running := supervisor.Status(); running {
		t.Fatal("the idle answer launched a child")
	}

	launched, err := http.Get(front.URL + "/health?swap=alpha")
	if err != nil {
		t.Fatal(err)
	}
	launched.Body.Close()
	if launched.StatusCode != http.StatusOK || launched.Header.Get("X-Overgo-Swap-Proxy") != "alpha" {
		t.Fatalf("named swap = %d %q", launched.StatusCode, launched.Header.Get("X-Overgo-Swap-Proxy"))
	}
	served, err := http.Get(front.URL + "/app.html")
	if err != nil {
		t.Fatal(err)
	}
	served.Body.Close()
	if served.Header.Get("X-Idle") != "" || served.Header.Get("X-Overgo-Swap-Proxy") != "alpha" {
		t.Fatalf("request after the launch = %q", served.Header)
	}
}
