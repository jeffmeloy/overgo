package modelswap

import (
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
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
			// A JSON body the proxy read for its model field arrives whole.
			body, _ := io.ReadAll(request.Body)
			io.WriteString(response, "idle"+string(body))
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
	posted, err := http.Post(front.URL+"/library/register", "application/json", strings.NewReader(`{"kind":"provider","name":"p"}`))
	if err != nil {
		t.Fatal(err)
	}
	echoed, _ := io.ReadAll(posted.Body)
	posted.Body.Close()
	if string(echoed) != `idle{"kind":"provider","name":"p"}` {
		t.Fatalf("idle handler received %q", echoed)
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
