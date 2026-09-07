package modelswap

import (
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

// TestModelSwapProxyAdmitsAsTheServerDoes pins the proxy's admission: a
// foreign origin cannot place a key or launch a child through the key
// route, a foreign Host on the real transport cannot launch through a
// model-less request or reach the idle handler, and a same-origin request
// passes to the intake and the child.
func TestModelSwapProxyAdmitsAsTheServerDoes(t *testing.T) {
	launcher := &upstreamLauncher{}
	supervisor, err := New(launcher, 0)
	if err != nil {
		t.Fatal(err)
	}
	defer supervisor.Close()
	keys := &keyRecorder{}
	alpha := Servable{Name: "alpha", Location: "alpha.gguf"}
	idleCalls := 0
	proxy := &Proxy{Supervisor: supervisor, Resolver: mapResolver{"alpha": alpha}, Keys: keys,
		Idle: http.HandlerFunc(func(http.ResponseWriter, *http.Request) { idleCalls++ })}
	front := httptest.NewServer(proxy)
	defer front.Close()
	body := `{"location":"remote://fake/vendor/model","key":"secret"}`
	send := func(method, path, host string, headers map[string]string) *http.Response {
		t.Helper()
		request, err := http.NewRequest(method, front.URL+path, strings.NewReader(body))
		if err != nil {
			t.Fatal(err)
		}
		request.Header.Set("Content-Type", "application/json")
		for name, value := range headers {
			request.Header.Set(name, value)
		}
		if host != "" {
			request.Host = host
		}
		response, err := front.Client().Do(request)
		if err != nil {
			t.Fatal(err)
		}
		t.Cleanup(func() { response.Body.Close() })
		return response
	}
	refusal := func(response *http.Response, want string) {
		t.Helper()
		answer, _ := io.ReadAll(response.Body)
		if response.StatusCode != http.StatusForbidden || !strings.Contains(string(answer), `"type":"`+want+`"`) {
			t.Fatalf("status=%d body=%s, want 403 %s", response.StatusCode, answer, want)
		}
	}

	// A foreign origin's key post: refused before the key lands and before any child launches.
	refusal(send(http.MethodPost, "/providers/key", "", map[string]string{"Origin": "https://attacker.invalid", "Sec-Fetch-Site": "cross-site"}), "cross_origin_request")
	// A rebound Host on the real transport: refused before the idle handler answers.
	refusal(send(http.MethodGet, "/app.html", "attacker.invalid", nil), "invalid_host")
	// A rebound Host naming a model: refused before the child launches.
	refusal(send(http.MethodGet, "/health?swap=alpha", "attacker.invalid", nil), "invalid_host")
	if keys.reference != "" || idleCalls != 0 {
		t.Fatalf("a refused request reached the intake (%+v) or the idle handler (%d)", *keys, idleCalls)
	}
	if _, running := supervisor.Status(); running {
		t.Fatal("a refused request launched a child")
	}
	// Same-origin: the idle handler answers while nothing serves, the key lands, the child launches.
	if response := send(http.MethodGet, "/app.html", "", map[string]string{"Sec-Fetch-Site": "same-origin"}); response.StatusCode != http.StatusOK || idleCalls != 1 {
		t.Fatalf("same-origin idle request = %d idle=%d", response.StatusCode, idleCalls)
	}
	if response := send(http.MethodGet, "/health?swap=alpha", "", nil); response.StatusCode != http.StatusOK {
		t.Fatalf("named swap = %d", response.StatusCode)
	}
	if response := send(http.MethodPost, "/providers/key", "", map[string]string{"Origin": "http://" + strings.TrimPrefix(front.URL, "http://")}); response.StatusCode != http.StatusOK || keys.key != "secret" {
		t.Fatalf("same-origin key post = %d keys=%+v", response.StatusCode, *keys)
	}
}
