package modelswap

import (
	"bufio"
	"context"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"overgo/internal/testcheck"
)

func TestModelSwapRoutesLiveEnvelopeBeforeEOF(t *testing.T) {
	supervisor := testcheck.Must(New(&fakeLauncher{}, 0))(t)
	defer supervisor.Close()
	wanted := Servable{Name: "native", Model: "exact-model", Location: "weights"}
	proxy := &Proxy{Supervisor: supervisor, Resolver: mapResolver{"exact-model": wanted}}
	reader, writer := io.Pipe()
	defer reader.Close()
	defer writer.Close()
	request := httptest.NewRequest(http.MethodPost, "/v1/audio/transcriptions", reader)
	request.Header.Set("Content-Type", "application/x-ndjson; charset=utf-8")
	first := "{\"model\":\"exact-model\",\"source\":{}}\n"
	done := make(chan error, 1)
	go func() {
		_, err := io.WriteString(writer, first)
		done <- err
	}()
	routed := make(chan Servable, 1)
	routeError := make(chan error, 1)
	go func() {
		servable, err := proxy.routeServable(request)
		if err != nil {
			routeError <- err
			return
		}
		routed <- servable
	}()
	select {
	case got := <-routed:
		if got != wanted || request.ContentLength != -1 {
			t.Fatalf("route=%+v length=%d", got, request.ContentLength)
		}
	case err := <-routeError:
		t.Fatal(err)
	}
	if err := <-done; err != nil {
		t.Fatal(err)
	}
	// The opening record names the child by its served name (servedAs); the stream after it is passed on unread.
	replay := bufio.NewReader(request.Body)
	if line, err := replay.ReadString('\n'); err != nil || line != "{\"model\":\"native\",\"source\":{}}\n" {
		t.Fatalf("opening envelope = %q %v, want it naming the child's served name", line, err)
	}
	next := "{\"sequence\":0}\n"
	go func() {
		_, err := io.WriteString(writer, next)
		done <- err
	}()
	if line, err := replay.ReadString('\n'); err != nil || line != next {
		t.Fatalf("continuation changed: %q %v", line, err)
	}
	if err := <-done; err != nil {
		t.Fatal(err)
	}
	testcheck.NoError(t, request.Body.Close())
	if _, err := writer.Write([]byte("closed")); err == nil {
		t.Fatal("replay lost original body close ownership")
	}
}

type mapResolver map[string]Servable

func (r mapResolver) Resolve(_ context.Context, name string) (Servable, bool, error) {
	servable, found := r[name]
	return servable, found, nil
}

// upstreamLauncher launches real loopback HTTP servers so the proxy
// test exercises genuine request streaming through the swap.
type upstreamLauncher struct {
	servers map[string]*httptest.Server
}

type upstreamProcess struct{ server *httptest.Server }

func (p upstreamProcess) URL() string                     { return p.server.URL }
func (p upstreamProcess) Ready(ctx context.Context) error { return ctx.Err() }
func (p upstreamProcess) Stop() error                     { p.server.Close(); return nil }
func (p upstreamProcess) Exited() <-chan struct{}         { return nil }

func (l *upstreamLauncher) Launch(_ context.Context, servable Servable) (Process, error) {
	name := servable.Name
	server := httptest.NewServer(http.HandlerFunc(func(response http.ResponseWriter, request *http.Request) {
		body, _ := io.ReadAll(request.Body)
		response.Write([]byte("served-by:" + name + " path:" + request.URL.Path + " body:" + string(body)))
	}))
	if l.servers == nil {
		l.servers = map[string]*httptest.Server{}
	}
	l.servers[name] = server
	return upstreamProcess{server: server}, nil
}

// TestModelSwapProxy pins the routing contract: a JSON body's model
// field swaps to and streams through the right child with the body
// replayed intact, a model-less request rides the running child, and
// a request with no model, no running child, and no default is a
// typed refusal.
func TestModelSwapProxy(t *testing.T) {
	launcher := &upstreamLauncher{}
	supervisor := testcheck.Must(New(launcher, 0))(t)
	defer supervisor.Close()
	proxy := &Proxy{
		Supervisor: supervisor,
		Resolver: mapResolver{
			"alpha": {Name: "alpha", Location: "alpha.gguf"},
			"beta":  {Name: "beta", Location: "beta.gguf"},
		},
	}
	front := httptest.NewServer(proxy)
	defer front.Close()

	refused := testcheck.Must(http.Get(front.URL + "/health"))(t)
	refused.Body.Close()
	if refused.StatusCode != http.StatusUnprocessableEntity {
		t.Fatalf("model-less request with nothing running = %d", refused.StatusCode)
	}

	first := testcheck.Must(http.Post(front.URL+"/v1/chat/completions", "application/json",
		strings.NewReader(`{"model":"alpha","messages":[{"role":"user","content":"hi"}]}`)))(t)
	firstBody, _ := io.ReadAll(first.Body)
	first.Body.Close()
	if !strings.Contains(string(firstBody), "served-by:alpha") ||
		!strings.Contains(string(firstBody), `"model":"alpha"`) {
		t.Fatalf("alpha response = %s", firstBody)
	}

	// Model-less requests ride the running child.
	plain := testcheck.Must(http.Get(front.URL + "/health"))(t)
	plainBody, _ := io.ReadAll(plain.Body)
	plain.Body.Close()
	if !strings.Contains(string(plainBody), "served-by:alpha path:/health") {
		t.Fatalf("model-less response = %s", plainBody)
	}

	// Naming beta swaps: alpha's server closes, beta serves.
	second := testcheck.Must(http.Post(front.URL+"/v1/chat/completions", "application/json",
		strings.NewReader(`{"model":"beta"}`)))(t)
	secondBody, _ := io.ReadAll(second.Body)
	second.Body.Close()
	if !strings.Contains(string(secondBody), "served-by:beta") {
		t.Fatalf("beta response = %s", secondBody)
	}
	if running, ok := supervisor.Status(); !ok || running.Name != "beta" {
		t.Fatalf("running after swap = (%+v, %t)", running, ok)
	}

	// The swap query parameter swaps too -- the GUI's picker rides a
	// health probe with ?swap= through the proxy to switch models.
	query := testcheck.Must(http.Get(front.URL + "/health?swap=alpha"))(t)
	queryBody, _ := io.ReadAll(query.Body)
	query.Body.Close()
	if !strings.Contains(string(queryBody), "served-by:alpha") {
		t.Fatalf("query-param swap response = %s", queryBody)
	}
	if running, ok := supervisor.Status(); !ok || running.Name != "alpha" {
		t.Fatalf("running after query-param swap = (%+v, %t)", running, ok)
	}
	missing := testcheck.Must(http.Get(front.URL + "/health?swap=missing-model"))(t)
	missing.Body.Close()
	if missing.StatusCode != http.StatusUnprocessableEntity {
		t.Fatalf("explicit unknown swap silently served current model: HTTP %d", missing.StatusCode)
	}

	// A model query parameter is the server's own vocabulary (analysis
	// targets); it rides the running child and never swaps.
	analyze := testcheck.Must(http.Get(front.URL + "/analyze/tensors?model=beta"))(t)
	analyzeBody, _ := io.ReadAll(analyze.Body)
	analyze.Body.Close()
	if !strings.Contains(string(analyzeBody), "served-by:alpha") {
		t.Fatalf("analyze response = %s", analyzeBody)
	}
	if running, ok := supervisor.Status(); !ok || running.Name != "alpha" {
		t.Fatalf("?model= must not swap; running = (%+v, %t)", running, ok)
	}

	// A chat body echoing the running child's own name never re-resolves
	// against the catalog -- alias collisions must not swap mid-chat.
	echo := testcheck.Must(http.Post(front.URL+"/v1/chat/completions", "application/json",
		strings.NewReader(`{"model":"ALPHA"}`)))(t)
	echoBody, _ := io.ReadAll(echo.Body)
	echo.Body.Close()
	if !strings.Contains(string(echoBody), "served-by:alpha") {
		t.Fatalf("self-echo response = %s", echoBody)
	}
}

// TestModelSwapProxyNamesItself pins the response header every proxied
// answer carries, the fact the shell reads to show its swap-proxy status.
func TestModelSwapProxyNamesItself(t *testing.T) {
	launcher := &upstreamLauncher{}
	supervisor := testcheck.Must(New(launcher, 0))(t)
	defer supervisor.Close()
	proxy := &Proxy{Supervisor: supervisor, Resolver: mapResolver{"alpha": {Name: "alpha", Location: "alpha.gguf"}}}
	front := httptest.NewServer(proxy)
	defer front.Close()
	answer := testcheck.Must(http.Get(front.URL + "/health?swap=alpha"))(t)
	answer.Body.Close()
	if answer.Header.Get("X-Overgo-Swap-Proxy") != "alpha" {
		t.Fatalf("proxied answer header = %q", answer.Header.Get("X-Overgo-Swap-Proxy"))
	}
}
