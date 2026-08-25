package modelswap

import (
	"context"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

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
	supervisor, err := New(launcher, 0)
	if err != nil {
		t.Fatal(err)
	}
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

	refused, err := http.Get(front.URL + "/health")
	if err != nil {
		t.Fatal(err)
	}
	refused.Body.Close()
	if refused.StatusCode != http.StatusUnprocessableEntity {
		t.Fatalf("model-less request with nothing running = %d", refused.StatusCode)
	}

	first, err := http.Post(front.URL+"/v1/chat/completions", "application/json",
		strings.NewReader(`{"model":"alpha","messages":[{"role":"user","content":"hi"}]}`))
	if err != nil {
		t.Fatal(err)
	}
	firstBody, _ := io.ReadAll(first.Body)
	first.Body.Close()
	if !strings.Contains(string(firstBody), "served-by:alpha") ||
		!strings.Contains(string(firstBody), `"model":"alpha"`) {
		t.Fatalf("alpha response = %s", firstBody)
	}

	// Model-less requests ride the running child.
	plain, err := http.Get(front.URL + "/health")
	if err != nil {
		t.Fatal(err)
	}
	plainBody, _ := io.ReadAll(plain.Body)
	plain.Body.Close()
	if !strings.Contains(string(plainBody), "served-by:alpha path:/health") {
		t.Fatalf("model-less response = %s", plainBody)
	}

	// Naming beta swaps: alpha's server closes, beta serves.
	second, err := http.Post(front.URL+"/v1/chat/completions", "application/json",
		strings.NewReader(`{"model":"beta"}`))
	if err != nil {
		t.Fatal(err)
	}
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
	query, err := http.Get(front.URL + "/health?swap=alpha")
	if err != nil {
		t.Fatal(err)
	}
	queryBody, _ := io.ReadAll(query.Body)
	query.Body.Close()
	if !strings.Contains(string(queryBody), "served-by:alpha") {
		t.Fatalf("query-param swap response = %s", queryBody)
	}
	if running, ok := supervisor.Status(); !ok || running.Name != "alpha" {
		t.Fatalf("running after query-param swap = (%+v, %t)", running, ok)
	}

	// A model query parameter is the server's own vocabulary (analysis
	// targets); it rides the running child and never swaps.
	analyze, err := http.Get(front.URL + "/analyze/tensors?model=beta")
	if err != nil {
		t.Fatal(err)
	}
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
	echo, err := http.Post(front.URL+"/v1/chat/completions", "application/json",
		strings.NewReader(`{"model":"ALPHA"}`))
	if err != nil {
		t.Fatal(err)
	}
	echoBody, _ := io.ReadAll(echo.Body)
	echo.Body.Close()
	if !strings.Contains(string(echoBody), "served-by:alpha") {
		t.Fatalf("self-echo response = %s", echoBody)
	}
}
