package modelswap

import (
	"context"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"overgo/internal/apimanifest"
)

// heldOpenLauncher launches children whose activity stream answers its headers, names itself held open as the
// workbench's server does, and then stays open until the child stops, like the page's own subscription.
type heldOpenLauncher struct{}

type heldOpenProcess struct {
	server *httptest.Server
	stop   chan struct{}
}

func (p heldOpenProcess) URL() string                     { return p.server.URL }
func (p heldOpenProcess) Ready(ctx context.Context) error { return ctx.Err() }
func (p heldOpenProcess) Stop() error                     { close(p.stop); p.server.Close(); return nil }

func (heldOpenLauncher) Launch(_ context.Context, servable Servable) (Process, error) {
	stop := make(chan struct{})
	server := httptest.NewServer(http.HandlerFunc(func(response http.ResponseWriter, request *http.Request) {
		if request.URL.Path == "/runtime/activity/stream" {
			response.Header().Set(apimanifest.HeldOpenHeader, "1")
			response.WriteHeader(http.StatusOK)
			response.(http.Flusher).Flush()
			select {
			case <-stop:
			case <-request.Context().Done():
			}
			return
		}
		_, _ = response.Write([]byte("served-by:" + servable.Name))
	}))
	return heldOpenProcess{server: server, stop: stop}, nil
}

// TestModelSwapProxyDoesNotWaitOnAHeldOpenStream holds the proxy to what choosing a model waits for: a page's own
// subscription, which the child names held open, is not waited for however long the drain grace is, and the answer
// to the request that swapped says what each part of the swap cost while an answer that did not swap says nothing.
func TestModelSwapProxyDoesNotWaitOnAHeldOpenStream(t *testing.T) {
	supervisor, err := New(heldOpenLauncher{}, 0)
	if err != nil {
		t.Fatal(err)
	}
	defer supervisor.Close()
	supervisor.mu.Lock()
	supervisor.grace = time.Hour
	supervisor.mu.Unlock()
	proxy := &Proxy{Supervisor: supervisor, Resolver: mapResolver{"alpha": {Name: "alpha", Location: "alpha.gguf"}, "beta": {Name: "beta", Location: "beta.gguf"}}}
	front := httptest.NewServer(proxy)
	defer front.Close()
	first, err := http.Get(front.URL + "/health?swap=alpha")
	if err != nil {
		t.Fatal(err)
	}
	first.Body.Close()
	if first.Header.Get(swapPhasesHeader) == "" {
		t.Fatal("the request that launched the first child was told nothing of its phases")
	}
	stream, err := http.Get(front.URL + "/runtime/activity/stream")
	if err != nil {
		t.Fatal(err)
	}
	defer stream.Body.Close()
	if stream.Header.Get(swapPhasesHeader) != "" {
		t.Fatalf("a request that did not swap reports phases %q", stream.Header.Get(swapPhasesHeader))
	}
	swapped := make(chan *http.Response, 1)
	go func() {
		answer, err := http.Get(front.URL + "/health?swap=beta")
		if err != nil {
			t.Error(err)
			return
		}
		swapped <- answer
	}()
	select {
	case answer := <-swapped:
		defer answer.Body.Close()
		body, _ := io.ReadAll(answer.Body)
		if !strings.Contains(string(body), "served-by:beta") || !strings.HasPrefix(answer.Header.Get(swapPhasesHeader), "drain=") {
			t.Fatalf("swap answer = %q with phases %q", body, answer.Header.Get(swapPhasesHeader))
		}
	case <-t.Context().Done():
		t.Fatal("the swap waited on the page's held-open stream")
	}
}
