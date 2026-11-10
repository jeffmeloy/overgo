package modelswap

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"overgo/internal/testcheck"
)

// servedNameLauncher launches loopback children that answer only a request naming them by their served name, as
// overgo's server answers any other "requested model is not loaded".
type servedNameLauncher struct{}

func (servedNameLauncher) Launch(_ context.Context, servable Servable) (Process, error) {
	server := httptest.NewServer(http.HandlerFunc(func(response http.ResponseWriter, request *http.Request) {
		var body struct {
			Model    string `json:"model"`
			Messages []any  `json:"messages"`
		}
		if err := json.NewDecoder(request.Body).Decode(&body); err != nil || body.Model != servable.Name {
			http.Error(response, "requested model is not loaded", http.StatusNotFound)
			return
		}
		fmt.Fprintf(response, "%s answered %d messages", servable.Name, len(body.Messages))
	}))
	return upstreamProcess{server: server}, nil
}

// TestARequestNamingAModelByIdentityIsAnsweredByThatModel names a model by its identity, as the local worker's requests
// did on 2026-11-09: the proxy swapped to the model and passed the identity on, and the child, which knows itself by
// its file name, answered "requested model is not loaded". The request that launches the child and the next, which
// finds it running, both reach it under its served name, the rest of the request as sent.
func TestARequestNamingAModelByIdentityIsAnsweredByThatModel(t *testing.T) {
	supervisor := testcheck.Must(New(servedNameLauncher{}, 0))(t)
	defer supervisor.Close()
	worker := Servable{Name: "worker.gguf", Location: "models/worker.gguf", Model: "model:sha256:feed"}
	front := httptest.NewServer(&Proxy{Supervisor: supervisor, Resolver: mapResolver{worker.Model: worker}})
	defer front.Close()
	for _, turn := range []string{"launches the child", "finds it running"} {
		answer := testcheck.Must(http.Post(front.URL+"/v1/chat/completions", "application/json",
			strings.NewReader(`{"model":"model:sha256:feed","messages":[{"role":"user","content":"hi"}]}`)))(t)
		said, _ := io.ReadAll(answer.Body)
		answer.Body.Close()
		if answer.StatusCode != http.StatusOK || string(said) != "worker.gguf answered 1 messages" {
			t.Fatalf("a request naming the model by identity, which %s, was answered %d %q", turn, answer.StatusCode, said)
		}
	}
}
