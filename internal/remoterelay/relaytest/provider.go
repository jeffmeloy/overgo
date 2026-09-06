// Package relaytest serves a fake OpenAI-compatible provider for relay
// tests: it checks the bearer key, records the request it received and
// streams the pieces it was given as chat completion chunks; given a
// listing, it answers the models route too.
package relaytest

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"
)

// Received is the last chat completion request the fake accepted.
type Received struct {
	Model    string `json:"model"`
	Messages []struct {
		Role    string `json:"role"`
		Content string `json:"content"`
	} `json:"messages"`
	Stream    bool     `json:"stream"`
	MaxTokens int      `json:"max_tokens"`
	Stop      []string `json:"stop"`
}

// Model is one entry of the fake's model listing, in the provider's
// wire shape.
type Model struct {
	ID            string `json:"id"`
	Name          string `json:"name,omitzero"`
	ContextLength uint32 `json:"context_length,omitzero"`
}

// Serve starts the fake under basePath answering pieces to every chat
// completion signed with key; the models route is absent.
func Serve(t testing.TB, basePath, key string, pieces []string) (*httptest.Server, *Received) {
	t.Helper()
	return ServeListing(t, basePath, key, pieces, nil)
}

// ServeListing starts the fake with a models listing beside its chat
// completions; a nil listing leaves the models route absent.
func ServeListing(t testing.TB, basePath, key string, pieces []string, models []Model) (*httptest.Server, *Received) {
	t.Helper()
	received := &Received{}
	server := httptest.NewServer(http.HandlerFunc(func(response http.ResponseWriter, request *http.Request) {
		if request.Header.Get("Authorization") != "Bearer "+key {
			http.Error(response, `{"error":"missing key"}`, http.StatusUnauthorized)
			return
		}
		if models != nil && request.URL.Path == basePath+"/models" && request.Method == http.MethodGet {
			response.Header().Set("Content-Type", "application/json")
			_ = json.NewEncoder(response).Encode(map[string]any{"data": models})
			return
		}
		if request.URL.Path != basePath+"/chat/completions" || request.Method != http.MethodPost {
			http.Error(response, "unexpected route", http.StatusNotFound)
			return
		}
		body, err := io.ReadAll(request.Body)
		if err != nil || json.Unmarshal(body, received) != nil {
			http.Error(response, "bad request", http.StatusBadRequest)
			return
		}
		response.Header().Set("Content-Type", "text/event-stream")
		for _, piece := range pieces {
			fmt.Fprintf(response, "data: {\"choices\":[{\"delta\":{\"content\":%q}}]}\n\n", piece)
		}
		fmt.Fprint(response, "data: [DONE]\n\n")
	}))
	t.Cleanup(server.Close)
	return server, received
}
