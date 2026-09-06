// Package relaytest serves a fake OpenAI-compatible provider for relay
// tests: it checks the bearer key, records the request it received and
// streams the pieces it was given as chat completion chunks.
package relaytest

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"
)

// Received is the last chat completions request the fake provider read.
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

// Serve starts the fake provider under the given API base path, admitting
// only the key and answering every completion with the pieces.
func Serve(t testing.TB, basePath, key string, pieces []string) (*httptest.Server, *Received) {
	t.Helper()
	received := &Received{}
	server := httptest.NewServer(http.HandlerFunc(func(response http.ResponseWriter, request *http.Request) {
		if request.URL.Path != basePath+"/chat/completions" || request.Method != http.MethodPost {
			http.Error(response, "unexpected route", http.StatusNotFound)
			return
		}
		if request.Header.Get("Authorization") != "Bearer "+key {
			http.Error(response, `{"error":"missing key"}`, http.StatusUnauthorized)
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
