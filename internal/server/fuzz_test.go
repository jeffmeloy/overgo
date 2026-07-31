package server

import (
	"bytes"
	"net/http"
	"net/http/httptest"
	"testing"
)

func FuzzJSONEndpointsNeverPanic(f *testing.F) {
	routes := []string{
		"/v1/completions",
		"/completion",
		"/v1/chat/completions",
		"/v1/embeddings",
		"/v1/responses",
		"/v1/messages",
		"/tokenize",
		"/detokenize",
		"/apply-template",
		"/lora-adapters",
	}
	f.Add(uint8(0), []byte(`{"prompt":"hello","max_tokens":1}`))
	f.Add(uint8(6), []byte(`{"content":["hello",1]}`))
	f.Add(uint8(255), []byte{0xff, 0, '{'})
	handler := newTestHandler(f, &fakeGenerator{})
	f.Fuzz(func(t *testing.T, route uint8, body []byte) {
		if len(body) > maxRequestBytes+1 {
			t.Skip()
		}
		path := routes[int(route)%len(routes)]
		request := httptest.NewRequest(http.MethodPost, path, bytes.NewReader(body))
		response := httptest.NewRecorder()
		handler.ServeHTTP(response, request)
		if response.Code < 100 || response.Code > 599 {
			t.Fatalf("invalid HTTP status %d", response.Code)
		}
	})
}
