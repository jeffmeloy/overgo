package server

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"overgo/internal/modelrecipe"
)

// statedGenerator is a generator whose model the store records as state says.
type statedGenerator struct {
	*fakeGenerator
	state modelrecipe.ModelState
}

func (g statedGenerator) ModelState() modelrecipe.ModelState { return g.state }

// TestAResponseCarriesItsModelsState: a chat and a completion say what the store records of the model that answered, in
// their body and, for a stream's sake, a header on every routed answer; the model listing gives the reason too.
func TestAResponseCarriesItsModelsState(t *testing.T) {
	t.Parallel()
	state := modelrecipe.ModelState{Verification: modelrecipe.StateBroken, Reason: "its evidence does not verify"}
	handler := newTestHandler(t, statedGenerator{&fakeGenerator{}, state})
	serve := func(method, path, body string) *httptest.ResponseRecorder {
		t.Helper()
		response := httptest.NewRecorder()
		handler.ServeHTTP(response, httptest.NewRequest(method, path, strings.NewReader(body)))
		if response.Code != http.StatusOK {
			t.Fatalf("%s: status = %d body=%s", path, response.Code, response.Body.String())
		}
		if got := response.Header().Get(modelVerificationHeader); got != string(state.Verification) {
			t.Fatalf("%s: header = %q", path, got)
		}
		return response
	}
	for path, body := range map[string]string{
		"/v1/chat/completions": `{"model":"test-model","messages":[{"role":"user","content":"hi"}],"max_tokens":2,"temperature":0}`,
		"/v1/completions":      `{"model":"test-model","prompt":"hi","max_tokens":2,"temperature":0}`,
	} {
		var answer struct {
			Verification string `json:"verification"`
		}
		if err := json.Unmarshal(serve(http.MethodPost, path, body).Body.Bytes(), &answer); err != nil {
			t.Fatal(err)
		}
		if answer.Verification != string(state.Verification) {
			t.Fatalf("%s: body says %q", path, answer.Verification)
		}
	}
	var listing struct {
		Data []struct {
			Meta struct {
				Verification string `json:"verification"`
				Reason       string `json:"verification_reason"`
			} `json:"meta"`
		} `json:"data"`
	}
	if err := json.Unmarshal(serve(http.MethodGet, "/v1/models", "").Body.Bytes(), &listing); err != nil {
		t.Fatal(err)
	}
	if len(listing.Data) != 1 || listing.Data[0].Meta.Verification != string(state.Verification) || listing.Data[0].Meta.Reason != state.Reason {
		t.Fatalf("listing = %+v", listing)
	}
}

// TestAResponseFromAnUnstatedGeneratorSaysNothing: a generator with no store behind it, a test's or a relay's, adds no
// field or header.
func TestAResponseFromAnUnstatedGeneratorSaysNothing(t *testing.T) {
	t.Parallel()
	handler := newTestHandler(t, &fakeGenerator{})
	response := httptest.NewRecorder()
	handler.ServeHTTP(response, httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(
		`{"model":"test-model","messages":[{"role":"user","content":"hi"}],"max_tokens":2,"temperature":0}`)))
	if response.Header().Get(modelVerificationHeader) != "" || strings.Contains(response.Body.String(), "verification") {
		t.Fatalf("header %q, body %s", response.Header().Get(modelVerificationHeader), response.Body.String())
	}
}
