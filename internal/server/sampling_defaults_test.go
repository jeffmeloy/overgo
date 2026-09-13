package server

import (
	"encoding/json"
	"net/http"
	"regexp"
	"slices"
	"testing"

	"overgo/internal/overgodb"
	"overgo/internal/sampling"
)

// TestRequestWithoutSamplersUsesDeclaredChain: a request that names no
// chain runs the recipe's declared chain, an explicit chain stays
// authoritative even when empty, temperature 0 under a logit bias draws
// the argmax, and the responses object names the resolved chain.
func TestRequestWithoutSamplersUsesDeclaredChain(t *testing.T) {
	declared := testRuntimePolicy().Serving.Sampling.Samplers
	if len(declared) == 0 {
		t.Fatal("inference runtime policy declares no sampler chain")
	}
	generator := &fakeGenerator{}
	handler := newTestHandler(t, generator)
	complete := func(body string) sampling.Config {
		t.Helper()
		response := serveTestRequest(handler, http.MethodPost, "/v1/completions", body)
		if response.Code != http.StatusOK {
			t.Fatalf("completion status=%d body=%s", response.Code, response.Body.String())
		}
		generator.mu.Lock()
		defer generator.mu.Unlock()
		return generator.sampling
	}
	if got := complete(`{"prompt":"hi","max_tokens":1}`); !slices.Equal(got.Samplers, declared) {
		t.Fatalf("request without samplers resolved %v, want the declared chain %v", got.Samplers, declared)
	}
	if got := complete(`{"prompt":"hi","max_tokens":1,"samplers":[]}`); len(got.Samplers) != 0 {
		t.Fatalf("explicit empty chain resolved %v", got.Samplers)
	}
	if got := complete(`{"prompt":"hi","max_tokens":1,"samplers":["top_k"]}`); !slices.Equal(got.Samplers, []sampling.SamplerStage{sampling.SamplerTopK}) {
		t.Fatalf("explicit chain resolved %v", got.Samplers)
	}
	biased := complete(`{"prompt":"hi","max_tokens":1,"temperature":0,"logit_bias":{"2":-100}}`)
	if biased.Temperature != 0 || !slices.Contains(biased.Samplers, sampling.SamplerTemperature) {
		t.Fatalf("temperature 0 with a logit bias resolved %+v", biased)
	}
	if len(biased.LogitBiases) != 1 {
		t.Fatalf("logit bias resolved as %+v", biased.LogitBiases)
	}
	sampler, err := sampling.New(biased)
	if err != nil {
		t.Fatal(err)
	}
	// The bias lands on the vocabulary's top logit, so the draw must move to the next one, every time.
	logits := make([]float32, 64)
	for index := range logits {
		logits[index] = float32(index) / 64
	}
	logits[biased.LogitBiases[0].Token] = 8
	want := 63
	if biased.LogitBiases[0].Token == 63 {
		want = 62
	}
	for range 8 {
		token, err := sampler.Sample(logits)
		if err != nil {
			t.Fatal(err)
		}
		if token != want {
			t.Fatalf("temperature 0 with a logit bias drew token %d, want the biased argmax %d under %+v", token, want, biased)
		}
	}

	store, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	recorded := newTestHandlerForRepository(t, store, responseRecipeGenerator(t, &fakeGenerator{pieces: []string{"Sampled answer"}}))
	defer recorded.Close()
	response := serveTestRequest(recorded, http.MethodPost, "/v1/responses", `{"input":"hi","max_output_tokens":1,"temperature":0.25}`)
	var result responsesResponse
	if response.Code != http.StatusOK || json.Unmarshal(response.Body.Bytes(), &result) != nil {
		t.Fatalf("responses status=%d body=%s", response.Code, response.Body.String())
	}
	if result.Sampling == nil || !slices.Equal(result.Sampling.Samplers, declared) || result.Sampling.Temperature != 0.25 {
		t.Fatalf("responses object sampling = %+v, want the declared chain at temperature 0.25", result.Sampling)
	}
	// The inspector reads the held turn, which only a streamed turn leaves behind.
	streamed := serveTestRequest(recorded, http.MethodPost, "/v1/responses", `{"input":"hi","max_output_tokens":1,"temperature":0.25,"stream":true}`)
	streamedID := regexp.MustCompile(`resp_[0-9a-z]+`).FindString(streamed.Body.String())
	if streamed.Code != http.StatusOK || streamedID == "" {
		t.Fatalf("streamed responses status=%d body=%s", streamed.Code, streamed.Body.String())
	}
	inspect := serveTestRequest(recorded, http.MethodGet, "/interactions/inspect?response="+streamedID, "")
	var record turnInspection
	if inspect.Code != http.StatusOK || json.Unmarshal(inspect.Body.Bytes(), &record) != nil {
		t.Fatalf("inspect status=%d body=%s", inspect.Code, inspect.Body.String())
	}
	if record.Sampling == nil || !slices.Equal(record.Sampling.Samplers, declared) || record.Sampling.Temperature != 0.25 {
		t.Fatalf("inspected turn sampling = %+v", record.Sampling)
	}
}
