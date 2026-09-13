package server

import (
	"encoding/json"
	"fmt"
	"net/http"
	"reflect"
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

// Omission selects a declaration; explicit zero and nonzero remain request values.
func TestRequestSamplingPresence(t *testing.T) {
	fields := []struct {
		field, wire        string
		declared, explicit float64
	}{
		{"Temperature", "temperature", .5, .25}, {"DynatempRange", "dynatemp_range", .5, .25},
		{"DynatempExponent", "dynatemp_exponent", .5, .25}, {"TopP", "top_p", .5, .25},
		{"TopK", "top_k", 2, 1}, {"MinP", "min_p", .5, .25}, {"TypicalP", "typical_p", .5, .25},
		{"TopNSigma", "top_n_sigma", .5, .25}, {"XTCProbability", "xtc_probability", .5, .25},
		{"XTCThreshold", "xtc_threshold", .5, .25}, {"MinKeep", "min_keep", 2, 1},
		{"AdaptiveTarget", "adaptive_target", .5, .25}, {"AdaptiveDecay", "adaptive_decay", .5, .25},
		{"RepeatLastN", "repeat_last_n", 2, 1}, {"RepeatPenalty", "repeat_penalty", .5, .25},
		{"PresencePenalty", "presence_penalty", .5, .25}, {"FrequencyPenalty", "frequency_penalty", .5, .25},
		{"DryMultiplier", "dry_multiplier", .5, .25}, {"DryBase", "dry_base", 2, 1},
		{"DryAllowedLength", "dry_allowed_length", 2, 1}, {"DryPenaltyLastN", "dry_penalty_last_n", 2, 1},
		{"Mirostat", "mirostat", 2, 1}, {"MirostatTau", "mirostat_tau", .5, .25},
		{"MirostatEta", "mirostat_eta", .5, .25}, {"Seed", "seed", 2, 1},
	}
	for _, field := range fields {
		t.Run(field.field, func(t *testing.T) {
			handler := &Handler{servingWorkspace: servingWorkspace{defaultSampling: sampling.Config{TypicalP: 1, RepeatPenalty: 1, DryBase: 2, MirostatTau: 1, MirostatEta: .1}}}
			target := reflect.ValueOf(&handler.defaultSampling).Elem().FieldByName(field.field)
			target.Set(reflect.ValueOf(field.declared).Convert(target.Type()))
			handler.config.DefaultTemperature = handler.defaultSampling.Temperature
			handler.config.DefaultTopP = handler.defaultSampling.TopP
			for _, variant := range []struct {
				name string
				body string
				want float64
			}{
				{"omitted", "{}", field.declared}, {"zero", fmt.Sprintf(`{"%s":0}`, field.wire), 0},
				{"explicit", fmt.Sprintf(`{"%s":%g}`, field.wire, field.explicit), field.explicit},
			} {
				t.Run(variant.name, func(t *testing.T) {
					var request samplingParameters
					if err := json.Unmarshal([]byte(variant.body), &request); err != nil {
						t.Fatal(err)
					}
					sampler, err := handler.newSampler(request)
					if err != nil {
						t.Fatal(err)
					}
					value := reflect.ValueOf(sampler.Config()).FieldByName(field.field)
					got := value.Convert(reflect.TypeFor[float64]()).Float()
					if got != variant.want {
						t.Fatalf("%s: got %g want %g", variant.body, got, variant.want)
					}
				})
			}
		})
	}
	t.Run("invalid explicit zero is refused", func(t *testing.T) {
		handler := &Handler{servingWorkspace: servingWorkspace{defaultSampling: sampling.Config{RepeatPenalty: 1, RepeatLastN: 1}}}
		var request samplingParameters
		if err := json.Unmarshal([]byte(`{"repeat_penalty":0}`), &request); err != nil {
			t.Fatal(err)
		}
		if _, err := handler.newSampler(request); err == nil {
			t.Fatal("explicit invalid zero was replaced by a valid default")
		}
	})
}
