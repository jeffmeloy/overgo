package server

import (
	"bytes"

	"context"

	"encoding/base64"

	"encoding/binary"

	"encoding/json"

	"errors"

	"fmt"

	"image"

	"image/color"

	"image/png"

	"llamacpp2go/internal/inference"
	"llamacpp2go/internal/projector"

	"llamacpp2go/internal/sampling"

	"llamacpp2go/internal/tokenizer"

	"math"
	"reflect"

	"net/http"

	"net/http/httptest"

	"slices"

	"strings"

	"testing"

	"time"
)

func TestNativeCompletionStreaming(t *testing.T) {
	handler := newTestHandler(t, &fakeGenerator{})
	request := httptest.NewRequest(
		http.MethodPost,
		"/completion",
		strings.NewReader(
			`{"prompt":"hi","n_predict":2,"stream":true,"return_progress":true,`+
				`"timings_per_token":true,"n_probs":2,"post_sampling_probs":true,`+
				`"sse_ping_interval":-1}`,
		),
	)
	response := httptest.NewRecorder()
	handler.ServeHTTP(response, request)
	if response.Code != http.StatusOK ||
		response.Header().Get("Content-Type") != "text/event-stream" {
		t.Fatalf(
			"status/content-type = %d %q body=%s",
			response.Code,
			response.Header().Get("Content-Type"),
			response.Body.String(),
		)
	}
	body := response.Body.String()
	for _, fragment := range []string{
		`"content":"A","tokens":[30],"stop":false,"id_slot":-1`,
		`"content":"B","tokens":[31],"stop":false,"id_slot":-1`,
		`"content":"","tokens":[],"id_slot":0,"stop":true`,
		`"stop":true`,
		`"stop_type":"limit"`,
		`"sse_ping_interval":-1`,
		`"return_progress":true`,
		`"timings_per_token":true`,
		`"n_probs":2`,
		`"post_sampling_probs":true`,
		`"prompt_progress":{"total":2,"cache":0,"processed":0,"time_ms":0}`,
		`"prompt_progress":{"total":2,"cache":0,"processed":2,"time_ms":1}`,
	} {
		if !strings.Contains(body, fragment) {
			t.Fatalf("stream lacks %q:\n%s", fragment, body)
		}
	}
	if strings.Contains(body, "[DONE]") {
		t.Fatalf("native stream unexpectedly contains OpenAI terminator:\n%s", body)
	}
	if count := strings.Count(
		body,
		`"timings":{"cache_n":0,"prompt_n":2,"prompt_ms":1`,
	); count < 3 {
		t.Fatalf("native stream timing event count = %d:\n%s", count, body)
	}
	if count := strings.Count(body, `"completion_probabilities"`); count != 3 {
		t.Fatalf("native stream probability event count = %d:\n%s", count, body)
	}
}

func TestNativeCompletionAcceptsJSONSchema(t *testing.T) {
	generator := &fakeGenerator{}
	handler := newTestHandler(t, generator)
	request := httptest.NewRequest(
		http.MethodPost,
		"/completion",
		strings.NewReader(
			`{"prompt":"hi","n_predict":0,"json_schema":{`+
				`"type":"object","properties":{"ok":{"type":"boolean"}},`+
				`"required":["ok"],"additionalProperties":false}}`,
		),
	)
	response := httptest.NewRecorder()
	handler.ServeHTTP(response, request)
	if response.Code != http.StatusOK {
		t.Fatalf("status = %d body=%s", response.Code, response.Body.String())
	}
	generator.mu.Lock()
	source, root := generator.gbnfSource, generator.gbnfRoot
	generator.mu.Unlock()
	if root != "root" ||
		!strings.Contains(source, `ok-kv ::= "\"ok\"" space ":" space boolean`) ||
		!strings.Contains(source, `root ::= "{" space ok-kv space "}"`) {
		t.Fatalf("JSON-schema GBNF source/root = %q/%q", source, root)
	}
}

func TestNativeCompletionReportsJSONSchemaCompileFailure(t *testing.T) {
	generator := &fakeGenerator{gbnfErr: errors.New("fixture compile failure")}
	handler := newTestHandler(t, generator)
	response := httptest.NewRecorder()
	handler.ServeHTTP(
		response,
		httptest.NewRequest(
			http.MethodPost,
			"/completion",
			strings.NewReader(
				`{"prompt":"hi","n_predict":0,"json_schema":{"type":"boolean"}}`,
			),
		),
	)
	if response.Code != http.StatusBadRequest ||
		!strings.Contains(response.Body.String(), "fixture compile failure") {
		t.Fatalf("status = %d body=%s", response.Code, response.Body.String())
	}
}

func TestNativeCompletionTreatsNullJSONSchemaAsAbsent(t *testing.T) {
	generator := &fakeGenerator{}
	handler := newTestHandler(t, generator)
	response := httptest.NewRecorder()
	handler.ServeHTTP(
		response,
		httptest.NewRequest(
			http.MethodPost,
			"/completion",
			strings.NewReader(`{"prompt":"hi","n_predict":0,"json_schema":null}`),
		),
	)
	if response.Code != http.StatusOK {
		t.Fatalf("status = %d body=%s", response.Code, response.Body.String())
	}
	generator.mu.Lock()
	source := generator.gbnfSource
	generator.mu.Unlock()
	if source != "" {
		t.Fatalf("null json_schema compiled %q", source)
	}
}

func TestOpenAICompletionAcceptsJSONSchema(t *testing.T) {
	generator := &fakeGenerator{}
	handler := newTestHandler(t, generator)
	response := httptest.NewRecorder()
	handler.ServeHTTP(
		response,
		httptest.NewRequest(
			http.MethodPost,
			"/v1/completions",
			strings.NewReader(
				`{"prompt":"hi","max_tokens":0,"json_schema":{`+
					`"type":"object","properties":{"ok":{"type":"boolean"}},`+
					`"required":["ok"],"additionalProperties":false}}`,
			),
		),
	)
	if response.Code != http.StatusOK {
		t.Fatalf("status = %d body=%s", response.Code, response.Body.String())
	}
	generator.mu.Lock()
	source, root := generator.gbnfSource, generator.gbnfRoot
	generator.mu.Unlock()
	if root != "root" ||
		!strings.Contains(source, `ok-kv ::= "\"ok\"" space ":" space boolean`) ||
		!strings.Contains(source, `root ::= "{" space ok-kv space "}"`) {
		t.Fatalf("JSON-schema GBNF source/root = %q/%q", source, root)
	}
}

func TestOpenAICompletionRejectsInvalidJSONSchemaOptions(t *testing.T) {
	handler := newTestHandler(t, &fakeGenerator{})
	cases := []string{
		`{"prompt":"hi","max_tokens":0,"json_schema":{"type":"string","pattern":"unanchored"}}`,
		`{"prompt":"hi","max_tokens":0,"json_schema":{},"grammar":"root ::= \"x\""}`,
		`{"prompt":"hi","max_tokens":0,"json_schema":{},"grammar_choices":["x"]}`,
	}
	for _, body := range cases {
		response := httptest.NewRecorder()
		handler.ServeHTTP(
			response,
			httptest.NewRequest(
				http.MethodPost,
				"/v1/completions",
				strings.NewReader(body),
			),
		)
		if response.Code != http.StatusBadRequest {
			t.Fatalf(
				"body %s status = %d response=%s",
				body,
				response.Code,
				response.Body.String(),
			)
		}
	}
}

func TestNativeCompletionRejectsUnsupportedAndInvalidOptions(t *testing.T) {
	handler := newTestHandler(t, &fakeGenerator{})
	cases := []string{
		`{"prompt":[1.5],"n_predict":1}`,
		`{"prompt":[64],"n_predict":1}`,
		`{"prompt":[{}],"n_predict":1}`,
		`{"prompt":"hi","n_predict":9}`,
		`{"prompt":"hi","n_predict":1,"n_cmpl":9}`,
		`{"prompt":"hi","n_predict":1,"n_probs":-1}`,
		`{"prompt":"hi","n_predict":1,"post_sampling_probs":true}`,
		`{"prompt":"hi","n_predict":1,"n_cache_reuse":1}`,
		`{"prompt":"hi","n_predict":1,"id_slot":2}`,
		`{"prompt":"hi","n_predict":1,"sse_ping_interval":-2}`,
		`{"prompt":"hi","n_predict":1,"sse_ping_interval":0.5}`,
		`{"prompt":"hi","n_predict":1,"t_max_predict_ms":-2}`,
		`{"prompt":"hi","n_predict":1,"n_indent":-1}`,
		`{"prompt":"hi","n_predict":1,"json_schema":{"type":"string","pattern":"unanchored"}}`,
		`{"prompt":"hi","n_predict":1,"json_schema":{},"grammar":"root ::= \"x\""}`,
		`{"prompt":"hi","n_predict":1,"json_schema":{},"grammar_choices":["x"]}`,
	}
	for _, body := range cases {
		response := httptest.NewRecorder()
		handler.ServeHTTP(
			response,
			httptest.NewRequest(http.MethodPost, "/completion", strings.NewReader(body)),
		)
		if response.Code != http.StatusBadRequest {
			t.Fatalf("body %s status = %d response=%s", body, response.Code, response.Body.String())
		}
	}
	tooMany := make([]string, 65)
	for index := range tooMany {
		tooMany[index] = "prompt"
	}
	encoded, err := json.Marshal(map[string]any{
		"prompt":    tooMany,
		"n_predict": 1,
	})
	if err != nil {
		t.Fatal(err)
	}
	response := httptest.NewRecorder()
	handler.ServeHTTP(
		response,
		httptest.NewRequest(http.MethodPost, "/completion", strings.NewReader(string(encoded))),
	)
	if response.Code != http.StatusBadRequest {
		t.Fatalf("oversized batch status = %d body=%s", response.Code, response.Body.String())
	}
	tooManyFields := make([]string, 65)
	for index := range tooManyFields {
		tooManyFields[index] = "content"
	}
	for _, fields := range [][]string{
		tooManyFields,
		{strings.Repeat("x", 257)},
		{"a/b/c/d/e/f/g/h/i/j/k/l/m/n/o/p/q"},
	} {
		encoded, err := json.Marshal(map[string]any{
			"prompt":          "hi",
			"n_predict":       1,
			"response_fields": fields,
		})
		if err != nil {
			t.Fatal(err)
		}
		response := httptest.NewRecorder()
		handler.ServeHTTP(
			response,
			httptest.NewRequest(http.MethodPost, "/completion", strings.NewReader(string(encoded))),
		)
		if response.Code != http.StatusBadRequest {
			t.Fatalf("fields %#v status = %d body=%s", fields, response.Code, response.Body.String())
		}
	}
	get := httptest.NewRecorder()
	handler.ServeHTTP(get, httptest.NewRequest(http.MethodGet, "/completion", nil))
	if get.Code != http.StatusMethodNotAllowed ||
		get.Header().Get("Allow") != http.MethodPost {
		t.Fatalf("GET status = %d Allow=%q", get.Code, get.Header().Get("Allow"))
	}
}

func TestSynchronizedSSEHeartbeatUsesPinnedCommentFrame(t *testing.T) {
	response := &signalingRecorder{
		ResponseRecorder: httptest.NewRecorder(),
		flushed:          make(chan struct{}),
	}
	stream := newSynchronizedSSE(response, response)
	ctx, cancel := context.WithCancel(context.Background())
	stop := stream.startHeartbeat(ctx, time.Millisecond)
	select {
	case <-response.flushed:
	case <-time.After(time.Second):
		t.Fatal("heartbeat did not flush")
	}
	stop()
	cancel()
	if body := response.Body.String(); !strings.Contains(body, ":\n\n") {
		t.Fatalf("heartbeat body = %q", body)
	}
	if !response.ResponseRecorder.Flushed {
		t.Fatal("heartbeat did not flush")
	}
}

func TestNativeCompletionAuthenticationAndTimeout(t *testing.T) {
	authenticated, err := New(Config{
		ModelID:            "test-model",
		MaxTokens:          8,
		DefaultTemperature: 1,
		DefaultTopP:        1,
		APIKey:             "test-secret",
	}, &fakeGenerator{})
	if err != nil {
		t.Fatal(err)
	}
	body := `{"prompt":"hi","n_predict":1}`
	unauthorized := httptest.NewRecorder()
	authenticated.ServeHTTP(
		unauthorized,
		httptest.NewRequest(http.MethodPost, "/completion", strings.NewReader(body)),
	)
	if unauthorized.Code != http.StatusUnauthorized {
		t.Fatalf("unauthorized status = %d body=%s", unauthorized.Code, unauthorized.Body.String())
	}
	authorizedRequest := httptest.NewRequest(
		http.MethodPost,
		"/completion",
		strings.NewReader(body),
	)
	authorizedRequest.Header.Set("Authorization", "Bearer test-secret")
	authorized := httptest.NewRecorder()
	authenticated.ServeHTTP(authorized, authorizedRequest)
	if authorized.Code != http.StatusOK {
		t.Fatalf("authorized status = %d body=%s", authorized.Code, authorized.Body.String())
	}

	blocked := &fakeGenerator{
		started: make(chan struct{}),
		release: make(chan struct{}),
	}
	timed, err := New(Config{
		ModelID:            "test-model",
		MaxTokens:          8,
		DefaultTemperature: 1,
		DefaultTopP:        1,
		RequestTimeout:     20 * time.Millisecond,
	}, blocked)
	if err != nil {
		t.Fatal(err)
	}
	timeout := httptest.NewRecorder()
	timed.ServeHTTP(
		timeout,
		httptest.NewRequest(http.MethodPost, "/completion", strings.NewReader(body)),
	)
	if timeout.Code != http.StatusRequestTimeout {
		t.Fatalf("timeout status = %d body=%s", timeout.Code, timeout.Body.String())
	}
}

func TestCompletionStopSequenceSpansTokens(t *testing.T) {
	handler := newTestHandler(t, &fakeGenerator{})
	request := httptest.NewRequest(
		http.MethodPost,
		"/v1/completions",
		strings.NewReader(`{"prompt":"hi","max_tokens":2,"stop":"AB"}`),
	)
	response := httptest.NewRecorder()
	handler.ServeHTTP(response, request)
	if response.Code != http.StatusOK {
		t.Fatalf("status = %d body=%s", response.Code, response.Body.String())
	}
	var result completionResponse
	if err := json.Unmarshal(response.Body.Bytes(), &result); err != nil {
		t.Fatal(err)
	}
	if result.Choices[0].Text != "" || result.Choices[0].FinishReason != "stop" {
		t.Fatalf("choice = %+v", result.Choices[0])
	}
	if result.Usage.CompletionTokens != 2 || result.Usage.PromptTokens != 2 {
		t.Fatalf("usage = %+v", result.Usage)
	}
}

func TestCompletionFlushesUnmatchedStopPrefix(t *testing.T) {
	handler := newTestHandler(t, &fakeGenerator{})
	request := httptest.NewRequest(
		http.MethodPost,
		"/v1/completions",
		strings.NewReader(`{"prompt":"hi","max_tokens":2,"stop":["ABC"]}`),
	)
	response := httptest.NewRecorder()
	handler.ServeHTTP(response, request)
	if response.Code != http.StatusOK {
		t.Fatalf("status = %d body=%s", response.Code, response.Body.String())
	}
	var result completionResponse
	if err := json.Unmarshal(response.Body.Bytes(), &result); err != nil {
		t.Fatal(err)
	}
	if result.Choices[0].Text != "AB" || result.Choices[0].FinishReason != "length" {
		t.Fatalf("choice = %+v", result.Choices[0])
	}
}

func TestCompletionMultipleChoices(t *testing.T) {
	handler := newTestHandler(t, &fakeGenerator{})
	request := httptest.NewRequest(
		http.MethodPost,
		"/v1/completions",
		strings.NewReader(`{"prompt":"hi","max_tokens":2,"n":3}`),
	)
	response := httptest.NewRecorder()
	handler.ServeHTTP(response, request)
	if response.Code != http.StatusOK {
		t.Fatalf("status = %d body=%s", response.Code, response.Body.String())
	}
	var result completionResponse
	if err := json.Unmarshal(response.Body.Bytes(), &result); err != nil {
		t.Fatal(err)
	}
	if len(result.Choices) != 3 {
		t.Fatalf("choices = %+v", result.Choices)
	}
	for index, choice := range result.Choices {
		if choice.Index != index || choice.Text != "AB" {
			t.Fatalf("choice %d = %+v", index, choice)
		}
	}
	if result.Usage != (completionUsage{PromptTokens: 2, CompletionTokens: 6, TotalTokens: 8}) {
		t.Fatalf("usage = %+v", result.Usage)
	}
}

func TestCompletionAcceptsGrammarChoices(t *testing.T) {
	generator := &fakeGenerator{}
	handler := newTestHandler(t, generator)
	request := httptest.NewRequest(
		http.MethodPost,
		"/v1/completions",
		strings.NewReader(
			`{"prompt":"hi","max_tokens":1,"grammar_choices":[" yes"," no"]}`,
		),
	)
	response := httptest.NewRecorder()
	handler.ServeHTTP(response, request)
	if response.Code != http.StatusOK {
		t.Fatalf("status = %d body=%s", response.Code, response.Body.String())
	}
	generator.mu.Lock()
	got := append([]string(nil), generator.grammar...)
	generator.mu.Unlock()
	if len(got) != 2 || got[0] != " yes" || got[1] != " no" {
		t.Fatalf("grammar choices = %#v", got)
	}
}

func TestCompletionAcceptsGBNF(t *testing.T) {
	generator := &fakeGenerator{}
	handler := newTestHandler(t, generator)
	request := httptest.NewRequest(
		http.MethodPost,
		"/v1/completions",
		strings.NewReader(
			`{"prompt":"hi","max_tokens":1,"grammar":"answer ::= \"a\"","grammar_root":"answer"}`,
		),
	)
	response := httptest.NewRecorder()
	handler.ServeHTTP(response, request)
	if response.Code != http.StatusOK {
		t.Fatalf("status = %d body=%s", response.Code, response.Body.String())
	}
	generator.mu.Lock()
	source, root := generator.gbnfSource, generator.gbnfRoot
	generator.mu.Unlock()
	if source != `answer ::= "a"` || root != "answer" {
		t.Fatalf("GBNF source/root = %q/%q", source, root)
	}
}

func TestCompletionAcceptsLazyGBNF(t *testing.T) {
	generator := &fakeGenerator{}
	handler := newTestHandler(t, generator)
	request := httptest.NewRequest(
		http.MethodPost,
		"/v1/completions",
		strings.NewReader(
			`{"prompt":"hi","max_tokens":1,"grammar":"root ::= \"a\"","grammar_lazy":true,"grammar_trigger_patterns":["(a)"],"grammar_trigger_tokens":[1]}`,
		),
	)
	response := httptest.NewRecorder()
	handler.ServeHTTP(response, request)
	if response.Code != http.StatusOK {
		t.Fatalf("status = %d body=%s", response.Code, response.Body.String())
	}
	generator.mu.Lock()
	patterns := append([]string(nil), generator.gbnfPatterns...)
	tokens := append([]tokenizer.TokenID(nil), generator.gbnfTokens...)
	generator.mu.Unlock()
	if !slices.Equal(patterns, []string{"(a)"}) ||
		!slices.Equal(tokens, []tokenizer.TokenID{1}) {
		t.Fatalf("lazy GBNF patterns/tokens = %v/%v", patterns, tokens)
	}
}

func TestCompletionAcceptsOrderedSamplers(t *testing.T) {
	generator := &fakeGenerator{}
	handler := newTestHandler(t, generator)
	request := httptest.NewRequest(
		http.MethodPost,
		"/v1/completions",
		strings.NewReader(
			`{"prompt":"hi","max_tokens":1,"samplers":["top_k","penalties","top_k","temperature"]}`,
		),
	)
	response := httptest.NewRecorder()
	handler.ServeHTTP(response, request)
	if response.Code != http.StatusOK {
		t.Fatalf("status = %d body=%s", response.Code, response.Body.String())
	}
	generator.mu.Lock()
	got := append([]sampling.SamplerStage(nil), generator.samplers...)
	generator.mu.Unlock()
	want := []sampling.SamplerStage{
		sampling.SamplerTopK,
		sampling.SamplerPenalties,
		sampling.SamplerTopK,
		sampling.SamplerTemperature,
	}
	if !slices.Equal(got, want) {
		t.Fatalf("samplers = %v, want %v", got, want)
	}
}

func TestCompletionRejectsUnsupportedSampler(t *testing.T) {
	handler := newTestHandler(t, &fakeGenerator{})
	request := httptest.NewRequest(
		http.MethodPost,
		"/v1/completions",
		strings.NewReader(`{"prompt":"hi","max_tokens":1,"samplers":["tfs_z"]}`),
	)
	response := httptest.NewRecorder()
	handler.ServeHTTP(response, request)
	if response.Code != http.StatusBadRequest {
		t.Fatalf("status = %d body=%s", response.Code, response.Body.String())
	}
}

func TestCompletionAcceptsInfillSampler(t *testing.T) {
	generator := &fakeGenerator{}
	handler := newTestHandler(t, generator)
	request := httptest.NewRequest(
		http.MethodPost,
		"/v1/completions",
		strings.NewReader(`{"prompt":"hi","max_tokens":1,"samplers":["infill"]}`),
	)
	response := httptest.NewRecorder()
	handler.ServeHTTP(response, request)
	if response.Code != http.StatusOK {
		t.Fatalf("status = %d body=%s", response.Code, response.Body.String())
	}
	generator.mu.Lock()
	config := generator.sampling
	generator.mu.Unlock()
	if !slices.Equal(config.Samplers, []sampling.SamplerStage{sampling.SamplerInfill}) ||
		config.Infill == nil ||
		len(config.Infill.Pieces) != 64 ||
		!config.Infill.EOG[63] {
		t.Fatalf("infill sampling config = %+v", config)
	}
}

func TestNativeInfillFormatsAndGeneratesExactPromptTokens(t *testing.T) {
	generator := &fakeGenerator{}
	handler, err := New(Config{
		ModelID:            "test-model",
		MaxTokens:          8,
		DefaultTemperature: 1,
		DefaultTopP:        1,
		InfillBatchSize:    16,
		SPMInfill:          true,
	}, generator)
	if err != nil {
		t.Fatal(err)
	}
	request := httptest.NewRequest(
		http.MethodPost,
		"/infill",
		strings.NewReader(
			`{"input_prefix":"before","input_suffix":"after",`+
				`"prompt":"<special>","input_extra":[{"filename":"x.go","text":"extra"}],`+
				`"n_predict":1,"samplers":["infill"]}`,
		),
	)
	response := httptest.NewRecorder()
	handler.ServeHTTP(response, request)
	if response.Code != http.StatusOK {
		t.Fatalf("status = %d body=%s", response.Code, response.Body.String())
	}
	generator.mu.Lock()
	prefix := append([]tokenizer.TokenID(nil), generator.infillPrefix...)
	suffix := append([]tokenizer.TokenID(nil), generator.infillSuffix...)
	prompt := append([]tokenizer.TokenID(nil), generator.infillPrompt...)
	extra := append([]inference.InfillExtra(nil), generator.infillExtra...)
	options := generator.infillOptions
	generatedPrompt := append([]tokenizer.TokenID(nil), generator.promptIDs...)
	generator.mu.Unlock()
	if !slices.Equal(prefix, []tokenizer.TokenID{10}) ||
		!slices.Equal(suffix, []tokenizer.TokenID{10}) ||
		!slices.Equal(prompt, []tokenizer.TokenID{10, 2}) {
		t.Fatalf("infill prefix/suffix/prompt = %v/%v/%v", prefix, suffix, prompt)
	}
	if len(extra) != 1 ||
		extra[0].Filename != "x.go" ||
		!slices.Equal(extra[0].Tokens, []tokenizer.TokenID{10}) {
		t.Fatalf("infill extra = %+v", extra)
	}
	if options.BatchSize != 16 ||
		options.MaxNewTokens != 1 ||
		!options.SuffixPrefix {
		t.Fatalf("infill options = %+v", options)
	}
	if !slices.Equal(generatedPrompt, []tokenizer.TokenID{1, 2, 3}) {
		t.Fatalf("generated prompt IDs = %v", generatedPrompt)
	}
}

func TestNativeInfillValidatesRequiredFieldsAndExtraChunks(t *testing.T) {
	handler := newTestHandler(t, &fakeGenerator{})
	for name, body := range map[string]string{
		"prefix": `{"input_suffix":"after"}`,
		"suffix": `{"input_prefix":"before"}`,
		"extra":  `{"input_prefix":"before","input_suffix":"after","input_extra":[{}]}`,
	} {
		t.Run(name, func(t *testing.T) {
			request := httptest.NewRequest(
				http.MethodPost,
				"/infill",
				strings.NewReader(body),
			)
			response := httptest.NewRecorder()
			handler.ServeHTTP(response, request)
			if response.Code != http.StatusBadRequest {
				t.Fatalf("status = %d body=%s", response.Code, response.Body.String())
			}
		})
	}
}

func TestCompletionAcceptsTopNSigmaXTCAndMinKeep(t *testing.T) {
	generator := &fakeGenerator{}
	handler := newTestHandler(t, generator)
	request := httptest.NewRequest(
		http.MethodPost,
		"/v1/completions",
		strings.NewReader(
			`{"prompt":"hi","max_tokens":1,"top_n_sigma":1.5,"xtc_probability":0.75,"xtc_threshold":0.2,"min_keep":3,"dynatemp_range":0.4,"dynatemp_exponent":2,"adaptive_target":0.25,"adaptive_decay":0.8,"samplers":["min_p","adaptive_p"]}`,
		),
	)
	response := httptest.NewRecorder()
	handler.ServeHTTP(response, request)
	if response.Code != http.StatusOK {
		t.Fatalf("status = %d body=%s", response.Code, response.Body.String())
	}
	generator.mu.Lock()
	config := generator.sampling
	generator.mu.Unlock()
	if config.TopNSigma != 1.5 ||
		config.XTCProbability != 0.75 ||
		config.XTCThreshold != 0.2 ||
		config.MinKeep != 3 ||
		config.DynatempRange != 0.4 ||
		config.DynatempExponent != 2 ||
		config.AdaptiveTarget != 0.25 ||
		config.AdaptiveDecay != 0.8 {
		t.Fatalf("sampling config = %+v", config)
	}
}

func TestCompletionAcceptsLogitBiasAndIgnoreEOS(t *testing.T) {
	generator := &fakeGenerator{}
	handler := newTestHandler(t, generator)
	request := httptest.NewRequest(
		http.MethodPost,
		"/v1/completions",
		strings.NewReader(
			`{"prompt":"hi","max_tokens":1,"logit_bias":{"2":1.5,"ban":false},"ignore_eos":true}`,
		),
	)
	response := httptest.NewRecorder()
	handler.ServeHTTP(response, request)
	if response.Code != http.StatusOK {
		t.Fatalf("status = %d body=%s", response.Code, response.Body.String())
	}
	generator.mu.Lock()
	biases := append([]sampling.LogitBias(nil), generator.sampling.LogitBiases...)
	generator.mu.Unlock()
	want := map[int]float32{
		2: 1.5,
		4: float32(math.Inf(-1)),
		5: float32(math.Inf(-1)),
		6: float32(math.Inf(-1)),
		7: float32(math.Inf(-1)),
	}
	if len(biases) != len(want) {
		t.Fatalf("logit biases = %+v", biases)
	}
	for _, bias := range biases {
		value, ok := want[bias.Token]
		if !ok || (math.IsInf(float64(value), -1) != math.IsInf(float64(bias.Bias), -1)) ||
			(!math.IsInf(float64(value), -1) && value != bias.Bias) {
			t.Fatalf("unexpected logit bias %+v in %+v", bias, biases)
		}
	}
}

func TestCompletionAcceptsArrayLogitBias(t *testing.T) {
	generator := &fakeGenerator{}
	handler := newTestHandler(t, generator)
	request := httptest.NewRequest(
		http.MethodPost,
		"/v1/completions",
		strings.NewReader(
			`{"prompt":"hi","max_tokens":1,"logit_bias":[[2,-1],["ban",false]]}`,
		),
	)
	response := httptest.NewRecorder()
	handler.ServeHTTP(response, request)
	if response.Code != http.StatusOK {
		t.Fatalf("status = %d body=%s", response.Code, response.Body.String())
	}
	generator.mu.Lock()
	biases := append([]sampling.LogitBias(nil), generator.sampling.LogitBiases...)
	generator.mu.Unlock()
	if len(biases) != 3 || biases[0] != (sampling.LogitBias{Token: 2, Bias: -1}) {
		t.Fatalf("array logit biases = %+v", biases)
	}
}

func TestStreamingCompletion(t *testing.T) {
	handler := newTestHandler(t, &fakeGenerator{})
	request := httptest.NewRequest(
		http.MethodPost,
		"/v1/completions",
		strings.NewReader(`{"prompt":"hi","max_tokens":2,"stream":true}`),
	)
	response := httptest.NewRecorder()
	handler.ServeHTTP(response, request)
	if response.Code != http.StatusOK {
		t.Fatalf("status = %d body=%s", response.Code, response.Body.String())
	}
	body := response.Body.String()
	for _, fragment := range []string{`"text":"A"`, `"text":"B"`, `"finish_reason":"length"`, "data: [DONE]"} {
		if !strings.Contains(body, fragment) {
			t.Fatalf("stream lacks %q: %s", fragment, body)
		}
	}
}

func TestStreamingCompletionDoesNotLeakStopPrefix(t *testing.T) {
	handler := newTestHandler(t, &fakeGenerator{})
	request := httptest.NewRequest(
		http.MethodPost,
		"/v1/completions",
		strings.NewReader(`{"prompt":"hi","max_tokens":2,"stream":true,"stop":"AB"}`),
	)
	response := httptest.NewRecorder()
	handler.ServeHTTP(response, request)
	if response.Code != http.StatusOK {
		t.Fatalf("status = %d body=%s", response.Code, response.Body.String())
	}
	body := response.Body.String()
	if strings.Contains(body, `"text":"A"`) || strings.Contains(body, `"text":"B"`) {
		t.Fatalf("stream leaked stop prefix: %s", body)
	}
	if !strings.Contains(body, `"finish_reason":"stop"`) {
		t.Fatalf("stream lacks stop reason: %s", body)
	}
}

func TestStreamingCompletionIndexesMultipleChoices(t *testing.T) {
	handler := newTestHandler(t, &fakeGenerator{})
	request := httptest.NewRequest(
		http.MethodPost,
		"/v1/completions",
		strings.NewReader(`{"prompt":"hi","max_tokens":1,"stream":true,"n":2}`),
	)
	response := httptest.NewRecorder()
	handler.ServeHTTP(response, request)
	if response.Code != http.StatusOK {
		t.Fatalf("status = %d body=%s", response.Code, response.Body.String())
	}
	body := response.Body.String()
	for _, fragment := range []string{`"text":"A","index":0`, `"text":"A","index":1`} {
		if !strings.Contains(body, fragment) {
			t.Fatalf("stream lacks %q: %s", fragment, body)
		}
	}
}

func TestStreamingCompletionBatchesUsePromptMajorIndices(t *testing.T) {
	handler := newTestHandler(t, &fakeGenerator{})
	request := httptest.NewRequest(
		http.MethodPost,
		"/v1/completions",
		strings.NewReader(`{"prompt":["one","two"],"max_tokens":1,"stream":true,"n":2}`),
	)
	response := httptest.NewRecorder()
	handler.ServeHTTP(response, request)
	if response.Code != http.StatusOK {
		t.Fatalf("status = %d body=%s", response.Code, response.Body.String())
	}
	body := response.Body.String()
	previous := -1
	for index := range 4 {
		fragment := fmt.Sprintf(`"text":"A","index":%d`, index)
		at := strings.Index(body, fragment)
		if at < 0 || at <= previous {
			t.Fatalf("fragment %q index=%d after=%d body=%s", fragment, at, previous, body)
		}
		previous = at
	}
	if !strings.HasSuffix(body, "data: [DONE]\n\n") {
		t.Fatalf("stream ending = %q", body)
	}
}

func TestCompletionRejectsInvalidStop(t *testing.T) {
	handler := newTestHandler(t, &fakeGenerator{})
	for _, body := range []string{
		`{"prompt":"hi","stop":""}`,
		`{"prompt":"hi","stop":[1]}`,
	} {
		request := httptest.NewRequest(
			http.MethodPost,
			"/v1/completions",
			strings.NewReader(body),
		)
		response := httptest.NewRecorder()
		handler.ServeHTTP(response, request)
		if response.Code != http.StatusBadRequest {
			t.Fatalf("body %s status = %d response=%s", body, response.Code, response.Body.String())
		}
	}
}

func TestCompletionRejectsInvalidChoiceCount(t *testing.T) {
	handler := newTestHandler(t, &fakeGenerator{})
	for _, n := range []int{-1, 9} {
		request := httptest.NewRequest(
			http.MethodPost,
			"/v1/completions",
			strings.NewReader(fmt.Sprintf(`{"prompt":"hi","n":%d}`, n)),
		)
		response := httptest.NewRecorder()
		handler.ServeHTTP(response, request)
		if response.Code != http.StatusBadRequest {
			t.Fatalf("n=%d status = %d response=%s", n, response.Code, response.Body.String())
		}
	}
}

func TestEmbeddings(t *testing.T) {
	handler := newTestHandler(t, &fakeGenerator{})
	request := httptest.NewRequest(
		http.MethodPost,
		"/v1/embeddings",
		strings.NewReader(`{"model":"test-model","input":["a","bc"]}`),
	)
	response := httptest.NewRecorder()
	handler.ServeHTTP(response, request)
	if response.Code != http.StatusOK {
		t.Fatalf("status = %d body=%s", response.Code, response.Body.String())
	}
	var result embeddingResponse
	if err := json.Unmarshal(response.Body.Bytes(), &result); err != nil {
		t.Fatal(err)
	}
	if len(result.Data) != 2 || result.Data[1].Index != 1 {
		t.Fatalf("embedding data = %+v", result.Data)
	}
	vector, ok := result.Data[0].Embedding.([]any)
	if !ok || len(vector) != 2 || vector[0] != 0.6 || vector[1] != 0.8 {
		t.Fatalf("embedding = %v", result.Data[0].Embedding)
	}
	if result.Usage != (embeddingUsage{PromptTokens: 3, TotalTokens: 3}) {
		t.Fatalf("usage = %+v", result.Usage)
	}
}

func TestOpenAIEmbeddingsBase64(t *testing.T) {
	handler := newTestHandler(t, &fakeGenerator{})
	response := httptest.NewRecorder()
	handler.ServeHTTP(
		response,
		httptest.NewRequest(
			http.MethodPost,
			"/v1/embeddings",
			strings.NewReader(`{"input":"a","encoding_format":"base64"}`),
		),
	)
	if response.Code != http.StatusOK {
		t.Fatalf("status = %d body=%s", response.Code, response.Body.String())
	}
	var result embeddingResponse
	if err := json.Unmarshal(response.Body.Bytes(), &result); err != nil {
		t.Fatal(err)
	}
	encoded, ok := result.Data[0].Embedding.(string)
	if !ok || result.Data[0].EncodingFormat != "base64" {
		t.Fatalf("base64 item = %+v", result.Data[0])
	}
	data, err := base64.StdEncoding.DecodeString(encoded)
	if err != nil {
		t.Fatal(err)
	}
	if len(data) != 8 ||
		math.Float32frombits(binary.LittleEndian.Uint32(data)) != 0.6 ||
		math.Float32frombits(binary.LittleEndian.Uint32(data[4:])) != 0.8 {
		t.Fatalf("decoded base64 = %v", data)
	}
}

func TestEmbeddingsRejectInvalidInput(t *testing.T) {
	handler := newTestHandler(t, &fakeGenerator{})
	for _, body := range []string{
		`{"input":""}`,
		`{"input":[]}`,
		`{"input":[64]}`,
		`{"input":[1.5]}`,
		`{"input":{}}`,
	} {
		request := httptest.NewRequest(http.MethodPost, "/v1/embeddings", strings.NewReader(body))
		response := httptest.NewRecorder()
		handler.ServeHTTP(response, request)
		if response.Code != http.StatusBadRequest {
			t.Fatalf("body %s status = %d, want 400", body, response.Code)
		}
	}
}

func TestEmbeddingsExactAndMixedTokenInputs(t *testing.T) {
	generator := &fakeGenerator{}
	handler := newTestHandler(t, generator)
	openAI := httptest.NewRecorder()
	handler.ServeHTTP(
		openAI,
		httptest.NewRequest(
			http.MethodPost,
			"/v1/embeddings",
			strings.NewReader(`{"input":[4,5]}`),
		),
	)
	if openAI.Code != http.StatusOK {
		t.Fatalf("OpenAI status = %d body=%s", openAI.Code, openAI.Body.String())
	}
	var openAIResult embeddingResponse
	if err := json.Unmarshal(openAI.Body.Bytes(), &openAIResult); err != nil {
		t.Fatal(err)
	}
	if openAIResult.Usage.TotalTokens != 2 {
		t.Fatalf("OpenAI usage = %+v", openAIResult.Usage)
	}

	native := httptest.NewRecorder()
	handler.ServeHTTP(
		native,
		httptest.NewRequest(
			http.MethodPost,
			"/embedding",
			strings.NewReader(`{"content":[4,"text",5]}`),
		),
	)
	if native.Code != http.StatusOK {
		t.Fatalf("native status = %d body=%s", native.Code, native.Body.String())
	}
	generator.mu.Lock()
	ids := append([]tokenizer.TokenID(nil), generator.promptIDs...)
	generator.mu.Unlock()
	if !slices.Equal(ids, []tokenizer.TokenID{4, 10, 5}) {
		t.Fatalf("native exact IDs = %v", ids)
	}
}

func TestNativeEmbeddings(t *testing.T) {
	handler := newTestHandler(t, &fakeGenerator{})
	for _, test := range []struct {
		path string
		body string
		want int
	}{
		{path: "/embedding", body: `{"content":"a","embd_normalize":2}`, want: 1},
		{path: "/embeddings", body: `{"input":["a","bc"],"encoding_format":"float"}`, want: 2},
	} {
		response := httptest.NewRecorder()
		handler.ServeHTTP(
			response,
			httptest.NewRequest(http.MethodPost, test.path, strings.NewReader(test.body)),
		)
		if response.Code != http.StatusOK {
			t.Fatalf("%s status = %d body=%s", test.path, response.Code, response.Body.String())
		}
		var result []nativeEmbeddingItem
		if err := json.Unmarshal(response.Body.Bytes(), &result); err != nil {
			t.Fatal(err)
		}
		if len(result) != test.want {
			t.Fatalf("%s result count = %d", test.path, len(result))
		}
		for index, item := range result {
			if item.Index != index ||
				len(item.Embedding) != 1 ||
				!slices.Equal(item.Embedding[0], []float32{0.6, 0.8}) {
				t.Fatalf("%s item %d = %+v", test.path, index, item)
			}
		}
	}
	advanced := httptest.NewRecorder()
	handler.ServeHTTP(
		advanced,
		httptest.NewRequest(
			http.MethodPost,
			"/embedding",
			strings.NewReader(`{"content":"a","pooling":"none","embd_normalize":-1}`),
		),
	)
	if advanced.Code != http.StatusOK {
		t.Fatalf("advanced status = %d body=%s", advanced.Code, advanced.Body.String())
	}
	var advancedResult []nativeEmbeddingItem
	if err := json.Unmarshal(advanced.Body.Bytes(), &advancedResult); err != nil {
		t.Fatal(err)
	}
	if len(advancedResult) != 1 ||
		len(advancedResult[0].Embedding) != 2 ||
		!slices.Equal(advancedResult[0].Embedding[0], []float32{3, 4}) {
		t.Fatalf("advanced result = %+v", advancedResult)
	}
}

func TestNativeEmbeddingsValidationAuthenticationAndMethod(t *testing.T) {
	handler := newTestHandler(t, &fakeGenerator{})
	for _, body := range []string{
		`{"content":""}`,
		`{"input":[]}`,
		`{"content":"a","encoding_format":"base64"}`,
		`{"content":"a","pooling":"rank"}`,
	} {
		response := httptest.NewRecorder()
		handler.ServeHTTP(
			response,
			httptest.NewRequest(http.MethodPost, "/embedding", strings.NewReader(body)),
		)
		if response.Code != http.StatusBadRequest {
			t.Fatalf("body %s status = %d response=%s", body, response.Code, response.Body.String())
		}
	}
	get := httptest.NewRecorder()
	handler.ServeHTTP(get, httptest.NewRequest(http.MethodGet, "/embedding", nil))
	if get.Code != http.StatusMethodNotAllowed ||
		get.Header().Get("Allow") != http.MethodPost {
		t.Fatalf("GET status = %d Allow=%q", get.Code, get.Header().Get("Allow"))
	}
	protected, err := New(Config{
		ModelID:            "test-model",
		MaxTokens:          8,
		DefaultTemperature: 1,
		DefaultTopP:        1,
		APIKey:             "test-secret",
		MaxEmbeddingInputs: 1,
	}, &fakeGenerator{})
	if err != nil {
		t.Fatal(err)
	}
	unauthorized := httptest.NewRecorder()
	protected.ServeHTTP(
		unauthorized,
		httptest.NewRequest(
			http.MethodPost,
			"/embeddings",
			strings.NewReader(`{"input":"a"}`),
		),
	)
	if unauthorized.Code != http.StatusUnauthorized {
		t.Fatalf("unauthorized status = %d body=%s", unauthorized.Code, unauthorized.Body.String())
	}
	request := httptest.NewRequest(
		http.MethodPost,
		"/embeddings",
		strings.NewReader(`{"input":["a","b"]}`),
	)
	request.Header.Set("Authorization", "Bearer test-secret")
	oversized := httptest.NewRecorder()
	protected.ServeHTTP(oversized, request)
	if oversized.Code != http.StatusBadRequest {
		t.Fatalf("oversized status = %d body=%s", oversized.Code, oversized.Body.String())
	}
}

func TestRerankJinaAndTEIFormats(t *testing.T) {
	handler := newTestHandler(t, &fakeGenerator{})
	jina := httptest.NewRecorder()
	handler.ServeHTTP(jina, httptest.NewRequest(
		http.MethodPost, "/v1/rerank",
		strings.NewReader(`{"model":"test-model","query":"q","documents":["a","longer"],"top_n":1}`),
	))
	if jina.Code != http.StatusOK {
		t.Fatalf("Jina status = %d body=%s", jina.Code, jina.Body.String())
	}
	var jinaResult rerankResponse
	if err := json.Unmarshal(jina.Body.Bytes(), &jinaResult); err != nil {
		t.Fatal(err)
	}
	if jinaResult.Model != "test-model" || jinaResult.Object != "list" ||
		len(jinaResult.Results) != 1 || jinaResult.Results[0].Index != 1 ||
		jinaResult.Results[0].RelevanceScore == nil || *jinaResult.Results[0].RelevanceScore != 0.6 ||
		jinaResult.Usage != (embeddingUsage{PromptTokens: 9, TotalTokens: 9}) {
		t.Fatalf("Jina rerank = %+v", jinaResult)
	}

	tei := httptest.NewRecorder()
	handler.ServeHTTP(tei, httptest.NewRequest(
		http.MethodPost, "/reranking",
		strings.NewReader(`{"query":"q","texts":["a","longer"],"return_text":true}`),
	))
	if tei.Code != http.StatusOK {
		t.Fatalf("TEI status = %d body=%s", tei.Code, tei.Body.String())
	}
	var teiResult []rerankItem
	if err := json.Unmarshal(tei.Body.Bytes(), &teiResult); err != nil {
		t.Fatal(err)
	}
	if len(teiResult) != 2 || teiResult[0].Index != 1 || teiResult[0].Score == nil ||
		*teiResult[0].Score != 0.6 || teiResult[0].Text == nil || *teiResult[0].Text != "longer" ||
		teiResult[0].RelevanceScore != nil {
		t.Fatalf("TEI rerank = %+v", teiResult)
	}
}

func TestRerankValidationCapabilityAndMethod(t *testing.T) {
	handler := newTestHandler(t, &fakeGenerator{})
	for _, body := range []string{
		`{"documents":["a"]}`,
		`{"query":"q","documents":[]}`,
		`{"query":"q","documents":[1]}`,
		`{"query":"q","documents":["a"],"top_n":-1}`,
	} {
		response := httptest.NewRecorder()
		handler.ServeHTTP(response, httptest.NewRequest(http.MethodPost, "/rerank", strings.NewReader(body)))
		if response.Code != http.StatusBadRequest {
			t.Fatalf("body %s status = %d response=%s", body, response.Code, response.Body.String())
		}
	}
	method := httptest.NewRecorder()
	handler.ServeHTTP(method, httptest.NewRequest(http.MethodGet, "/rerank", nil))
	if method.Code != http.StatusMethodNotAllowed || method.Header().Get("Allow") != http.MethodPost {
		t.Fatalf("GET status = %d Allow=%q", method.Code, method.Header().Get("Allow"))
	}
	disabled := newTestHandler(t, &rankDisabledGenerator{fakeGenerator: &fakeGenerator{}})
	unavailable := httptest.NewRecorder()
	disabled.ServeHTTP(unavailable, httptest.NewRequest(
		http.MethodPost, "/rerank", strings.NewReader(`{"query":"q","documents":["a"]}`),
	))
	if unavailable.Code != http.StatusNotImplemented {
		t.Fatalf("disabled status = %d body=%s", unavailable.Code, unavailable.Body.String())
	}
}

func TestChatCompletion(t *testing.T) {
	handler := newTestHandler(t, &fakeGenerator{})
	request := httptest.NewRequest(
		http.MethodPost,
		"/v1/chat/completions",
		strings.NewReader(`{"model":"test-model","messages":[{"role":"user","content":"hi"}],"max_tokens":2,"temperature":0}`),
	)
	response := httptest.NewRecorder()
	handler.ServeHTTP(response, request)
	if response.Code != http.StatusOK {
		t.Fatalf("status = %d body=%s", response.Code, response.Body.String())
	}
	var result chatResponse
	if err := json.Unmarshal(response.Body.Bytes(), &result); err != nil {
		t.Fatal(err)
	}
	choice := result.Choices[0]
	if choice.Message.Role != "assistant" || choice.Message.Content != "AB" ||
		choice.FinishReason != "length" {
		t.Fatalf("chat choice = %+v", choice)
	}
}

func TestChatCompletionStructuredResponseFormats(t *testing.T) {
	const schema = `{"type":"object","properties":{"ok":{"type":"boolean"}},` +
		`"required":["ok"],"additionalProperties":false}`
	cases := []struct {
		name            string
		fields          string
		wantGrammar     bool
		wantSchemaField bool
	}{
		{
			name:            "top-level JSON schema",
			fields:          `,"json_schema":` + schema,
			wantGrammar:     true,
			wantSchemaField: true,
		},
		{
			name:        "JSON object",
			fields:      `,"response_format":{"type":"json_object"}`,
			wantGrammar: true,
		},
		{
			name:            "JSON object schema",
			fields:          `,"response_format":{"type":"json_object","schema":` + schema + `}`,
			wantGrammar:     true,
			wantSchemaField: true,
		},
		{
			name: "wrapped JSON schema",
			fields: `,"response_format":{"type":"json_schema","json_schema":{` +
				`"name":"answer","strict":true,"schema":` + schema + `}}`,
			wantGrammar:     true,
			wantSchemaField: true,
		},
		{
			name:   "text",
			fields: `,"response_format":{"type":"text"}`,
		},
		{
			name:   "null top-level JSON schema",
			fields: `,"json_schema":null`,
		},
	}
	for _, test := range cases {
		t.Run(test.name, func(t *testing.T) {
			generator := &fakeGenerator{}
			handler := newTestHandler(t, generator)
			response := httptest.NewRecorder()
			handler.ServeHTTP(
				response,
				httptest.NewRequest(
					http.MethodPost,
					"/v1/chat/completions",
					strings.NewReader(
						`{"messages":[{"role":"user","content":"hi"}],`+
							`"max_tokens":0`+test.fields+`}`,
					),
				),
			)
			if response.Code != http.StatusOK {
				t.Fatalf(
					"status = %d body=%s",
					response.Code,
					response.Body.String(),
				)
			}
			generator.mu.Lock()
			source, root := generator.gbnfSource, generator.gbnfRoot
			generator.mu.Unlock()
			if test.wantGrammar != (source != "") {
				t.Fatalf("GBNF source = %q", source)
			}
			if test.wantGrammar && root != "root" {
				t.Fatalf("GBNF root = %q", root)
			}
			if test.wantSchemaField &&
				!strings.Contains(source, `ok-kv ::= "\"ok\"" space ":" space boolean`) {
				t.Fatalf("schema-specific GBNF source = %q", source)
			}
		})
	}
}

func TestChatCompletionRejectsInvalidStructuredResponseFormats(t *testing.T) {
	handler := newTestHandler(t, &fakeGenerator{})
	cases := []string{
		`,"response_format":"json"`,
		`,"response_format":{"type":1}`,
		`,"response_format":{"type":"yaml"}`,
		`,"response_format":{"type":"json_schema","json_schema":null}`,
		`,"response_format":{"type":"json_object"},"grammar":"root ::= \"x\""`,
		`,"json_schema":{},"grammar_choices":["x"]`,
		`,"response_format":{"type":"json_object","schema":` +
			`{"type":"string","pattern":"unanchored"}}`,
	}
	for _, fields := range cases {
		response := httptest.NewRecorder()
		handler.ServeHTTP(
			response,
			httptest.NewRequest(
				http.MethodPost,
				"/v1/chat/completions",
				strings.NewReader(
					`{"messages":[{"role":"user","content":"hi"}],`+
						`"max_tokens":0`+fields+`}`,
				),
			),
		)
		if response.Code != http.StatusBadRequest {
			t.Fatalf(
				"fields %s status = %d response=%s",
				fields,
				response.Code,
				response.Body.String(),
			)
		}
	}
}

func TestChatCompletionAliasAndInputTokens(t *testing.T) {
	handler := newTestHandler(t, &fakeGenerator{})
	alias := httptest.NewRecorder()
	handler.ServeHTTP(
		alias,
		httptest.NewRequest(
			http.MethodPost,
			"/chat/completions",
			strings.NewReader(
				`{"messages":[{"role":"user","content":"hi"}],"max_tokens":1}`,
			),
		),
	)
	if alias.Code != http.StatusOK {
		t.Fatalf("alias status = %d body=%s", alias.Code, alias.Body.String())
	}
	for _, path := range []string{
		"/chat/completions/input_tokens",
		"/v1/chat/completions/input_tokens",
	} {
		response := httptest.NewRecorder()
		handler.ServeHTTP(
			response,
			httptest.NewRequest(
				http.MethodPost,
				path,
				strings.NewReader(
					`{"messages":[{"role":"user","content":"hi"}],"max_tokens":7}`,
				),
			),
		)
		if response.Code != http.StatusOK {
			t.Fatalf("%s status = %d body=%s", path, response.Code, response.Body.String())
		}
		var result struct {
			Object      string `json:"object"`
			InputTokens int    `json:"input_tokens"`
		}
		if err := json.Unmarshal(response.Body.Bytes(), &result); err != nil {
			t.Fatal(err)
		}
		if result.Object != "response.input_tokens" || result.InputTokens != 3 {
			t.Fatalf("%s response = %+v", path, result)
		}
	}
}

func TestPreparedPromptGenerationAndTokenCountParity(t *testing.T) {
	tests := []struct {
		name           string
		generationPath string
		generationBody string
		countPath      string
		countBody      string
		responsesUsage bool
	}{
		{
			name:           "chat",
			generationPath: "/v1/chat/completions",
			generationBody: `{"messages":[{"role":"user","content":"hi"}],"max_tokens":1}`,
			countPath:      "/v1/chat/completions/input_tokens",
			countBody:      `{"messages":[{"role":"user","content":"hi"}]}`,
		},
		{
			name:           "responses",
			generationPath: "/v1/responses",
			generationBody: `{"input":"hi","max_output_tokens":1}`,
			countPath:      "/v1/responses/input_tokens",
			countBody:      `{"input":"hi"}`,
			responsesUsage: true,
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			generator := &fakeGenerator{}
			handler := newTestHandler(t, generator)
			generated := httptest.NewRecorder()
			handler.ServeHTTP(
				generated,
				httptest.NewRequest(
					http.MethodPost,
					test.generationPath,
					strings.NewReader(test.generationBody),
				),
			)
			if generated.Code != http.StatusOK {
				t.Fatalf("generation status = %d body=%s", generated.Code, generated.Body.String())
			}
			var generation struct {
				Usage struct {
					PromptTokens int `json:"prompt_tokens"`
					InputTokens  int `json:"input_tokens"`
				} `json:"usage"`
			}
			if err := json.Unmarshal(generated.Body.Bytes(), &generation); err != nil {
				t.Fatal(err)
			}
			counted := httptest.NewRecorder()
			handler.ServeHTTP(
				counted,
				httptest.NewRequest(
					http.MethodPost,
					test.countPath,
					strings.NewReader(test.countBody),
				),
			)
			if counted.Code != http.StatusOK {
				t.Fatalf("count status = %d body=%s", counted.Code, counted.Body.String())
			}
			var count struct {
				InputTokens int `json:"input_tokens"`
			}
			if err := json.Unmarshal(counted.Body.Bytes(), &count); err != nil {
				t.Fatal(err)
			}
			used := generation.Usage.PromptTokens
			if test.responsesUsage {
				used = generation.Usage.InputTokens
			}
			if used != count.InputTokens || count.InputTokens != 3 {
				t.Fatalf("usage = %v count = %d", generation.Usage, count.InputTokens)
			}
			if len(generator.promptIDs) != count.InputTokens {
				t.Fatalf("prepared prompt IDs = %v count = %d", generator.promptIDs, count.InputTokens)
			}
		})
	}
}

func TestChatToolSchemasAreCountedAndBufferedCallsAreStructured(t *testing.T) {
	generator := &fakeGenerator{
		pieces: []string{
			`<tool_call>{"name":"weather","arguments":{"city":"Paris"}}</tool_call>`,
		},
	}
	handler := newTestHandler(t, generator)
	body := `{"messages":[{"role":"user","content":"weather?"}],"tools":[{"type":"function","function":{"name":"weather","parameters":{"type":"object"}}}]}`
	count := httptest.NewRecorder()
	handler.ServeHTTP(
		count,
		httptest.NewRequest(
			http.MethodPost,
			"/chat/completions/input_tokens",
			strings.NewReader(body),
		),
	)
	if count.Code != http.StatusOK {
		t.Fatalf("count status = %d body=%s", count.Code, count.Body.String())
	}
	generator.mu.Lock()
	toolCount := len(generator.chatOptions.Tools)
	generator.mu.Unlock()
	if toolCount != 1 {
		t.Fatalf("formatted tool count = %d", toolCount)
	}
	completion := httptest.NewRecorder()
	handler.ServeHTTP(
		completion,
		httptest.NewRequest(
			http.MethodPost,
			"/v1/chat/completions",
			strings.NewReader(strings.TrimSuffix(body, "}")+`,"max_tokens":1}`),
		),
	)
	if completion.Code != http.StatusOK {
		t.Fatalf(
			"completion status = %d body=%s",
			completion.Code,
			completion.Body.String(),
		)
	}
	var result chatResponse
	if err := json.Unmarshal(completion.Body.Bytes(), &result); err != nil {
		t.Fatal(err)
	}
	if len(result.Choices) != 1 ||
		result.Choices[0].FinishReason != "tool_calls" ||
		len(result.Choices[0].Message.ToolCalls) != 1 ||
		result.Choices[0].Message.ToolCalls[0].ID == "" ||
		result.Choices[0].Message.ToolCalls[0].Function.Name != "weather" {
		t.Fatalf("completion response = %+v", result)
	}
	for _, suffix := range []string{
		`,"tool_choice":"invalid"`,
		`,"tool_choice":{"type":"function","function":{"name":"unknown"}}`,
		`,"response_format":{"type":"json_object"}`,
		`,"grammar":"root ::= \"x\""`,
	} {
		rejected := httptest.NewRecorder()
		handler.ServeHTTP(
			rejected,
			httptest.NewRequest(
				http.MethodPost,
				"/v1/chat/completions",
				strings.NewReader(strings.TrimSuffix(body, "}")+suffix+"}"),
			),
		)
		if rejected.Code != http.StatusBadRequest {
			t.Fatalf(
				"suffix %s status = %d body=%s",
				suffix,
				rejected.Code,
				rejected.Body.String(),
			)
		}
	}
	for _, suffix := range []string{
		`,"tool_choice":"required"`,
		`,"tool_choice":{"type":"function","function":{"name":"weather"}}`,
	} {
		accepted := httptest.NewRecorder()
		handler.ServeHTTP(
			accepted,
			httptest.NewRequest(
				http.MethodPost,
				"/v1/chat/completions",
				strings.NewReader(
					strings.TrimSuffix(body, "}")+
						`,"max_tokens":1`+suffix+"}",
				),
			),
		)
		if accepted.Code != http.StatusOK {
			t.Fatalf(
				"suffix %s status = %d body=%s",
				suffix,
				accepted.Code,
				accepted.Body.String(),
			)
		}
	}
}

func TestStreamingChatToolCallsAreStructured(t *testing.T) {
	handler := newTestHandler(t, &fakeGenerator{
		pieces: []string{
			`<tool_call><function=weather><parameter=city>`,
			`Par`,
			`is</parameter></function></tool_call>`,
		},
	})
	response := httptest.NewRecorder()
	handler.ServeHTTP(
		response,
		httptest.NewRequest(
			http.MethodPost,
			"/v1/chat/completions",
			strings.NewReader(
				`{"messages":[{"role":"user","content":"weather?"}],`+
					`"tools":[{"type":"function","function":{"name":"weather",`+
					`"parameters":{"type":"object"}}}],`+
					`"tool_choice":"required","max_tokens":3,"stream":true}`,
			),
		),
	)
	if response.Code != http.StatusOK {
		t.Fatalf(
			"stream status = %d body=%s",
			response.Code,
			response.Body.String(),
		)
	}
	body := response.Body.String()
	for _, fragment := range []string{
		`"role":"assistant"`,
		`"tool_calls":[{"index":0,"id":"call_`,
		`"type":"function","function":{"name":"weather","arguments":"{\"city\":\""}`,
		`"function":{"arguments":"Par"}`,
		`"function":{"arguments":"is\"}"}`,
		`"finish_reason":"tool_calls"`,
		"data: [DONE]",
	} {
		if !strings.Contains(body, fragment) {
			t.Fatalf("tool-call stream lacks %q:\n%s", fragment, body)
		}
	}
	if strings.Contains(body, "<tool_call>") {
		t.Fatalf("tool-call stream leaked template syntax:\n%s", body)
	}
}

func TestChatTextContentParts(t *testing.T) {
	handler := newTestHandler(t, &fakeGenerator{})
	for _, path := range []string{"/v1/chat/completions", "/chat/completions/input_tokens"} {
		response := httptest.NewRecorder()
		handler.ServeHTTP(
			response,
			httptest.NewRequest(
				http.MethodPost,
				path,
				strings.NewReader(
					`{"messages":[{"role":"user","content":[{"type":"text","text":"hello"},{"type":"text","text":" world"}]}],"max_tokens":1}`,
				),
			),
		)
		if response.Code != http.StatusOK {
			t.Fatalf("%s status = %d body=%s", path, response.Code, response.Body.String())
		}
	}
	response := httptest.NewRecorder()
	handler.ServeHTTP(
		response,
		httptest.NewRequest(
			http.MethodPost,
			"/v1/chat/completions",
			strings.NewReader(
				`{"messages":[{"role":"user","content":[{"type":"image_url","image_url":{"url":"x"}}]}]}`,
			),
		),
	)
	if response.Code != http.StatusBadRequest {
		t.Fatalf("image content status = %d body=%s", response.Code, response.Body.String())
	}
	if !strings.Contains(response.Body.String(), "no image projector") {
		t.Fatalf("image content response = %s", response.Body.String())
	}
}

func TestChatImageContentPartProjectsPrompt(t *testing.T) {
	generator := &fakeGenerator{}
	vision := &fakeQwen3VLProjector{}
	handler, err := New(Config{
		ModelID: "test-model", MaxTokens: 8,
		DefaultTemperature: 1, DefaultTopP: 1,
		ImageProjector: vision,
	}, generator)
	if err != nil {
		t.Fatal(err)
	}
	input := image.NewRGBA(image.Rect(0, 0, 1, 1))
	input.SetRGBA(0, 0, color.RGBA{R: 10, G: 20, B: 30, A: 255})
	var encoded bytes.Buffer
	if err := png.Encode(&encoded, input); err != nil {
		t.Fatal(err)
	}
	dataURI := "data:image/png;base64," + base64.StdEncoding.EncodeToString(encoded.Bytes())
	body, err := json.Marshal(map[string]any{
		"messages": []any{map[string]any{
			"role": "user",
			"content": []any{
				map[string]any{"type": "text", "text": "Look "},
				map[string]any{"type": "image_url", "image_url": map[string]any{"url": dataURI}},
				map[string]any{"type": "text", "text": " now"},
			},
		}},
		"max_tokens": 1,
	})
	if err != nil {
		t.Fatal(err)
	}
	response := httptest.NewRecorder()
	handler.ServeHTTP(
		response,
		httptest.NewRequest(http.MethodPost, "/v1/chat/completions", bytes.NewReader(body)),
	)
	if response.Code != http.StatusOK {
		t.Fatalf("status = %d body=%s", response.Code, response.Body.String())
	}
	if vision.before != "Look " || vision.after != " now" {
		t.Fatalf("projector text = %q, %q", vision.before, vision.after)
	}
	if !slices.Equal(generator.promptIDs, []tokenizer.TokenID{1, 2, 2, 3}) ||
		generator.projectedInputs == nil ||
		len(generator.projectedInputs.EmbeddingOverrides) != 2 ||
		generator.projectedInputs.MultiAxisPositions == nil {
		t.Fatalf("prompt IDs = %v, projected = %+v", generator.promptIDs, generator.projectedInputs)
	}
}

func TestChatMultipleImagesPreservesContentOrder(t *testing.T) {
	generator := &fakeGenerator{}
	vision := &fakeQwen3VLProjector{}
	handler, err := New(Config{
		ModelID: "test-model", MaxTokens: 8, DefaultTemperature: 1, DefaultTopP: 1,
		ImageProjector: vision,
	}, generator)
	if err != nil {
		t.Fatal(err)
	}
	input := image.NewRGBA(image.Rect(0, 0, 1, 1))
	var encoded bytes.Buffer
	if err := png.Encode(&encoded, input); err != nil {
		t.Fatal(err)
	}
	dataURI := "data:image/png;base64," + base64.StdEncoding.EncodeToString(encoded.Bytes())
	body, err := json.Marshal(map[string]any{
		"messages": []any{map[string]any{
			"role": "user", "content": []any{
				map[string]any{"type": "text", "text": "A"},
				map[string]any{"type": "image_url", "image_url": map[string]any{"url": dataURI}},
				map[string]any{"type": "text", "text": "B"},
				map[string]any{"type": "image_url", "image_url": map[string]any{"url": dataURI}},
				map[string]any{"type": "text", "text": "C"},
			},
		}}, "max_tokens": 1,
	})
	if err != nil {
		t.Fatal(err)
	}
	response := httptest.NewRecorder()
	handler.ServeHTTP(response, httptest.NewRequest(http.MethodPost, "/v1/chat/completions", bytes.NewReader(body)))
	if response.Code != http.StatusOK {
		t.Fatalf("status = %d body=%s", response.Code, response.Body.String())
	}
	if vision.images != 2 || !slices.Equal(vision.text, []string{"A", "B", "C"}) {
		t.Fatalf("projector sequence = images %d text %q", vision.images, vision.text)
	}
}

func TestStreamingChatAudioContentPartProjectsPrompt(t *testing.T) {
	generator := &fakeGenerator{}
	audio := &fakeAudioProjector{}
	handler, err := New(Config{
		ModelID: "test-model", MaxTokens: 8,
		DefaultTemperature: 1, DefaultTopP: 1,
		AudioProjector: audio,
	}, generator)
	if err != nil {
		t.Fatal(err)
	}
	wav := tinyPCM16WAV()
	body, err := json.Marshal(map[string]any{
		"messages": []any{map[string]any{
			"role": "user",
			"content": []any{
				map[string]any{"type": "text", "text": "Hear "},
				map[string]any{"type": "input_audio", "input_audio": map[string]any{
					"data": base64.StdEncoding.EncodeToString(wav), "format": "wav",
				}},
				map[string]any{"type": "text", "text": " now"},
			},
		}},
		"max_tokens": 1,
		"stream":     true,
	})
	if err != nil {
		t.Fatal(err)
	}
	response := httptest.NewRecorder()
	handler.ServeHTTP(
		response,
		httptest.NewRequest(http.MethodPost, "/v1/chat/completions", bytes.NewReader(body)),
	)
	if response.Code != http.StatusOK || !strings.Contains(response.Body.String(), "data: [DONE]") {
		t.Fatalf("status = %d body=%s", response.Code, response.Body.String())
	}
	if audio.before != "Hear " || audio.after != " now" ||
		!slices.Equal(audio.samples, []float32{0.5, -0.5}) {
		t.Fatalf("projector input = %q, %q, %v", audio.before, audio.after, audio.samples)
	}
	if !slices.Equal(generator.promptIDs, []tokenizer.TokenID{1, 4, 4, 3}) ||
		generator.projectedInputs == nil ||
		len(generator.projectedInputs.EmbeddingOverrides) != 2 {
		t.Fatalf("prompt IDs = %v, projected = %+v", generator.promptIDs, generator.projectedInputs)
	}
}

func TestChatMultimodalValidation(t *testing.T) {
	vision := &fakeQwen3VLProjector{}
	handler, err := New(Config{
		ModelID: "test-model", MaxTokens: 8,
		DefaultTemperature: 1, DefaultTopP: 1,
		ImageProjector: vision,
	}, &fakeGenerator{})
	if err != nil {
		t.Fatal(err)
	}
	imagePart := `{"type":"image_url","image_url":{"url":"data:image/png;base64,AA=="}}`
	for _, test := range []struct {
		path string
		body string
		want string
	}{
		{"/v1/chat/completions", `{"messages":[{"role":"user","content":[` + imagePart + `]}],"n":2}`, "requires n=1"},
		{"/v1/chat/completions/input_tokens", `{"messages":[{"role":"user","content":[` + imagePart + `]}],"n":2}`, "requires n=1"},
		{"/v1/chat/completions", `{"messages":[{"role":"system","content":"x"},{"role":"user","content":[` + imagePart + `]}]}`, "image 0 is unsupported"},
		{"/v1/chat/completions", `{"messages":[{"role":"user","content":[{"type":"image_url","image_url":{"url":"https://example.com/x.png"}}]}]}`, "remote media URLs are disabled"},
	} {
		response := httptest.NewRecorder()
		handler.ServeHTTP(
			response,
			httptest.NewRequest(http.MethodPost, test.path, strings.NewReader(test.body)),
		)
		if response.Code != http.StatusBadRequest || !strings.Contains(response.Body.String(), test.want) {
			t.Fatalf("%s body %s: status = %d response=%s", test.path, test.body, response.Code, response.Body.String())
		}
	}
}

func TestChatImageHistoryPreservesTurnPositionAndReplay(t *testing.T) {
	base := &fakeGenerator{}
	generator := &historyGenerator{fakeGenerator: base}
	vision := &fakeHistoryProjector{}
	handler, err := New(Config{
		ModelID: "test-model", MaxTokens: 8,
		DefaultTemperature: 1, DefaultTopP: 1,
		ImageProjector: vision,
	}, generator)
	if err != nil {
		t.Fatal(err)
	}
	input := image.NewRGBA(image.Rect(0, 0, 1, 1))
	var encoded bytes.Buffer
	if err := png.Encode(&encoded, input); err != nil {
		t.Fatal(err)
	}
	dataURI := "data:image/png;base64," + base64.StdEncoding.EncodeToString(encoded.Bytes())
	body, err := json.Marshal(map[string]any{
		"messages": []any{
			map[string]any{"role": "user", "content": []any{
				map[string]any{"type": "text", "text": "See "},
				map[string]any{"type": "image_url", "image_url": map[string]any{"url": dataURI}},
				map[string]any{"type": "text", "text": " now"},
			}},
			map[string]any{"role": "assistant", "content": "seen"},
			map[string]any{"role": "user", "content": "recall"},
		},
		"max_tokens": 1,
	})
	if err != nil {
		t.Fatal(err)
	}
	wantText := []string{
		"<chat><user>See ",
		" now</user><assistant>seen</assistant><user>recall</user><assistant>",
	}
	for run := 0; run < 2; run++ {
		response := httptest.NewRecorder()
		handler.ServeHTTP(response, httptest.NewRequest(http.MethodPost, "/v1/chat/completions", bytes.NewReader(body)))
		if response.Code != http.StatusOK {
			t.Fatalf("run %d status = %d body=%s", run, response.Code, response.Body.String())
		}
		if vision.images != 1 || !slices.Equal(vision.historyText, wantText) {
			t.Fatalf("run %d projector = images %d text %q", run, vision.images, vision.historyText)
		}
		if !slices.Equal(base.promptIDs, []tokenizer.TokenID{7, 8, 8, 9}) ||
			!base.cachePrompt ||
			base.projectedInputs == nil || len(base.projectedInputs.EmbeddingOverrides) != 2 ||
			len(base.projectedInputs.BidirectionalAttentionBlocks) != 1 {
			t.Fatalf("run %d prompt = %v projected=%+v", run, base.promptIDs, base.projectedInputs)
		}
	}
	if vision.historyRuns != 2 {
		t.Fatalf("history runs = %d", vision.historyRuns)
	}
}

func TestChatMixedImageAudioPreservesChunkOrder(t *testing.T) {
	base := &fakeGenerator{}
	generator := &historyGenerator{fakeGenerator: base}
	vision := &fakeHistoryProjector{}
	audio := &fakeAudioProjector{}
	handler, err := New(Config{
		ModelID: "test-model", MaxTokens: 8,
		DefaultTemperature: 1, DefaultTopP: 1,
		ImageProjector: vision, AudioProjector: audio,
	}, generator)
	if err != nil {
		t.Fatal(err)
	}
	input := image.NewRGBA(image.Rect(0, 0, 1, 1))
	var encodedImage bytes.Buffer
	if err := png.Encode(&encodedImage, input); err != nil {
		t.Fatal(err)
	}
	wav := silentPCM16WAV()
	body, err := json.Marshal(map[string]any{
		"messages": []any{map[string]any{
			"role": "user",
			"content": []any{
				map[string]any{"type": "text", "text": "A"},
				map[string]any{"type": "image_url", "image_url": map[string]any{
					"url": "data:image/png;base64," + base64.StdEncoding.EncodeToString(encodedImage.Bytes()),
				}},
				map[string]any{"type": "text", "text": "B"},
				map[string]any{"type": "input_audio", "input_audio": map[string]any{
					"data": base64.StdEncoding.EncodeToString(wav), "format": "wav",
				}},
				map[string]any{"type": "text", "text": "C"},
			},
		}},
		"max_tokens": 1,
	})
	if err != nil {
		t.Fatal(err)
	}
	response := httptest.NewRecorder()
	handler.ServeHTTP(response, httptest.NewRequest(http.MethodPost, "/v1/chat/completions", bytes.NewReader(body)))
	if response.Code != http.StatusOK {
		t.Fatalf("status = %d body=%s", response.Code, response.Body.String())
	}
	if !slices.Equal(vision.mediaKinds, []projector.MediaKind{projector.MediaImage, projector.MediaAudio}) ||
		!slices.Equal(vision.historyText, []string{"<chat><user>A", "B", "C</user><assistant>"}) {
		t.Fatalf("media = %v text=%q", vision.mediaKinds, vision.historyText)
	}
	if base.projectedInputs == nil || len(base.projectedInputs.EmbeddingOverrides) != 2 ||
		len(base.projectedInputs.BidirectionalAttentionBlocks) != 1 || !base.cachePrompt {
		t.Fatalf("projected = %+v cache=%v", base.projectedInputs, base.cachePrompt)
	}
}

func TestChatRemoteImageUsesExplicitPolicy(t *testing.T) {
	var encoded bytes.Buffer
	if err := png.Encode(&encoded, image.NewRGBA(image.Rect(0, 0, 1, 1))); err != nil {
		t.Fatal(err)
	}
	mediaServer := httptest.NewServer(http.HandlerFunc(func(response http.ResponseWriter, _ *http.Request) {
		response.Header().Set("Content-Type", "image/png")
		_, _ = response.Write(encoded.Bytes())
	}))
	defer mediaServer.Close()
	vision := &fakeQwen3VLProjector{}
	generator := &fakeGenerator{}
	handler, err := New(Config{
		ModelID: "test-model", MaxTokens: 8,
		DefaultTemperature: 1, DefaultTopP: 1,
		ImageProjector: vision, RemoteMediaPolicy: testRemoteMediaPolicy(t, mediaServer.URL),
	}, generator)
	if err != nil {
		t.Fatal(err)
	}
	body, err := json.Marshal(map[string]any{
		"messages": []any{map[string]any{
			"role": "user", "content": []any{map[string]any{
				"type": "image_url", "image_url": map[string]any{"url": mediaServer.URL + "/image.png"},
			}},
		}},
		"max_tokens": 1,
	})
	if err != nil {
		t.Fatal(err)
	}
	response := httptest.NewRecorder()
	handler.ServeHTTP(response, httptest.NewRequest(http.MethodPost, "/v1/chat/completions", bytes.NewReader(body)))
	if response.Code != http.StatusOK || generator.projectedInputs == nil {
		t.Fatalf("status = %d body=%s projected=%+v", response.Code, response.Body.String(), generator.projectedInputs)
	}
}

func TestChatImageHistoryOmissionUsesTextPath(t *testing.T) {
	base := &fakeGenerator{}
	generator := &historyGenerator{fakeGenerator: base}
	vision := &fakeHistoryProjector{}
	handler, err := New(Config{
		ModelID: "test-model", MaxTokens: 8,
		DefaultTemperature: 1, DefaultTopP: 1,
		ImageProjector: vision,
	}, generator)
	if err != nil {
		t.Fatal(err)
	}
	response := httptest.NewRecorder()
	handler.ServeHTTP(response, httptest.NewRequest(
		http.MethodPost,
		"/v1/chat/completions",
		strings.NewReader(`{"messages":[{"role":"user","content":"See it"},{"role":"assistant","content":"seen"},{"role":"user","content":"recall"}],"max_tokens":1}`),
	))
	if response.Code != http.StatusOK {
		t.Fatalf("status = %d body=%s", response.Code, response.Body.String())
	}
	if vision.historyRuns != 0 || base.projectedInputs != nil {
		t.Fatalf("history runs = %d projected=%+v", vision.historyRuns, base.projectedInputs)
	}
}

func TestMultimodalInputTokenCounting(t *testing.T) {
	vision := &fakeQwen3VLProjector{}
	handler, err := New(Config{
		ModelID: "test-model", MaxTokens: 8,
		DefaultTemperature: 1, DefaultTopP: 1,
		ImageProjector: vision,
	}, &fakeGenerator{})
	if err != nil {
		t.Fatal(err)
	}
	input := image.NewRGBA(image.Rect(0, 0, 1, 1))
	var encoded bytes.Buffer
	if err := png.Encode(&encoded, input); err != nil {
		t.Fatal(err)
	}
	dataURI := "data:image/png;base64," + base64.StdEncoding.EncodeToString(encoded.Bytes())
	requests := []struct {
		path string
		body map[string]any
	}{
		{
			path: "/v1/chat/completions/input_tokens",
			body: map[string]any{"messages": []any{map[string]any{
				"role": "user", "content": []any{
					map[string]any{"type": "image_url", "image_url": map[string]any{"url": dataURI}},
					map[string]any{"type": "text", "text": "B"},
					map[string]any{"type": "image_url", "image_url": map[string]any{"url": dataURI}},
				},
			}}},
		},
		{
			path: "/v1/responses/input_tokens",
			body: map[string]any{"input": []any{map[string]any{
				"role": "user", "content": []any{
					map[string]any{"type": "input_image", "image_url": dataURI},
					map[string]any{"type": "input_text", "text": "B"},
					map[string]any{"type": "input_image", "image_url": dataURI},
				},
			}}},
		},
	}
	for _, test := range requests {
		body, err := json.Marshal(test.body)
		if err != nil {
			t.Fatal(err)
		}
		response := httptest.NewRecorder()
		handler.ServeHTTP(response, httptest.NewRequest(http.MethodPost, test.path, bytes.NewReader(body)))
		if response.Code != http.StatusOK {
			t.Fatalf("%s status = %d body=%s", test.path, response.Code, response.Body.String())
		}
		var result struct {
			Object      string `json:"object"`
			InputTokens int    `json:"input_tokens"`
		}
		if err := json.Unmarshal(response.Body.Bytes(), &result); err != nil {
			t.Fatal(err)
		}
		if result.Object != "response.input_tokens" || result.InputTokens != 7 {
			t.Fatalf("%s response = %+v", test.path, result)
		}
	}
}

func TestResponsesInputTokensAliasesAndValidation(t *testing.T) {
	handler := newTestHandler(t, &fakeGenerator{})
	for _, path := range []string{"/responses/input_tokens", "/v1/responses/input_tokens"} {
		for _, body := range []string{
			`{"model":"test-model","instructions":"be concise","input":"hello"}`,
			`{"input":[{"type":"message","role":"user","content":"hello"}]}`,
			`{"input":[{"role":"user","content":[{"type":"input_text","text":"hello"}]}]}`,
		} {
			response := httptest.NewRecorder()
			handler.ServeHTTP(
				response,
				httptest.NewRequest(http.MethodPost, path, strings.NewReader(body)),
			)
			if response.Code != http.StatusOK {
				t.Fatalf("%s body %s: status = %d response=%s", path, body, response.Code, response.Body.String())
			}
			var result struct {
				Object      string `json:"object"`
				InputTokens int    `json:"input_tokens"`
			}
			if err := json.Unmarshal(response.Body.Bytes(), &result); err != nil {
				t.Fatal(err)
			}
			if result.Object != "response.input_tokens" || result.InputTokens != 3 {
				t.Fatalf("%s response = %+v", path, result)
			}
		}
	}
	for _, body := range []string{
		`{}`,
		`{"input":[]}`,
		`{"input":"hello","previous_response_id":"resp-old"}`,
		`{"model":"missing","input":"hello"}`,
		`{"input":[{"type":"function_call","role":"assistant","content":"x"}]}`,
		`{"input":[{"role":"user","content":[{"type":"input_image","image_url":"x"}]}]}`,
	} {
		response := httptest.NewRecorder()
		handler.ServeHTTP(
			response,
			httptest.NewRequest(
				http.MethodPost,
				"/responses/input_tokens",
				strings.NewReader(body),
			),
		)
		if response.Code < 400 {
			t.Fatalf("body %s status = %d response=%s", body, response.Code, response.Body.String())
		}
	}
	get := httptest.NewRecorder()
	handler.ServeHTTP(
		get,
		httptest.NewRequest(http.MethodGet, "/responses/input_tokens", nil),
	)
	if get.Code != http.StatusMethodNotAllowed || get.Header().Get("Allow") != http.MethodPost {
		t.Fatalf("GET status = %d Allow=%q", get.Code, get.Header().Get("Allow"))
	}
}

func TestResponsesInputTokensIncludeToolsAndCallHistory(t *testing.T) {
	generator := &fakeGenerator{}
	handler := newTestHandler(t, generator)
	response := httptest.NewRecorder()
	handler.ServeHTTP(
		response,
		httptest.NewRequest(
			http.MethodPost,
			"/v1/responses/input_tokens",
			strings.NewReader(
				`{"tools":[{"type":"function","name":"weather",`+
					`"parameters":{"type":"object"}}],`+
					`"input":[`+
					`{"role":"user","content":"weather?"},`+
					`{"type":"message","id":"msg_1","role":"assistant","status":"completed",`+
					`"content":[{"type":"output_text","text":"Checking.","annotations":[],"logprobs":[]}]},`+
					`{"type":"function_call","id":"fc_1","call_id":"call_1",`+
					`"name":"weather","arguments":"{\"city\":\"Paris\"}","status":"completed"},`+
					`{"type":"function_call_output","call_id":"call_1","output":"Sunny"}`+
					`]}`,
			),
		),
	)
	if response.Code != http.StatusOK {
		t.Fatalf("status = %d body=%s", response.Code, response.Body.String())
	}
	generator.mu.Lock()
	defer generator.mu.Unlock()
	if len(generator.chatOptions.Tools) != 1 ||
		len(generator.chatMessages) != 3 ||
		len(generator.chatMessages[1].ToolCalls) != 1 ||
		generator.chatMessages[1].Content != "Checking." ||
		generator.chatMessages[1].ToolCalls[0].ID != "call_1" ||
		generator.chatMessages[2].Role != "tool" ||
		generator.chatMessages[2].ToolCallID != "call_1" {
		t.Fatalf(
			"formatted options/messages = %+v / %+v",
			generator.chatOptions,
			generator.chatMessages,
		)
	}
}

func TestBufferedResponsesAliases(t *testing.T) {
	handler := newTestHandler(t, &fakeGenerator{})
	for _, path := range []string{"/responses", "/v1/responses"} {
		response := httptest.NewRecorder()
		handler.ServeHTTP(
			response,
			httptest.NewRequest(
				http.MethodPost,
				path,
				strings.NewReader(
					`{"model":"test-model","instructions":"brief","input":"hello","max_output_tokens":1,"temperature":0}`,
				),
			),
		)
		if response.Code != http.StatusOK {
			t.Fatalf("%s status = %d body=%s", path, response.Code, response.Body.String())
		}
		var result responsesResponse
		if err := json.Unmarshal(response.Body.Bytes(), &result); err != nil {
			t.Fatal(err)
		}
		if !strings.HasPrefix(result.ID, "resp_") ||
			result.Object != "response" ||
			result.Status != "completed" ||
			len(result.Output) != 1 ||
			!strings.HasPrefix(result.Output[0].ID, "msg_") ||
			result.Output[0].Content[0].Text != "A" ||
			result.Usage.InputTokens != 3 ||
			result.Usage.OutputTokens != 1 ||
			result.Usage.TotalTokens != 4 {
			t.Fatalf("%s response = %+v body=%s", path, result, response.Body.String())
		}
	}
}

func TestResponsesPreviousResponseContinuation(t *testing.T) {
	generator := &fakeGenerator{}
	handler := newTestHandler(t, generator)
	first := httptest.NewRecorder()
	handler.ServeHTTP(
		first,
		httptest.NewRequest(
			http.MethodPost,
			"/v1/responses",
			strings.NewReader(`{"instructions":"old","input":"hello","max_output_tokens":1}`),
		),
	)
	if first.Code != http.StatusOK {
		t.Fatalf("first status = %d body=%s", first.Code, first.Body.String())
	}
	var initial responsesResponse
	if err := json.Unmarshal(first.Body.Bytes(), &initial); err != nil {
		t.Fatal(err)
	}
	second := httptest.NewRecorder()
	handler.ServeHTTP(
		second,
		httptest.NewRequest(
			http.MethodPost,
			"/v1/responses",
			strings.NewReader(
				`{"instructions":"new","input":"next","max_output_tokens":1,"previous_response_id":"`+
					initial.ID+`"}`,
			),
		),
	)
	if second.Code != http.StatusOK {
		t.Fatalf("second status = %d body=%s", second.Code, second.Body.String())
	}
	generator.mu.Lock()
	defer generator.mu.Unlock()
	want := []inference.ChatMessage{
		{Role: "system", Content: "new"},
		{Role: "user", Content: "hello"},
		{Role: "assistant", Content: "A"},
		{Role: "user", Content: "next"},
	}
	if !reflect.DeepEqual(generator.chatMessages, want) {
		t.Fatalf("continuation messages = %+v, want %+v", generator.chatMessages, want)
	}
}

func TestResponsesStoreFalseDisablesContinuation(t *testing.T) {
	handler := newTestHandler(t, &fakeGenerator{})
	first := httptest.NewRecorder()
	handler.ServeHTTP(
		first,
		httptest.NewRequest(
			http.MethodPost,
			"/v1/responses",
			strings.NewReader(`{"input":"hello","max_output_tokens":1,"store":false}`),
		),
	)
	if first.Code != http.StatusOK {
		t.Fatalf("first status = %d body=%s", first.Code, first.Body.String())
	}
	var initial responsesResponse
	if err := json.Unmarshal(first.Body.Bytes(), &initial); err != nil {
		t.Fatal(err)
	}
	continuation := httptest.NewRecorder()
	handler.ServeHTTP(
		continuation,
		httptest.NewRequest(
			http.MethodPost,
			"/v1/responses",
			strings.NewReader(
				`{"input":"next","previous_response_id":"`+initial.ID+`"}`,
			),
		),
	)
	if continuation.Code != http.StatusNotFound ||
		!strings.Contains(continuation.Body.String(), "previous response not found") {
		t.Fatalf("continuation status = %d body=%s", continuation.Code, continuation.Body.String())
	}
}

func TestResponsesContinuationRetainsGeneratedToolCallID(t *testing.T) {
	generator := &fakeGenerator{pieces: []string{
		`<tool_call><function=weather><parameter=city>`,
		`Paris</parameter></function></tool_call>`,
	}}
	handler := newTestHandler(t, generator)
	first := httptest.NewRecorder()
	handler.ServeHTTP(
		first,
		httptest.NewRequest(
			http.MethodPost,
			"/v1/responses",
			strings.NewReader(
				`{"input":"weather?","max_output_tokens":2,"tool_choice":"required",`+
					`"tools":[{"type":"function","name":"weather","parameters":{"type":"object"}}]}`,
			),
		),
	)
	if first.Code != http.StatusOK {
		t.Fatalf("first status = %d body=%s", first.Code, first.Body.String())
	}
	var initial responsesResponse
	if err := json.Unmarshal(first.Body.Bytes(), &initial); err != nil {
		t.Fatal(err)
	}
	if len(initial.Output) != 1 || initial.Output[0].CallID == "" {
		t.Fatalf("first output = %+v", initial.Output)
	}
	second := httptest.NewRecorder()
	handler.ServeHTTP(
		second,
		httptest.NewRequest(
			http.MethodPost,
			"/v1/responses",
			strings.NewReader(
				`{"input":[{"type":"function_call_output","call_id":"`+
					initial.Output[0].CallID+`","output":"sunny"}],"max_output_tokens":1,`+
					`"previous_response_id":"`+initial.ID+`"}`,
			),
		),
	)
	if second.Code != http.StatusOK {
		t.Fatalf("second status = %d body=%s", second.Code, second.Body.String())
	}
	generator.mu.Lock()
	defer generator.mu.Unlock()
	if len(generator.chatMessages) != 3 ||
		len(generator.chatMessages[1].ToolCalls) != 1 ||
		generator.chatMessages[1].ToolCalls[0].ID != initial.Output[0].CallID ||
		generator.chatMessages[2].Role != "tool" ||
		generator.chatMessages[2].ToolCallID != initial.Output[0].CallID {
		t.Fatalf("continuation tool history = %+v", generator.chatMessages)
	}
}
