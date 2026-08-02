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
	"math"
	"net/http"
	"net/http/httptest"
	"slices"
	"strings"
	"sync"
	"testing"
	"time"

	"llamacpp2go/internal/cuda/driver"
	"llamacpp2go/internal/inference"
	"llamacpp2go/internal/projector"
	"llamacpp2go/internal/sampling"
	"llamacpp2go/internal/tokenizer"
)

type fakeGenerator struct {
	mu              sync.Mutex
	started         chan struct{}
	release         chan struct{}
	grammar         []string
	contextShift    bool
	gbnfSource      string
	gbnfRoot        string
	gbnfPatterns    []string
	gbnfTokens      []tokenizer.TokenID
	gbnfErr         error
	samplers        []sampling.SamplerStage
	sampling        sampling.Config
	promptIDs       []tokenizer.TokenID
	cachePrompt     bool
	keepTokens      int
	discardTokens   int
	minCacheReuse   int
	promptCached    int
	pieces          []string
	infillPrefix    []tokenizer.TokenID
	infillSuffix    []tokenizer.TokenID
	infillPrompt    []tokenizer.TokenID
	infillExtra     []inference.InfillExtra
	infillOptions   inference.InfillFormatOptions
	chatMessages    []inference.ChatMessage
	chatOptions     inference.ChatFormatOptions
	grammarParallel bool
	tokenDelay      time.Duration
	lora            []inference.LoRAScale
	loraConfigured  bool
	projectedInputs *inference.ProjectedInputs
}

type fakeQwen3VLProjector struct {
	before string
	after  string
	images int
	text   []string
}

func (f *fakeQwen3VLProjector) BuildImagesPrompt(
	_ context.Context,
	_ projector.ImageTokenizer,
	images []image.Image,
	text []string,
	_ bool,
) (projector.MultimodalPrompt, error) {
	f.images = len(images)
	f.text = append([]string(nil), text...)
	positions := [4][]uint32{
		{0, 1, 1, 2, 3, 3, 4}, {0, 1, 1, 2, 3, 3, 4},
		{0, 1, 2, 2, 3, 4, 4}, {0, 0, 0, 2, 0, 0, 4},
	}
	return projector.MultimodalPrompt{
		TokenIDs:   []tokenizer.TokenID{1, 2, 2, 3, 4, 4, 5},
		Embeddings: make([]float32, 4*2560), EmbeddingWidth: 2560,
		EmbeddingTokenIndices: []uint32{1, 2, 4, 5}, MultiAxisPositions: positions,
	}, nil
}

type fakeAudioProjector struct {
	before  string
	after   string
	samples []float32
}

func (f *fakeAudioProjector) BuildAudioPrompt(
	_ context.Context,
	_ projector.ImageTokenizer,
	samples []float32,
	before, after string,
) (projector.MultimodalPrompt, error) {
	f.before, f.after = before, after
	f.samples = append([]float32(nil), samples...)
	return projector.MultimodalPrompt{
		TokenIDs:   []tokenizer.TokenID{1, 4, 4, 3},
		Embeddings: make([]float32, 2*2560), EmbeddingWidth: 2560,
		EmbeddingTokenIndices: []uint32{1, 2},
	}, nil
}

func (f *fakeQwen3VLProjector) BuildQwen35ImagePrompt(
	_ context.Context,
	_ projector.Qwen3VLTokenizer,
	_ image.Image,
	before, after string,
	_ bool,
) (projector.Qwen3VLPrompt, error) {
	f.before, f.after = before, after
	positions := [4][]uint32{
		{0, 1, 1, 2}, {0, 1, 1, 2}, {0, 1, 2, 2}, {0, 0, 0, 2},
	}
	return projector.Qwen3VLPrompt{
		TokenIDs:   []tokenizer.TokenID{1, 2, 2, 3},
		Embeddings: make([]float32, 2*2560), EmbeddingWidth: 2560,
		EmbeddingStart: 1, EmbeddingTokenIndices: []uint32{1, 2}, MultiAxisPositions: positions,
	}, nil
}

func (f *fakeQwen3VLProjector) BuildImagePrompt(
	ctx context.Context,
	tokenizer projector.ImageTokenizer,
	input image.Image,
	before, after string,
	thinking bool,
) (projector.MultimodalPrompt, error) {
	return f.BuildQwen35ImagePrompt(ctx, tokenizer, input, before, after, thinking)
}

type failingMemoryGenerator struct {
	*fakeGenerator
}

type rankDisabledGenerator struct {
	*fakeGenerator
}

func (f *rankDisabledGenerator) SupportsRank() bool {
	return false
}

type fakeLoRAGenerator struct {
	*fakeGenerator
	adapters  []inference.LoRAAdapterInfo
	requested []inference.LoRAScale
}

func (f *fakeLoRAGenerator) LoRAAdapters() []inference.LoRAAdapterInfo {
	return append([]inference.LoRAAdapterInfo(nil), f.adapters...)
}

func (f *fakeLoRAGenerator) SetLoRAScales(scales []inference.LoRAScale) error {
	f.requested = append([]inference.LoRAScale(nil), scales...)
	return nil
}

type signalingRecorder struct {
	*httptest.ResponseRecorder
	flushed chan struct{}
}

func (recorder *signalingRecorder) Flush() {
	recorder.ResponseRecorder.Flush()
	select {
	case <-recorder.flushed:
	default:
		close(recorder.flushed)
	}
}

func (f *failingMemoryGenerator) DeviceMemoryStats(context.Context) (driver.MemoryStats, error) {
	return driver.MemoryStats{}, errors.New("metrics unavailable")
}

func (f *fakeGenerator) Generate(
	ctx context.Context,
	_ string,
	options inference.GenerateOptions,
) ([]tokenizer.TokenID, string, error) {
	f.mu.Lock()
	f.contextShift = options.ContextShift
	f.samplers = options.Sampler.Config().Samplers
	f.sampling = options.Sampler.Config()
	f.promptIDs = append([]tokenizer.TokenID(nil), options.PromptTokenIDs...)
	f.cachePrompt = options.CachePrompt
	f.keepTokens = options.KeepTokens
	f.discardTokens = options.DiscardTokens
	f.minCacheReuse = options.MinCacheReuse
	f.lora = append([]inference.LoRAScale(nil), options.LoRA...)
	f.loraConfigured = options.LoRAConfigured
	f.projectedInputs = options.ProjectedInputs
	f.mu.Unlock()
	if f.started != nil {
		f.mu.Lock()
		select {
		case <-f.started:
		default:
			close(f.started)
		}
		f.mu.Unlock()
		select {
		case <-ctx.Done():
			return nil, "", ctx.Err()
		case <-f.release:
		}
	}
	ids := []tokenizer.TokenID{10, 20}
	if options.PromptTokenIDs != nil {
		ids = append([]tokenizer.TokenID(nil), options.PromptTokenIDs...)
	}
	if options.OnPromptEvaluated != nil {
		options.OnPromptEvaluated(inference.PromptEvaluation{
			Tokens:   len(ids),
			Cached:   f.promptCached,
			Duration: time.Millisecond,
		})
	}
	pieces := f.pieces
	if pieces == nil {
		pieces = []string{"A", "B"}
	}
	for index := 0; index < options.MaxNewTokens && index < len(pieces); index++ {
		if f.tokenDelay > 0 {
			time.Sleep(f.tokenDelay)
		}
		ids = append(ids, tokenizer.TokenID(30+index))
		logits := make([]float32, 64)
		logits[30+index] = 4
		event := inference.TokenEvent{
			ID:     tokenizer.TokenID(30 + index),
			Piece:  pieces[index],
			Index:  index,
			Logits: logits,
		}
		if options.PostSamplingProbabilities > 0 {
			event.SelectedProbability = 0.8
			event.TopProbabilities = []sampling.TokenProbability{{
				ID:          30 + index,
				Probability: 0.8,
			}}
			if options.PostSamplingProbabilities > 1 {
				event.TopProbabilities = append(
					event.TopProbabilities,
					sampling.TokenProbability{
						ID:          40 + index,
						Probability: 0.2,
					},
				)
			}
		}
		if options.OnToken != nil {
			if err := options.OnToken(event); err != nil {
				return nil, "", err
			}
		}
		if options.ShouldStop != nil && options.ShouldStop(event) {
			break
		}
	}
	return ids, "promptAB", nil
}

func TestServerContextShiftOptionReachesGenerator(t *testing.T) {
	generator := &fakeGenerator{}
	handler, err := New(Config{
		ModelID:            "test-model",
		MaxTokens:          8,
		DefaultTemperature: 1,
		DefaultTopP:        1,
		ContextShift:       true,
	}, generator)
	if err != nil {
		t.Fatal(err)
	}
	request := httptest.NewRequest(
		http.MethodPost,
		"/v1/completions",
		strings.NewReader(`{"prompt":"hi","max_tokens":1}`),
	)
	response := httptest.NewRecorder()
	handler.ServeHTTP(response, request)
	if response.Code != http.StatusOK {
		t.Fatalf("status = %d, body = %s", response.Code, response.Body.String())
	}
	generator.mu.Lock()
	contextShift := generator.contextShift
	generator.mu.Unlock()
	if !contextShift {
		t.Fatal("server did not enable context shifting on generation")
	}
}

func TestNativeCompletionNKeepReachesGenerator(t *testing.T) {
	generator := &fakeGenerator{}
	handler, err := New(Config{
		ModelID:            "test-model",
		MaxTokens:          8,
		DefaultTemperature: 1,
		DefaultTopP:        1,
		ContextShift:       true,
	}, generator)
	if err != nil {
		t.Fatal(err)
	}
	request := httptest.NewRequest(
		http.MethodPost,
		"/completion",
		strings.NewReader(`{"prompt":"hi","n_predict":1,"n_keep":3,"n_discard":5}`),
	)
	response := httptest.NewRecorder()
	handler.ServeHTTP(response, request)
	if response.Code != http.StatusOK {
		t.Fatalf("status = %d, body = %s", response.Code, response.Body.String())
	}
	var result nativeCompletionResponse
	if err := json.Unmarshal(response.Body.Bytes(), &result); err != nil {
		t.Fatal(err)
	}
	generator.mu.Lock()
	keepTokens := generator.keepTokens
	discardTokens := generator.discardTokens
	generator.mu.Unlock()
	if keepTokens != 3 {
		t.Fatalf("generator keep tokens = %d, want 3", keepTokens)
	}
	if result.GenerationSettings["n_keep"] != float64(3) {
		t.Fatalf("generation settings = %#v", result.GenerationSettings)
	}
	if discardTokens != 5 ||
		result.GenerationSettings["n_discard"] != float64(5) {
		t.Fatalf("discard tokens/settings = %d/%#v",
			discardTokens, result.GenerationSettings)
	}

	request = httptest.NewRequest(
		http.MethodPost,
		"/completion",
		strings.NewReader(`{"prompt":"hi","n_predict":1,"n_keep":-2}`),
	)
	response = httptest.NewRecorder()
	handler.ServeHTTP(response, request)
	if response.Code != http.StatusBadRequest {
		t.Fatalf("invalid n_keep status = %d, body = %s",
			response.Code, response.Body.String())
	}

	request = httptest.NewRequest(
		http.MethodPost,
		"/completion",
		strings.NewReader(`{"prompt":"hi","n_predict":1,"n_discard":-1}`),
	)
	response = httptest.NewRecorder()
	handler.ServeHTTP(response, request)
	if response.Code != http.StatusBadRequest {
		t.Fatalf("invalid n_discard status = %d, body = %s",
			response.Code, response.Body.String())
	}
}

func TestServerRequestTimeoutCancelsGeneration(t *testing.T) {
	generator := &fakeGenerator{
		started: make(chan struct{}),
		release: make(chan struct{}),
	}
	handler, err := New(Config{
		ModelID:            "test-model",
		MaxTokens:          8,
		DefaultTemperature: 1,
		DefaultTopP:        1,
		RequestTimeout:     20 * time.Millisecond,
	}, generator)
	if err != nil {
		t.Fatal(err)
	}
	request := httptest.NewRequest(
		http.MethodPost,
		"/v1/completions",
		strings.NewReader(`{"prompt":"hi","max_tokens":1}`),
	)
	response := httptest.NewRecorder()
	handler.ServeHTTP(response, request)
	if response.Code != http.StatusRequestTimeout {
		t.Fatalf("status = %d body=%s", response.Code, response.Body.String())
	}
	if !strings.Contains(response.Body.String(), "deadline exceeded") {
		t.Fatalf("timeout response = %s", response.Body.String())
	}
}

func TestServerRejectsNegativeRequestTimeout(t *testing.T) {
	if _, err := New(Config{RequestTimeout: -time.Second}, &fakeGenerator{}); err == nil {
		t.Fatal("negative request timeout was accepted")
	}
}

func TestServerRejectsExcessiveSlotCount(t *testing.T) {
	if _, err := New(Config{MaxConcurrent: 65537}, &fakeGenerator{}); err == nil {
		t.Fatal("excessive slot count was accepted")
	}
}

func (f *fakeGenerator) Embed(
	ctx context.Context,
	text string,
) ([]float32, int, error) {
	if err := ctx.Err(); err != nil {
		return nil, 0, err
	}
	return []float32{0.6, 0.8}, len(text), nil
}

func (f *fakeGenerator) EmbedTokens(
	ctx context.Context,
	tokens []tokenizer.TokenID,
) ([]float32, int, error) {
	if err := ctx.Err(); err != nil {
		return nil, 0, err
	}
	f.mu.Lock()
	f.promptIDs = append([]tokenizer.TokenID(nil), tokens...)
	f.mu.Unlock()
	return []float32{0.6, 0.8}, len(tokens), nil
}

func (f *fakeGenerator) EmbedAdvanced(
	ctx context.Context,
	text string,
	options inference.EmbeddingOptions,
) (inference.EmbeddingResult, error) {
	return fakeAdvancedEmbedding(ctx, len(text), options)
}

func (f *fakeGenerator) EmbedTokensAdvanced(
	ctx context.Context,
	tokens []tokenizer.TokenID,
	options inference.EmbeddingOptions,
) (inference.EmbeddingResult, error) {
	f.mu.Lock()
	f.promptIDs = append([]tokenizer.TokenID(nil), tokens...)
	f.mu.Unlock()
	return fakeAdvancedEmbedding(ctx, len(tokens), options)
}

func (f *fakeGenerator) SupportsRank() bool {
	return true
}

func (f *fakeGenerator) RankPair(
	ctx context.Context,
	query string,
	document string,
) (inference.RankResult, error) {
	if err := ctx.Err(); err != nil {
		return inference.RankResult{}, err
	}
	return inference.RankResult{
		Scores: []float32{float32(len(document)) / 10},
		Labels: []string{"0"},
		Tokens: len(query) + len(document),
	}, nil
}

func fakeAdvancedEmbedding(
	ctx context.Context,
	tokens int,
	options inference.EmbeddingOptions,
) (inference.EmbeddingResult, error) {
	if err := ctx.Err(); err != nil {
		return inference.EmbeddingResult{}, err
	}
	if options.Pooling == inference.EmbeddingPoolingNone {
		return inference.EmbeddingResult{
			Vectors: [][]float32{{3, 4}, {-6, 8}},
			Tokens:  tokens,
		}, nil
	}
	vector := []float32{0.6, 0.8}
	if options.Normalize == -1 {
		vector = []float32{3, 4}
	}
	return inference.EmbeddingResult{Vectors: [][]float32{vector}, Tokens: tokens}, nil
}

func (f *fakeGenerator) FormatChat(messages []inference.ChatMessage) (string, error) {
	if len(messages) == 0 {
		return "", errors.New("chat message list is empty")
	}
	return "formatted-chat", nil
}

func (f *fakeGenerator) FormatChatWithOptions(
	messages []inference.ChatMessage,
	options inference.ChatFormatOptions,
) (string, error) {
	if len(messages) == 0 {
		return "", errors.New("chat message list is empty")
	}
	f.mu.Lock()
	f.chatMessages = append([]inference.ChatMessage(nil), messages...)
	f.chatOptions = options
	f.chatOptions.Tools = append([]inference.ChatTool(nil), options.Tools...)
	f.mu.Unlock()
	return "formatted-chat-tools", nil
}

func (f *fakeGenerator) ParseChatOutput(
	output string,
	_ []inference.ChatTool,
) (inference.ChatMessage, error) {
	message := inference.ChatMessage{Role: "assistant", Content: output}
	if strings.Contains(output, "<tool_call>") {
		message.Content = ""
		message.ToolCalls = []inference.ChatToolCall{{
			Type: "function",
			Function: inference.ChatToolFunction{
				Name:      "weather",
				Arguments: `{"city":"Paris"}`,
			},
		}}
	}
	return message, nil
}

func (f *fakeGenerator) ChatToolGrammar(
	_ []inference.ChatTool,
	required bool,
	_ bool,
	parallel bool,
) (string, string, []string, error) {
	f.mu.Lock()
	f.grammarParallel = parallel
	f.mu.Unlock()
	if required {
		return `root ::= "a"`, "root", nil, nil
	}
	return `root ::= "a"`, "root", []string{`(<tool_call>)`}, nil
}

func (f *fakeGenerator) TokenizeGrammarChoices(choices []string) (*sampling.TokenGrammar, error) {
	f.mu.Lock()
	f.grammar = append([]string(nil), choices...)
	f.mu.Unlock()
	tokenChoices := make([][]int, len(choices))
	for index := range choices {
		tokenChoices[index] = []int{index + 1}
	}
	return sampling.NewChoiceGrammar(tokenChoices, []int{63}, 64)
}

func (f *fakeGenerator) CompileGBNF(
	source, root string,
) (*sampling.GBNFGrammar, error) {
	f.mu.Lock()
	f.gbnfSource = source
	f.gbnfRoot = root
	err := f.gbnfErr
	f.mu.Unlock()
	if err != nil {
		return nil, err
	}
	return sampling.NewGBNFGrammar(
		source,
		root,
		[][]byte{[]byte("a"), nil},
		[]int{1},
	)
}

func (f *fakeGenerator) CompileLazyGBNF(
	source, root string,
	patterns []string,
	tokens []tokenizer.TokenID,
) (*sampling.GBNFGrammar, error) {
	f.mu.Lock()
	f.gbnfSource = source
	f.gbnfRoot = root
	f.gbnfPatterns = append([]string(nil), patterns...)
	f.gbnfTokens = append([]tokenizer.TokenID(nil), tokens...)
	f.mu.Unlock()
	triggerTokens := make([]int, len(tokens))
	for index, token := range tokens {
		triggerTokens[index] = int(token)
	}
	return sampling.NewGBNFGrammarWithOptions(
		source,
		root,
		[][]byte{[]byte("a"), []byte("trigger"), nil},
		[]int{2},
		nil,
		sampling.GBNFLazyOptions{
			Enabled:  true,
			Patterns: patterns,
			Tokens:   triggerTokens,
		},
	)
}

func (f *fakeGenerator) TokenizeSamplingText(text string) ([]tokenizer.TokenID, error) {
	if text == "ban" {
		return []tokenizer.TokenID{4, 5}, nil
	}
	return []tokenizer.TokenID{3}, nil
}

func (f *fakeGenerator) SamplingEOGTokens() []tokenizer.TokenID {
	return []tokenizer.TokenID{6, 7}
}

func (f *fakeGenerator) SamplingVocabularySize() int {
	return 64
}

func (f *fakeGenerator) SamplingInfillVocabulary() (*sampling.InfillVocabulary, error) {
	pieces := make([]string, 64)
	eog := make([]bool, 64)
	for index := range pieces {
		pieces[index] = fmt.Sprintf("<%d>", index)
	}
	eog[63] = true
	return &sampling.InfillVocabulary{
		Pieces: pieces,
		EOG:    eog,
		EOT:    63,
		EOS:    63,
	}, nil
}

func (f *fakeGenerator) FormatInfillTokens(
	prefix, suffix, prompt []tokenizer.TokenID,
	extra []inference.InfillExtra,
	options inference.InfillFormatOptions,
) ([]tokenizer.TokenID, error) {
	f.mu.Lock()
	f.infillPrefix = append([]tokenizer.TokenID(nil), prefix...)
	f.infillSuffix = append([]tokenizer.TokenID(nil), suffix...)
	f.infillPrompt = append([]tokenizer.TokenID(nil), prompt...)
	f.infillExtra = append([]inference.InfillExtra(nil), extra...)
	f.infillOptions = options
	f.mu.Unlock()
	return []tokenizer.TokenID{1, 2, 3}, nil
}

func (f *fakeGenerator) TokenizeText(
	text string,
	addSpecial, parseSpecial bool,
) ([]tokenizer.TokenID, error) {
	result := []tokenizer.TokenID{10}
	if addSpecial {
		result = append([]tokenizer.TokenID{1}, result...)
	}
	if parseSpecial {
		result = append(result, 2)
	}
	return result, nil
}

func (f *fakeGenerator) DetokenizeTokens(tokens []tokenizer.TokenID) (string, error) {
	var result strings.Builder
	for _, token := range tokens {
		fmt.Fprintf(&result, "[%d]", token)
	}
	return result.String(), nil
}

func (f *fakeGenerator) TokenPiece(token tokenizer.TokenID) (string, error) {
	if token == 63 {
		return string([]byte{0xc3}), nil
	}
	return fmt.Sprintf("<%d>", token), nil
}

func (f *fakeGenerator) DetokenizePromptTokens(tokens []tokenizer.TokenID) (string, error) {
	return f.DetokenizeTokens(tokens)
}

func (f *fakeGenerator) ModelProperties() inference.ModelProperties {
	return inference.ModelProperties{
		Path:              "fixture.gguf",
		Name:              "fixture-name",
		Architecture:      "qwen3",
		FileType:          "Q8_0",
		ContextLength:     32768,
		EmbeddingLength:   2560,
		FeedForwardLength: 9728,
		BlockCount:        36,
		HeadCount:         32,
		HeadCountKV:       8,
		VocabularySize:    64,
		VocabularyType:    "gpt2",
		ParameterCount:    4000000000,
		ModelSize:         4200000000,
		ChatTemplate:      "{{ messages }}",
		BOSToken:          "<bos>",
		EOSToken:          "<eos>",
	}
}

func (f *fakeGenerator) DeviceMemoryStats(context.Context) (driver.MemoryStats, error) {
	return driver.MemoryStats{CurrentBytes: 123, PeakBytes: 456, Allocations: 2}, nil
}

func (f *fakeGenerator) DeviceExecutionStats(context.Context) (driver.ExecutionStats, error) {
	return driver.ExecutionStats{
		KernelLaunches:          12,
		StreamSynchronizations:  3,
		ContextSynchronizations: 1,
		HostToDeviceBytes:       1024,
		DeviceToHostBytes:       2048,
		DeviceToDeviceBytes:     3072,
		DeviceMemsetBytes:       4096,
	}, nil
}

func newTestHandler(t testing.TB, generator Generator) *Handler {
	t.Helper()
	handler, err := New(Config{
		ModelID:            "test-model",
		MaxTokens:          8,
		DefaultTemperature: 1,
		DefaultTopP:        1,
	}, generator)
	if err != nil {
		t.Fatal(err)
	}
	return handler
}

func TestHealth(t *testing.T) {
	handler := newTestHandler(t, &fakeGenerator{})
	for _, path := range []string{"/health", "/healthz", "/v1/health"} {
		request := httptest.NewRequest(http.MethodGet, path, nil)
		response := httptest.NewRecorder()
		handler.ServeHTTP(response, request)
		if response.Code != http.StatusOK {
			t.Fatalf("%s status = %d, want 200", path, response.Code)
		}
		if !strings.Contains(response.Body.String(), `"model":"test-model"`) {
			t.Fatalf("%s health body = %s", path, response.Body.String())
		}
	}
}

func TestMetricsOmitDeviceMemoryWhenSnapshotFails(t *testing.T) {
	handler := newTestHandler(t, &failingMemoryGenerator{fakeGenerator: &fakeGenerator{}})
	response := httptest.NewRecorder()
	handler.ServeHTTP(response, httptest.NewRequest(http.MethodGet, "/metrics", nil))
	if response.Code != http.StatusOK {
		t.Fatalf("status = %d body=%s", response.Code, response.Body.String())
	}
	if strings.Contains(response.Body.String(), "llamacpp2go_cuda_memory_") {
		t.Fatalf("failed device metrics leaked into response: %s", response.Body.String())
	}
}

func TestLoraAdaptersEmptyControlPlane(t *testing.T) {
	handler := newTestHandler(t, &fakeGenerator{})
	get := httptest.NewRecorder()
	handler.ServeHTTP(get, httptest.NewRequest(http.MethodGet, "/lora-adapters", nil))
	if get.Code != http.StatusOK || strings.TrimSpace(get.Body.String()) != "[]" {
		t.Fatalf("GET status/body = %d %s", get.Code, get.Body.String())
	}
	disable := httptest.NewRecorder()
	handler.ServeHTTP(
		disable,
		httptest.NewRequest(http.MethodPost, "/lora-adapters", strings.NewReader(`[]`)),
	)
	if disable.Code != http.StatusOK || !strings.Contains(disable.Body.String(), `"success":true`) {
		t.Fatalf("POST [] status/body = %d %s", disable.Code, disable.Body.String())
	}
	enable := httptest.NewRecorder()
	handler.ServeHTTP(
		enable,
		httptest.NewRequest(
			http.MethodPost,
			"/lora-adapters",
			strings.NewReader(`[{"id":0,"scale":1}]`),
		),
	)
	if enable.Code != http.StatusNotImplemented {
		t.Fatalf("POST adapter status/body = %d %s", enable.Code, enable.Body.String())
	}
}

func TestLoraAdaptersLoadedControlPlane(t *testing.T) {
	generator := &fakeLoRAGenerator{
		fakeGenerator: &fakeGenerator{},
		adapters:      []inference.LoRAAdapterInfo{{ID: 0, Path: "adapter.gguf", Scale: 1}},
	}
	handler := newTestHandler(t, generator)
	get := httptest.NewRecorder()
	handler.ServeHTTP(get, httptest.NewRequest(http.MethodGet, "/lora-adapters", nil))
	if get.Code != http.StatusOK || !strings.Contains(get.Body.String(), `"path":"adapter.gguf"`) {
		t.Fatalf("GET status/body = %d %s", get.Code, get.Body.String())
	}
	post := httptest.NewRecorder()
	handler.ServeHTTP(post, httptest.NewRequest(
		http.MethodPost, "/lora-adapters", strings.NewReader(`[{"id":0,"scale":0.25}]`),
	))
	if post.Code != http.StatusOK || len(generator.requested) != 1 || generator.requested[0].Scale != 0.25 {
		t.Fatalf("POST status/body/request = %d %s %v", post.Code, post.Body.String(), generator.requested)
	}
}

func TestNativeCompletionPerRequestLoRA(t *testing.T) {
	generator := &fakeLoRAGenerator{
		fakeGenerator: &fakeGenerator{},
		adapters:      []inference.LoRAAdapterInfo{{ID: 0, Path: "adapter.gguf", Scale: 1}},
	}
	handler := newTestHandler(t, generator)
	response := httptest.NewRecorder()
	handler.ServeHTTP(response, httptest.NewRequest(
		http.MethodPost,
		"/completion",
		strings.NewReader(`{"prompt":"hello","n_predict":1,"lora":[{"id":0,"scale":0.25}]}`),
	))
	if response.Code != http.StatusOK {
		t.Fatalf("status/body = %d %s", response.Code, response.Body.String())
	}
	if !generator.loraConfigured || len(generator.lora) != 1 || generator.lora[0].Scale != 0.25 {
		t.Fatalf("request LoRA = configured:%t values:%v", generator.loraConfigured, generator.lora)
	}
	invalid := httptest.NewRecorder()
	handler.ServeHTTP(invalid, httptest.NewRequest(
		http.MethodPost,
		"/completion",
		strings.NewReader(`{"prompt":"hello","n_predict":1,"lora":[{"id":1,"scale":1}]}`),
	))
	if invalid.Code != http.StatusBadRequest || !strings.Contains(invalid.Body.String(), "not loaded") {
		t.Fatalf("invalid status/body = %d %s", invalid.Code, invalid.Body.String())
	}
}

func TestModels(t *testing.T) {
	handler := newTestHandler(t, &fakeGenerator{})
	request := httptest.NewRequest(http.MethodGet, "/v1/models", nil)
	response := httptest.NewRecorder()
	handler.ServeHTTP(response, request)
	if response.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200", response.Code)
	}
	if !strings.Contains(response.Body.String(), `"id":"test-model"`) {
		t.Fatalf("models body = %s", response.Body.String())
	}
	var result struct {
		Models []struct {
			Name         string         `json:"name"`
			Capabilities []string       `json:"capabilities"`
			Details      map[string]any `json:"details"`
		} `json:"models"`
		Data []struct {
			Meta map[string]any `json:"meta"`
		} `json:"data"`
	}
	if err := json.Unmarshal(response.Body.Bytes(), &result); err != nil {
		t.Fatal(err)
	}
	if len(result.Models) != 1 ||
		result.Models[0].Name != "test-model" ||
		!slices.Contains(result.Models[0].Capabilities, "completion") ||
		result.Models[0].Details["format"] != "gguf" ||
		result.Models[0].Details["family"] != "qwen3" ||
		len(result.Data) != 1 ||
		result.Data[0].Meta["n_vocab"] != float64(64) ||
		result.Data[0].Meta["n_params"] != float64(4000000000) ||
		result.Data[0].Meta["size"] != float64(4200000000) ||
		result.Data[0].Meta["ftype"] != "Q8_0" {
		t.Fatalf("models response = %#v", result)
	}
}

func TestModelsArePublicAndStrictlyGet(t *testing.T) {
	handler, err := New(Config{
		ModelID:            "test-model",
		MaxTokens:          8,
		DefaultTemperature: 1,
		DefaultTopP:        1,
		APIKey:             "test-secret",
	}, &fakeGenerator{})
	if err != nil {
		t.Fatal(err)
	}
	for _, path := range []string{"/models", "/v1/models"} {
		response := httptest.NewRecorder()
		handler.ServeHTTP(response, httptest.NewRequest(http.MethodGet, path, nil))
		if response.Code != http.StatusOK {
			t.Fatalf("%s status = %d body=%s", path, response.Code, response.Body.String())
		}
	}
	post := httptest.NewRecorder()
	handler.ServeHTTP(post, httptest.NewRequest(http.MethodPost, "/models", nil))
	if post.Code != http.StatusMethodNotAllowed ||
		post.Header().Get("Allow") != http.MethodGet {
		t.Fatalf("POST status = %d Allow=%q", post.Code, post.Header().Get("Allow"))
	}
}

func TestProperties(t *testing.T) {
	handler := newTestHandler(t, &fakeGenerator{})
	request := httptest.NewRequest(http.MethodGet, "/props", nil)
	response := httptest.NewRecorder()
	handler.ServeHTTP(response, request)
	if response.Code != http.StatusOK {
		t.Fatalf("status = %d body=%s", response.Code, response.Body.String())
	}
	var result propertiesResponse
	if err := json.Unmarshal(response.Body.Bytes(), &result); err != nil {
		t.Fatal(err)
	}
	if result.TotalSlots != 1 ||
		result.ModelAlias != "test-model" ||
		result.ModelFType != "Q8_0" ||
		result.ModelPath != "fixture.gguf" ||
		result.ModelMetadata.Architecture != "qwen3" ||
		result.DefaultGenerationSettings.NCtx != 32768 ||
		result.DefaultGenerationSettings.Params.NPredict != 16 ||
		result.DefaultGenerationSettings.Params.TopP != 1 ||
		result.BOSToken != "<bos>" ||
		result.EOSToken != "<eos>" ||
		result.ChatTemplate != "{{ messages }}" ||
		!result.EndpointMetrics ||
		!result.EndpointSlots ||
		result.EndpointProps {
		t.Fatalf("properties = %+v", result)
	}
	if result.Modalities["vision"] ||
		result.Modalities["video"] ||
		result.Modalities["audio"] {
		t.Fatalf("modalities = %#v", result.Modalities)
	}
}

func TestSlotsReportStableBusyAndIdleState(t *testing.T) {
	generator := &fakeGenerator{
		started: make(chan struct{}),
		release: make(chan struct{}),
	}
	handler, err := New(Config{
		ModelID:            "test-model",
		MaxTokens:          8,
		MaxConcurrent:      2,
		DefaultTemperature: 1,
		DefaultTopP:        1,
		APIKey:             "test-secret",
	}, generator)
	if err != nil {
		t.Fatal(err)
	}
	var wait sync.WaitGroup
	responses := make([]*httptest.ResponseRecorder, 2)
	for index := range responses {
		wait.Add(1)
		go func(index int) {
			defer wait.Done()
			request := httptest.NewRequest(
				http.MethodPost,
				"/v1/completions",
				strings.NewReader(`{"prompt":"hi","max_tokens":1}`),
			)
			request.Header.Set("Authorization", "Bearer test-secret")
			responses[index] = httptest.NewRecorder()
			handler.ServeHTTP(responses[index], request)
		}(index)
	}
	deadline := time.Now().Add(time.Second)
	for len(handler.slots) != 0 && time.Now().Before(deadline) {
		time.Sleep(time.Millisecond)
	}
	if len(handler.slots) != 0 {
		t.Fatal("generation requests did not acquire both slots")
	}

	slotRequest := httptest.NewRequest(http.MethodGet, "/slots", nil)
	slotRequest.Header.Set("Authorization", "Bearer test-secret")
	slotResponse := httptest.NewRecorder()
	handler.ServeHTTP(slotResponse, slotRequest)
	if slotResponse.Code != http.StatusOK {
		t.Fatalf("slots status = %d body=%s", slotResponse.Code, slotResponse.Body.String())
	}
	var busy []slotStatusItem
	if err := json.Unmarshal(slotResponse.Body.Bytes(), &busy); err != nil {
		t.Fatal(err)
	}
	if len(busy) != 2 {
		t.Fatalf("slot count = %d", len(busy))
	}
	for id, slot := range busy {
		if slot.ID != id ||
			!slot.IsProcessing ||
			slot.IDTask == nil ||
			*slot.IDTask == 0 ||
			slot.NCtx != 32768 {
			t.Fatalf("busy slot %d = %+v", id, slot)
		}
	}
	if *busy[0].IDTask == *busy[1].IDTask {
		t.Fatalf("slot task IDs collide: %+v", busy)
	}

	failRequest := httptest.NewRequest(http.MethodGet, "/slots?fail_on_no_slot=1", nil)
	failRequest.Header.Set("Authorization", "Bearer test-secret")
	failResponse := httptest.NewRecorder()
	handler.ServeHTTP(failResponse, failRequest)
	if failResponse.Code != http.StatusServiceUnavailable {
		t.Fatalf("fail-on-no-slot status = %d body=%s", failResponse.Code, failResponse.Body.String())
	}

	close(generator.release)
	wait.Wait()
	for index, response := range responses {
		if response.Code != http.StatusOK {
			t.Fatalf("generation %d status = %d body=%s", index, response.Code, response.Body.String())
		}
	}
	idleRequest := httptest.NewRequest(http.MethodGet, "/slots?fail_on_no_slot=1", nil)
	idleRequest.Header.Set("Authorization", "Bearer test-secret")
	idleResponse := httptest.NewRecorder()
	handler.ServeHTTP(idleResponse, idleRequest)
	if idleResponse.Code != http.StatusOK {
		t.Fatalf("idle slots status = %d body=%s", idleResponse.Code, idleResponse.Body.String())
	}
	var idle []slotStatusItem
	if err := json.Unmarshal(idleResponse.Body.Bytes(), &idle); err != nil {
		t.Fatal(err)
	}
	for id, slot := range idle {
		if slot.IsProcessing || slot.IDTask != nil {
			t.Fatalf("idle slot %d = %+v", id, slot)
		}
		if slot.NPromptTokens != 2 ||
			slot.NPromptTokensProcessed != 2 ||
			slot.NPromptTokensCache != 0 ||
			slot.Prompt != "hi" ||
			slot.Generated != "A" ||
			slot.Timings == nil ||
			slot.Timings.CacheN != 0 ||
			slot.Timings.PromptN != 2 ||
			slot.Timings.PromptMS != 1 ||
			slot.Timings.PredictedN != 1 ||
			slot.Timings.PredictedMS <= 0 ||
			slot.Timings.PromptPerTokenMS != 0.5 ||
			slot.Timings.PromptPerSecond != 2000 ||
			slot.Timings.PredictedPerTokenMS <= 0 ||
			slot.Timings.PredictedPerSecond <= 0 ||
			slot.NextToken == nil ||
			slot.NextToken.HasNextToken ||
			slot.NextToken.HasNewLine ||
			slot.NextToken.NRemain != 0 ||
			slot.NextToken.NDecoded != 1 {
			t.Fatalf("idle slot metrics %d = %+v", id, slot)
		}
	}
}

func TestSlotsRequireAuthenticationAndGET(t *testing.T) {
	handler, err := New(Config{
		ModelID:            "test-model",
		MaxTokens:          8,
		DefaultTemperature: 1,
		DefaultTopP:        1,
		APIKey:             "test-secret",
	}, &fakeGenerator{})
	if err != nil {
		t.Fatal(err)
	}
	unauthorized := httptest.NewRecorder()
	handler.ServeHTTP(unauthorized, httptest.NewRequest(http.MethodGet, "/slots", nil))
	if unauthorized.Code != http.StatusUnauthorized {
		t.Fatalf("unauthorized status = %d body=%s", unauthorized.Code, unauthorized.Body.String())
	}
	request := httptest.NewRequest(http.MethodPost, "/slots", nil)
	request.Header.Set("Authorization", "Bearer test-secret")
	method := httptest.NewRecorder()
	handler.ServeHTTP(method, request)
	if method.Code != http.StatusMethodNotAllowed ||
		method.Header().Get("Allow") != http.MethodGet {
		t.Fatalf("POST status = %d Allow=%q", method.Code, method.Header().Get("Allow"))
	}
}

func TestNativeCompletionHonorsRequestedSlot(t *testing.T) {
	handler, err := New(Config{
		ModelID:            "test-model",
		MaxTokens:          8,
		MaxConcurrent:      2,
		DefaultTemperature: 1,
		DefaultTopP:        1,
	}, &fakeGenerator{})
	if err != nil {
		t.Fatal(err)
	}
	request := httptest.NewRequest(
		http.MethodPost,
		"/completion",
		strings.NewReader(`{"prompt":"hi","n_predict":1,"id_slot":1}`),
	)
	response := httptest.NewRecorder()
	handler.ServeHTTP(response, request)
	if response.Code != http.StatusOK {
		t.Fatalf("status = %d body=%s", response.Code, response.Body.String())
	}
	var result nativeCompletionResponse
	if err := json.Unmarshal(response.Body.Bytes(), &result); err != nil {
		t.Fatal(err)
	}
	if result.IDSlot != 1 {
		t.Fatalf("id_slot = %d, want 1", result.IDSlot)
	}

	slot, ok := handler.acquireSlot(1)
	if !ok || slot != 1 {
		t.Fatalf("could not reserve slot 1: %d/%v", slot, ok)
	}
	defer handler.releaseSlot(slot)
	busy := httptest.NewRecorder()
	handler.ServeHTTP(
		busy,
		httptest.NewRequest(
			http.MethodPost,
			"/completion",
			strings.NewReader(`{"prompt":"hi","n_predict":1,"id_slot":1}`),
		),
	)
	if busy.Code != http.StatusTooManyRequests {
		t.Fatalf("busy requested slot status = %d body=%s", busy.Code, busy.Body.String())
	}
}

func TestPropertiesRejectsPost(t *testing.T) {
	handler := newTestHandler(t, &fakeGenerator{})
	response := httptest.NewRecorder()
	handler.ServeHTTP(response, httptest.NewRequest(http.MethodPost, "/props", nil))
	if response.Code != http.StatusMethodNotAllowed ||
		response.Header().Get("Allow") != http.MethodGet {
		t.Fatalf(
			"status = %d Allow=%q body=%s",
			response.Code,
			response.Header().Get("Allow"),
			response.Body.String(),
		)
	}
}

func TestPropertiesRequiresConfiguredBearerToken(t *testing.T) {
	handler, err := New(Config{
		ModelID:            "test-model",
		MaxTokens:          8,
		DefaultTemperature: 1,
		DefaultTopP:        1,
		APIKey:             "test-secret",
	}, &fakeGenerator{})
	if err != nil {
		t.Fatal(err)
	}
	unauthorized := httptest.NewRecorder()
	handler.ServeHTTP(
		unauthorized,
		httptest.NewRequest(http.MethodGet, "/props", nil),
	)
	if unauthorized.Code != http.StatusUnauthorized {
		t.Fatalf("unauthorized status = %d body=%s", unauthorized.Code, unauthorized.Body.String())
	}
	request := httptest.NewRequest(http.MethodGet, "/props", nil)
	request.Header.Set("Authorization", "Bearer test-secret")
	authorized := httptest.NewRecorder()
	handler.ServeHTTP(authorized, request)
	if authorized.Code != http.StatusOK {
		t.Fatalf("authorized status = %d body=%s", authorized.Code, authorized.Body.String())
	}
}

func TestBearerAuthentication(t *testing.T) {
	handler, err := New(Config{
		ModelID:            "test-model",
		MaxTokens:          8,
		DefaultTemperature: 1,
		DefaultTopP:        1,
		APIKey:             "test-secret",
	}, &fakeGenerator{})
	if err != nil {
		t.Fatal(err)
	}
	body := `{"prompt":"hi","max_tokens":1}`
	unauthorized := httptest.NewRecorder()
	handler.ServeHTTP(
		unauthorized,
		httptest.NewRequest(http.MethodPost, "/v1/completions", strings.NewReader(body)),
	)
	if unauthorized.Code != http.StatusUnauthorized ||
		unauthorized.Header().Get("WWW-Authenticate") != "Bearer" {
		t.Fatalf(
			"unauthorized response = %d, WWW-Authenticate %q",
			unauthorized.Code,
			unauthorized.Header().Get("WWW-Authenticate"),
		)
	}
	authorizedRequest := httptest.NewRequest(
		http.MethodPost,
		"/v1/completions",
		strings.NewReader(body),
	)
	authorizedRequest.Header.Set("Authorization", "Bearer test-secret")
	authorized := httptest.NewRecorder()
	handler.ServeHTTP(authorized, authorizedRequest)
	if authorized.Code != http.StatusOK {
		t.Fatalf("authorized status = %d body=%s", authorized.Code, authorized.Body.String())
	}
	health := httptest.NewRecorder()
	handler.ServeHTTP(health, httptest.NewRequest(http.MethodGet, "/health", nil))
	if health.Code != http.StatusOK {
		t.Fatalf("authenticated server health status = %d", health.Code)
	}
}

func TestCompletion(t *testing.T) {
	handler := newTestHandler(t, &fakeGenerator{})
	request := httptest.NewRequest(
		http.MethodPost,
		"/v1/completions",
		strings.NewReader(`{"model":"test-model","prompt":"hi","max_tokens":2,"temperature":0}`),
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
	if result.Usage != (completionUsage{PromptTokens: 2, CompletionTokens: 2, TotalTokens: 4}) {
		t.Fatalf("usage = %+v", result.Usage)
	}
	metricsRequest := httptest.NewRequest(http.MethodGet, "/metrics", nil)
	metricsResponse := httptest.NewRecorder()
	handler.ServeHTTP(metricsResponse, metricsRequest)
	if metricsResponse.Code != http.StatusOK {
		t.Fatalf("metrics status = %d", metricsResponse.Code)
	}
	for _, metric := range []string{
		"llamacpp2go_up 1",
		"llamacpp2go_http_requests_total 2",
		"llamacpp2go_http_requests_active 1",
		"llamacpp2go_generation_requests_total 1",
		"llamacpp2go_generation_errors_total 0",
		"llamacpp2go_generated_tokens_total 2",
		"llamacpp2go_cuda_memory_current_bytes 123",
		"llamacpp2go_cuda_memory_peak_bytes 456",
		"llamacpp2go_cuda_allocations_current 2",
		"llamacpp2go_cuda_custom_kernel_launches_total 12",
		"llamacpp2go_cuda_stream_synchronizations_total 3",
		"llamacpp2go_cuda_context_synchronizations_total 1",
		"llamacpp2go_cuda_host_to_device_bytes_total 1024",
		"llamacpp2go_cuda_device_to_host_bytes_total 2048",
		"llamacpp2go_cuda_device_to_device_bytes_total 3072",
		"llamacpp2go_cuda_device_memset_bytes_total 4096",
	} {
		if !strings.Contains(metricsResponse.Body.String(), metric) {
			t.Fatalf("metrics lack %q:\n%s", metric, metricsResponse.Body.String())
		}
	}
}

func TestOpenAICompletionExactMixedAndBatchedPrompts(t *testing.T) {
	for _, test := range []struct {
		name        string
		prompt      string
		n           int
		wantIDs     []tokenizer.TokenID
		wantChoices int
		wantUsage   completionUsage
	}{
		{
			name: "exact tokens", prompt: `[5,6]`, n: 1,
			wantIDs: []tokenizer.TokenID{5, 6}, wantChoices: 1,
			wantUsage: completionUsage{PromptTokens: 2, CompletionTokens: 1, TotalTokens: 3},
		},
		{
			name: "mixed tokens", prompt: `[5,"tail"]`, n: 1,
			wantIDs: []tokenizer.TokenID{5, 10}, wantChoices: 1,
			wantUsage: completionUsage{PromptTokens: 2, CompletionTokens: 1, TotalTokens: 3},
		},
		{
			name: "string batch", prompt: `["one","two"]`, n: 2,
			wantChoices: 4,
			wantUsage:   completionUsage{PromptTokens: 4, CompletionTokens: 4, TotalTokens: 8},
		},
		{
			name: "token batch", prompt: `[[5],[6]]`, n: 1,
			wantIDs: []tokenizer.TokenID{6}, wantChoices: 2,
			wantUsage: completionUsage{PromptTokens: 2, CompletionTokens: 2, TotalTokens: 4},
		},
	} {
		t.Run(test.name, func(t *testing.T) {
			generator := &fakeGenerator{}
			handler := newTestHandler(t, generator)
			request := httptest.NewRequest(
				http.MethodPost,
				"/v1/completions",
				strings.NewReader(fmt.Sprintf(
					`{"prompt":%s,"max_tokens":1,"n":%d,"temperature":0}`,
					test.prompt,
					test.n,
				)),
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
			if len(result.Choices) != test.wantChoices || result.Usage != test.wantUsage {
				t.Fatalf("result = %+v", result)
			}
			for index, choice := range result.Choices {
				if choice.Index != index || choice.Text != "A" {
					t.Fatalf("choice %d = %+v", index, choice)
				}
			}
			if test.wantIDs != nil {
				generator.mu.Lock()
				gotIDs := append([]tokenizer.TokenID(nil), generator.promptIDs...)
				generator.mu.Unlock()
				if !slices.Equal(gotIDs, test.wantIDs) {
					t.Fatalf("prompt IDs = %v, want %v", gotIDs, test.wantIDs)
				}
			}
		})
	}
}

func TestNativeCompletion(t *testing.T) {
	handler := newTestHandler(t, &fakeGenerator{})
	request := httptest.NewRequest(
		http.MethodPost,
		"/completion",
		strings.NewReader(
			`{"prompt":"hi","n_predict":2,"temperature":0,"return_tokens":true}`,
		),
	)
	response := httptest.NewRecorder()
	handler.ServeHTTP(response, request)
	if response.Code != http.StatusOK {
		t.Fatalf("status = %d body=%s", response.Code, response.Body.String())
	}
	var result nativeCompletionResponse
	if err := json.Unmarshal(response.Body.Bytes(), &result); err != nil {
		t.Fatal(err)
	}
	if result.Content != "AB" ||
		!slices.Equal(result.Tokens, []tokenizer.TokenID{30, 31}) ||
		result.StopType != "limit" ||
		result.StoppingWord != "" ||
		result.TokensPredicted != 2 ||
		result.TokensEvaluated != 2 ||
		result.Model != "test-model" ||
		result.Prompt != "hi" ||
		result.Timings.PredictedN != 2 {
		t.Fatalf("native completion = %+v", result)
	}
	if result.GenerationSettings["n_predict"] != float64(2) ||
		result.GenerationSettings["model"] != "test-model" {
		t.Fatalf("generation settings = %#v", result.GenerationSettings)
	}
}

func TestNativeCompletionProjectedInputs(t *testing.T) {
	generator := &fakeGenerator{}
	handler := newTestHandler(t, generator)
	request := httptest.NewRequest(
		http.MethodPost,
		"/completion",
		strings.NewReader(`{
  "prompt": [5, 6],
  "n_predict": 1,
  "projected_inputs": {
    "embedding_overrides": [{"token_index": 1, "embedding": [1, 2]}],
    "multi_axis_positions": [[0, 1], [0, 1], [0, 1], [0, 1]],
    "deepstack_embeddings": [{"shape": [2, 2], "data": [1, 2, 3, 4]}]
  }
}`),
	)
	response := httptest.NewRecorder()
	handler.ServeHTTP(response, request)
	if response.Code != http.StatusOK {
		t.Fatalf("status = %d body=%s", response.Code, response.Body.String())
	}
	if generator.projectedInputs == nil ||
		len(generator.projectedInputs.EmbeddingOverrides) != 1 ||
		generator.projectedInputs.MultiAxisPositions == nil ||
		len(generator.projectedInputs.DeepstackEmbeddings) != 1 {
		t.Fatalf("projected inputs = %+v", generator.projectedInputs)
	}
	var result nativeCompletionResponse
	if err := json.Unmarshal(response.Body.Bytes(), &result); err != nil {
		t.Fatal(err)
	}
	if result.GenerationSettings["projected_inputs"] != true {
		t.Fatalf("generation settings = %#v", result.GenerationSettings)
	}
}

func TestNativeCompletionImageProjectorMultimodalPrompt(t *testing.T) {
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
	body, err := json.Marshal(map[string]any{
		"prompt": map[string]any{
			"prompt_string":   "Look <__media__> now",
			"multimodal_data": []string{base64.StdEncoding.EncodeToString(encoded.Bytes())},
		},
		"n_predict": 1,
	})
	if err != nil {
		t.Fatal(err)
	}
	request := httptest.NewRequest(http.MethodPost, "/completion", bytes.NewReader(body))
	response := httptest.NewRecorder()
	handler.ServeHTTP(response, request)
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
	var result nativeCompletionResponse
	if err := json.Unmarshal(response.Body.Bytes(), &result); err != nil {
		t.Fatal(err)
	}
	if result.GenerationSettings["multimodal"] != true || result.Prompt != "Look <__media__> now" {
		t.Fatalf("result = %+v", result)
	}
}

func TestNativeCompletionMultipleImagesPreservesOrder(t *testing.T) {
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
	value := base64.StdEncoding.EncodeToString(encoded.Bytes())
	body, err := json.Marshal(map[string]any{
		"prompt": map[string]any{
			"prompt_string": "A<__media__>B<__media__>C", "multimodal_data": []string{value, value},
		},
		"n_predict": 1,
	})
	if err != nil {
		t.Fatal(err)
	}
	response := httptest.NewRecorder()
	handler.ServeHTTP(response, httptest.NewRequest(http.MethodPost, "/completion", bytes.NewReader(body)))
	if response.Code != http.StatusOK {
		t.Fatalf("status = %d body=%s", response.Code, response.Body.String())
	}
	if vision.images != 2 || !slices.Equal(vision.text, []string{"A", "B", "C"}) {
		t.Fatalf("projector sequence = images %d text %q", vision.images, vision.text)
	}
	if generator.projectedInputs == nil || len(generator.projectedInputs.EmbeddingOverrides) != 4 ||
		generator.projectedInputs.MultiAxisPositions == nil {
		t.Fatalf("projected = %+v", generator.projectedInputs)
	}
}

func TestNativeCompletionAudioProjectorMultimodalPrompt(t *testing.T) {
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
	wav := make([]byte, 48)
	copy(wav[0:4], "RIFF")
	binary.LittleEndian.PutUint32(wav[4:8], 40)
	copy(wav[8:12], "WAVE")
	copy(wav[12:16], "fmt ")
	binary.LittleEndian.PutUint32(wav[16:20], 16)
	binary.LittleEndian.PutUint16(wav[20:22], 1)
	binary.LittleEndian.PutUint16(wav[22:24], 1)
	binary.LittleEndian.PutUint32(wav[24:28], 16000)
	binary.LittleEndian.PutUint32(wav[28:32], 32000)
	binary.LittleEndian.PutUint16(wav[32:34], 2)
	binary.LittleEndian.PutUint16(wav[34:36], 16)
	copy(wav[36:40], "data")
	binary.LittleEndian.PutUint32(wav[40:44], 4)
	binary.LittleEndian.PutUint16(wav[44:46], uint16(16384))
	binary.LittleEndian.PutUint16(wav[46:48], uint16(49152))
	body, err := json.Marshal(map[string]any{
		"prompt": map[string]any{
			"prompt_string": "Hear <__media__> now",
			"multimodal_data": []string{
				"data:audio/wav;base64," + base64.StdEncoding.EncodeToString(wav),
			},
		},
		"n_predict": 1,
	})
	if err != nil {
		t.Fatal(err)
	}
	request := httptest.NewRequest(http.MethodPost, "/completion", bytes.NewReader(body))
	response := httptest.NewRecorder()
	handler.ServeHTTP(response, request)
	if response.Code != http.StatusOK {
		t.Fatalf("status = %d body=%s", response.Code, response.Body.String())
	}
	if audio.before != "Hear " || audio.after != " now" || !slices.Equal(audio.samples, []float32{0.5, -0.5}) {
		t.Fatalf("projector input = %q, %q, %v", audio.before, audio.after, audio.samples)
	}
	if !slices.Equal(generator.promptIDs, []tokenizer.TokenID{1, 4, 4, 3}) ||
		generator.projectedInputs == nil || len(generator.projectedInputs.EmbeddingOverrides) != 2 {
		t.Fatalf("prompt IDs = %v, projected = %+v", generator.promptIDs, generator.projectedInputs)
	}
}

func TestNativeCompletionMultimodalRequiresProjector(t *testing.T) {
	handler := newTestHandler(t, &fakeGenerator{})
	request := httptest.NewRequest(
		http.MethodPost,
		"/completion",
		strings.NewReader(`{"prompt":{"prompt_string":"<__media__>","multimodal_data":["AA=="]}}`),
	)
	response := httptest.NewRecorder()
	handler.ServeHTTP(response, request)
	if response.Code != http.StatusBadRequest || !strings.Contains(response.Body.String(), "no multimodal projector") {
		t.Fatalf("status = %d body=%s", response.Code, response.Body.String())
	}
}

func TestNativeCompletionProjectedInputsRejectsPromptCache(t *testing.T) {
	handler := newTestHandler(t, &fakeGenerator{})
	request := httptest.NewRequest(
		http.MethodPost,
		"/completion",
		strings.NewReader(`{
  "prompt": [5],
  "cache_prompt": true,
  "projected_inputs": {"embedding_overrides": [{"token_index": 0, "embedding": [1]}]}
}`),
	)
	response := httptest.NewRecorder()
	handler.ServeHTTP(response, request)
	if response.Code != http.StatusBadRequest || !strings.Contains(response.Body.String(), "cannot use cache_prompt") {
		t.Fatalf("status = %d body=%s", response.Code, response.Body.String())
	}
}

func TestNativeCompletionPromptCacheAccounting(t *testing.T) {
	generator := &fakeGenerator{promptCached: 1}
	handler := newTestHandler(t, generator)
	request := httptest.NewRequest(
		http.MethodPost,
		"/completion",
		strings.NewReader(
			`{"prompt":[4,5],"n_predict":1,"cache_prompt":true,"n_cache_reuse":1,`+
				`"temperature":0.25,"top_k":7,"top_p":0.75,"min_p":0.05,`+
				`"repeat_penalty":1.1,"stop":["done"]}`,
		),
	)
	response := httptest.NewRecorder()
	handler.ServeHTTP(response, request)
	if response.Code != http.StatusOK {
		t.Fatalf("status = %d body=%s", response.Code, response.Body.String())
	}
	var result nativeCompletionResponse
	if err := json.Unmarshal(response.Body.Bytes(), &result); err != nil {
		t.Fatal(err)
	}
	if !generator.cachePrompt ||
		generator.minCacheReuse != 1 ||
		result.TokensCached != 1 ||
		result.Timings.CacheN != 1 ||
		result.Timings.PromptN != 1 ||
		result.GenerationSettings["cache_prompt"] != true {
		t.Fatalf("cache result=%+v generator=%+v", result, generator)
	}
	slotsResponse := httptest.NewRecorder()
	handler.ServeHTTP(
		slotsResponse,
		httptest.NewRequest(http.MethodGet, "/slots", nil),
	)
	if slotsResponse.Code != http.StatusOK {
		t.Fatalf("slots status = %d body=%s", slotsResponse.Code, slotsResponse.Body.String())
	}
	var slots []slotStatusItem
	if err := json.Unmarshal(slotsResponse.Body.Bytes(), &slots); err != nil {
		t.Fatal(err)
	}
	if len(slots) != 1 ||
		slots[0].NPromptTokens != 2 ||
		slots[0].NPromptTokensProcessed != 1 ||
		slots[0].NPromptTokensCache != 1 ||
		slots[0].Generated != "A" ||
		slots[0].Params == nil ||
		slots[0].Params.MaxTokens != 1 ||
		slots[0].Params.NPredict != 1 ||
		slots[0].Params.Temperature != 0.25 ||
		slots[0].Params.TopK != 7 ||
		slots[0].Params.TopP != 0.75 ||
		slots[0].Params.MinP != 0.05 ||
		slots[0].Params.RepeatPenalty != 1.1 ||
		!slices.Equal(slots[0].Params.Stop, []string{"done"}) ||
		!slots[0].Params.CachePrompt ||
		slots[0].Params.NCacheReuse != 1 ||
		slots[0].NextToken == nil ||
		slots[0].NextToken.HasNextToken ||
		slots[0].NextToken.HasNewLine ||
		slots[0].NextToken.NRemain != 0 ||
		slots[0].NextToken.NDecoded != 1 ||
		slots[0].Timings == nil ||
		slots[0].Timings.CacheN != 1 ||
		slots[0].Timings.PromptN != 1 ||
		slots[0].Timings.PredictedN != 1 {
		t.Fatalf("slots = %+v", slots)
	}
}

func TestNativeCompletionPredictionTimeLimitStopsOnNewline(t *testing.T) {
	handler := newTestHandler(t, &fakeGenerator{
		pieces:     []string{"A", "\n", "C"},
		tokenDelay: 5 * time.Millisecond,
	})
	request := httptest.NewRequest(
		http.MethodPost,
		"/completion",
		strings.NewReader(
			`{"prompt":"hi","n_predict":3,"t_max_predict_ms":1}`,
		),
	)
	response := httptest.NewRecorder()
	handler.ServeHTTP(response, request)
	if response.Code != http.StatusOK {
		t.Fatalf("status = %d body=%s", response.Code, response.Body.String())
	}
	var result nativeCompletionResponse
	if err := json.Unmarshal(response.Body.Bytes(), &result); err != nil {
		t.Fatal(err)
	}
	if result.Content != "A\n" ||
		result.TokensPredicted != 2 ||
		result.StopType != "limit" ||
		!result.HasNewLine ||
		result.GenerationSettings["t_max_predict_ms"] != float64(1) {
		t.Fatalf("result = %+v", result)
	}
}

func TestNativeCompletionIndentationLimitTrimsOffendingSuffix(t *testing.T) {
	handler := newTestHandler(t, &fakeGenerator{
		pieces: []string{"x", "\n", " y", "z"},
	})
	request := httptest.NewRequest(
		http.MethodPost,
		"/completion",
		strings.NewReader(`{"prompt":"hi","n_predict":4,"n_indent":2}`),
	)
	response := httptest.NewRecorder()
	handler.ServeHTTP(response, request)
	if response.Code != http.StatusOK {
		t.Fatalf("status = %d body=%s", response.Code, response.Body.String())
	}
	var result nativeCompletionResponse
	if err := json.Unmarshal(response.Body.Bytes(), &result); err != nil {
		t.Fatal(err)
	}
	if result.Content != "x\n " ||
		result.TokensPredicted != 3 ||
		result.StopType != "limit" ||
		!result.HasNewLine ||
		result.GenerationSettings["n_indent"] != float64(2) {
		t.Fatalf("result = %+v", result)
	}
}

func TestNativeCompletionPreSamplingProbabilities(t *testing.T) {
	handler := newTestHandler(t, &fakeGenerator{})
	request := httptest.NewRequest(
		http.MethodPost,
		"/completion",
		strings.NewReader(`{"prompt":"hi","n_predict":2,"n_probs":3}`),
	)
	response := httptest.NewRecorder()
	handler.ServeHTTP(response, request)
	if response.Code != http.StatusOK {
		t.Fatalf("status = %d body=%s", response.Code, response.Body.String())
	}
	var result nativeCompletionResponse
	if err := json.Unmarshal(response.Body.Bytes(), &result); err != nil {
		t.Fatal(err)
	}
	if len(result.CompletionProbabilities) != 2 ||
		result.GenerationSettings["n_probs"] != float64(3) {
		t.Fatalf("result = %+v", result)
	}
	for index, probability := range result.CompletionProbabilities {
		wantID := tokenizer.TokenID(30 + index)
		if probability.ID != wantID ||
			probability.Token != fmt.Sprintf("<%d>", wantID) ||
			!slices.Equal(
				probability.Bytes,
				[]int{60, 51, 48 + index, 62},
			) ||
			probability.LogProb == nil ||
			*probability.LogProb >= 0 ||
			len(probability.TopLogProbs) != 3 ||
			probability.TopLogProbs[0].ID != wantID ||
			probability.TopLogProbs[0].LogProb != *probability.LogProb {
			t.Fatalf("probability %d = %+v", index, probability)
		}
	}
}

func TestNativeCompletionPostSamplingProbabilities(t *testing.T) {
	handler := newTestHandler(t, &fakeGenerator{})
	request := httptest.NewRequest(
		http.MethodPost,
		"/completion",
		strings.NewReader(
			`{"prompt":"hi","n_predict":2,"n_probs":2,"post_sampling_probs":true}`,
		),
	)
	response := httptest.NewRecorder()
	handler.ServeHTTP(response, request)
	if response.Code != http.StatusOK {
		t.Fatalf("status = %d body=%s", response.Code, response.Body.String())
	}
	var result nativeCompletionResponse
	if err := json.Unmarshal(response.Body.Bytes(), &result); err != nil {
		t.Fatal(err)
	}
	if len(result.CompletionProbabilities) != 2 ||
		result.GenerationSettings["post_sampling_probs"] != true {
		t.Fatalf("result = %+v", result)
	}
	for index, probability := range result.CompletionProbabilities {
		wantID := tokenizer.TokenID(30 + index)
		if probability.ID != wantID ||
			probability.LogProb != nil ||
			probability.Prob == nil ||
			*probability.Prob != 0.8 ||
			len(probability.TopLogProbs) != 0 ||
			len(probability.TopProbs) != 2 ||
			probability.TopProbs[0].ID != wantID ||
			probability.TopProbs[0].Prob != 0.8 ||
			probability.TopProbs[1].ID != tokenizer.TokenID(40+index) ||
			probability.TopProbs[1].Prob != 0.2 {
			t.Fatalf("probability %d = %+v", index, probability)
		}
	}
}

func TestNativeCompletionStopAndMultipleChoices(t *testing.T) {
	handler := newTestHandler(t, &fakeGenerator{})
	request := httptest.NewRequest(
		http.MethodPost,
		"/completions",
		strings.NewReader(`{"prompt":"hi","n_predict":2,"n_cmpl":2,"stop":["AB"]}`),
	)
	response := httptest.NewRecorder()
	handler.ServeHTTP(response, request)
	if response.Code != http.StatusOK {
		t.Fatalf("status = %d body=%s", response.Code, response.Body.String())
	}
	var results []nativeCompletionResponse
	if err := json.Unmarshal(response.Body.Bytes(), &results); err != nil {
		t.Fatal(err)
	}
	if len(results) != 2 {
		t.Fatalf("result count = %d", len(results))
	}
	for index, result := range results {
		if result.Index != index ||
			result.Content != "" ||
			result.StopType != "word" ||
			result.StoppingWord != "AB" ||
			result.TokensPredicted != 2 ||
			len(result.Tokens) != 0 {
			t.Fatalf("result %d = %+v", index, result)
		}
	}
}

func TestNativeCompletionExactAndMixedTokenPrompts(t *testing.T) {
	generator := &fakeGenerator{}
	handler := newTestHandler(t, generator)
	cases := []struct {
		prompt string
		want   []tokenizer.TokenID
	}{
		{prompt: `[4,5]`, want: []tokenizer.TokenID{4, 5}},
		{prompt: `[4,"text",5]`, want: []tokenizer.TokenID{4, 10, 5}},
		{prompt: `["text",4]`, want: []tokenizer.TokenID{1, 10, 4}},
	}
	for _, test := range cases {
		request := httptest.NewRequest(
			http.MethodPost,
			"/completion",
			strings.NewReader(`{"prompt":`+test.prompt+`,"n_predict":1}`),
		)
		response := httptest.NewRecorder()
		handler.ServeHTTP(response, request)
		if response.Code != http.StatusOK {
			t.Fatalf("prompt %s status = %d body=%s", test.prompt, response.Code, response.Body.String())
		}
		var result nativeCompletionResponse
		if err := json.Unmarshal(response.Body.Bytes(), &result); err != nil {
			t.Fatal(err)
		}
		generator.mu.Lock()
		got := append([]tokenizer.TokenID(nil), generator.promptIDs...)
		generator.mu.Unlock()
		if !slices.Equal(got, test.want) ||
			result.TokensEvaluated != len(test.want) {
			t.Fatalf(
				"prompt %s IDs=%v evaluated=%d, want %v",
				test.prompt,
				got,
				result.TokensEvaluated,
				test.want,
			)
		}
		wantPrompt, _ := generator.DetokenizePromptTokens(test.want)
		if result.Prompt != wantPrompt {
			t.Fatalf("prompt %s response prompt = %#v, want %q", test.prompt, result.Prompt, wantPrompt)
		}
	}
}

func TestNativeCompletionPromptBatchTimesChoiceCount(t *testing.T) {
	handler := newTestHandler(t, &fakeGenerator{})
	request := httptest.NewRequest(
		http.MethodPost,
		"/completion",
		strings.NewReader(
			`{"prompt":["one",[4,5]],"n_predict":2,"n_cmpl":2,"stop":["AB"]}`,
		),
	)
	response := httptest.NewRecorder()
	handler.ServeHTTP(response, request)
	if response.Code != http.StatusOK {
		t.Fatalf("status = %d body=%s", response.Code, response.Body.String())
	}
	var results []nativeCompletionResponse
	if err := json.Unmarshal(response.Body.Bytes(), &results); err != nil {
		t.Fatal(err)
	}
	if len(results) != 4 {
		t.Fatalf("result count = %d", len(results))
	}
	wantPrompts := []string{"one", "one", "[4][5]", "[4][5]"}
	for index, result := range results {
		if result.Index != index ||
			result.Prompt != wantPrompts[index] ||
			result.Content != "" ||
			result.StopType != "word" ||
			result.StoppingWord != "AB" {
			t.Fatalf("result %d = %+v", index, result)
		}
	}
}

func TestNativeCompletionStreamingPromptBatch(t *testing.T) {
	handler := newTestHandler(t, &fakeGenerator{})
	request := httptest.NewRequest(
		http.MethodPost,
		"/completion",
		strings.NewReader(`{"prompt":["one","two"],"n_predict":1,"n_cmpl":2,"stream":true}`),
	)
	response := httptest.NewRecorder()
	handler.ServeHTTP(response, request)
	if response.Code != http.StatusOK {
		t.Fatalf("status = %d body=%s", response.Code, response.Body.String())
	}
	body := response.Body.String()
	if strings.Count(body, `"stop":true`) != 4 {
		t.Fatalf("final event count is not 4:\n%s", body)
	}
	for index := range 4 {
		if !strings.Contains(body, fmt.Sprintf(`"index":%d`, index)) {
			t.Fatalf("stream lacks index %d:\n%s", index, body)
		}
	}
}

func TestNativeCompletionResponseFields(t *testing.T) {
	handler := newTestHandler(t, &fakeGenerator{})
	request := httptest.NewRequest(
		http.MethodPost,
		"/completion",
		strings.NewReader(
			`{"prompt":"hi","n_predict":2,"response_fields":`+
				`["content","generation_settings/n_predict","missing","content"]}`,
		),
	)
	response := httptest.NewRecorder()
	handler.ServeHTTP(response, request)
	if response.Code != http.StatusOK {
		t.Fatalf("status = %d body=%s", response.Code, response.Body.String())
	}
	var result map[string]any
	if err := json.Unmarshal(response.Body.Bytes(), &result); err != nil {
		t.Fatal(err)
	}
	if len(result) != 2 ||
		result["content"] != "AB" ||
		result["generation_settings/n_predict"] != float64(2) {
		t.Fatalf("projected response = %#v", result)
	}
}

func TestNativeCompletionResponseFieldsBatchAndStream(t *testing.T) {
	handler := newTestHandler(t, &fakeGenerator{})
	batch := httptest.NewRecorder()
	handler.ServeHTTP(
		batch,
		httptest.NewRequest(
			http.MethodPost,
			"/completion",
			strings.NewReader(
				`{"prompt":["one","two"],"n_predict":1,`+
					`"response_fields":["index","prompt"]}`,
			),
		),
	)
	if batch.Code != http.StatusOK {
		t.Fatalf("batch status = %d body=%s", batch.Code, batch.Body.String())
	}
	var projected []map[string]any
	if err := json.Unmarshal(batch.Body.Bytes(), &projected); err != nil {
		t.Fatal(err)
	}
	if len(projected) != 2 ||
		projected[0]["index"] != float64(0) ||
		projected[0]["prompt"] != "one" ||
		projected[1]["index"] != float64(1) ||
		projected[1]["prompt"] != "two" {
		t.Fatalf("projected batch = %#v", projected)
	}

	stream := httptest.NewRecorder()
	handler.ServeHTTP(
		stream,
		httptest.NewRequest(
			http.MethodPost,
			"/completion",
			strings.NewReader(
				`{"prompt":"hi","n_predict":1,"stream":true,`+
					`"response_fields":["stop","generation_settings/n_predict"]}`,
			),
		),
	)
	if stream.Code != http.StatusOK {
		t.Fatalf("stream status = %d body=%s", stream.Code, stream.Body.String())
	}
	lines := strings.Split(strings.TrimSpace(stream.Body.String()), "\n")
	var events []map[string]any
	for _, line := range lines {
		if !strings.HasPrefix(line, "data: ") {
			continue
		}
		var event map[string]any
		if err := json.Unmarshal([]byte(strings.TrimPrefix(line, "data: ")), &event); err != nil {
			t.Fatal(err)
		}
		events = append(events, event)
	}
	if len(events) != 2 ||
		events[0]["content"] != "A" ||
		events[1]["stop"] != true ||
		events[1]["generation_settings/n_predict"] != float64(1) ||
		len(events[1]) != 2 {
		t.Fatalf("stream events = %#v", events)
	}
}

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
			`<tool_call><function=weather><parameter=city>Paris</parameter></function></tool_call>`,
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
					`"tool_choice":"required","max_tokens":1,"stream":true}`,
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
		`"type":"function","function":{"name":"weather","arguments":"{\"city\":\"Paris\"}"}`,
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
	wav := make([]byte, 48)
	copy(wav[0:4], "RIFF")
	binary.LittleEndian.PutUint32(wav[4:8], 40)
	copy(wav[8:12], "WAVE")
	copy(wav[12:16], "fmt ")
	binary.LittleEndian.PutUint32(wav[16:20], 16)
	binary.LittleEndian.PutUint16(wav[20:22], 1)
	binary.LittleEndian.PutUint16(wav[22:24], 1)
	binary.LittleEndian.PutUint32(wav[24:28], 16000)
	binary.LittleEndian.PutUint32(wav[28:32], 32000)
	binary.LittleEndian.PutUint16(wav[32:34], 2)
	binary.LittleEndian.PutUint16(wav[34:36], 16)
	copy(wav[36:40], "data")
	binary.LittleEndian.PutUint32(wav[40:44], 4)
	binary.LittleEndian.PutUint16(wav[44:46], uint16(16384))
	binary.LittleEndian.PutUint16(wav[46:48], uint16(49152))
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
		{"/v1/chat/completions", `{"messages":[{"role":"system","content":"x"},{"role":"user","content":[` + imagePart + `]}]}`, "requires one user message"},
		{"/v1/chat/completions", `{"messages":[{"role":"user","content":[{"type":"image_url","image_url":{"url":"https://example.com/x.png"}}]}]}`, "base64 image data URI"},
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
			result.Usage.InputTokens != 2 ||
			result.Usage.OutputTokens != 1 ||
			result.Usage.TotalTokens != 3 {
			t.Fatalf("%s response = %+v body=%s", path, result, response.Body.String())
		}
	}
}

func TestResponsesImageContentPartProjectsPrompt(t *testing.T) {
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
	for _, stream := range []bool{false, true} {
		body, err := json.Marshal(map[string]any{
			"input": []any{map[string]any{
				"role": "user",
				"content": []any{
					map[string]any{"type": "input_text", "text": "Look "},
					map[string]any{"type": "input_image", "image_url": dataURI, "detail": "auto"},
					map[string]any{"type": "input_text", "text": " now"},
				},
			}},
			"max_output_tokens": 1,
			"stream":            stream,
		})
		if err != nil {
			t.Fatal(err)
		}
		response := httptest.NewRecorder()
		handler.ServeHTTP(
			response,
			httptest.NewRequest(http.MethodPost, "/v1/responses", bytes.NewReader(body)),
		)
		if response.Code != http.StatusOK {
			t.Fatalf("stream %v status = %d body=%s", stream, response.Code, response.Body.String())
		}
		if stream && !strings.Contains(response.Body.String(), "event: response.completed") {
			t.Fatalf("stream body = %s", response.Body.String())
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
}

func TestResponsesMultipleImagesPreservesContentOrder(t *testing.T) {
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
	var encoded bytes.Buffer
	if err := png.Encode(&encoded, input); err != nil {
		t.Fatal(err)
	}
	dataURI := "data:image/png;base64," + base64.StdEncoding.EncodeToString(encoded.Bytes())
	body, err := json.Marshal(map[string]any{
		"input": []any{map[string]any{
			"role": "user", "content": []any{
				map[string]any{"type": "input_text", "text": "A"},
				map[string]any{"type": "input_image", "image_url": dataURI},
				map[string]any{"type": "input_text", "text": "B"},
				map[string]any{"type": "input_image", "image_url": dataURI},
				map[string]any{"type": "input_text", "text": "C"},
			},
		}}, "max_output_tokens": 1,
	})
	if err != nil {
		t.Fatal(err)
	}
	response := httptest.NewRecorder()
	handler.ServeHTTP(response, httptest.NewRequest(http.MethodPost, "/v1/responses", bytes.NewReader(body)))
	if response.Code != http.StatusOK {
		t.Fatalf("status = %d body=%s", response.Code, response.Body.String())
	}
	if vision.images != 2 || !slices.Equal(vision.text, []string{"A", "B", "C"}) {
		t.Fatalf("projector sequence = images %d text %q", vision.images, vision.text)
	}
	if !slices.Equal(generator.promptIDs, []tokenizer.TokenID{1, 2, 2, 3, 4, 4, 5}) ||
		generator.projectedInputs == nil || len(generator.projectedInputs.EmbeddingOverrides) != 4 {
		t.Fatalf("prompt IDs = %v, projected = %+v", generator.promptIDs, generator.projectedInputs)
	}
}

func TestResponsesMultimodalValidation(t *testing.T) {
	vision := &fakeQwen3VLProjector{}
	handler, err := New(Config{
		ModelID: "test-model", MaxTokens: 8,
		DefaultTemperature: 1, DefaultTopP: 1,
		ImageProjector: vision,
	}, &fakeGenerator{})
	if err != nil {
		t.Fatal(err)
	}
	imagePart := `{"type":"input_image","image_url":"data:image/png;base64,AA=="}`
	for _, test := range []struct {
		path string
		body string
		want string
	}{
		{"/v1/responses", `{"instructions":"brief","input":[{"role":"user","content":[` + imagePart + `]}]}`, "requires one user message"},
		{"/v1/responses", `{"input":[{"role":"user","content":[` + imagePart + `]}],"tools":[{"type":"function","name":"x","parameters":{}}]}`, "cannot use tools"},
		{"/v1/responses", `{"input":[{"role":"user","content":[{"type":"input_image","image_url":"https://example.com/x.png"}]}]}`, "base64 image data URI"},
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

func TestBufferedResponsesFunctionCall(t *testing.T) {
	generator := &fakeGenerator{
		pieces: []string{
			`<tool_call><function=weather><parameter=city>Paris</parameter></function></tool_call>`,
		},
	}
	handler := newTestHandler(t, generator)
	response := httptest.NewRecorder()
	handler.ServeHTTP(
		response,
		httptest.NewRequest(
			http.MethodPost,
			"/v1/responses",
			strings.NewReader(
				`{"input":"weather?","max_output_tokens":1,`+
					`"tool_choice":{"type":"function","name":"weather"},`+
					`"tools":[{"type":"function","name":"weather",`+
					`"description":"forecast","parameters":{"type":"object",`+
					`"properties":{"city":{"type":"string"}}},"strict":true}]}`,
			),
		),
	)
	if response.Code != http.StatusOK {
		t.Fatalf("status = %d body=%s", response.Code, response.Body.String())
	}
	var result responsesResponse
	if err := json.Unmarshal(response.Body.Bytes(), &result); err != nil {
		t.Fatal(err)
	}
	if len(result.Output) != 1 ||
		result.Output[0].Type != "function_call" ||
		result.Output[0].ID == "" ||
		result.Output[0].CallID == "" ||
		result.Output[0].Name != "weather" ||
		result.Output[0].Arguments != `{"city":"Paris"}` {
		t.Fatalf("function response = %+v body=%s", result, response.Body.String())
	}
	generator.mu.Lock()
	defer generator.mu.Unlock()
	if len(generator.chatOptions.Tools) != 1 ||
		generator.chatOptions.Tools[0].Function.Name != "weather" ||
		generator.grammarParallel {
		t.Fatalf("formatted tools = %+v", generator.chatOptions.Tools)
	}
}

func TestStreamingResponsesLifecycle(t *testing.T) {
	handler := newTestHandler(t, &fakeGenerator{})
	response := httptest.NewRecorder()
	handler.ServeHTTP(
		response,
		httptest.NewRequest(
			http.MethodPost,
			"/v1/responses",
			strings.NewReader(`{"input":"hello","max_output_tokens":2,"stream":true}`),
		),
	)
	if response.Code != http.StatusOK ||
		response.Header().Get("Content-Type") != "text/event-stream" {
		t.Fatalf("status/headers = %d %#v body=%s", response.Code, response.Header(), response.Body.String())
	}
	body := response.Body.String()
	events := []string{
		"event: response.created",
		"event: response.in_progress",
		"event: response.output_item.added",
		"event: response.content_part.added",
		`"delta":"A"`,
		`"delta":"B"`,
		"event: response.output_text.done",
		"event: response.content_part.done",
		"event: response.output_item.done",
		"event: response.completed",
	}
	previous := -1
	for _, event := range events {
		index := strings.Index(body, event)
		if index < 0 || index <= previous {
			t.Fatalf("event %q index=%d after=%d body=%s", event, index, previous, body)
		}
		previous = index
	}
	if strings.Contains(body, "[DONE]") ||
		!strings.Contains(body, `"output_tokens":2`) ||
		!strings.Contains(body, `"total_tokens":4`) {
		t.Fatalf("stream body = %s", body)
	}
}

func TestStreamingResponsesFunctionCallLifecycle(t *testing.T) {
	generator := &fakeGenerator{
		pieces: []string{
			`<tool_call><function=weather><parameter=city>Paris</parameter></function></tool_call>`,
		},
	}
	handler := newTestHandler(t, generator)
	response := httptest.NewRecorder()
	handler.ServeHTTP(
		response,
		httptest.NewRequest(
			http.MethodPost,
			"/v1/responses",
			strings.NewReader(
				`{"input":"weather?","max_output_tokens":1,"stream":true,`+
					`"tool_choice":"required","parallel_tool_calls":false,`+
					`"tools":[{"type":"function",`+
					`"name":"weather","parameters":{"type":"object"}}]}`,
			),
		),
	)
	if response.Code != http.StatusOK {
		t.Fatalf("status = %d body=%s", response.Code, response.Body.String())
	}
	body := response.Body.String()
	events := []string{
		"event: response.created",
		"event: response.in_progress",
		"event: response.output_item.added",
		`"type":"function_call"`,
		"event: response.function_call_arguments.delta",
		`"delta":"{\"city\":\"Paris\"}"`,
		"event: response.function_call_arguments.done",
		"event: response.output_item.done",
		"event: response.completed",
	}
	previous := -1
	for _, event := range events {
		index := strings.Index(body, event)
		if index < 0 || index <= previous {
			t.Fatalf(
				"event %q index=%d after=%d body=%s",
				event,
				index,
				previous,
				body,
			)
		}
		previous = index
	}
	if strings.Contains(body, "<tool_call>") {
		t.Fatalf("function stream leaked template syntax:\n%s", body)
	}
	generator.mu.Lock()
	defer generator.mu.Unlock()
	if generator.grammarParallel {
		t.Fatal("parallel_tool_calls:false reached the grammar as parallel")
	}
}

func TestResponsesValidationAndMethod(t *testing.T) {
	handler := newTestHandler(t, &fakeGenerator{})
	for _, body := range []string{
		`{}`,
		`{"input":"hello","max_output_tokens":9}`,
		`{"input":"hello","previous_response_id":"resp_old"}`,
		`{"model":"missing","input":"hello"}`,
		`{"input":"hello","unknown":true}`,
		`{"input":"hello","tools":[{"type":"custom","name":"shell"}]}`,
		`{"input":"hello","tool_choice":{"type":"function","name":"missing"},` +
			`"tools":[{"type":"function","name":"weather","parameters":{"type":"object"}}]}`,
	} {
		response := httptest.NewRecorder()
		handler.ServeHTTP(
			response,
			httptest.NewRequest(http.MethodPost, "/v1/responses", strings.NewReader(body)),
		)
		if response.Code < 400 {
			t.Fatalf("body %s status = %d response=%s", body, response.Code, response.Body.String())
		}
	}
	get := httptest.NewRecorder()
	handler.ServeHTTP(get, httptest.NewRequest(http.MethodGet, "/responses", nil))
	if get.Code != http.StatusMethodNotAllowed || get.Header().Get("Allow") != http.MethodPost {
		t.Fatalf("GET status = %d Allow=%q", get.Code, get.Header().Get("Allow"))
	}
}

func TestResponsesInputTokensAuthentication(t *testing.T) {
	handler, err := New(Config{
		ModelID:            "test-model",
		MaxTokens:          8,
		DefaultTemperature: 1,
		DefaultTopP:        1,
		APIKey:             "test-secret",
	}, &fakeGenerator{})
	if err != nil {
		t.Fatal(err)
	}
	for _, path := range []string{
		"/responses",
		"/v1/responses",
		"/responses/input_tokens",
		"/v1/responses/input_tokens",
		"/v1/messages",
		"/v1/messages/count_tokens",
		"/lora-adapters",
	} {
		response := httptest.NewRecorder()
		handler.ServeHTTP(
			response,
			httptest.NewRequest(http.MethodPost, path, strings.NewReader(`{"input":"hello"}`)),
		)
		if response.Code != http.StatusUnauthorized {
			t.Fatalf("%s status = %d body=%s", path, response.Code, response.Body.String())
		}
	}
	health := httptest.NewRecorder()
	handler.ServeHTTP(health, httptest.NewRequest(http.MethodGet, "/v1/health", nil))
	if health.Code != http.StatusOK {
		t.Fatalf("public /v1/health status = %d", health.Code)
	}
}

func TestAnthropicInputTokensTextForms(t *testing.T) {
	handler := newTestHandler(t, &fakeGenerator{})
	for _, body := range []string{
		`{"model":"test-model","messages":[{"role":"user","content":"hello"}]}`,
		`{"system":"brief","messages":[{"role":"user","content":[{"type":"text","text":"hello"},{"type":"text","text":" world"}]}]}`,
		`{"system":[{"type":"text","text":"brief"}],"messages":[{"role":"user","content":"hello"},{"role":"assistant","content":"hi"},{"role":"user","content":"again"}],"tools":[]}`,
	} {
		response := httptest.NewRecorder()
		handler.ServeHTTP(
			response,
			httptest.NewRequest(
				http.MethodPost,
				"/v1/messages/count_tokens",
				strings.NewReader(body),
			),
		)
		if response.Code != http.StatusOK {
			t.Fatalf("body %s status = %d response=%s", body, response.Code, response.Body.String())
		}
		var result map[string]int
		if err := json.Unmarshal(response.Body.Bytes(), &result); err != nil {
			t.Fatal(err)
		}
		if result["input_tokens"] != 3 {
			t.Fatalf("body %s result = %+v", body, result)
		}
	}
}

func TestAnthropicInputTokensIncludeToolsAndResults(t *testing.T) {
	generator := &fakeGenerator{}
	handler := newTestHandler(t, generator)
	response := httptest.NewRecorder()
	handler.ServeHTTP(
		response,
		httptest.NewRequest(
			http.MethodPost,
			"/v1/messages/count_tokens",
			strings.NewReader(
				`{"tools":[{"name":"weather","input_schema":{"type":"object"}}],`+
					`"messages":[`+
					`{"role":"assistant","content":[{"type":"tool_use","id":"toolu_1","name":"weather","input":{"city":"Paris"}}]},`+
					`{"role":"user","content":[{"type":"tool_result","tool_use_id":"toolu_1","content":"Sunny"}]}`+
					`]}`,
			),
		),
	)
	if response.Code != http.StatusOK {
		t.Fatalf("status = %d body=%s", response.Code, response.Body.String())
	}
	var result map[string]int
	if err := json.Unmarshal(response.Body.Bytes(), &result); err != nil {
		t.Fatal(err)
	}
	if result["input_tokens"] != 3 {
		t.Fatalf("result = %+v", result)
	}
	generator.mu.Lock()
	defer generator.mu.Unlock()
	if len(generator.chatOptions.Tools) != 1 ||
		len(generator.chatMessages) != 2 ||
		len(generator.chatMessages[0].ToolCalls) != 1 ||
		generator.chatMessages[1].Role != "tool" {
		t.Fatalf(
			"formatted options/messages = %+v / %+v",
			generator.chatOptions,
			generator.chatMessages,
		)
	}
}

func TestBufferedAnthropicMessages(t *testing.T) {
	handler := newTestHandler(t, &fakeGenerator{})
	response := httptest.NewRecorder()
	handler.ServeHTTP(
		response,
		httptest.NewRequest(
			http.MethodPost,
			"/v1/messages",
			strings.NewReader(
				`{"model":"test-model","system":"brief","max_tokens":2,"messages":[{"role":"user","content":[{"type":"text","text":"hello"}]}]}`,
			),
		),
	)
	if response.Code != http.StatusOK {
		t.Fatalf("status = %d body=%s", response.Code, response.Body.String())
	}
	var result anthropicResponse
	if err := json.Unmarshal(response.Body.Bytes(), &result); err != nil {
		t.Fatal(err)
	}
	if !strings.HasPrefix(result.ID, "msg_") ||
		result.Type != "message" ||
		result.Role != "assistant" ||
		len(result.Content) != 1 ||
		result.Content[0].Type != "text" ||
		result.Content[0].Text != "AB" ||
		result.StopReason != "max_tokens" ||
		result.StopSequence != nil ||
		result.Usage.InputTokens != 2 ||
		result.Usage.OutputTokens != 2 {
		t.Fatalf("response = %+v body=%s", result, response.Body.String())
	}
}

func TestBufferedAnthropicToolUse(t *testing.T) {
	generator := &fakeGenerator{
		pieces: []string{
			`<tool_call><function=weather><parameter=city>Paris</parameter></function></tool_call>`,
		},
	}
	handler := newTestHandler(t, generator)
	response := httptest.NewRecorder()
	handler.ServeHTTP(
		response,
		httptest.NewRequest(
			http.MethodPost,
			"/v1/messages",
			strings.NewReader(
				`{"max_tokens":1,"tool_choice":{"type":"any"},`+
					`"tools":[{"name":"weather","description":"forecast",`+
					`"input_schema":{"type":"object","properties":{"city":{"type":"string"}}}}],`+
					`"messages":[{"role":"user","content":"weather?"}]}`,
			),
		),
	)
	if response.Code != http.StatusOK {
		t.Fatalf("status = %d body=%s", response.Code, response.Body.String())
	}
	var result anthropicResponse
	if err := json.Unmarshal(response.Body.Bytes(), &result); err != nil {
		t.Fatal(err)
	}
	if result.StopReason != "tool_use" ||
		len(result.Content) != 1 ||
		result.Content[0].Type != "tool_use" ||
		result.Content[0].ID == "" ||
		result.Content[0].Name != "weather" ||
		string(result.Content[0].Input) != `{"city":"Paris"}` {
		t.Fatalf("tool response = %+v body=%s", result, response.Body.String())
	}
	generator.mu.Lock()
	defer generator.mu.Unlock()
	if len(generator.chatOptions.Tools) != 1 ||
		generator.chatOptions.Tools[0].Function.Name != "weather" {
		t.Fatalf("formatted tools = %+v", generator.chatOptions.Tools)
	}
}

func TestAnthropicMessagesStopAndValidation(t *testing.T) {
	handler := newTestHandler(t, &fakeGenerator{})
	stopped := httptest.NewRecorder()
	handler.ServeHTTP(
		stopped,
		httptest.NewRequest(
			http.MethodPost,
			"/v1/messages",
			strings.NewReader(
				`{"max_tokens":2,"stop_sequences":["AB"],"messages":[{"role":"user","content":"hello"}]}`,
			),
		),
	)
	if stopped.Code != http.StatusOK {
		t.Fatalf("stop status = %d body=%s", stopped.Code, stopped.Body.String())
	}
	var result anthropicResponse
	if err := json.Unmarshal(stopped.Body.Bytes(), &result); err != nil {
		t.Fatal(err)
	}
	if result.StopReason != "end_turn" ||
		result.StopSequence == nil ||
		*result.StopSequence != "AB" ||
		len(result.Content) != 0 {
		t.Fatalf("stopped response = %+v", result)
	}
	for _, body := range []string{
		`{"messages":[{"role":"user","content":"hello"}]}`,
		`{"max_tokens":9,"messages":[{"role":"user","content":"hello"}]}`,
		`{"max_tokens":1,"tools":[{"name":"x"}],"messages":[{"role":"user","content":"hello"}]}`,
	} {
		response := httptest.NewRecorder()
		handler.ServeHTTP(
			response,
			httptest.NewRequest(http.MethodPost, "/v1/messages", strings.NewReader(body)),
		)
		if response.Code < 400 {
			t.Fatalf("body %s status = %d response=%s", body, response.Code, response.Body.String())
		}
	}
}

func TestStreamingAnthropicMessagesLifecycle(t *testing.T) {
	handler := newTestHandler(t, &fakeGenerator{})
	response := httptest.NewRecorder()
	handler.ServeHTTP(
		response,
		httptest.NewRequest(
			http.MethodPost,
			"/v1/messages",
			strings.NewReader(
				`{"max_tokens":2,"stream":true,"messages":[{"role":"user","content":"hello"}]}`,
			),
		),
	)
	if response.Code != http.StatusOK ||
		response.Header().Get("Content-Type") != "text/event-stream" {
		t.Fatalf("status/headers = %d %#v body=%s", response.Code, response.Header(), response.Body.String())
	}
	body := response.Body.String()
	events := []string{
		"event: message_start",
		"event: content_block_start",
		`"text":"A"`,
		`"text":"B"`,
		"event: content_block_stop",
		"event: message_delta",
		"event: message_stop",
	}
	previous := -1
	for _, event := range events {
		index := strings.Index(body, event)
		if index < 0 || index <= previous {
			t.Fatalf("event %q index=%d after=%d body=%s", event, index, previous, body)
		}
		previous = index
	}
	if strings.Contains(body, "timings") ||
		!strings.Contains(body, `"stop_reason":"max_tokens"`) ||
		!strings.Contains(body, `"output_tokens":2`) {
		t.Fatalf("stream body = %s", body)
	}
}

func TestStreamingAnthropicToolUseLifecycle(t *testing.T) {
	generator := &fakeGenerator{
		pieces: []string{
			`<tool_call><function=weather><parameter=city>Paris</parameter></function></tool_call>`,
		},
	}
	handler := newTestHandler(t, generator)
	response := httptest.NewRecorder()
	handler.ServeHTTP(
		response,
		httptest.NewRequest(
			http.MethodPost,
			"/v1/messages",
			strings.NewReader(
				`{"max_tokens":1,"stream":true,"tool_choice":{"type":"tool","name":"weather",`+
					`"disable_parallel_tool_use":true},`+
					`"tools":[{"name":"weather","input_schema":{"type":"object"}}],`+
					`"messages":[{"role":"user","content":"weather?"}]}`,
			),
		),
	)
	if response.Code != http.StatusOK {
		t.Fatalf("status = %d body=%s", response.Code, response.Body.String())
	}
	body := response.Body.String()
	for _, fragment := range []string{
		"event: message_start",
		`"id":"toolu_`,
		`"type":"tool_use"`,
		`"name":"weather"`,
		`"type":"input_json_delta"`,
		`"partial_json":"{\"city\":\"Paris\"}"`,
		`"stop_reason":"tool_use"`,
		"event: message_stop",
	} {
		if !strings.Contains(body, fragment) {
			t.Fatalf("tool stream lacks %q:\n%s", fragment, body)
		}
	}
	if strings.Contains(body, "<tool_call>") {
		t.Fatalf("tool stream leaked template syntax:\n%s", body)
	}
	generator.mu.Lock()
	defer generator.mu.Unlock()
	if generator.grammarParallel {
		t.Fatal("disable_parallel_tool_use reached the grammar as parallel")
	}
}

func TestParseAnthropicToolHistory(t *testing.T) {
	messages, err := parseAnthropicMessages(
		nil,
		json.RawMessage(`[
			{"role":"assistant","content":[
				{"type":"text","text":"Checking."},
				{"type":"tool_use","id":"toolu_1","name":"weather","input":{"city":"Paris"}}
			]},
			{"role":"user","content":[
				{"type":"tool_result","tool_use_id":"toolu_1","content":[{"type":"text","text":"Sunny"}],"is_error":true},
				{"type":"text","text":"Thanks"}
			]}
		]`),
	)
	if err != nil {
		t.Fatal(err)
	}
	if len(messages) != 3 ||
		messages[0].Content != "Checking." ||
		len(messages[0].ToolCalls) != 1 ||
		messages[0].ToolCalls[0].ID != "toolu_1" ||
		messages[1].Role != "tool" ||
		messages[1].ToolCallID != "toolu_1" ||
		!messages[1].ToolResultError ||
		messages[1].Content != "Sunny" ||
		messages[2].Role != "user" ||
		messages[2].Content != "Thanks" {
		t.Fatalf("parsed messages = %+v", messages)
	}
}

func TestAnthropicInputTokensValidationAndMethod(t *testing.T) {
	handler := newTestHandler(t, &fakeGenerator{})
	for _, body := range []string{
		`{}`,
		`{"messages":[]}`,
		`{"model":"missing","messages":[{"role":"user","content":"hi"}]}`,
		`{"messages":[{"role":"tool","content":"hi"}]}`,
		`{"messages":[{"role":"user","content":[{"type":"image","source":{}}]}]}`,
		`{"messages":[{"role":"user","content":"hi"}],"tools":[{"name":"x"}]}`,
		`{"messages":[{"role":"user","content":"hi"}],"thinking":{"type":"enabled"}}`,
	} {
		response := httptest.NewRecorder()
		handler.ServeHTTP(
			response,
			httptest.NewRequest(
				http.MethodPost,
				"/v1/messages/count_tokens",
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
		httptest.NewRequest(http.MethodGet, "/v1/messages/count_tokens", nil),
	)
	if get.Code != http.StatusMethodNotAllowed || get.Header().Get("Allow") != http.MethodPost {
		t.Fatalf("GET status = %d Allow=%q", get.Code, get.Header().Get("Allow"))
	}
}

func TestChatInputTokensValidationAuthenticationAndMethod(t *testing.T) {
	handler := newTestHandler(t, &fakeGenerator{})
	for _, body := range []string{
		`{"messages":[]}`,
		`{"model":"missing","messages":[{"role":"user","content":"hi"}]}`,
	} {
		response := httptest.NewRecorder()
		handler.ServeHTTP(
			response,
			httptest.NewRequest(
				http.MethodPost,
				"/chat/completions/input_tokens",
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
		httptest.NewRequest(http.MethodGet, "/chat/completions/input_tokens", nil),
	)
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
	}, &fakeGenerator{})
	if err != nil {
		t.Fatal(err)
	}
	unauthorized := httptest.NewRecorder()
	protected.ServeHTTP(
		unauthorized,
		httptest.NewRequest(
			http.MethodPost,
			"/chat/completions/input_tokens",
			strings.NewReader(`{"messages":[{"role":"user","content":"hi"}]}`),
		),
	)
	if unauthorized.Code != http.StatusUnauthorized {
		t.Fatalf("unauthorized status = %d body=%s", unauthorized.Code, unauthorized.Body.String())
	}
}

func TestCompletionRejectsInvalidPromptShapesAndBatchBounds(t *testing.T) {
	handler := newTestHandler(t, &fakeGenerator{})
	largeBatch := make([]string, 65)
	for index := range largeBatch {
		largeBatch[index] = "prompt"
	}
	largeJSON, err := json.Marshal(map[string]any{"prompt": largeBatch})
	if err != nil {
		t.Fatal(err)
	}
	for _, body := range []string{
		`{}`,
		`{"prompt":[]}`,
		`{"prompt":[64]}`,
		`{"prompt":[[5],[]]}`,
		string(largeJSON),
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

func TestApplyTemplate(t *testing.T) {
	handler := newTestHandler(t, &fakeGenerator{})
	request := httptest.NewRequest(
		http.MethodPost,
		"/apply-template",
		strings.NewReader(`{"messages":[{"role":"user","content":"hi"}]}`),
	)
	response := httptest.NewRecorder()
	handler.ServeHTTP(response, request)
	if response.Code != http.StatusOK {
		t.Fatalf("status = %d body=%s", response.Code, response.Body.String())
	}
	var result map[string]string
	if err := json.Unmarshal(response.Body.Bytes(), &result); err != nil {
		t.Fatal(err)
	}
	if result["prompt"] != "formatted-chat" {
		t.Fatalf("apply-template response = %#v", result)
	}
}

func TestApplyTemplateSuppliesToolsAndTemplateOptions(t *testing.T) {
	generator := &fakeGenerator{}
	handler := newTestHandler(t, generator)
	body := `{
		"messages":[
			{"role":"user","content":"weather?"},
			{"role":"assistant","content":null,"tool_calls":[
				{"id":"call_1","type":"function","function":{"name":"weather","arguments":{"city":"Paris"}}}
			]},
			{"role":"tool","tool_call_id":"call_1","content":"sunny"}
		],
		"tools":[{"type":"function","function":{
			"name":"weather",
			"description":"Get weather",
			"parameters":{"type":"object","properties":{"city":{"type":"string"}}}
		}}],
		"add_generation_prompt":false,
		"chat_template_kwargs":{"enable_thinking":false}
	}`
	response := httptest.NewRecorder()
	handler.ServeHTTP(
		response,
		httptest.NewRequest(http.MethodPost, "/apply-template", strings.NewReader(body)),
	)
	if response.Code != http.StatusOK {
		t.Fatalf("status = %d body=%s", response.Code, response.Body.String())
	}
	var result map[string]string
	if err := json.Unmarshal(response.Body.Bytes(), &result); err != nil {
		t.Fatal(err)
	}
	if result["prompt"] != "formatted-chat-tools" {
		t.Fatalf("apply-template response = %#v", result)
	}
	generator.mu.Lock()
	defer generator.mu.Unlock()
	if len(generator.chatOptions.Tools) != 1 ||
		generator.chatOptions.Tools[0].Function.Name != "weather" ||
		generator.chatOptions.AddGenerationPrompt ||
		generator.chatOptions.EnableThinking ||
		len(generator.chatMessages) != 3 ||
		len(generator.chatMessages[1].ToolCalls) != 1 {
		t.Fatalf(
			"messages=%+v options=%+v",
			generator.chatMessages,
			generator.chatOptions,
		)
	}
}

func TestApplyTemplateRejectsMethodAndInvalidMessages(t *testing.T) {
	handler := newTestHandler(t, &fakeGenerator{})
	get := httptest.NewRequest(http.MethodGet, "/apply-template", nil)
	getResponse := httptest.NewRecorder()
	handler.ServeHTTP(getResponse, get)
	if getResponse.Code != http.StatusMethodNotAllowed {
		t.Fatalf("GET status = %d", getResponse.Code)
	}
	post := httptest.NewRequest(
		http.MethodPost,
		"/apply-template",
		strings.NewReader(`{"messages":[]}`),
	)
	postResponse := httptest.NewRecorder()
	handler.ServeHTTP(postResponse, post)
	if postResponse.Code != http.StatusBadRequest {
		t.Fatalf("empty messages status = %d body=%s", postResponse.Code, postResponse.Body.String())
	}
}

func TestTokenizeAndDetokenizeEndpoints(t *testing.T) {
	handler := newTestHandler(t, &fakeGenerator{})
	tokenize := httptest.NewRequest(
		http.MethodPost,
		"/tokenize",
		strings.NewReader(`{"content":"hello","add_special":true,"parse_special":true}`),
	)
	tokenizeResponse := httptest.NewRecorder()
	handler.ServeHTTP(tokenizeResponse, tokenize)
	if tokenizeResponse.Code != http.StatusOK ||
		!strings.Contains(tokenizeResponse.Body.String(), `"tokens":[1,10,2]`) {
		t.Fatalf("tokenize status/body = %d %s", tokenizeResponse.Code, tokenizeResponse.Body.String())
	}
	detokenize := httptest.NewRequest(
		http.MethodPost,
		"/detokenize",
		strings.NewReader(`{"tokens":[1,10,2]}`),
	)
	detokenizeResponse := httptest.NewRecorder()
	handler.ServeHTTP(detokenizeResponse, detokenize)
	var detokenizeResult map[string]string
	detokenizeErr := json.Unmarshal(detokenizeResponse.Body.Bytes(), &detokenizeResult)
	if detokenizeResponse.Code != http.StatusOK ||
		detokenizeErr != nil ||
		detokenizeResult["content"] != "<1><10><2>" {
		t.Fatalf(
			"detokenize status/body = %d %s",
			detokenizeResponse.Code,
			detokenizeResponse.Body.String(),
		)
	}
}

func TestTokenizeMixedContentAndPinnedDefaults(t *testing.T) {
	handler := newTestHandler(t, &fakeGenerator{})
	for _, test := range []struct {
		name string
		body string
		want string
	}{
		{
			name: "strings and token",
			body: `{"content":["first",5,"second"],"add_special":true}`,
			want: `"tokens":[1,10,2,5,10,2]`,
		},
		{
			name: "leading token suppresses special",
			body: `{"content":[5,"second"],"add_special":true}`,
			want: `"tokens":[5,10,2]`,
		},
		{
			name: "parse special override",
			body: `{"content":"plain","parse_special":false}`,
			want: `"tokens":[10]`,
		},
		{
			name: "missing content",
			body: `{}`,
			want: `"tokens":[]`,
		},
	} {
		t.Run(test.name, func(t *testing.T) {
			request := httptest.NewRequest(http.MethodPost, "/tokenize", strings.NewReader(test.body))
			response := httptest.NewRecorder()
			handler.ServeHTTP(response, request)
			if response.Code != http.StatusOK || !strings.Contains(response.Body.String(), test.want) {
				t.Fatalf("status/body = %d %s, want %s", response.Code, response.Body.String(), test.want)
			}
		})
	}
}

func TestTokenizeWithPiecesPreservesInvalidUTF8AsBytes(t *testing.T) {
	handler := newTestHandler(t, &fakeGenerator{})
	request := httptest.NewRequest(
		http.MethodPost,
		"/tokenize",
		strings.NewReader(`{"content":[5,63],"with_pieces":true}`),
	)
	response := httptest.NewRecorder()
	handler.ServeHTTP(response, request)
	if response.Code != http.StatusOK {
		t.Fatalf("status = %d body=%s", response.Code, response.Body.String())
	}
	var result struct {
		Tokens []struct {
			ID    int             `json:"id"`
			Piece json.RawMessage `json:"piece"`
		} `json:"tokens"`
	}
	if err := json.Unmarshal(response.Body.Bytes(), &result); err != nil {
		t.Fatal(err)
	}
	if len(result.Tokens) != 2 {
		t.Fatalf("pieces = %s", response.Body.String())
	}
	var firstPiece string
	if err := json.Unmarshal(result.Tokens[0].Piece, &firstPiece); err != nil {
		t.Fatal(err)
	}
	if result.Tokens[0].ID != 5 ||
		firstPiece != "<5>" ||
		result.Tokens[1].ID != 63 ||
		string(result.Tokens[1].Piece) != `[195]` {
		t.Fatalf("pieces = %s", response.Body.String())
	}
}

func TestTokenizeRejectsInvalidMixedContent(t *testing.T) {
	handler := newTestHandler(t, &fakeGenerator{})
	for _, body := range []string{
		`{"content":[64]}`,
		`{"content":[1.5]}`,
		`{"content":{"text":"bad"}}`,
	} {
		request := httptest.NewRequest(http.MethodPost, "/tokenize", strings.NewReader(body))
		response := httptest.NewRecorder()
		handler.ServeHTTP(response, request)
		if response.Code != http.StatusBadRequest {
			t.Fatalf("body %s: status = %d response=%s", body, response.Code, response.Body.String())
		}
	}
}

func TestDetokenizeRejectsOutOfRangeToken(t *testing.T) {
	handler := newTestHandler(t, &fakeGenerator{})
	request := httptest.NewRequest(
		http.MethodPost,
		"/detokenize",
		strings.NewReader(`{"tokens":[64]}`),
	)
	response := httptest.NewRecorder()
	handler.ServeHTTP(response, request)
	if response.Code != http.StatusBadRequest {
		t.Fatalf("status = %d body=%s", response.Code, response.Body.String())
	}
}

func TestChatCompletionMultipleChoices(t *testing.T) {
	handler := newTestHandler(t, &fakeGenerator{})
	request := httptest.NewRequest(
		http.MethodPost,
		"/v1/chat/completions",
		strings.NewReader(`{"messages":[{"role":"user","content":"hi"}],"max_tokens":1,"n":2}`),
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
	if len(result.Choices) != 2 ||
		result.Choices[0].Index != 0 ||
		result.Choices[1].Index != 1 {
		t.Fatalf("chat choices = %+v", result.Choices)
	}
	if result.Usage.CompletionTokens != 2 {
		t.Fatalf("chat usage = %+v", result.Usage)
	}
}

func TestStreamingChatCompletion(t *testing.T) {
	handler := newTestHandler(t, &fakeGenerator{})
	request := httptest.NewRequest(
		http.MethodPost,
		"/v1/chat/completions",
		strings.NewReader(`{"messages":[{"role":"user","content":"hi"}],"max_tokens":2,"stream":true}`),
	)
	response := httptest.NewRecorder()
	handler.ServeHTTP(response, request)
	if response.Code != http.StatusOK {
		t.Fatalf("status = %d body=%s", response.Code, response.Body.String())
	}
	body := response.Body.String()
	for _, fragment := range []string{
		`"role":"assistant"`,
		`"content":"A"`,
		`"content":"B"`,
		`"finish_reason":"length"`,
		"data: [DONE]",
	} {
		if !strings.Contains(body, fragment) {
			t.Fatalf("chat stream lacks %q: %s", fragment, body)
		}
	}
}

func TestRejectsInvalidRequest(t *testing.T) {
	handler := newTestHandler(t, &fakeGenerator{})
	for _, body := range []string{
		`{"prompt":""}`,
		`{"prompt":"x","max_tokens":9}`,
		`{"prompt":"x","unknown":true}`,
		`{"prompt":"x","temperature":-1}`,
	} {
		request := httptest.NewRequest(http.MethodPost, "/v1/completions", strings.NewReader(body))
		response := httptest.NewRecorder()
		handler.ServeHTTP(response, request)
		if response.Code != http.StatusBadRequest {
			t.Fatalf("body %s status = %d, want 400", body, response.Code)
		}
	}
}

func TestAdmissionControl(t *testing.T) {
	blocking := &fakeGenerator{started: make(chan struct{}), release: make(chan struct{})}
	handler := newTestHandler(t, blocking)
	firstDone := make(chan struct{})
	go func() {
		defer close(firstDone)
		request := httptest.NewRequest(
			http.MethodPost,
			"/v1/completions",
			strings.NewReader(`{"prompt":"first","max_tokens":1}`),
		)
		handler.ServeHTTP(httptest.NewRecorder(), request)
	}()
	<-blocking.started
	request := httptest.NewRequest(
		http.MethodPost,
		"/v1/completions",
		strings.NewReader(`{"prompt":"second","max_tokens":1}`),
	)
	response := httptest.NewRecorder()
	handler.ServeHTTP(response, request)
	if response.Code != http.StatusTooManyRequests {
		t.Fatalf("status = %d, want 429", response.Code)
	}
	close(blocking.release)
	<-firstDone
}

func TestCancellationPropagates(t *testing.T) {
	blocking := &fakeGenerator{started: make(chan struct{}), release: make(chan struct{})}
	handler := newTestHandler(t, blocking)
	ctx, cancel := context.WithCancel(context.Background())
	request := httptest.NewRequest(
		http.MethodPost,
		"/v1/completions",
		strings.NewReader(`{"prompt":"cancel","max_tokens":1}`),
	).WithContext(ctx)
	response := httptest.NewRecorder()
	done := make(chan struct{})
	go func() {
		defer close(done)
		handler.ServeHTTP(response, request)
	}()
	<-blocking.started
	cancel()
	<-done
	if response.Code != http.StatusRequestTimeout {
		t.Fatalf("status = %d, want 408", response.Code)
	}
	var envelope apiErrorEnvelope
	if err := json.Unmarshal(response.Body.Bytes(), &envelope); err != nil {
		t.Fatal(err)
	}
	if !errors.Is(ctx.Err(), context.Canceled) || envelope.Error.Type != "request_cancelled" {
		t.Fatalf("error = %+v, context = %v", envelope, ctx.Err())
	}
}
