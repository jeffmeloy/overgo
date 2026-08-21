package server

import (
	"bytes"

	"context"

	"encoding/base64"

	"encoding/binary"

	"encoding/json"

	"errors"

	"fmt"

	"hash/crc32"

	"image"

	"image/color"

	"image/png"

	"overgo/internal/cuda/driver"

	"overgo/internal/inference"

	"overgo/internal/projector"

	"overgo/internal/sampling"
	"overgo/internal/testutil"

	"overgo/internal/tokenizer"

	"net/http"

	"net/http/httptest"

	"slices"

	"strings"

	"sync"

	"testing"

	"time"
)

const (
	testModelID            = "test-model"
	testMaxTokens          = 8
	testNeutralTemperature = 1
	testFullTopP           = 1
	testAPIKey             = "test-secret"
	testBearerToken        = "Bearer " + testAPIKey

	testAudioSampleRate = 16_000
	testAudioAmplitude  = 16_384

	pngIHDRPayloadBytes  = 13
	pngTruecolorBitDepth = 8
	pngTruecolorType     = 2
	pngDeflateMethod     = 0
	pngAdaptiveFilter    = 0
	pngNoInterlace       = 0

	testAnalysisSamples       = 512
	testAnalysisReadBytes     = 1 << 20
	testAnalysisPositions     = 64
	testAnalysisMDSIterations = 1000
	testAnalysisMDSTolerance  = 1e-6
)

var testAnalysisPolicy = AnalysisPolicy{
	TensorSamples: testAnalysisSamples, TensorReadBytes: testAnalysisReadBytes,
	StatePositions: testAnalysisPositions, MDSIterations: testAnalysisMDSIterations,
	MDSTolerance: testAnalysisMDSTolerance,
}

func tinyPCM16WAV() []byte {
	return testutil.MonoPCM16WAV(
		testAudioSampleRate,
		[]int16{testAudioAmplitude, -testAudioAmplitude},
	)
}

func silentPCM16WAV() []byte {
	return testutil.MonoPCM16WAV(testAudioSampleRate, []int16{0, 0})
}

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

// fakeSessionStub: embeddable projector.Session base; every prompt surface
// errors until a fake overrides it.
type fakeSessionStub struct{}

func (fakeSessionStub) BuildImagePrompt(
	context.Context, projector.ImageTokenizer, image.Image, string, string, bool,
) (projector.MultimodalPrompt, error) {
	return projector.MultimodalPrompt{}, errors.New("fake session: image prompts unsupported")
}

func (fakeSessionStub) BuildImagesPrompt(
	context.Context, projector.ImageTokenizer, []image.Image, []string, projector.PromptOptions,
) (projector.MultimodalPrompt, error) {
	return projector.MultimodalPrompt{}, errors.New("fake session: image prompts unsupported")
}

func (fakeSessionStub) BuildVideoPrompt(
	context.Context, projector.ImageTokenizer, []image.Image, string, string, float64, bool,
) (projector.MultimodalPrompt, error) {
	return projector.MultimodalPrompt{}, errors.New("fake session: video prompts unsupported")
}

func (fakeSessionStub) BuildAudioPrompt(
	context.Context, projector.ImageTokenizer, []float32, string, string,
) (projector.MultimodalPrompt, error) {
	return projector.MultimodalPrompt{}, errors.New("fake session: audio prompts unsupported")
}

func (fakeSessionStub) BuildMediaHistoryPrompt(
	context.Context, projector.ImageTokenizer, []projector.MediaInput, []string,
) (projector.MultimodalPrompt, error) {
	return projector.MultimodalPrompt{}, errors.New("fake session: media history prompts unsupported")
}

func (fakeSessionStub) Capabilities() projector.SessionCapabilities {
	return projector.SessionCapabilities{}
}

func (fakeSessionStub) Close() error { return nil }

type fakeQwen3VLProjector struct {
	fakeSessionStub
	before string
	after  string
	images int
	text   []string
}

func (*fakeQwen3VLProjector) Capabilities() projector.SessionCapabilities {
	return projector.SessionCapabilities{Image: true, MultiImage: true}
}

type fakeVideoProjector struct {
	fakeQwen3VLProjector
	frames int
	fps    float64
}

func (*fakeVideoProjector) Capabilities() projector.SessionCapabilities {
	return projector.SessionCapabilities{Image: true, MultiImage: true, Video: true}
}

func (f *fakeVideoProjector) BuildVideoPrompt(
	_ context.Context,
	_ projector.ImageTokenizer,
	frames []image.Image,
	before, after string,
	fps float64,
	_ bool,
) (projector.MultimodalPrompt, error) {
	f.frames = len(frames)
	f.fps = fps
	f.before, f.after = before, after
	return projector.MultimodalPrompt{
		TokenIDs: []tokenizer.TokenID{1, 2, 3}, Embeddings: make([]float32, 2560),
		EmbeddingWidth: 2560, EmbeddingTokenIndices: []uint32{1},
	}, nil
}

func (*fakeVideoProjector) Close() error { return nil }

type fakeHistoryProjector struct {
	fakeQwen3VLProjector
	historyText []string
	historyRuns int
	mediaKinds  []projector.MediaKind
}

func (*fakeHistoryProjector) Capabilities() projector.SessionCapabilities {
	return projector.SessionCapabilities{Image: true, MultiImage: true, MediaHistory: true}
}

func (f *fakeHistoryProjector) BuildImagesPrompt(
	_ context.Context,
	_ projector.ImageTokenizer,
	images []image.Image,
	text []string,
	options projector.PromptOptions,
) (projector.MultimodalPrompt, error) {
	if !options.History {
		return f.fakeQwen3VLProjector.BuildImagesPrompt(context.Background(), nil, images, text, options)
	}
	f.images = len(images)
	f.historyText = append([]string(nil), text...)
	f.historyRuns++
	return projector.MultimodalPrompt{
		TokenIDs:              []tokenizer.TokenID{7, 8, 8, 9},
		Embeddings:            make([]float32, 2*2560),
		EmbeddingWidth:        2560,
		EmbeddingTokenIndices: []uint32{1, 2},
		AttentionBlocks:       []projector.AttentionBlock{{Start: 1, End: 3}},
	}, nil
}

func (f *fakeHistoryProjector) BuildMediaHistoryPrompt(
	_ context.Context,
	_ projector.ImageTokenizer,
	media []projector.MediaInput,
	text []string,
) (projector.MultimodalPrompt, error) {
	f.historyText = append([]string(nil), text...)
	f.historyRuns++
	f.mediaKinds = f.mediaKinds[:0]
	for _, item := range media {
		f.mediaKinds = append(f.mediaKinds, item.Kind)
	}
	return projector.MultimodalPrompt{
		TokenIDs:              []tokenizer.TokenID{7, 8, 9, 10},
		Embeddings:            make([]float32, 2*2560),
		EmbeddingWidth:        2560,
		EmbeddingTokenIndices: []uint32{1, 2},
		AttentionBlocks:       []projector.AttentionBlock{{Start: 1, End: 2}},
	}, nil
}

type historyGenerator struct {
	*fakeGenerator
}

type reasoningGenerator struct {
	*fakeGenerator
}

func (g *reasoningGenerator) ParseChatOutput(
	output string,
	tools []inference.ChatTool,
) (inference.ChatMessage, error) {
	if len(tools) != 0 {
		return g.fakeGenerator.ParseChatOutput(output, tools)
	}
	message := inference.ChatMessage{Role: "assistant", Content: output}
	if closeIndex := strings.Index(output, "</think>"); closeIndex >= 0 {
		message.ReasoningContent = strings.TrimSpace(output[:closeIndex])
		message.Content = strings.TrimSpace(output[closeIndex+len("</think>"):])
	}
	return message, nil
}

func (g *historyGenerator) FormatChat(messages []inference.ChatMessage) (string, error) {
	if len(messages) == 0 {
		return "", errors.New("chat message list is empty")
	}
	var result strings.Builder
	result.WriteString("<chat>")
	for _, message := range messages {
		fmt.Fprintf(&result, "<%s>%s</%s>", message.Role, message.Content, message.Role)
	}
	result.WriteString("<assistant>")
	return result.String(), nil
}

func (f *fakeQwen3VLProjector) BuildImagesPrompt(
	_ context.Context,
	_ projector.ImageTokenizer,
	images []image.Image,
	text []string,
	_ projector.PromptOptions,
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
	fakeSessionStub
	before  string
	after   string
	samples []float32
}

func (*fakeAudioProjector) Capabilities() projector.SessionCapabilities {
	return projector.SessionCapabilities{Audio: true}
}

func (*fakeQwen3VLProjector) Close() error { return nil }

func (*fakeAudioProjector) Close() error { return nil }

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

func (f *fakeQwen3VLProjector) BuildImagePrompt(
	_ context.Context,
	_ projector.ImageTokenizer,
	_ image.Image,
	before, after string,
	_ bool,
) (projector.MultimodalPrompt, error) {
	f.before, f.after = before, after
	positions := [4][]uint32{
		{0, 1, 1, 2}, {0, 1, 1, 2}, {0, 1, 2, 2}, {0, 0, 0, 2},
	}
	return projector.MultimodalPrompt{
		TokenIDs:   []tokenizer.TokenID{1, 2, 2, 3},
		Embeddings: make([]float32, 2*2560), EmbeddingWidth: 2560,
		EmbeddingStart: 1, EmbeddingTokenIndices: []uint32{1, 2}, MultiAxisPositions: positions,
	}, nil
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

func (f *fakeLoRAGenerator) SetLoRAScales(_ context.Context, scales []inference.LoRAScale) error {
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
		ModelID:            testModelID,
		MaxTokens:          testMaxTokens,
		DefaultTemperature: testNeutralTemperature,
		DefaultTopP:        testFullTopP,
		Analysis:           testAnalysisPolicy,
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
		ModelID:            testModelID,
		MaxTokens:          testMaxTokens,
		DefaultTemperature: testNeutralTemperature,
		DefaultTopP:        testFullTopP,
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
		ModelID:            testModelID,
		MaxTokens:          testMaxTokens,
		DefaultTemperature: testNeutralTemperature,
		DefaultTopP:        testFullTopP,
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
	f.mu.Lock()
	f.chatMessages = cloneResponseMessages(messages)
	f.mu.Unlock()
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

func (f *fakeGenerator) NewChatOutputStream(
	tools []inference.ChatTool,
) (inference.ChatOutputStream, error) {
	return inference.NewChatOutputStream(
		`<function=example_function_name>`,
		tools,
	), nil
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

func (f *fakeGenerator) Detokenize(tokens []tokenizer.TokenID, _ inference.TokenRenderMode) (string, error) {
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
		ModelID:            testModelID,
		MaxTokens:          testMaxTokens,
		DefaultTemperature: testNeutralTemperature,
		DefaultTopP:        testFullTopP,
		Analysis:           testAnalysisPolicy,
	}, generator)
	if err != nil {
		t.Fatal(err)
	}
	return handler
}

func pngConfigFixture(width, height uint32) []byte {
	chunk := make([]byte, 0, len("IHDR")+pngIHDRPayloadBytes)
	chunk = append(chunk, "IHDR"...)
	chunk = binary.BigEndian.AppendUint32(chunk, width)
	chunk = binary.BigEndian.AppendUint32(chunk, height)
	chunk = append(
		chunk,
		pngTruecolorBitDepth,
		pngTruecolorType,
		pngDeflateMethod,
		pngAdaptiveFilter,
		pngNoInterlace,
	)

	data := []byte("\x89PNG\r\n\x1a\n")
	data = binary.BigEndian.AppendUint32(data, pngIHDRPayloadBytes)
	data = append(data, chunk...)
	data = binary.BigEndian.AppendUint32(data, crc32.ChecksumIEEE(chunk))
	return data
}

func TestJSONRequestBudgets(t *testing.T) {
	body := `{"value":"` + strings.Repeat("x", maxRequestBytes) + `"}`
	var ordinary struct {
		Value string `json:"value"`
	}
	response := httptest.NewRecorder()
	request := httptest.NewRequest(http.MethodPost, "/", strings.NewReader(body))
	if newTestHandler(t, &fakeGenerator{}).decodeBoundedJSON(response, request, &ordinary) {
		t.Fatal("ordinary request accepted oversized JSON")
	}
	if response.Code != http.StatusBadRequest {
		t.Fatalf("ordinary status = %d body=%s", response.Code, response.Body.String())
	}

	var multimodal struct {
		Value string `json:"value"`
	}
	response = httptest.NewRecorder()
	request = httptest.NewRequest(http.MethodPost, "/", strings.NewReader(body))
	if !newTestHandler(t, &fakeGenerator{}).decodeMultimodalJSON(response, request, &multimodal) {
		t.Fatalf("multimodal status = %d body=%s", response.Code, response.Body.String())
	}
	if multimodal.Value != strings.Repeat("x", maxRequestBytes) {
		t.Fatalf("multimodal value length = %d", len(multimodal.Value))
	}
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
	response := serveTestRequest(handler, http.MethodGet, "/metrics", "")
	if response.Code != http.StatusOK {
		t.Fatalf("status = %d body=%s", response.Code, response.Body.String())
	}
	if strings.Contains(response.Body.String(), "overgo_cuda_memory_") {
		t.Fatalf("failed device metrics leaked into response: %s", response.Body.String())
	}
}

func TestLoraAdaptersEmptyControlPlane(t *testing.T) {
	handler := newTestHandler(t, &fakeGenerator{})
	get := serveTestRequest(handler, http.MethodGet, "/lora-adapters", "")
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
	get := serveTestRequest(handler, http.MethodGet, "/lora-adapters", "")
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
		result.Models[0].Name != testModelID ||
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
		ModelID:            testModelID,
		MaxTokens:          testMaxTokens,
		DefaultTemperature: testNeutralTemperature,
		DefaultTopP:        testFullTopP,
		APIKey:             testAPIKey,
	}, &fakeGenerator{})
	if err != nil {
		t.Fatal(err)
	}
	for _, path := range []string{"/models", "/v1/models"} {
		response := serveTestRequest(handler, http.MethodGet, path, "")
		if response.Code != http.StatusOK {
			t.Fatalf("%s status = %d body=%s", path, response.Code, response.Body.String())
		}
	}
	post := serveTestRequest(handler, http.MethodPost, "/models", "")
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
		result.ModelAlias != testModelID ||
		result.ModelFType != "Q8_0" ||
		result.ModelPath != "fixture.gguf" ||
		result.ModelMetadata.Architecture != "qwen3" ||
		result.DefaultGenerationSettings.NCtx != 32768 ||
		result.DefaultGenerationSettings.Params.NPredict != defaultProtocolMaxTokens ||
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
		ModelID:            testModelID,
		MaxTokens:          testMaxTokens,
		MaxConcurrent:      2,
		DefaultTemperature: testNeutralTemperature,
		DefaultTopP:        testFullTopP,
		APIKey:             testAPIKey,
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
			request.Header.Set("Authorization", testBearerToken)
			responses[index] = httptest.NewRecorder()
			handler.ServeHTTP(responses[index], request)
		}(index)
	}
	deadline := time.Now().Add(time.Second)
	for handler.sessions.Available() != 0 && time.Now().Before(deadline) {
		time.Sleep(time.Millisecond)
	}
	if handler.sessions.Available() != 0 {
		t.Fatal("generation requests did not acquire both slots")
	}

	slotRequest := httptest.NewRequest(http.MethodGet, "/slots", nil)
	slotRequest.Header.Set("Authorization", testBearerToken)
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
	failRequest.Header.Set("Authorization", testBearerToken)
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
	idleRequest.Header.Set("Authorization", testBearerToken)
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
			slot.Prompt != "" ||
			slot.Generated != "" ||
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
		ModelID:            testModelID,
		MaxTokens:          testMaxTokens,
		DefaultTemperature: testNeutralTemperature,
		DefaultTopP:        testFullTopP,
		APIKey:             testAPIKey,
	}, &fakeGenerator{})
	if err != nil {
		t.Fatal(err)
	}
	unauthorized := serveTestRequest(handler, http.MethodGet, "/slots", "")
	if unauthorized.Code != http.StatusUnauthorized {
		t.Fatalf("unauthorized status = %d body=%s", unauthorized.Code, unauthorized.Body.String())
	}
	request := httptest.NewRequest(http.MethodPost, "/slots", nil)
	request.Header.Set("Authorization", testBearerToken)
	method := httptest.NewRecorder()
	handler.ServeHTTP(method, request)
	if method.Code != http.StatusMethodNotAllowed ||
		method.Header().Get("Allow") != http.MethodGet {
		t.Fatalf("POST status = %d Allow=%q", method.Code, method.Header().Get("Allow"))
	}
}

func TestSlotsRedactTextByDefault(t *testing.T) {
	handler := newTestHandler(t, &fakeGenerator{})
	generation := serveTestRequest(
		handler,
		http.MethodPost,
		"/completion",
		`{"prompt":"private","n_predict":1}`,
	)
	if generation.Code != http.StatusOK {
		t.Fatalf("generation status = %d body=%s", generation.Code, generation.Body.String())
	}

	redacted := serveTestRequest(handler, http.MethodGet, "/slots", "")
	var fields []map[string]json.RawMessage
	if err := json.Unmarshal(redacted.Body.Bytes(), &fields); err != nil {
		t.Fatal(err)
	}
	if len(fields) != 1 {
		t.Fatalf("slot count = %d", len(fields))
	}
	for _, name := range []string{"prompt", "generated"} {
		if _, ok := fields[0][name]; ok {
			t.Errorf("default slot status exposes %s", name)
		}
	}

	included := serveTestRequest(handler, http.MethodGet, "/slots?include_text=1", "")
	var slots []slotStatusItem
	if err := json.Unmarshal(included.Body.Bytes(), &slots); err != nil {
		t.Fatal(err)
	}
	if len(slots) != 1 || slots[0].Prompt != "private" || slots[0].Generated != "A" {
		t.Fatalf("explicit slot text = %+v", slots)
	}
}

func TestNativeCompletionHonorsRequestedSlot(t *testing.T) {
	handler, err := New(Config{
		ModelID:            testModelID,
		MaxTokens:          testMaxTokens,
		MaxConcurrent:      2,
		DefaultTemperature: testNeutralTemperature,
		DefaultTopP:        testFullTopP,
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

	session, ok := handler.acquireSession(1)
	if !ok || session.ID != 1 {
		t.Fatalf("could not reserve slot 1: %v/%v", session, ok)
	}
	resumed := make(chan *httptest.ResponseRecorder, 1)
	go func() {
		response := httptest.NewRecorder()
		handler.ServeHTTP(
			response,
			httptest.NewRequest(
				http.MethodPost,
				"/completion",
				strings.NewReader(`{"prompt":"hi","n_predict":1,"id_slot":1}`),
			),
		)
		resumed <- response
	}()
	waitForServerSessionWaiter(t, handler)
	handler.releaseSession(session)
	if response := <-resumed; response.Code != http.StatusOK {
		t.Fatalf("resumed requested slot status = %d body=%s", response.Code, response.Body.String())
	}
}

func TestPropertiesRejectsPost(t *testing.T) {
	handler := newTestHandler(t, &fakeGenerator{})
	response := serveTestRequest(handler, http.MethodPost, "/props", "")
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
		ModelID:            testModelID,
		MaxTokens:          testMaxTokens,
		DefaultTemperature: testNeutralTemperature,
		DefaultTopP:        testFullTopP,
		APIKey:             testAPIKey,
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
	request.Header.Set("Authorization", testBearerToken)
	authorized := httptest.NewRecorder()
	handler.ServeHTTP(authorized, request)
	if authorized.Code != http.StatusOK {
		t.Fatalf("authorized status = %d body=%s", authorized.Code, authorized.Body.String())
	}
}

func TestBearerAuthentication(t *testing.T) {
	handler, err := New(Config{
		ModelID:            testModelID,
		MaxTokens:          testMaxTokens,
		DefaultTemperature: testNeutralTemperature,
		DefaultTopP:        testFullTopP,
		APIKey:             testAPIKey,
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
	authorizedRequest.Header.Set("Authorization", testBearerToken)
	authorized := httptest.NewRecorder()
	handler.ServeHTTP(authorized, authorizedRequest)
	if authorized.Code != http.StatusOK {
		t.Fatalf("authorized status = %d body=%s", authorized.Code, authorized.Body.String())
	}
	health := serveTestRequest(handler, http.MethodGet, "/health", "")
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
		"overgo_up 1",
		"overgo_http_requests_total 2",
		"overgo_http_requests_active 1",
		"overgo_generation_requests_total 1",
		"overgo_generation_errors_total 0",
		"overgo_generated_tokens_total 2",
		"overgo_cuda_memory_current_bytes 123",
		"overgo_cuda_memory_peak_bytes 456",
		"overgo_cuda_allocations_current 2",
		"overgo_cuda_custom_kernel_launches_total 12",
		"overgo_cuda_stream_synchronizations_total 3",
		"overgo_cuda_context_synchronizations_total 1",
		"overgo_cuda_host_to_device_bytes_total 1024",
		"overgo_cuda_device_to_host_bytes_total 2048",
		"overgo_cuda_device_to_device_bytes_total 3072",
		"overgo_cuda_device_memset_bytes_total 4096",
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
		result.Model != testModelID ||
		result.Prompt != "hi" ||
		result.Timings.PredictedN != 2 {
		t.Fatalf("native completion = %+v", result)
	}
	if result.GenerationSettings["n_predict"] != float64(2) ||
		result.GenerationSettings["model"] != testModelID {
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

func TestConvertProjectedPromptExpandsDeepstack(t *testing.T) {
	handler := newTestHandler(t, &fakeGenerator{})
	stream := make([]float32, 2*2560)
	stream[0], stream[2560] = 1, 2
	inputs, err := handler.convertProjectedPrompt(projector.MultimodalPrompt{
		TokenIDs:              []tokenizer.TokenID{1, 2, 3, 4, 5},
		Embeddings:            make([]float32, 2*2560),
		DeepstackEmbeddings:   [][]float32{stream},
		EmbeddingWidth:        2560,
		EmbeddingTokenIndices: []uint32{1, 3},
	})
	if err != nil {
		t.Fatal(err)
	}
	if len(inputs.DeepstackEmbeddings) != 1 ||
		inputs.DeepstackEmbeddings[0].Shape.Dims[0] != 2560 ||
		inputs.DeepstackEmbeddings[0].Shape.Dims[1] != 5 {
		t.Fatalf("deepstack inputs = %+v", inputs.DeepstackEmbeddings)
	}
	data := inputs.DeepstackEmbeddings[0].Data
	if data[0] != 0 || data[2560] != 1 || data[2*2560] != 0 || data[3*2560] != 2 || data[4*2560] != 0 {
		t.Fatalf("deepstack token starts = %v", []float32{data[0], data[2560], data[2*2560], data[3*2560], data[4*2560]})
	}
}

func TestNativeCompletionImageProjectorMultimodalPrompt(t *testing.T) {
	generator := &fakeGenerator{}
	vision := &fakeQwen3VLProjector{}
	handler, err := New(Config{
		ModelID: testModelID, MaxTokens: testMaxTokens,
		DefaultTemperature: testNeutralTemperature, DefaultTopP: testFullTopP,
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
		ModelID: testModelID, MaxTokens: testMaxTokens, DefaultTemperature: testNeutralTemperature, DefaultTopP: testFullTopP,
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

func TestValidateMultimodalImagesEnforcesGeometryBudgets(t *testing.T) {
	oversizedImage := make([]byte, maxImageBytes+1)
	aggregateImage := func() []byte {
		data := make([]byte, maxMediaBytes/2+1)
		copy(data, pngConfigFixture(1, 1))
		return data
	}
	tests := []struct {
		name   string
		images [][]byte
		want   string
	}{
		{
			name:   "per-image bytes",
			images: [][]byte{oversizedImage},
			want:   "decoded image limit",
		},
		{
			name:   "aggregate bytes",
			images: [][]byte{aggregateImage(), aggregateImage()},
			want:   "decoded media limit",
		},
		{
			name:   "dimension",
			images: [][]byte{pngConfigFixture(maxImageDimension+1, 1)},
			want:   "dimensions exceed limit",
		},
		{
			name:   "per-image pixels",
			images: [][]byte{pngConfigFixture(4096, 4097)},
			want:   "pixel count exceeds limit",
		},
		{
			name: "aggregate pixels",
			images: [][]byte{
				pngConfigFixture(4096, 4096),
				pngConfigFixture(4096, 4096),
				pngConfigFixture(4096, 4096),
			},
			want: "aggregate pixel limit",
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			err := validateMultimodalImages(test.images)
			if err == nil || !strings.Contains(err.Error(), test.want) {
				t.Fatalf("error = %v; want %q", err, test.want)
			}
		})
	}
}

func TestNativeCompletionRejectsOversizedImageGeometry(t *testing.T) {
	handler, err := New(Config{
		ModelID: testModelID, MaxTokens: testMaxTokens,
		DefaultTemperature: testNeutralTemperature, DefaultTopP: testFullTopP,
		ImageProjector: &fakeQwen3VLProjector{},
	}, &fakeGenerator{})
	if err != nil {
		t.Fatal(err)
	}
	body, err := json.Marshal(map[string]any{
		"prompt": map[string]any{
			"prompt_string": "Look <__media__>",
			"multimodal_data": []string{
				base64.StdEncoding.EncodeToString(pngConfigFixture(maxImageDimension+1, 1)),
			},
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	response := httptest.NewRecorder()
	handler.ServeHTTP(
		response,
		httptest.NewRequest(http.MethodPost, "/completion", bytes.NewReader(body)),
	)
	if response.Code != http.StatusBadRequest ||
		!strings.Contains(response.Body.String(), "dimensions exceed limit") {
		t.Fatalf("status = %d body=%s", response.Code, response.Body.String())
	}
}

func TestNativeCompletionAudioProjectorMultimodalPrompt(t *testing.T) {
	generator := &fakeGenerator{}
	audio := &fakeAudioProjector{}
	handler, err := New(Config{
		ModelID: testModelID, MaxTokens: testMaxTokens,
		DefaultTemperature: testNeutralTemperature, DefaultTopP: testFullTopP,
		AudioProjector: audio,
	}, generator)
	if err != nil {
		t.Fatal(err)
	}
	wav := tinyPCM16WAV()
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

func TestNativeCompletionMixedMediaPreservesChunkOrder(t *testing.T) {
	generator := &fakeGenerator{}
	vision := &fakeHistoryProjector{}
	audio := &fakeAudioProjector{}
	handler, err := New(Config{
		ModelID: testModelID, MaxTokens: testMaxTokens,
		DefaultTemperature: testNeutralTemperature, DefaultTopP: testFullTopP,
		ImageProjector: vision, AudioProjector: audio,
	}, generator)
	if err != nil {
		t.Fatal(err)
	}
	var encodedImage bytes.Buffer
	if err := png.Encode(&encodedImage, image.NewRGBA(image.Rect(0, 0, 1, 1))); err != nil {
		t.Fatal(err)
	}
	wav := silentPCM16WAV()
	body, err := json.Marshal(map[string]any{
		"prompt": map[string]any{
			"prompt_string": "A<__media__>B<__media__>C",
			"multimodal_data": []string{
				"data:image/png;base64," + base64.StdEncoding.EncodeToString(encodedImage.Bytes()),
				"data:audio/wav;base64," + base64.StdEncoding.EncodeToString(wav),
			},
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
	if !slices.Equal(vision.mediaKinds, []projector.MediaKind{projector.MediaImage, projector.MediaAudio}) ||
		!slices.Equal(vision.historyText, []string{"A", "B", "C"}) {
		t.Fatalf("media = %v text=%q", vision.mediaKinds, vision.historyText)
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

func TestNativeCompletionProjectedInputsAllowSignedPromptCache(t *testing.T) {
	generator := &fakeGenerator{}
	handler := newTestHandler(t, generator)
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
	if response.Code != http.StatusOK || !generator.cachePrompt || generator.projectedInputs == nil {
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
		slots[0].Generated != "" ||
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
		wantPrompt, _ := generator.Detokenize(test.want, inference.RenderPrompt)
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
