package server

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"image"
	_ "image/gif"
	_ "image/jpeg"
	_ "image/png"
	"math"
	"net/http"
	"slices"
	"sort"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"overgo/internal/cuda/driver"
	"overgo/internal/inference"
	"overgo/internal/projector"
	"overgo/internal/sampling"
	"overgo/internal/tokenizer"
)

const (
	DefaultModelID             = "overgo"
	DefaultMaxTokens           = 4096
	DefaultMaxConcurrent       = 1
	DefaultMaxEmbeddingInputs  = 16
	DefaultVideoFPS            = 2
	DefaultVideoFrameLimit     = 32
	DefaultSamplingTemperature = float32(1)
	DefaultSamplingTopP        = float32(1)
	DefaultSamplingTopK        = 40
	maxRequestBytes            = 1 << 20
	maxMultimodalRequestBytes  = 32 << 20
	maxImageBytes              = 16 << 20
	maxMediaBytes              = 24 << 20
	maxImageDimension          = 16384
	maxImagePixels             = 16 << 20
	maxRequestImagePixels      = 32 << 20
	maxConcurrentRequests      = 1 << 16
	maxStoredResponses         = 1 << 16
	maxResponseStoreBytes      = 1 << 30
	maxTokenListLength         = 1 << 20
	maxCompletionChoices       = 8
	defaultProtocolMaxTokens   = 16
	maxProtocolTools           = 128
	maxVideoFrameLimit         = 256
)

type Generator interface {
	Generate(
		context.Context,
		string,
		inference.GenerateOptions,
	) ([]tokenizer.TokenID, string, error)
}

type ContinuousGeneratorFactory interface {
	NewContinuousGenerator(
		inference.ContinuousGeneratorOptions,
	) (*inference.ContinuousGenerator, error)
}

type Embedder interface {
	Embed(context.Context, string) ([]float32, int, error)
}

type TokenEmbedder interface {
	EmbedTokens(context.Context, []tokenizer.TokenID) ([]float32, int, error)
}

type AdvancedEmbedder interface {
	EmbedAdvanced(
		context.Context,
		string,
		inference.EmbeddingOptions,
	) (inference.EmbeddingResult, error)
	EmbedTokensAdvanced(
		context.Context,
		[]tokenizer.TokenID,
		inference.EmbeddingOptions,
	) (inference.EmbeddingResult, error)
}

type Ranker interface {
	RankPair(context.Context, string, string) (inference.RankResult, error)
}

type RankCapability interface {
	SupportsRank() bool
}

type ChatFormatter interface {
	FormatChat([]inference.ChatMessage) (string, error)
}

type ToolAwareChatFormatter interface {
	FormatChatWithOptions(
		[]inference.ChatMessage,
		inference.ChatFormatOptions,
	) (string, error)
}

type ChatOutputParser interface {
	ParseChatOutput(
		string,
		[]inference.ChatTool,
	) (inference.ChatMessage, error)
}

type ChatOutputStreamProvider interface {
	NewChatOutputStream(
		[]inference.ChatTool,
	) (inference.ChatOutputStream, error)
}

type ChatToolGrammarProvider interface {
	ChatToolGrammar(
		[]inference.ChatTool,
		bool,
		bool,
		bool,
	) (source, root string, triggerPatterns []string, err error)
}

type DryBreakerTokenizer interface {
	TokenizeDryBreakers([]string) ([][]int, error)
}

type GrammarChoiceTokenizer interface {
	TokenizeGrammarChoices([]string) (*sampling.TokenGrammar, error)
}

type GBNFCompiler interface {
	CompileGBNF(string, string) (*sampling.GBNFGrammar, error)
}

type LazyGBNFCompiler interface {
	CompileLazyGBNF(
		string,
		string,
		[]string,
		[]tokenizer.TokenID,
	) (*sampling.GBNFGrammar, error)
}

type SamplingVocabulary interface {
	TokenizeSamplingText(string) ([]tokenizer.TokenID, error)
	SamplingEOGTokens() []tokenizer.TokenID
	SamplingVocabularySize() int
}

type InfillVocabularyProvider interface {
	SamplingInfillVocabulary() (*sampling.InfillVocabulary, error)
}

type InfillFormatter interface {
	FormatInfillTokens(
		[]tokenizer.TokenID,
		[]tokenizer.TokenID,
		[]tokenizer.TokenID,
		[]inference.InfillExtra,
		inference.InfillFormatOptions,
	) ([]tokenizer.TokenID, error)
}

type TokenizationAPI interface {
	TokenizeText(string, bool, bool) ([]tokenizer.TokenID, error)
	DetokenizeTokens([]tokenizer.TokenID) (string, error)
	SamplingVocabularySize() int
}

type TokenPieceAPI interface {
	TokenPiece(tokenizer.TokenID) (string, error)
}

type PromptTokenDecoder interface {
	DetokenizePromptTokens([]tokenizer.TokenID) (string, error)
}

type ModelPropertiesAPI interface {
	ModelProperties() inference.ModelProperties
}

type LoRAControlAPI interface {
	LoRAAdapters() []inference.LoRAAdapterInfo
	SetLoRAScales([]inference.LoRAScale) error
}

type DeviceMemoryAPI interface {
	DeviceMemoryStats(context.Context) (driver.MemoryStats, error)
}

type DeviceExecutionAPI interface {
	DeviceExecutionStats(context.Context) (driver.ExecutionStats, error)
}

type Qwen3VLProjector interface {
	BuildQwen35ImagePrompt(
		context.Context,
		projector.Qwen3VLTokenizer,
		image.Image,
		string,
		string,
		bool,
	) (projector.Qwen3VLPrompt, error)
}

type ImageProjector interface {
	BuildImagePrompt(
		context.Context,
		projector.ImageTokenizer,
		image.Image,
		string,
		string,
		bool,
	) (projector.MultimodalPrompt, error)
}

type AudioProjector interface {
	BuildAudioPrompt(
		context.Context,
		projector.ImageTokenizer,
		[]float32,
		string,
		string,
	) (projector.MultimodalPrompt, error)
}

type Config struct {
	ModelID            string
	MaxTokens          int
	MaxConcurrent      int
	DefaultTemperature float32
	DefaultTopP        float32
	DefaultTopK        int
	MaxEmbeddingInputs int
	APIKey             string
	ContextShift       bool
	RequestTimeout     time.Duration
	InfillBatchSize    int
	SPMInfill          bool
	Qwen3VLProjector   Qwen3VLProjector
	ImageProjector     ImageProjector
	AudioProjector     AudioProjector
	RemoteMediaPolicy  *RemoteMediaPolicy
	ResponseFiles      ResponseFileResolver
	ResponseToolPolicy ResponseToolPolicy
	MaxStoredResponses int
	ResponseStoreBytes int
	FFmpegPath         string
	VideoFPS           float64
	VideoMaxFrames     int
	// Browse surfaces (read-only): the datasets root (holding manifest.json) and
	// the RepoDB store path (for run/artifact browsing). Empty disables the
	// corresponding /datasets or /runs endpoint.
	DatasetsRoot string
	RepoDBPath   string
}

type slotRuntimeStats struct {
	promptTokens    atomic.Uint64
	cachedTokens    atomic.Uint64
	generatedTokens atomic.Uint64
	startedNanos    atomic.Int64
	promptNanos     atomic.Int64
	totalNanos      atomic.Int64

	textMu    sync.RWMutex
	prompt    string
	generated strings.Builder
	params    slotStatusParams
}

func (stats *slotRuntimeStats) reset(started time.Time) {
	stats.promptTokens.Store(0)
	stats.cachedTokens.Store(0)
	stats.generatedTokens.Store(0)
	stats.startedNanos.Store(started.UnixNano())
	stats.promptNanos.Store(0)
	stats.totalNanos.Store(0)
	stats.textMu.Lock()
	stats.prompt = ""
	stats.generated.Reset()
	stats.params = slotStatusParams{}
	stats.textMu.Unlock()
}

func (stats *slotRuntimeStats) beginGeneration(
	prompt string,
	options inference.GenerateOptions,
) {
	stats.textMu.Lock()
	stats.prompt = prompt
	if options.Sampler != nil {
		config := options.Sampler.Config()
		stats.params = slotStatusParams{
			Seed:             config.Seed,
			Temperature:      config.Temperature,
			DynatempRange:    config.DynatempRange,
			DynatempExponent: config.DynatempExponent,
			TopK:             config.TopK,
			TopP:             config.TopP,
			MinP:             config.MinP,
			TypicalP:         config.TypicalP,
			TopNSigma:        config.TopNSigma,
			XTCProbability:   config.XTCProbability,
			XTCThreshold:     config.XTCThreshold,
			MinKeep:          config.MinKeep,
			AdaptiveTarget:   config.AdaptiveTarget,
			AdaptiveDecay:    config.AdaptiveDecay,
			RepeatLastN:      config.RepeatLastN,
			RepeatPenalty:    config.RepeatPenalty,
			PresencePenalty:  config.PresencePenalty,
			FrequencyPenalty: config.FrequencyPenalty,
			DryMultiplier:    config.DryMultiplier,
			DryBase:          config.DryBase,
			DryAllowedLength: config.DryAllowedLength,
			DryPenaltyLastN:  config.DryPenaltyLastN,
			Mirostat:         config.Mirostat,
			MirostatTau:      config.MirostatTau,
			MirostatEta:      config.MirostatEta,
			MaxTokens:        options.MaxNewTokens,
			NPredict:         options.MaxNewTokens,
			Stop:             slices.Clone(options.StopSequences),
			Samplers:         slices.Clone(config.Samplers),
			CachePrompt:      options.CachePrompt,
			NKeep:            options.KeepTokens,
			NDiscard:         options.DiscardTokens,
			NCacheReuse:      options.MinCacheReuse,
			ContextShift:     options.ContextShift,
		}
	}
	stats.textMu.Unlock()
}

func (stats *slotRuntimeStats) appendGenerated(piece string) {
	stats.textMu.Lock()
	stats.generated.WriteString(piece)
	stats.textMu.Unlock()
}

func (stats *slotRuntimeStats) snapshot() (string, string, slotStatusParams) {
	stats.textMu.RLock()
	defer stats.textMu.RUnlock()
	params := stats.params
	params.Stop = slices.Clone(params.Stop)
	params.Samplers = slices.Clone(params.Samplers)
	return stats.prompt, stats.generated.String(), params
}

type Handler struct {
	config             Config
	generator          Generator
	generation         Generator
	continuous         *inference.ContinuousGenerator
	defaultSampling    sampling.Config
	slots              chan int
	slotMu             sync.Mutex
	slotBusy           []atomic.Bool
	slotTasks          []atomic.Uint64
	slotStats          []slotRuntimeStats
	nextTask           atomic.Uint64
	nextID             atomic.Uint64
	started            time.Time
	requestsTotal      atomic.Uint64
	requestsActive     atomic.Int64
	generationRequests atomic.Uint64
	generationErrors   atomic.Uint64
	generatedTokens    atomic.Uint64
	mediaFetcher       *remoteMediaFetcher
	responseHistory    *responseHistoryStore
	responseFiles      ResponseFileResolver
	thinkingSigner     *anthropicThinkingSigner
}

func New(config Config, generator Generator) (*Handler, error) {
	if generator == nil {
		return nil, errors.New("server: generator is nil")
	}
	if config.ModelID == "" {
		config.ModelID = DefaultModelID
	}
	if config.MaxTokens == 0 {
		config.MaxTokens = DefaultMaxTokens
	}
	if config.MaxTokens < 0 {
		return nil, errors.New("server: max tokens must be positive")
	}
	if config.MaxConcurrent == 0 {
		config.MaxConcurrent = DefaultMaxConcurrent
	}
	if config.MaxConcurrent < 0 {
		return nil, errors.New("server: max concurrent requests must be positive")
	}
	if config.MaxConcurrent > maxConcurrentRequests {
		return nil, fmt.Errorf("server: max concurrent requests exceeds %d", maxConcurrentRequests)
	}
	if config.MaxEmbeddingInputs == 0 {
		config.MaxEmbeddingInputs = DefaultMaxEmbeddingInputs
	}
	if config.MaxEmbeddingInputs < 0 {
		return nil, errors.New("server: max embedding inputs must be positive")
	}
	if config.DefaultTemperature == 0 {
		config.DefaultTemperature = DefaultSamplingTemperature
	}
	if config.DefaultTopP == 0 {
		config.DefaultTopP = DefaultSamplingTopP
	}
	if config.RequestTimeout < 0 {
		return nil, errors.New("server: request timeout must be non-negative")
	}
	if config.InfillBatchSize == 0 {
		config.InfillBatchSize = inference.DefaultInfillBatchSize
	}
	if config.InfillBatchSize < 0 {
		return nil, errors.New("server: infill batch size must be positive")
	}
	if (config.ResponseToolPolicy.Hosted != "" && config.ResponseToolPolicy.Hosted != "deny") ||
		(config.ResponseToolPolicy.Custom != "" && config.ResponseToolPolicy.Custom != "deny") {
		return nil, errors.New("server: response tool policy requires an external executor for non-deny modes")
	}
	if config.MaxStoredResponses == 0 {
		config.MaxStoredResponses = DefaultStoredResponses
	}
	if config.MaxStoredResponses < 0 || config.MaxStoredResponses > maxStoredResponses {
		return nil, fmt.Errorf("server: stored response count must be in [1,%d]", maxStoredResponses)
	}
	if config.ResponseStoreBytes == 0 {
		config.ResponseStoreBytes = DefaultResponseStoreBytes
	}
	if config.ResponseStoreBytes < 0 || config.ResponseStoreBytes > maxResponseStoreBytes {
		return nil, fmt.Errorf("server: response store bytes must be in [1,%d]", maxResponseStoreBytes)
	}
	if config.VideoFPS == 0 {
		config.VideoFPS = DefaultVideoFPS
	}
	if config.VideoFPS <= 0 || math.IsNaN(config.VideoFPS) || math.IsInf(config.VideoFPS, 0) {
		return nil, errors.New("server: video FPS must be finite and positive")
	}
	if config.VideoMaxFrames == 0 {
		config.VideoMaxFrames = DefaultVideoFrameLimit
	}
	if config.VideoMaxFrames < 0 || config.VideoMaxFrames > maxVideoFrameLimit {
		return nil, fmt.Errorf("server: video frame limit must be in [1,%d]", maxVideoFrameLimit)
	}
	mediaFetcher, err := newRemoteMediaFetcher(config.RemoteMediaPolicy)
	if err != nil {
		return nil, fmt.Errorf("server remote media policy: %w", err)
	}
	thinkingSigner, err := newAnthropicThinkingSigner()
	if err != nil {
		return nil, err
	}
	defaultSampler, err := sampling.New(sampling.Config{
		Temperature: config.DefaultTemperature,
		TopK:        config.DefaultTopK,
		TopP:        config.DefaultTopP,
	})
	if err != nil {
		return nil, fmt.Errorf("server defaults: %w", err)
	}
	slots := make(chan int, config.MaxConcurrent)
	for id := range config.MaxConcurrent {
		slots <- id
	}
	generation := generator
	var continuous *inference.ContinuousGenerator
	if config.MaxConcurrent > 1 {
		if factory, ok := generator.(ContinuousGeneratorFactory); ok {
			candidate, schedulerErr := factory.NewContinuousGenerator(
				inference.ContinuousGeneratorOptions{
					MaxSequences: config.MaxConcurrent,
					ContextShift: config.ContextShift,
				},
			)
			if schedulerErr == nil {
				continuous = candidate
				generation = candidate
			}
		}
	}
	return &Handler{
		config:          config,
		generator:       generator,
		generation:      generation,
		continuous:      continuous,
		defaultSampling: defaultSampler.Config(),
		slots:           slots,
		slotBusy:        make([]atomic.Bool, config.MaxConcurrent),
		slotTasks:       make([]atomic.Uint64, config.MaxConcurrent),
		slotStats:       make([]slotRuntimeStats, config.MaxConcurrent),
		started:         time.Now(),
		mediaFetcher:    mediaFetcher,
		responseHistory: newResponseHistoryStore(
			config.MaxStoredResponses,
			config.ResponseStoreBytes,
		),
		responseFiles:  config.ResponseFiles,
		thinkingSigner: thinkingSigner,
	}, nil
}

// Close: release fused scheduler state.
func (h *Handler) Close() error {
	if h == nil || h.continuous == nil {
		return nil
	}
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	return h.continuous.Close(ctx)
}

func (h *Handler) acquireSlot(requested int) (int, bool) {
	h.slotMu.Lock()
	defer h.slotMu.Unlock()
	id := -1
	if requested < 0 {
		select {
		case id = <-h.slots:
		default:
			return -1, false
		}
	} else {
		available := len(h.slots)
		others := make([]int, 0, available)
		for range available {
			candidate := <-h.slots
			if candidate == requested {
				id = candidate
			} else {
				others = append(others, candidate)
			}
		}
		for _, candidate := range others {
			h.slots <- candidate
		}
		if id < 0 {
			return -1, false
		}
	}
	h.slotTasks[id].Store(h.nextTask.Add(1))
	h.slotStats[id].reset(time.Now())
	h.slotBusy[id].Store(true)
	return id, true
}

func (h *Handler) acquireRequestSlot(response http.ResponseWriter, requested int) (int, bool) {
	id, acquired := h.acquireSlot(requested)
	if acquired {
		return id, true
	}
	response.Header().Set("Retry-After", "1")
	writeError(response, http.StatusTooManyRequests, "server_busy", "generation capacity is busy")
	return -1, false
}

func (h *Handler) releaseSlot(id int) {
	if id < 0 || id >= len(h.slotBusy) {
		return
	}
	h.slotMu.Lock()
	started := h.slotStats[id].startedNanos.Load()
	if started > 0 {
		h.slotStats[id].totalNanos.Store(max(time.Now().UnixNano()-started, 0))
	}
	h.slotBusy[id].Store(false)
	h.slots <- id
	h.slotMu.Unlock()
}

func (h *Handler) ServeHTTP(response http.ResponseWriter, request *http.Request) {
	if h.config.RequestTimeout > 0 {
		ctx, cancel := context.WithTimeout(request.Context(), h.config.RequestTimeout)
		defer cancel()
		request = request.WithContext(ctx)
	}
	h.requestsTotal.Add(1)
	h.requestsActive.Add(1)
	defer h.requestsActive.Add(-1)
	protectedV1 := strings.HasPrefix(request.URL.Path, "/v1/") &&
		request.URL.Path != "/v1/models" &&
		request.URL.Path != "/v1/health"
	if (protectedV1 ||
		request.URL.Path == "/apply-template" ||
		request.URL.Path == "/tokenize" ||
		request.URL.Path == "/detokenize" ||
		request.URL.Path == "/props" ||
		request.URL.Path == "/analyze/model" ||
		request.URL.Path == "/analyze/vocab" ||
		request.URL.Path == "/analyze/states" ||
		request.URL.Path == "/analyze/attention" ||
		request.URL.Path == "/datasets" ||
		request.URL.Path == "/runs" ||
		request.URL.Path == "/completion" ||
		request.URL.Path == "/completions" ||
		request.URL.Path == "/infill" ||
		request.URL.Path == "/responses" ||
		request.URL.Path == "/embedding" ||
		request.URL.Path == "/embeddings" ||
		request.URL.Path == "/rerank" ||
		request.URL.Path == "/reranking" ||
		request.URL.Path == "/slots" ||
		request.URL.Path == "/chat/completions" ||
		request.URL.Path == "/chat/completions/input_tokens" ||
		request.URL.Path == "/responses/input_tokens" ||
		request.URL.Path == "/lora-adapters") &&
		!h.authorized(request) {
		response.Header().Set("WWW-Authenticate", "Bearer")
		writeError(
			response,
			http.StatusUnauthorized,
			"invalid_api_key",
			"missing or invalid bearer token",
		)
		return
	}
	switch request.URL.Path {
	case "/health", "/healthz", "/v1/health":
		h.health(response, request)
	case "/metrics":
		h.metrics(response, request)
	case "/v1/models":
		h.models(response, request)
	case "/models":
		h.models(response, request)
	case "/v1/completions":
		h.completions(response, request)
	case "/completion", "/completions":
		h.nativeCompletions(response, request)
	case "/infill":
		h.infill(response, request)
	case "/v1/chat/completions":
		h.chatCompletions(response, request)
	case "/chat/completions":
		h.chatCompletions(response, request)
	case "/responses", "/v1/responses":
		h.responses(response, request)
	case "/chat/completions/input_tokens", "/v1/chat/completions/input_tokens":
		h.chatInputTokens(response, request)
	case "/responses/input_tokens", "/v1/responses/input_tokens":
		h.responsesInputTokens(response, request)
	case "/v1/messages/count_tokens":
		h.anthropicInputTokens(response, request)
	case "/v1/messages":
		h.anthropicMessages(response, request)
	case "/v1/embeddings":
		h.embeddings(response, request)
	case "/embedding", "/embeddings":
		h.nativeEmbeddings(response, request)
	case "/rerank", "/reranking", "/v1/rerank", "/v1/reranking":
		h.rerank(response, request)
	case "/apply-template":
		h.applyTemplate(response, request)
	case "/tokenize":
		h.tokenize(response, request)
	case "/detokenize":
		h.detokenize(response, request)
	case "/props":
		h.properties(response, request)
	case "/analyze/model":
		h.analyzeModel(response, request)
	case "/analyze/vocab":
		h.analyzeVocab(response, request)
	case "/analyze/states":
		h.analyzeStates(response, request)
	case "/analyze/attention":
		h.analyzeAttention(response, request)
	case "/datasets":
		h.browseDatasets(response, request)
	case "/runs":
		h.browseRuns(response, request)
	case "/slots":
		h.slotStatus(response, request)
	case "/lora-adapters":
		h.loraAdapters(response, request)
	default:
		h.serveWebUI(response, request)
	}
}

func rawJSONConfigured(raw json.RawMessage) bool {
	return len(raw) != 0 && string(raw) != "null"
}

func (h *Handler) newSampler(body samplingParameters) (*sampling.Sampler, error) {
	temperature := h.config.DefaultTemperature
	if body.Temperature != nil {
		temperature = *body.Temperature
	}
	topP := h.config.DefaultTopP
	if body.TopP != nil {
		topP = *body.TopP
	}
	topK := h.config.DefaultTopK
	if body.TopK != nil {
		topK = *body.TopK
	}
	adaptiveTarget := float32(-1)
	if body.AdaptiveTarget != nil {
		adaptiveTarget = *body.AdaptiveTarget
	}
	adaptiveDecay := float32(0.9)
	if body.AdaptiveDecay != nil {
		adaptiveDecay = *body.AdaptiveDecay
	}
	var dryBreakers [][]int
	breakerStrings := body.DryBreakers
	if body.DryMultiplier != 0 && breakerStrings == nil {
		breakerStrings = []string{"\n", ":", "\"", "*"}
	}
	if len(breakerStrings) > 0 {
		breakerTokenizer, ok := h.generator.(DryBreakerTokenizer)
		if !ok {
			return nil, errors.New("server: generator does not support DRY string breakers")
		}
		var err error
		dryBreakers, err = breakerTokenizer.TokenizeDryBreakers(breakerStrings)
		if err != nil {
			return nil, fmt.Errorf("server: tokenize DRY breakers: %w", err)
		}
	}
	var grammar *sampling.TokenGrammar
	if len(body.GrammarChoices) > 0 {
		grammarTokenizer, ok := h.generator.(GrammarChoiceTokenizer)
		if !ok {
			return nil, errors.New("server: generator does not support grammar choices")
		}
		var err error
		grammar, err = grammarTokenizer.TokenizeGrammarChoices(body.GrammarChoices)
		if err != nil {
			return nil, fmt.Errorf("server: tokenize grammar choices: %w", err)
		}
	}
	var gbnf *sampling.GBNFGrammar
	if body.Grammar != "" {
		var err error
		if body.GrammarLazy {
			compiler, ok := h.generator.(LazyGBNFCompiler)
			if !ok {
				return nil, errors.New("server: generator does not support lazy GBNF")
			}
			triggerTokens := make([]tokenizer.TokenID, len(body.GrammarTriggerTokens))
			for index, token := range body.GrammarTriggerTokens {
				if token < 0 {
					return nil, errors.New("server: grammar trigger token is negative")
				}
				triggerTokens[index] = tokenizer.TokenID(token)
			}
			gbnf, err = compiler.CompileLazyGBNF(
				body.Grammar,
				body.GrammarRoot,
				body.GrammarTriggerPatterns,
				triggerTokens,
			)
		} else {
			if len(body.GrammarTriggerPatterns) > 0 ||
				len(body.GrammarTriggerTokens) > 0 {
				return nil, errors.New("server: grammar triggers require grammar_lazy")
			}
			compiler, ok := h.generator.(GBNFCompiler)
			if !ok {
				return nil, errors.New("server: generator does not support GBNF")
			}
			gbnf, err = compiler.CompileGBNF(body.Grammar, body.GrammarRoot)
		}
		if err != nil {
			return nil, fmt.Errorf("server: compile GBNF: %w", err)
		}
	} else if body.GrammarLazy ||
		len(body.GrammarTriggerPatterns) > 0 ||
		len(body.GrammarTriggerTokens) > 0 {
		return nil, errors.New("server: lazy grammar options require grammar")
	}
	var samplerOrder []sampling.SamplerStage
	if body.Samplers != nil {
		var err error
		samplerOrder, err = sampling.ParseSamplerNames(body.Samplers)
		if err != nil {
			return nil, fmt.Errorf("server: parse samplers: %w", err)
		}
	}
	var infillVocabulary *sampling.InfillVocabulary
	if slices.Contains(samplerOrder, sampling.SamplerInfill) {
		provider, ok := h.generator.(InfillVocabularyProvider)
		if !ok {
			return nil, errors.New("server: generator does not expose infill vocabulary")
		}
		var err error
		infillVocabulary, err = provider.SamplingInfillVocabulary()
		if err != nil {
			return nil, fmt.Errorf("server: load infill vocabulary: %w", err)
		}
	}
	logitBiases, err := h.parseLogitBias(body.LogitBias)
	if err != nil {
		return nil, err
	}
	if body.IgnoreEOS {
		vocabulary, ok := h.generator.(SamplingVocabulary)
		if !ok {
			return nil, errors.New("server: generator does not expose EOG tokens")
		}
		for _, token := range vocabulary.SamplingEOGTokens() {
			logitBiases = append(logitBiases, sampling.LogitBias{
				Token: int(token),
				Bias:  float32(math.Inf(-1)),
			})
		}
	}
	return sampling.New(sampling.Config{
		Temperature:      temperature,
		DynatempRange:    body.DynatempRange,
		DynatempExponent: body.DynatempExponent,
		TopK:             topK,
		TopP:             topP,
		MinP:             body.MinP,
		TypicalP:         body.TypicalP,
		TopNSigma:        body.TopNSigma,
		XTCProbability:   body.XTCProbability,
		XTCThreshold:     body.XTCThreshold,
		MinKeep:          body.MinKeep,
		AdaptiveTarget:   adaptiveTarget,
		AdaptiveDecay:    adaptiveDecay,
		RepeatLastN:      body.RepeatLastN,
		RepeatPenalty:    body.RepeatPenalty,
		PresencePenalty:  body.PresencePenalty,
		FrequencyPenalty: body.FrequencyPenalty,
		DryMultiplier:    body.DryMultiplier,
		DryBase:          body.DryBase,
		DryAllowedLength: body.DryAllowedLength,
		DryPenaltyLastN:  body.DryPenaltyLastN,
		DryBreakers:      dryBreakers,
		Mirostat:         body.Mirostat,
		MirostatTau:      body.MirostatTau,
		MirostatEta:      body.MirostatEta,
		Seed:             body.Seed,
		Grammar:          grammar,
		GBNF:             gbnf,
		Samplers:         samplerOrder,
		LogitBiases:      logitBiases,
		Infill:           infillVocabulary,
	})
}

func samplerForChoice(base *sampling.Sampler, index int) (*sampling.Sampler, error) {
	if index == 0 {
		return base, nil
	}
	config := base.Config()
	config.Seed += int64(index)
	return sampling.New(config)
}

func (h *Handler) parseLogitBias(raw json.RawMessage) ([]sampling.LogitBias, error) {
	if len(raw) == 0 || string(raw) == "null" {
		return nil, nil
	}
	vocabulary, _ := h.generator.(SamplingVocabulary)
	add := func(
		result []sampling.LogitBias,
		key json.RawMessage,
		value json.RawMessage,
	) ([]sampling.LogitBias, error) {
		bias, err := parseBiasValue(value)
		if err != nil {
			return nil, err
		}
		var token int
		if err := json.Unmarshal(key, &token); err == nil {
			if token < 0 ||
				(vocabulary != nil && token >= vocabulary.SamplingVocabularySize()) {
				return nil, fmt.Errorf("server: logit-bias token %d is out of range", token)
			}
			return append(result, sampling.LogitBias{Token: token, Bias: bias}), nil
		}
		var text string
		if err := json.Unmarshal(key, &text); err != nil {
			return nil, errors.New("server: logit-bias key must be a token ID or string")
		}
		if vocabulary == nil {
			return nil, errors.New("server: generator cannot tokenize logit-bias text")
		}
		tokens, err := vocabulary.TokenizeSamplingText(text)
		if err != nil {
			return nil, fmt.Errorf("server: tokenize logit-bias text: %w", err)
		}
		for _, id := range tokens {
			result = append(result, sampling.LogitBias{Token: int(id), Bias: bias})
		}
		return result, nil
	}

	var pairs []json.RawMessage
	if err := json.Unmarshal(raw, &pairs); err == nil {
		result := make([]sampling.LogitBias, 0, len(pairs))
		for index, encodedPair := range pairs {
			var pair []json.RawMessage
			if err := json.Unmarshal(encodedPair, &pair); err != nil || len(pair) != 2 {
				return nil, fmt.Errorf("server: logit-bias pair %d must contain key and bias", index)
			}
			result, err = add(result, pair[0], pair[1])
			if err != nil {
				return nil, fmt.Errorf("server: logit-bias pair %d: %w", index, err)
			}
		}
		return result, nil
	}

	var object map[string]json.RawMessage
	if err := json.Unmarshal(raw, &object); err != nil {
		return nil, errors.New("server: logit_bias must be an array or object")
	}
	keys := make([]string, 0, len(object))
	for key := range object {
		keys = append(keys, key)
	}
	sort.Strings(keys)
	result := make([]sampling.LogitBias, 0, len(keys))
	for _, key := range keys {
		var encodedKey []byte
		if token, err := strconv.Atoi(key); err == nil {
			encodedKey, _ = json.Marshal(token)
		} else {
			encodedKey, _ = json.Marshal(key)
		}
		var err error
		result, err = add(result, encodedKey, object[key])
		if err != nil {
			return nil, fmt.Errorf("server: logit-bias key %q: %w", key, err)
		}
	}
	return result, nil
}

func parseBiasValue(raw json.RawMessage) (float32, error) {
	var enabled bool
	if err := json.Unmarshal(raw, &enabled); err == nil {
		if enabled {
			return 0, errors.New("server: true is not a valid logit bias")
		}
		return float32(math.Inf(-1)), nil
	}
	var value float32
	if err := json.Unmarshal(raw, &value); err != nil ||
		math.IsNaN(float64(value)) ||
		math.IsInf(float64(value), 0) {
		return 0, errors.New("server: logit bias must be a finite number or false")
	}
	return value, nil
}

type completionChoice struct {
	Text         string `json:"text"`
	Index        int    `json:"index"`
	FinishReason string `json:"finish_reason"`
}

type completionUsage struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
	TotalTokens      int `json:"total_tokens"`
}

type completionResponse struct {
	ID      string             `json:"id"`
	Object  string             `json:"object"`
	Created int64              `json:"created"`
	Model   string             `json:"model"`
	Choices []completionChoice `json:"choices"`
	Usage   completionUsage    `json:"usage"`
}

func (h *Handler) complete(
	response http.ResponseWriter,
	plan *protocolBatchGenerationPlan,
	id string,
	n int,
) {
	choices := make([]completionChoice, 0, len(plan.prompts)*n)
	promptTokens := 0
	totalCompletionTokens := 0
	err := plan.run(n, nil, func(
		promptChoice, choiceIndex int,
		result protocolGenerationResult,
	) error {
		if promptChoice == 0 {
			promptTokens += result.promptTokens()
		}
		finishReason := result.pump.finishReason(plan.maxTokens, "stop", "length")
		totalCompletionTokens += result.pump.completion
		choices = append(choices, completionChoice{
			Text:         result.pump.text(),
			Index:        choiceIndex,
			FinishReason: finishReason,
		})
		return nil
	})
	if err != nil {
		writeGenerationError(response, err)
		return
	}
	writeJSON(response, http.StatusOK, completionResponse{
		ID:      id,
		Object:  "text_completion",
		Created: time.Now().Unix(),
		Model:   h.config.ModelID,
		Choices: choices,
		Usage: completionUsage{
			PromptTokens:     promptTokens,
			CompletionTokens: totalCompletionTokens,
			TotalTokens:      promptTokens + totalCompletionTokens,
		},
	})
}

type streamChoice struct {
	Text         string  `json:"text"`
	Index        int     `json:"index"`
	FinishReason *string `json:"finish_reason"`
}

type streamResponse struct {
	ID      string         `json:"id"`
	Object  string         `json:"object"`
	Created int64          `json:"created"`
	Model   string         `json:"model"`
	Choices []streamChoice `json:"choices"`
}

func (h *Handler) streamCompletion(
	response http.ResponseWriter,
	request *http.Request,
	plan *protocolBatchGenerationPlan,
	id string,
	n int,
) {
	flusher, ok := beginSSE(response)
	if !ok {
		return
	}
	stream := newSSEEmitter(request.Context(), response, flusher)
	created := time.Now().Unix()
	err := plan.run(
		n,
		func(choiceIndex int, piece string) error {
			if piece == "" {
				return request.Context().Err()
			}
			chunk := streamResponse{
				ID:      id,
				Object:  "text_completion",
				Created: created,
				Model:   h.config.ModelID,
				Choices: []streamChoice{{
					Text:  piece,
					Index: choiceIndex,
				}},
			}
			return stream.write(chunk)
		},
		func(_ int, choiceIndex int, result protocolGenerationResult) error {
			reason := result.pump.finishReason(plan.maxTokens, "stop", "length")
			_ = stream.write(streamResponse{
				ID:      id,
				Object:  "text_completion",
				Created: created,
				Model:   h.config.ModelID,
				Choices: []streamChoice{{
					Index:        choiceIndex,
					FinishReason: &reason,
				}},
			})
			return nil
		},
	)
	if err != nil {
		_ = emitGenerationError(stream.write, err)
	}
	_ = stream.done()
}
