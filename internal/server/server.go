package server

import (
	"bytes"

	"context"

	"crypto/subtle"

	"encoding/base64"

	"encoding/binary"

	"encoding/json"

	"errors"

	"fmt"

	"image"

	"io"

	"llamacpp2go/internal/cuda/driver"

	"llamacpp2go/internal/inference"

	"llamacpp2go/internal/projector"

	"llamacpp2go/internal/sampling"

	"llamacpp2go/internal/tokenizer"

	"math"

	"net/http"

	"slices"

	"sort"

	"strconv"

	"strings"

	"sync"

	"sync/atomic"

	"time"

	"unicode/utf8"

	_ "image/gif"

	_ "image/jpeg"

	_ "image/png"
)

const (
	maxRequestBytes           = 1 << 20
	maxMultimodalRequestBytes = 32 << 20
	maxImageBytes             = 16 << 20
	maxMediaBytes             = 24 << 20
	maxImageDimension         = 16384
	maxImagePixels            = 16 << 20
	maxRequestImagePixels     = 32 << 20
)

type Generator interface {
	Generate(
		context.Context,
		string,
		inference.GenerateOptions,
	) ([]tokenizer.TokenID, string, error)
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
			Stop:             append([]string(nil), options.StopSequences...),
			Samplers:         append([]sampling.SamplerStage(nil), config.Samplers...),
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
	params.Stop = append([]string(nil), params.Stop...)
	params.Samplers = append([]sampling.SamplerStage(nil), params.Samplers...)
	return stats.prompt, stats.generated.String(), params
}

type Handler struct {
	config             Config
	generator          Generator
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
}

func New(config Config, generator Generator) (*Handler, error) {
	if generator == nil {
		return nil, errors.New("server: generator is nil")
	}
	if config.ModelID == "" {
		config.ModelID = "llamacpp2go"
	}
	if config.MaxTokens == 0 {
		config.MaxTokens = 4096
	}
	if config.MaxTokens < 0 {
		return nil, errors.New("server: max tokens must be positive")
	}
	if config.MaxConcurrent == 0 {
		config.MaxConcurrent = 1
	}
	if config.MaxConcurrent < 0 {
		return nil, errors.New("server: max concurrent requests must be positive")
	}
	if config.MaxConcurrent > 1<<16 {
		return nil, errors.New("server: max concurrent requests exceeds 65536")
	}
	if config.MaxEmbeddingInputs == 0 {
		config.MaxEmbeddingInputs = 16
	}
	if config.MaxEmbeddingInputs < 0 {
		return nil, errors.New("server: max embedding inputs must be positive")
	}
	if config.DefaultTemperature == 0 {
		config.DefaultTemperature = 1
	}
	if config.DefaultTopP == 0 {
		config.DefaultTopP = 1
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
	return &Handler{
		config:          config,
		generator:       generator,
		defaultSampling: defaultSampler.Config(),
		slots:           slots,
		slotBusy:        make([]atomic.Bool, config.MaxConcurrent),
		slotTasks:       make([]atomic.Uint64, config.MaxConcurrent),
		slotStats:       make([]slotRuntimeStats, config.MaxConcurrent),
		started:         time.Now(),
	}, nil
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
	case "/slots":
		h.slotStatus(response, request)
	case "/lora-adapters":
		h.loraAdapters(response, request)
	default:
		writeError(response, http.StatusNotFound, "not_found", "route not found")
	}
}

type propertiesSamplingParams struct {
	NPredict         int                     `json:"n_predict"`
	Seed             int64                   `json:"seed"`
	Temperature      float32                 `json:"temperature"`
	DynatempRange    float32                 `json:"dynatemp_range"`
	DynatempExponent float32                 `json:"dynatemp_exponent"`
	TopK             int                     `json:"top_k"`
	TopP             float32                 `json:"top_p"`
	MinP             float32                 `json:"min_p"`
	TopNSigma        float32                 `json:"top_n_sigma"`
	XTCProbability   float32                 `json:"xtc_probability"`
	XTCThreshold     float32                 `json:"xtc_threshold"`
	TypicalP         float32                 `json:"typical_p"`
	RepeatLastN      int                     `json:"repeat_last_n"`
	RepeatPenalty    float32                 `json:"repeat_penalty"`
	PresencePenalty  float32                 `json:"presence_penalty"`
	FrequencyPenalty float32                 `json:"frequency_penalty"`
	DryMultiplier    float32                 `json:"dry_multiplier"`
	DryBase          float32                 `json:"dry_base"`
	DryAllowedLength int                     `json:"dry_allowed_length"`
	DryPenaltyLastN  int                     `json:"dry_penalty_last_n"`
	DryBreakers      []string                `json:"dry_sequence_breakers"`
	Mirostat         int                     `json:"mirostat"`
	MirostatTau      float32                 `json:"mirostat_tau"`
	MirostatEta      float32                 `json:"mirostat_eta"`
	Stop             []string                `json:"stop"`
	MaxTokens        int                     `json:"max_tokens"`
	IgnoreEOS        bool                    `json:"ignore_eos"`
	Stream           bool                    `json:"stream"`
	MinKeep          int                     `json:"min_keep"`
	Grammar          string                  `json:"grammar"`
	Samplers         []sampling.SamplerStage `json:"samplers"`
}

type propertiesModelMetadata struct {
	Name              string `json:"name"`
	Architecture      string `json:"architecture"`
	ContextLength     uint32 `json:"context_length"`
	EmbeddingLength   uint32 `json:"embedding_length"`
	FeedForwardLength uint32 `json:"feed_forward_length"`
	BlockCount        uint32 `json:"block_count"`
	HeadCount         uint32 `json:"head_count"`
	HeadCountKV       uint32 `json:"head_count_kv"`
	VocabularySize    uint32 `json:"vocabulary_size"`
}

type propertiesResponse struct {
	DefaultGenerationSettings struct {
		Params propertiesSamplingParams `json:"params"`
		NCtx   uint32                   `json:"n_ctx"`
	} `json:"default_generation_settings"`
	TotalSlots       int                     `json:"total_slots"`
	ModelAlias       string                  `json:"model_alias"`
	ModelFType       string                  `json:"model_ftype"`
	ModelPath        string                  `json:"model_path"`
	ModelMetadata    propertiesModelMetadata `json:"model_metadata"`
	Modalities       map[string]bool         `json:"modalities"`
	MediaMarker      string                  `json:"media_marker"`
	EndpointSlots    bool                    `json:"endpoint_slots"`
	EndpointProps    bool                    `json:"endpoint_props"`
	EndpointMetrics  bool                    `json:"endpoint_metrics"`
	UI               bool                    `json:"ui"`
	UISettings       map[string]any          `json:"ui_settings"`
	ChatTemplate     string                  `json:"chat_template"`
	ChatTemplateCaps map[string]bool         `json:"chat_template_caps"`
	BOSToken         string                  `json:"bos_token"`
	EOSToken         string                  `json:"eos_token"`
	BuildInfo        string                  `json:"build_info"`
	IsSleeping       bool                    `json:"is_sleeping"`
	CORSProxyEnabled bool                    `json:"cors_proxy_enabled"`
}

func (h *Handler) properties(response http.ResponseWriter, request *http.Request) {
	if request.Method != http.MethodGet {
		response.Header().Set("Allow", http.MethodGet)
		writeError(response, http.StatusMethodNotAllowed, "method_not_allowed", "GET required")
		return
	}
	api, ok := h.generator.(ModelPropertiesAPI)
	if !ok {
		writeError(response, http.StatusNotImplemented, "unsupported_operation", "model properties are unavailable")
		return
	}
	model := api.ModelProperties()
	samplingConfig := h.defaultSampling
	result := propertiesResponse{
		TotalSlots: h.config.MaxConcurrent,
		ModelAlias: h.config.ModelID,
		ModelFType: model.FileType,
		ModelPath:  model.Path,
		ModelMetadata: propertiesModelMetadata{
			Name:              model.Name,
			Architecture:      model.Architecture,
			ContextLength:     model.ContextLength,
			EmbeddingLength:   model.EmbeddingLength,
			FeedForwardLength: model.FeedForwardLength,
			BlockCount:        model.BlockCount,
			HeadCount:         model.HeadCount,
			HeadCountKV:       model.HeadCountKV,
			VocabularySize:    model.VocabularySize,
		},
		Modalities: map[string]bool{
			"vision": false,
			"video":  false,
			"audio":  false,
		},
		MediaMarker:      "",
		EndpointSlots:    true,
		EndpointProps:    false,
		EndpointMetrics:  true,
		UI:               false,
		UISettings:       map[string]any{},
		ChatTemplate:     model.ChatTemplate,
		ChatTemplateCaps: map[string]bool{},
		BOSToken:         model.BOSToken,
		EOSToken:         model.EOSToken,
		BuildInfo:        "llamacpp2go",
		IsSleeping:       false,
	}
	result.DefaultGenerationSettings.NCtx = model.ContextLength
	result.DefaultGenerationSettings.Params = propertiesSamplingParams{
		NPredict:         16,
		Seed:             samplingConfig.Seed,
		Temperature:      samplingConfig.Temperature,
		DynatempRange:    samplingConfig.DynatempRange,
		DynatempExponent: samplingConfig.DynatempExponent,
		TopK:             samplingConfig.TopK,
		TopP:             samplingConfig.TopP,
		MinP:             samplingConfig.MinP,
		TopNSigma:        samplingConfig.TopNSigma,
		XTCProbability:   samplingConfig.XTCProbability,
		XTCThreshold:     samplingConfig.XTCThreshold,
		TypicalP:         samplingConfig.TypicalP,
		RepeatLastN:      samplingConfig.RepeatLastN,
		RepeatPenalty:    samplingConfig.RepeatPenalty,
		PresencePenalty:  samplingConfig.PresencePenalty,
		FrequencyPenalty: samplingConfig.FrequencyPenalty,
		DryMultiplier:    samplingConfig.DryMultiplier,
		DryBase:          samplingConfig.DryBase,
		DryAllowedLength: samplingConfig.DryAllowedLength,
		DryPenaltyLastN:  samplingConfig.DryPenaltyLastN,
		DryBreakers:      []string{},
		Mirostat:         samplingConfig.Mirostat,
		MirostatTau:      samplingConfig.MirostatTau,
		MirostatEta:      samplingConfig.MirostatEta,
		Stop:             []string{},
		MaxTokens:        16,
		IgnoreEOS:        false,
		Stream:           false,
		MinKeep:          samplingConfig.MinKeep,
		Grammar:          "",
		Samplers:         append([]sampling.SamplerStage(nil), samplingConfig.Samplers...),
	}
	writeJSON(response, http.StatusOK, result)
}

type slotStatusItem struct {
	ID                     int                `json:"id"`
	IDTask                 *uint64            `json:"id_task,omitempty"`
	NCtx                   uint32             `json:"n_ctx"`
	Speculative            bool               `json:"speculative"`
	IsProcessing           bool               `json:"is_processing"`
	NPromptTokens          uint64             `json:"n_prompt_tokens,omitempty"`
	NPromptTokensProcessed uint64             `json:"n_prompt_tokens_processed,omitempty"`
	NPromptTokensCache     uint64             `json:"n_prompt_tokens_cache,omitempty"`
	Prompt                 string             `json:"prompt,omitempty"`
	Generated              string             `json:"generated,omitempty"`
	Params                 *slotStatusParams  `json:"params,omitempty"`
	NextToken              *slotNextToken     `json:"next_token,omitempty"`
	Timings                *slotStatusTimings `json:"timings,omitempty"`
}

type slotNextToken struct {
	HasNextToken bool   `json:"has_next_token"`
	HasNewLine   bool   `json:"has_new_line"`
	NRemain      int    `json:"n_remain"`
	NDecoded     uint64 `json:"n_decoded"`
}

type slotStatusParams struct {
	Seed             int64                   `json:"seed"`
	Temperature      float32                 `json:"temperature"`
	DynatempRange    float32                 `json:"dynatemp_range"`
	DynatempExponent float32                 `json:"dynatemp_exponent"`
	TopK             int                     `json:"top_k"`
	TopP             float32                 `json:"top_p"`
	MinP             float32                 `json:"min_p"`
	TypicalP         float32                 `json:"typical_p"`
	TopNSigma        float32                 `json:"top_n_sigma"`
	XTCProbability   float32                 `json:"xtc_probability"`
	XTCThreshold     float32                 `json:"xtc_threshold"`
	MinKeep          int                     `json:"min_keep"`
	AdaptiveTarget   float32                 `json:"adaptive_target"`
	AdaptiveDecay    float32                 `json:"adaptive_decay"`
	RepeatLastN      int                     `json:"repeat_last_n"`
	RepeatPenalty    float32                 `json:"repeat_penalty"`
	PresencePenalty  float32                 `json:"presence_penalty"`
	FrequencyPenalty float32                 `json:"frequency_penalty"`
	DryMultiplier    float32                 `json:"dry_multiplier"`
	DryBase          float32                 `json:"dry_base"`
	DryAllowedLength int                     `json:"dry_allowed_length"`
	DryPenaltyLastN  int                     `json:"dry_penalty_last_n"`
	Mirostat         int                     `json:"mirostat"`
	MirostatTau      float32                 `json:"mirostat_tau"`
	MirostatEta      float32                 `json:"mirostat_eta"`
	MaxTokens        int                     `json:"max_tokens"`
	NPredict         int                     `json:"n_predict"`
	Stop             []string                `json:"stop"`
	Samplers         []sampling.SamplerStage `json:"samplers"`
	CachePrompt      bool                    `json:"cache_prompt"`
	NKeep            int                     `json:"n_keep"`
	NDiscard         int                     `json:"n_discard"`
	NCacheReuse      int                     `json:"n_cache_reuse"`
	ContextShift     bool                    `json:"context_shift"`
}

type slotStatusTimings struct {
	CacheN              uint64  `json:"cache_n"`
	PromptN             uint64  `json:"prompt_n"`
	PromptMS            float64 `json:"prompt_ms"`
	PromptPerTokenMS    float64 `json:"prompt_per_token_ms"`
	PromptPerSecond     float64 `json:"prompt_per_second"`
	PredictedN          uint64  `json:"predicted_n"`
	PredictedMS         float64 `json:"predicted_ms"`
	PredictedPerTokenMS float64 `json:"predicted_per_token_ms"`
	PredictedPerSecond  float64 `json:"predicted_per_second"`
}

func perTokenAndRate(tokens uint64, durationMS float64) (float64, float64) {
	if tokens == 0 || durationMS <= 0 {
		return 0, 0
	}
	return durationMS / float64(tokens), 1000 * float64(tokens) / durationMS
}

func (h *Handler) slotStatus(response http.ResponseWriter, request *http.Request) {
	if request.Method != http.MethodGet {
		response.Header().Set("Allow", http.MethodGet)
		writeError(response, http.StatusMethodNotAllowed, "method_not_allowed", "GET required")
		return
	}
	if request.URL.Query().Has("fail_on_no_slot") && len(h.slots) == 0 {
		writeError(response, http.StatusServiceUnavailable, "server_busy", "no slot available")
		return
	}
	contextLength := uint32(0)
	if api, ok := h.generator.(ModelPropertiesAPI); ok {
		contextLength = api.ModelProperties().ContextLength
	}
	result := make([]slotStatusItem, len(h.slotBusy))
	for id := range h.slotBusy {
		processing := h.slotBusy[id].Load()
		stats := &h.slotStats[id]
		task := h.slotTasks[id].Load()
		promptTokens := stats.promptTokens.Load()
		cachedTokens := min(stats.cachedTokens.Load(), promptTokens)
		processedTokens := promptTokens - cachedTokens
		generatedTokens := stats.generatedTokens.Load()
		promptNanos := max(stats.promptNanos.Load(), 0)
		totalNanos := max(stats.totalNanos.Load(), 0)
		if processing {
			started := stats.startedNanos.Load()
			if started > 0 {
				totalNanos = max(time.Now().UnixNano()-started, 0)
			}
		}
		predictedNanos := max(totalNanos-promptNanos, 0)
		promptMS := float64(promptNanos) / float64(time.Millisecond)
		predictedMS := float64(predictedNanos) / float64(time.Millisecond)
		promptPerTokenMS, promptPerSecond := perTokenAndRate(processedTokens, promptMS)
		predictedPerTokenMS, predictedPerSecond := perTokenAndRate(generatedTokens, predictedMS)
		prompt, generated, params := stats.snapshot()
		result[id] = slotStatusItem{
			ID:                     id,
			NCtx:                   contextLength,
			Speculative:            false,
			IsProcessing:           processing,
			NPromptTokens:          promptTokens,
			NPromptTokensProcessed: processedTokens,
			NPromptTokensCache:     cachedTokens,
			Prompt:                 prompt,
			Generated:              generated,
		}
		if processing && task > 0 {
			result[id].IDTask = &task
		}
		if task > 0 {
			result[id].Params = &params
			result[id].NextToken = &slotNextToken{
				// Generate drives sampling synchronously and never leaves
				// sampled token buffered between calls
				HasNextToken: false,
				HasNewLine:   false,
				NRemain:      max(params.MaxTokens-int(generatedTokens), 0),
				NDecoded:     generatedTokens,
			}
			result[id].Timings = &slotStatusTimings{
				CacheN:              cachedTokens,
				PromptN:             processedTokens,
				PromptMS:            promptMS,
				PromptPerTokenMS:    promptPerTokenMS,
				PromptPerSecond:     promptPerSecond,
				PredictedN:          generatedTokens,
				PredictedMS:         predictedMS,
				PredictedPerTokenMS: predictedPerTokenMS,
				PredictedPerSecond:  predictedPerSecond,
			}
		}
	}
	writeJSON(response, http.StatusOK, result)
}

func (h *Handler) loraAdapters(response http.ResponseWriter, request *http.Request) {
	switch request.Method {
	case http.MethodGet:
		if controller, ok := h.generator.(LoRAControlAPI); ok {
			writeJSON(response, http.StatusOK, controller.LoRAAdapters())
			return
		}
		writeJSON(response, http.StatusOK, []any{})
	case http.MethodPost:
		request.Body = http.MaxBytesReader(response, request.Body, maxRequestBytes)
		decoder := json.NewDecoder(request.Body)
		decoder.DisallowUnknownFields()
		var adapters []inference.LoRAScale
		if err := decoder.Decode(&adapters); err != nil {
			writeError(response, http.StatusBadRequest, "invalid_request_error", "invalid JSON request: "+err.Error())
			return
		}
		if err := requireEOF(decoder); err != nil {
			writeError(response, http.StatusBadRequest, "invalid_request_error", err.Error())
			return
		}
		controller, ok := h.generator.(LoRAControlAPI)
		if !ok && len(adapters) != 0 {
			writeError(
				response,
				http.StatusNotImplemented,
				"unsupported_operation",
				"LoRA adapter loading and execution are unavailable",
			)
			return
		}
		if ok {
			if err := controller.SetLoRAScales(adapters); err != nil {
				writeError(response, http.StatusBadRequest, "invalid_request_error", err.Error())
				return
			}
		}
		writeJSON(response, http.StatusOK, map[string]bool{"success": true})
	default:
		response.Header().Set("Allow", http.MethodGet+", "+http.MethodPost)
		writeError(response, http.StatusMethodNotAllowed, "method_not_allowed", "GET or POST required")
	}
}

type tokenizeRequest struct {
	Content      json.RawMessage `json:"content"`
	AddSpecial   bool            `json:"add_special"`
	ParseSpecial *bool           `json:"parse_special"`
	WithPieces   bool            `json:"with_pieces"`
}

type tokenPieceResponse struct {
	ID    tokenizer.TokenID `json:"id"`
	Piece any               `json:"piece"`
}

func (h *Handler) tokenize(response http.ResponseWriter, request *http.Request) {
	if request.Method != http.MethodPost {
		response.Header().Set("Allow", http.MethodPost)
		writeError(response, http.StatusMethodNotAllowed, "method_not_allowed", "POST required")
		return
	}
	api, ok := h.generator.(TokenizationAPI)
	if !ok {
		writeError(response, http.StatusNotImplemented, "unsupported_operation", "tokenization is unavailable")
		return
	}
	var body tokenizeRequest
	if !h.decodeBoundedJSON(response, request, &body) {
		return
	}
	parseSpecial := true
	if body.ParseSpecial != nil {
		parseSpecial = *body.ParseSpecial
	}
	tokens, err := tokenizeMixed(api, body.Content, body.AddSpecial, parseSpecial)
	if err != nil {
		writeError(response, http.StatusBadRequest, "invalid_request_error", err.Error())
		return
	}
	if body.WithPieces {
		pieceAPI, ok := h.generator.(TokenPieceAPI)
		if !ok {
			writeError(response, http.StatusNotImplemented, "unsupported_operation", "token pieces are unavailable")
			return
		}
		pieces := make([]tokenPieceResponse, len(tokens))
		for index, token := range tokens {
			piece, err := pieceAPI.TokenPiece(token)
			if err != nil {
				writeError(response, http.StatusBadRequest, "invalid_request_error", err.Error())
				return
			}
			var rendered any = piece
			if !utf8.ValidString(piece) {
				values := make([]int, len(piece))
				for byteIndex := range piece {
					values[byteIndex] = int(piece[byteIndex])
				}
				rendered = values
			}
			pieces[index] = tokenPieceResponse{ID: token, Piece: rendered}
		}
		writeJSON(response, http.StatusOK, map[string]any{"tokens": pieces})
		return
	}
	writeJSON(response, http.StatusOK, map[string]any{"tokens": tokens})
}

func tokenizeMixed(
	api TokenizationAPI,
	raw json.RawMessage,
	addSpecial, parseSpecial bool,
) ([]tokenizer.TokenID, error) {
	if len(raw) == 0 {
		return []tokenizer.TokenID{}, nil
	}
	var text string
	if err := json.Unmarshal(raw, &text); err == nil {
		tokens, err := api.TokenizeText(text, addSpecial, parseSpecial)
		if err != nil {
			return nil, err
		}
		if len(tokens) > 1<<20 {
			return nil, errors.New("token count exceeds 1048576")
		}
		return tokens, nil
	}
	var parts []json.RawMessage
	if err := json.Unmarshal(raw, &parts); err != nil {
		return nil, errors.New("content must be a string or token/string sequence")
	}
	tokens := make([]tokenizer.TokenID, 0, len(parts))
	first := true
	for index, part := range parts {
		var token int64
		if err := json.Unmarshal(part, &token); err == nil {
			if token < 0 || token >= int64(api.SamplingVocabularySize()) {
				return nil, fmt.Errorf("content token %d is out of range", token)
			}
			tokens = append(tokens, tokenizer.TokenID(token))
		} else {
			var partText string
			if err := json.Unmarshal(part, &partText); err != nil {
				return nil, fmt.Errorf(
					"content element %d must be an integer token ID or string",
					index,
				)
			}
			expanded, err := api.TokenizeText(partText, addSpecial && first, parseSpecial)
			if err != nil {
				return nil, fmt.Errorf("tokenize content element %d: %w", index, err)
			}
			tokens = append(tokens, expanded...)
		}
		first = false
		if len(tokens) > 1<<20 {
			return nil, errors.New("token count exceeds 1048576")
		}
	}
	return tokens, nil
}

type detokenizeRequest struct {
	Tokens []int `json:"tokens"`
}

func (h *Handler) detokenize(response http.ResponseWriter, request *http.Request) {
	if request.Method != http.MethodPost {
		response.Header().Set("Allow", http.MethodPost)
		writeError(response, http.StatusMethodNotAllowed, "method_not_allowed", "POST required")
		return
	}
	api, ok := h.generator.(TokenizationAPI)
	if !ok {
		writeError(response, http.StatusNotImplemented, "unsupported_operation", "detokenization is unavailable")
		return
	}
	var body detokenizeRequest
	if !h.decodeBoundedJSON(response, request, &body) {
		return
	}
	if len(body.Tokens) > 1<<20 {
		writeError(response, http.StatusBadRequest, "invalid_request_error", "token count exceeds 1048576")
		return
	}
	tokens := make([]tokenizer.TokenID, len(body.Tokens))
	for index, token := range body.Tokens {
		if token < 0 || token >= api.SamplingVocabularySize() {
			writeError(
				response,
				http.StatusBadRequest,
				"invalid_request_error",
				fmt.Sprintf("token %d is out of range", token),
			)
			return
		}
		tokens[index] = tokenizer.TokenID(token)
	}
	content := ""
	if pieceAPI, ok := h.generator.(TokenPieceAPI); ok {
		var result strings.Builder
		for _, token := range tokens {
			piece, err := pieceAPI.TokenPiece(token)
			if err != nil {
				writeError(response, http.StatusBadRequest, "invalid_request_error", err.Error())
				return
			}
			result.WriteString(piece)
		}
		content = result.String()
	} else {
		var err error
		content, err = api.DetokenizeTokens(tokens)
		if err != nil {
			writeError(response, http.StatusBadRequest, "invalid_request_error", err.Error())
			return
		}
	}
	writeJSON(response, http.StatusOK, map[string]string{"content": content})
}

func (h *Handler) decodeBoundedJSON(
	response http.ResponseWriter,
	request *http.Request,
	target any,
) bool {
	return h.decodeJSONWithLimit(response, request, target, maxRequestBytes)
}

func (h *Handler) decodeMultimodalJSON(
	response http.ResponseWriter,
	request *http.Request,
	target any,
) bool {
	return h.decodeJSONWithLimit(response, request, target, maxMultimodalRequestBytes)
}

func (h *Handler) decodeJSONWithLimit(
	response http.ResponseWriter,
	request *http.Request,
	target any,
	limit int64,
) bool {
	request.Body = http.MaxBytesReader(response, request.Body, limit)
	decoder := json.NewDecoder(request.Body)
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(target); err != nil {
		writeError(response, http.StatusBadRequest, "invalid_request_error", "invalid JSON request: "+err.Error())
		return false
	}
	if err := requireEOF(decoder); err != nil {
		writeError(response, http.StatusBadRequest, "invalid_request_error", err.Error())
		return false
	}
	return true
}

type applyTemplateRequest struct {
	Messages            []inference.ChatMessage `json:"messages"`
	Tools               []inference.ChatTool    `json:"tools"`
	ToolChoice          json.RawMessage         `json:"tool_choice"`
	ParallelTools       *bool                   `json:"parallel_tool_calls"`
	AddGenerationPrompt *bool                   `json:"add_generation_prompt"`
	ChatTemplateKwargs  map[string]any          `json:"chat_template_kwargs"`
}

func (h *Handler) applyTemplate(response http.ResponseWriter, request *http.Request) {
	if request.Method != http.MethodPost {
		response.Header().Set("Allow", http.MethodPost)
		writeError(response, http.StatusMethodNotAllowed, "method_not_allowed", "POST required")
		return
	}
	formatter, ok := h.generator.(ChatFormatter)
	if !ok {
		writeError(response, http.StatusNotImplemented, "unsupported_operation", "chat formatting is unavailable")
		return
	}
	request.Body = http.MaxBytesReader(response, request.Body, maxRequestBytes)
	decoder := json.NewDecoder(request.Body)
	decoder.DisallowUnknownFields()
	var body applyTemplateRequest
	if err := decoder.Decode(&body); err != nil {
		writeError(response, http.StatusBadRequest, "invalid_request_error", "invalid JSON request: "+err.Error())
		return
	}
	if err := requireEOF(decoder); err != nil {
		writeError(response, http.StatusBadRequest, "invalid_request_error", err.Error())
		return
	}
	selection, err := selectChatTools(chatCompletionRequest{
		Messages:       body.Messages,
		Tools:          body.Tools,
		ToolChoice:     body.ToolChoice,
		ParallelTools:  body.ParallelTools,
		AddPrompt:      body.AddGenerationPrompt,
		TemplateKwargs: body.ChatTemplateKwargs,
	})
	if err != nil {
		writeError(response, http.StatusBadRequest, "invalid_request_error", err.Error())
		return
	}
	prompt, err := formatChatRequest(
		formatter,
		body.Messages,
		selection.prompt,
		body.AddGenerationPrompt,
		body.ChatTemplateKwargs,
	)
	if err != nil {
		writeError(response, http.StatusBadRequest, "invalid_request_error", err.Error())
		return
	}
	writeJSON(response, http.StatusOK, map[string]string{"prompt": prompt})
}

func (h *Handler) authorized(request *http.Request) bool {
	if h.config.APIKey == "" {
		return true
	}
	const prefix = "Bearer "
	header := request.Header.Get("Authorization")
	if !strings.HasPrefix(header, prefix) {
		return false
	}
	provided := header[len(prefix):]
	return subtle.ConstantTimeCompare([]byte(provided), []byte(h.config.APIKey)) == 1
}

func (h *Handler) metrics(response http.ResponseWriter, request *http.Request) {
	if request.Method != http.MethodGet {
		response.Header().Set("Allow", http.MethodGet)
		writeError(response, http.StatusMethodNotAllowed, "method_not_allowed", "GET required")
		return
	}
	response.Header().Set("Content-Type", "text/plain; version=0.0.4; charset=utf-8")
	response.WriteHeader(http.StatusOK)
	_, _ = fmt.Fprintf(
		response,
		"# HELP llamacpp2go_up Whether the server is running.\n"+
			"# TYPE llamacpp2go_up gauge\n"+
			"llamacpp2go_up 1\n"+
			"# TYPE llamacpp2go_http_requests_total counter\n"+
			"llamacpp2go_http_requests_total %d\n"+
			"# TYPE llamacpp2go_http_requests_active gauge\n"+
			"llamacpp2go_http_requests_active %d\n"+
			"# TYPE llamacpp2go_generation_requests_total counter\n"+
			"llamacpp2go_generation_requests_total %d\n"+
			"# TYPE llamacpp2go_generation_errors_total counter\n"+
			"llamacpp2go_generation_errors_total %d\n"+
			"# TYPE llamacpp2go_generated_tokens_total counter\n"+
			"llamacpp2go_generated_tokens_total %d\n"+
			"# TYPE llamacpp2go_process_uptime_seconds gauge\n"+
			"llamacpp2go_process_uptime_seconds %.3f\n",
		h.requestsTotal.Load(),
		h.requestsActive.Load(),
		h.generationRequests.Load(),
		h.generationErrors.Load(),
		h.generatedTokens.Load(),
		time.Since(h.started).Seconds(),
	)
	if api, ok := h.generator.(DeviceMemoryAPI); ok {
		if stats, err := api.DeviceMemoryStats(request.Context()); err == nil {
			_, _ = fmt.Fprintf(
				response,
				"# HELP llamacpp2go_cuda_memory_current_bytes CUDA bytes currently allocated by this Runner.\n"+
					"# TYPE llamacpp2go_cuda_memory_current_bytes gauge\n"+
					"llamacpp2go_cuda_memory_current_bytes %d\n"+
					"# HELP llamacpp2go_cuda_memory_peak_bytes Lifetime high-water CUDA bytes allocated by this Runner.\n"+
					"# TYPE llamacpp2go_cuda_memory_peak_bytes gauge\n"+
					"llamacpp2go_cuda_memory_peak_bytes %d\n"+
					"# TYPE llamacpp2go_cuda_allocations_current gauge\n"+
					"llamacpp2go_cuda_allocations_current %d\n",
				stats.CurrentBytes,
				stats.PeakBytes,
				stats.Allocations,
			)
		}
	}
	if api, ok := h.generator.(DeviceExecutionAPI); ok {
		if stats, err := api.DeviceExecutionStats(request.Context()); err == nil {
			_, _ = fmt.Fprintf(
				response,
				"# TYPE llamacpp2go_cuda_custom_kernel_launches_total counter\n"+
					"llamacpp2go_cuda_custom_kernel_launches_total %d\n"+
					"# TYPE llamacpp2go_cuda_stream_synchronizations_total counter\n"+
					"llamacpp2go_cuda_stream_synchronizations_total %d\n"+
					"# TYPE llamacpp2go_cuda_context_synchronizations_total counter\n"+
					"llamacpp2go_cuda_context_synchronizations_total %d\n"+
					"# TYPE llamacpp2go_cuda_host_to_device_bytes_total counter\n"+
					"llamacpp2go_cuda_host_to_device_bytes_total %d\n"+
					"# TYPE llamacpp2go_cuda_device_to_host_bytes_total counter\n"+
					"llamacpp2go_cuda_device_to_host_bytes_total %d\n"+
					"# TYPE llamacpp2go_cuda_device_to_device_bytes_total counter\n"+
					"llamacpp2go_cuda_device_to_device_bytes_total %d\n"+
					"# TYPE llamacpp2go_cuda_device_memset_bytes_total counter\n"+
					"llamacpp2go_cuda_device_memset_bytes_total %d\n",
				stats.KernelLaunches,
				stats.StreamSynchronizations,
				stats.ContextSynchronizations,
				stats.HostToDeviceBytes,
				stats.DeviceToHostBytes,
				stats.DeviceToDeviceBytes,
				stats.DeviceMemsetBytes,
			)
		}
	}
}

func (h *Handler) generate(
	ctx context.Context,
	slotID int,
	prompt string,
	options inference.GenerateOptions,
) ([]tokenizer.TokenID, string, error) {
	h.generationRequests.Add(1)
	var stats *slotRuntimeStats
	if slotID >= 0 && slotID < len(h.slotStats) {
		stats = &h.slotStats[slotID]
		stats.beginGeneration(prompt, options)
	}
	onPromptEvaluated := options.OnPromptEvaluated
	options.OnPromptEvaluated = func(evaluation inference.PromptEvaluation) {
		if stats != nil {
			stats.promptTokens.Add(uint64(max(evaluation.Tokens, 0)))
			stats.cachedTokens.Add(uint64(max(evaluation.Cached, 0)))
			stats.promptNanos.Add(max(evaluation.Duration.Nanoseconds(), 0))
		}
		if onPromptEvaluated != nil {
			onPromptEvaluated(evaluation)
		}
	}
	onToken := options.OnToken
	options.OnToken = func(event inference.TokenEvent) error {
		h.generatedTokens.Add(1)
		if stats != nil {
			stats.generatedTokens.Add(1)
			stats.appendGenerated(event.Piece)
		}
		if onToken != nil {
			return onToken(event)
		}
		return nil
	}
	ids, text, err := h.generator.Generate(ctx, prompt, options)
	if err != nil {
		h.generationErrors.Add(1)
	}
	return ids, text, err
}

func (h *Handler) models(response http.ResponseWriter, request *http.Request) {
	if request.Method != http.MethodGet {
		response.Header().Set("Allow", http.MethodGet)
		writeError(response, http.StatusMethodNotAllowed, "method_not_allowed", "GET required")
		return
	}
	model := inference.ModelProperties{}
	if api, ok := h.generator.(ModelPropertiesAPI); ok {
		model = api.ModelProperties()
	}
	created := time.Now().Unix()
	capabilities := []string{"completion", "embedding"}
	if capability, ok := h.generator.(RankCapability); ok && capability.SupportsRank() {
		capabilities = append(capabilities, "rerank")
	}
	writeJSON(response, http.StatusOK, map[string]any{
		"models": []map[string]any{{
			"name":         h.config.ModelID,
			"model":        h.config.ModelID,
			"modified_at":  "",
			"size":         "",
			"digest":       "",
			"type":         "model",
			"description":  "",
			"tags":         []string{},
			"capabilities": capabilities,
			"parameters":   "",
			"details": map[string]any{
				"parent_model":       "",
				"format":             "gguf",
				"family":             model.Architecture,
				"families":           []string{model.Architecture},
				"parameter_size":     strconv.FormatUint(model.ParameterCount, 10),
				"quantization_level": model.FileType,
			},
		}},
		"object": "list",
		"data": []map[string]any{{
			"id":       h.config.ModelID,
			"aliases":  []string{h.config.ModelID},
			"tags":     []string{},
			"object":   "model",
			"created":  created,
			"owned_by": "llamacpp2go",
			"meta": map[string]any{
				"vocab_type":  model.VocabularyType,
				"n_vocab":     model.VocabularySize,
				"n_ctx":       model.ContextLength,
				"n_ctx_train": model.ContextLength,
				"n_embd":      model.EmbeddingLength,
				"n_params":    model.ParameterCount,
				"size":        model.ModelSize,
				"ftype":       model.FileType,
			},
		}},
	})
}

func (h *Handler) health(response http.ResponseWriter, request *http.Request) {
	if request.Method != http.MethodGet {
		response.Header().Set("Allow", http.MethodGet)
		writeError(response, http.StatusMethodNotAllowed, "method_not_allowed", "GET required")
		return
	}
	writeJSON(response, http.StatusOK, map[string]any{
		"status": "ok",
		"model":  h.config.ModelID,
	})
}

type embeddingRequest struct {
	Model          string          `json:"model"`
	Input          json.RawMessage `json:"input"`
	EncodingFormat string          `json:"encoding_format"`
}

type embeddingItem struct {
	Object         string `json:"object"`
	Embedding      any    `json:"embedding"`
	Index          int    `json:"index"`
	EncodingFormat string `json:"encoding_format,omitempty"`
}

type embeddingResponse struct {
	Object string          `json:"object"`
	Data   []embeddingItem `json:"data"`
	Model  string          `json:"model"`
	Usage  embeddingUsage  `json:"usage"`
}

type embeddingUsage struct {
	PromptTokens int `json:"prompt_tokens"`
	TotalTokens  int `json:"total_tokens"`
}

type rerankRequest struct {
	Model      string   `json:"model"`
	Query      *string  `json:"query"`
	Documents  []string `json:"documents"`
	Texts      []string `json:"texts"`
	TopN       *int     `json:"top_n"`
	ReturnText bool     `json:"return_text"`
}

type rerankItem struct {
	Index          int      `json:"index"`
	RelevanceScore *float32 `json:"relevance_score,omitempty"`
	Score          *float32 `json:"score,omitempty"`
	Text           *string  `json:"text,omitempty"`
}

type rerankResponse struct {
	Model   string         `json:"model"`
	Object  string         `json:"object"`
	Usage   embeddingUsage `json:"usage"`
	Results []rerankItem   `json:"results"`
}

func (h *Handler) rerank(response http.ResponseWriter, request *http.Request) {
	if request.Method != http.MethodPost {
		response.Header().Set("Allow", http.MethodPost)
		writeError(response, http.StatusMethodNotAllowed, "method_not_allowed", "POST required")
		return
	}
	ranker, ok := h.generator.(Ranker)
	if !ok {
		writeError(response, http.StatusNotImplemented, "unsupported_operation", "reranking is unavailable")
		return
	}
	if capability, ok := h.generator.(RankCapability); ok && !capability.SupportsRank() {
		writeError(response, http.StatusNotImplemented, "unsupported_operation", "reranking is unavailable")
		return
	}
	var body rerankRequest
	if !h.decodeBoundedJSON(response, request, &body) {
		return
	}
	if body.Model != "" && body.Model != h.config.ModelID {
		writeError(response, http.StatusNotFound, "model_not_found", "requested model is not loaded")
		return
	}
	if body.Query == nil {
		writeError(response, http.StatusBadRequest, "invalid_request_error", "query must be provided")
		return
	}
	tei := body.Texts != nil
	documents := body.Documents
	if documents == nil {
		documents = body.Texts
	}
	if len(documents) == 0 {
		writeError(response, http.StatusBadRequest, "invalid_request_error", "documents must be a non-empty string array")
		return
	}
	if len(documents) > h.config.MaxEmbeddingInputs {
		writeError(response, http.StatusBadRequest, "invalid_request_error",
			fmt.Sprintf("rerank document count exceeds %d", h.config.MaxEmbeddingInputs))
		return
	}
	topN := len(documents)
	if body.TopN != nil {
		if *body.TopN < 0 {
			writeError(response, http.StatusBadRequest, "invalid_request_error", "top_n must be non-negative")
			return
		}
		topN = min(*body.TopN, len(documents))
	}
	slotID, acquired := h.acquireSlot(-1)
	if !acquired {
		response.Header().Set("Retry-After", "1")
		writeError(response, http.StatusTooManyRequests, "server_busy", "generation capacity is busy")
		return
	}
	defer h.releaseSlot(slotID)
	items := make([]rerankItem, len(documents))
	usage := embeddingUsage{}
	for index, document := range documents {
		result, err := ranker.RankPair(request.Context(), *body.Query, document)
		if err != nil {
			writeGenerationError(response, err)
			return
		}
		if len(result.Scores) == 0 {
			writeGenerationError(response, errors.New("server: rerank result has no scores"))
			return
		}
		score := result.Scores[0]
		item := rerankItem{Index: index}
		if tei {
			item.Score = &score
			if body.ReturnText {
				text := document
				item.Text = &text
			}
		} else {
			item.RelevanceScore = &score
		}
		items[index] = item
		usage.PromptTokens += result.Tokens
		usage.TotalTokens += result.Tokens
	}
	sort.SliceStable(items, func(left, right int) bool {
		leftScore, rightScore := items[left].RelevanceScore, items[right].RelevanceScore
		if tei {
			leftScore, rightScore = items[left].Score, items[right].Score
		}
		return *leftScore > *rightScore
	})
	items = items[:topN]
	if tei {
		writeJSON(response, http.StatusOK, items)
		return
	}
	writeJSON(response, http.StatusOK, rerankResponse{
		Model: h.config.ModelID, Object: "list", Usage: usage, Results: items,
	})
}

func (h *Handler) embeddings(response http.ResponseWriter, request *http.Request) {
	if request.Method != http.MethodPost {
		response.Header().Set("Allow", http.MethodPost)
		writeError(response, http.StatusMethodNotAllowed, "method_not_allowed", "POST required")
		return
	}
	embedder, ok := h.generator.(Embedder)
	if !ok {
		writeError(response, http.StatusNotImplemented, "unsupported_operation", "embeddings are unavailable")
		return
	}
	request.Body = http.MaxBytesReader(response, request.Body, maxRequestBytes)
	decoder := json.NewDecoder(request.Body)
	decoder.DisallowUnknownFields()
	var body embeddingRequest
	if err := decoder.Decode(&body); err != nil {
		writeError(response, http.StatusBadRequest, "invalid_request_error", "invalid JSON request: "+err.Error())
		return
	}
	if err := requireEOF(decoder); err != nil {
		writeError(response, http.StatusBadRequest, "invalid_request_error", err.Error())
		return
	}
	if body.Model != "" && body.Model != h.config.ModelID {
		writeError(response, http.StatusNotFound, "model_not_found", "requested model is not loaded")
		return
	}
	if body.EncodingFormat != "" &&
		body.EncodingFormat != "float" &&
		body.EncodingFormat != "base64" {
		writeError(
			response,
			http.StatusBadRequest,
			"invalid_request_error",
			"encoding_format must be float or base64",
		)
		return
	}
	inputs, err := h.parseNativePrompts(body.Input)
	if err != nil {
		writeError(response, http.StatusBadRequest, "invalid_request_error", "embedding input: "+err.Error())
		return
	}
	if len(inputs) > h.config.MaxEmbeddingInputs {
		writeError(
			response,
			http.StatusBadRequest,
			"invalid_request_error",
			fmt.Sprintf("embedding input count exceeds %d", h.config.MaxEmbeddingInputs),
		)
		return
	}
	slotID, acquired := h.acquireSlot(-1)
	if !acquired {
		response.Header().Set("Retry-After", "1")
		writeError(response, http.StatusTooManyRequests, "server_busy", "generation capacity is busy")
		return
	}
	defer h.releaseSlot(slotID)
	result := embeddingResponse{
		Object: "list",
		Data:   make([]embeddingItem, len(inputs)),
		Model:  h.config.ModelID,
	}
	for index, input := range inputs {
		vector, tokens, embedErr := h.embedPrompt(request.Context(), embedder, input)
		if embedErr != nil {
			writeGenerationError(response, embedErr)
			return
		}
		var encoded any = vector
		format := ""
		if body.EncodingFormat == "base64" {
			encoded = encodeFloat32Base64(vector)
			format = "base64"
		}
		result.Data[index] = embeddingItem{
			Object:         "embedding",
			Embedding:      encoded,
			Index:          index,
			EncodingFormat: format,
		}
		result.Usage.PromptTokens += tokens
		result.Usage.TotalTokens += tokens
	}
	writeJSON(response, http.StatusOK, result)
}

type nativeEmbeddingRequest struct {
	Model          string          `json:"model"`
	Input          json.RawMessage `json:"input"`
	Content        json.RawMessage `json:"content"`
	EmbdNormalize  *int            `json:"embd_normalize"`
	EncodingFormat string          `json:"encoding_format"`
	Pooling        string          `json:"pooling"`
}

type nativeEmbeddingItem struct {
	Index     int         `json:"index"`
	Embedding [][]float32 `json:"embedding"`
}

func (h *Handler) nativeEmbeddings(response http.ResponseWriter, request *http.Request) {
	if request.Method != http.MethodPost {
		response.Header().Set("Allow", http.MethodPost)
		writeError(response, http.StatusMethodNotAllowed, "method_not_allowed", "POST required")
		return
	}
	embedder, ok := h.generator.(Embedder)
	if !ok {
		writeError(response, http.StatusNotImplemented, "unsupported_operation", "embeddings are unavailable")
		return
	}
	var body nativeEmbeddingRequest
	if !h.decodeBoundedJSON(response, request, &body) {
		return
	}
	if body.Model != "" && body.Model != h.config.ModelID {
		writeError(response, http.StatusNotFound, "model_not_found", "requested model is not loaded")
		return
	}
	if body.EncodingFormat != "" && body.EncodingFormat != "float" {
		writeError(
			response,
			http.StatusBadRequest,
			"invalid_request_error",
			"encoding_format must be float for native embeddings",
		)
		return
	}
	pooling := inference.EmbeddingPooling(body.Pooling)
	if pooling == "" {
		pooling = inference.EmbeddingPoolingMean
	}
	if pooling != inference.EmbeddingPoolingMean &&
		pooling != inference.EmbeddingPoolingLast &&
		pooling != inference.EmbeddingPoolingNone {
		writeError(
			response,
			http.StatusBadRequest,
			"invalid_request_error",
			"pooling must be mean, last, or none",
		)
		return
	}
	normalize := 2
	if body.EmbdNormalize != nil {
		normalize = *body.EmbdNormalize
	}
	raw := body.Input
	if len(raw) == 0 {
		raw = body.Content
	}
	inputs, err := h.parseNativePrompts(raw)
	if err != nil {
		writeError(response, http.StatusBadRequest, "invalid_request_error", "embedding input: "+err.Error())
		return
	}
	if len(inputs) > h.config.MaxEmbeddingInputs {
		writeError(
			response,
			http.StatusBadRequest,
			"invalid_request_error",
			fmt.Sprintf("embedding input count exceeds %d", h.config.MaxEmbeddingInputs),
		)
		return
	}
	slotID, acquired := h.acquireSlot(-1)
	if !acquired {
		response.Header().Set("Retry-After", "1")
		writeError(response, http.StatusTooManyRequests, "server_busy", "generation capacity is busy")
		return
	}
	defer h.releaseSlot(slotID)
	result := make([]nativeEmbeddingItem, len(inputs))
	for index, input := range inputs {
		embedded, embedErr := h.embedPromptAdvanced(
			request.Context(),
			embedder,
			input,
			inference.EmbeddingOptions{Pooling: pooling, Normalize: normalize},
		)
		if embedErr != nil {
			writeGenerationError(response, embedErr)
			return
		}
		result[index] = nativeEmbeddingItem{
			Index:     index,
			Embedding: embedded.Vectors,
		}
	}
	writeJSON(response, http.StatusOK, result)
}

func (h *Handler) embedPromptAdvanced(
	ctx context.Context,
	embedder Embedder,
	prompt nativePrompt,
	options inference.EmbeddingOptions,
) (inference.EmbeddingResult, error) {
	if advanced, ok := h.generator.(AdvancedEmbedder); ok {
		if prompt.TokenIDs == nil {
			return advanced.EmbedAdvanced(ctx, prompt.Text, options)
		}
		return advanced.EmbedTokensAdvanced(ctx, prompt.TokenIDs, options)
	}
	if options.Pooling != inference.EmbeddingPoolingMean || options.Normalize != 2 {
		return inference.EmbeddingResult{}, errors.New(
			"generator does not expose advanced embedding modes",
		)
	}
	vector, tokens, err := h.embedPrompt(ctx, embedder, prompt)
	if err != nil {
		return inference.EmbeddingResult{}, err
	}
	return inference.EmbeddingResult{Vectors: [][]float32{vector}, Tokens: tokens}, nil
}

func encodeFloat32Base64(values []float32) string {
	data := make([]byte, len(values)*4)
	for index, value := range values {
		binary.LittleEndian.PutUint32(data[index*4:], math.Float32bits(value))
	}
	return base64.StdEncoding.EncodeToString(data)
}

func (h *Handler) embedPrompt(
	ctx context.Context,
	embedder Embedder,
	prompt nativePrompt,
) ([]float32, int, error) {
	if prompt.TokenIDs == nil {
		return embedder.Embed(ctx, prompt.Text)
	}
	tokenEmbedder, ok := h.generator.(TokenEmbedder)
	if !ok {
		return nil, 0, errors.New("server: exact-token embeddings are unavailable")
	}
	return tokenEmbedder.EmbedTokens(ctx, prompt.TokenIDs)
}

func parseStopSequences(raw json.RawMessage) ([]string, error) {
	if len(raw) == 0 || string(raw) == "null" {
		return nil, nil
	}
	var single string
	if err := json.Unmarshal(raw, &single); err == nil {
		if single == "" {
			return nil, errors.New("stop sequence must not be empty")
		}
		return []string{single}, nil
	}
	var multiple []string
	if err := json.Unmarshal(raw, &multiple); err != nil {
		return nil, errors.New("stop must be a string or string array")
	}
	if len(multiple) > 256 {
		return nil, errors.New("stop sequence count exceeds 256")
	}
	for _, stop := range multiple {
		if stop == "" {
			return nil, errors.New("stop sequence must not be empty")
		}
	}
	return multiple, nil
}

type stopFilter struct {
	stops   []string
	pending string
	stopped bool
	word    string
}

func newStopFilter(stops []string) *stopFilter {
	return &stopFilter{stops: append([]string(nil), stops...)}
}

func (filter *stopFilter) Accept(piece string) string {
	if filter == nil || filter.stopped {
		return ""
	}
	if len(filter.stops) == 0 {
		return piece
	}
	filter.pending += piece
	match := -1
	for _, stop := range filter.stops {
		if index := strings.Index(filter.pending, stop); index >= 0 &&
			(match < 0 || index < match) {
			match = index
			filter.word = stop
		}
	}
	if match >= 0 {
		safe := filter.pending[:match]
		filter.pending = ""
		filter.stopped = true
		return safe
	}
	hold := 0
	for _, stop := range filter.stops {
		limit := min(len(filter.pending), len(stop)-1)
		for length := limit; length > hold; length-- {
			if strings.HasSuffix(filter.pending, stop[:length]) {
				hold = length
				break
			}
		}
	}
	safeLength := len(filter.pending) - hold
	safe := filter.pending[:safeLength]
	filter.pending = filter.pending[safeLength:]
	return safe
}

func (filter *stopFilter) Flush() string {
	if filter == nil || filter.stopped {
		return ""
	}
	safe := filter.pending
	filter.pending = ""
	return safe
}

func (filter *stopFilter) Stopped() bool {
	return filter != nil && filter.stopped
}

func (filter *stopFilter) StoppingWord() string {
	if filter == nil {
		return ""
	}
	return filter.word
}

type samplingParameters struct {
	Temperature            *float32        `json:"temperature"`
	DynatempRange          float32         `json:"dynatemp_range"`
	DynatempExponent       float32         `json:"dynatemp_exponent"`
	TopP                   *float32        `json:"top_p"`
	TopK                   *int            `json:"top_k"`
	MinP                   float32         `json:"min_p"`
	TypicalP               float32         `json:"typical_p"`
	TopNSigma              float32         `json:"top_n_sigma"`
	XTCProbability         float32         `json:"xtc_probability"`
	XTCThreshold           float32         `json:"xtc_threshold"`
	MinKeep                int             `json:"min_keep"`
	AdaptiveTarget         *float32        `json:"adaptive_target"`
	AdaptiveDecay          *float32        `json:"adaptive_decay"`
	RepeatLastN            int             `json:"repeat_last_n"`
	RepeatPenalty          float32         `json:"repeat_penalty"`
	PresencePenalty        float32         `json:"presence_penalty"`
	FrequencyPenalty       float32         `json:"frequency_penalty"`
	DryMultiplier          float32         `json:"dry_multiplier"`
	DryBase                float32         `json:"dry_base"`
	DryAllowedLength       int             `json:"dry_allowed_length"`
	DryPenaltyLastN        int             `json:"dry_penalty_last_n"`
	DryBreakers            []string        `json:"dry_sequence_breakers"`
	Mirostat               int             `json:"mirostat"`
	MirostatTau            float32         `json:"mirostat_tau"`
	MirostatEta            float32         `json:"mirostat_eta"`
	Seed                   int64           `json:"seed"`
	GrammarChoices         []string        `json:"grammar_choices"`
	Grammar                string          `json:"grammar"`
	GrammarRoot            string          `json:"grammar_root"`
	GrammarLazy            bool            `json:"grammar_lazy"`
	GrammarTriggerPatterns []string        `json:"grammar_trigger_patterns"`
	GrammarTriggerTokens   []int           `json:"grammar_trigger_tokens"`
	Samplers               []string        `json:"samplers"`
	LogitBias              json.RawMessage `json:"logit_bias"`
	IgnoreEOS              bool            `json:"ignore_eos"`
}

type completionRequest struct {
	Model      string          `json:"model"`
	Prompt     json.RawMessage `json:"prompt"`
	MaxTokens  *int            `json:"max_tokens"`
	Stop       json.RawMessage `json:"stop"`
	N          int             `json:"n"`
	JSONSchema json.RawMessage `json:"json_schema"`
	samplingParameters
	Stream bool `json:"stream"`
}

func (h *Handler) completions(response http.ResponseWriter, request *http.Request) {
	if request.Method != http.MethodPost {
		response.Header().Set("Allow", http.MethodPost)
		writeError(response, http.StatusMethodNotAllowed, "method_not_allowed", "POST required")
		return
	}
	request.Body = http.MaxBytesReader(response, request.Body, maxRequestBytes)
	decoder := json.NewDecoder(request.Body)
	decoder.DisallowUnknownFields()
	var body completionRequest
	if err := decoder.Decode(&body); err != nil {
		writeError(response, http.StatusBadRequest, "invalid_request_error", "invalid JSON request: "+err.Error())
		return
	}
	if err := requireEOF(decoder); err != nil {
		writeError(response, http.StatusBadRequest, "invalid_request_error", err.Error())
		return
	}
	prompts, err := h.parseNativePrompts(body.Prompt)
	if err != nil {
		writeError(response, http.StatusBadRequest, "invalid_request_error", err.Error())
		return
	}
	if body.Model != "" && body.Model != h.config.ModelID {
		writeError(response, http.StatusNotFound, "model_not_found", "requested model is not loaded")
		return
	}
	if body.N == 0 {
		body.N = 1
	}
	if body.N < 1 || body.N > 8 {
		writeError(response, http.StatusBadRequest, "invalid_request_error", "n must be in [1,8]")
		return
	}
	maxTokens := 16
	if body.MaxTokens != nil {
		maxTokens = *body.MaxTokens
	}
	if maxTokens < 0 || maxTokens > h.config.MaxTokens {
		writeError(
			response,
			http.StatusBadRequest,
			"invalid_request_error",
			fmt.Sprintf("max_tokens must be in [0,%d]", h.config.MaxTokens),
		)
		return
	}
	stops, err := parseStopSequences(body.Stop)
	if err != nil {
		writeError(response, http.StatusBadRequest, "invalid_request_error", err.Error())
		return
	}
	if err := prepareStructuredOutput(
		&body.samplingParameters,
		body.JSONSchema,
		nil,
	); err != nil {
		writeError(response, http.StatusBadRequest, "invalid_request_error", err.Error())
		return
	}
	sampler, err := h.newSampler(body.samplingParameters)
	if err != nil {
		writeError(response, http.StatusBadRequest, "invalid_request_error", err.Error())
		return
	}
	slotID, acquired := h.acquireSlot(-1)
	if !acquired {
		response.Header().Set("Retry-After", "1")
		writeError(response, http.StatusTooManyRequests, "server_busy", "generation capacity is busy")
		return
	}
	defer h.releaseSlot(slotID)
	id := "cmpl-" + strconv.FormatUint(h.nextID.Add(1), 10)
	if body.Stream {
		h.streamCompletion(response, request, slotID, prompts, sampler, maxTokens, id, stops, body.N)
		return
	}
	h.complete(response, request, slotID, prompts, sampler, maxTokens, id, stops, body.N)
}

type nativeCompletionRequest struct {
	Prompt            json.RawMessage                `json:"prompt"`
	Model             string                         `json:"model"`
	NPredict          *int                           `json:"n_predict"`
	NCmpl             int                            `json:"n_cmpl"`
	Stop              json.RawMessage                `json:"stop"`
	Stream            bool                           `json:"stream"`
	ReturnTokens      bool                           `json:"return_tokens"`
	CachePrompt       *bool                          `json:"cache_prompt"`
	NIndent           int                            `json:"n_indent"`
	NKeep             int                            `json:"n_keep"`
	NDiscard          int                            `json:"n_discard"`
	NCacheReuse       int                            `json:"n_cache_reuse"`
	NProbs            int                            `json:"n_probs"`
	TMaxPredictMS     int                            `json:"t_max_predict_ms"`
	IDSlot            *int                           `json:"id_slot"`
	TimingsPerToken   bool                           `json:"timings_per_token"`
	ReturnProgress    bool                           `json:"return_progress"`
	SSEPingInterval   *float64                       `json:"sse_ping_interval"`
	PostSamplingProbs bool                           `json:"post_sampling_probs"`
	ResponseFields    []string                       `json:"response_fields"`
	JSONSchema        json.RawMessage                `json:"json_schema"`
	LoRA              json.RawMessage                `json:"lora"`
	ProjectedInputs   *inference.ProjectedInputsJSON `json:"projected_inputs"`
	samplingParameters
}

type infillExtraRequest struct {
	Text     *string `json:"text"`
	Filename *string `json:"filename"`
}

type infillRequest struct {
	InputPrefix json.RawMessage `json:"input_prefix"`
	InputSuffix json.RawMessage `json:"input_suffix"`
	InputExtra  json.RawMessage `json:"input_extra"`
	nativeCompletionRequest
}

func (h *Handler) infill(
	response http.ResponseWriter,
	request *http.Request,
) {
	if request.Method != http.MethodPost {
		response.Header().Set("Allow", http.MethodPost)
		writeError(
			response,
			http.StatusMethodNotAllowed,
			"method_not_allowed",
			"POST required",
		)
		return
	}
	formatter, ok := h.generator.(InfillFormatter)
	if !ok {
		writeError(
			response,
			http.StatusNotImplemented,
			"unsupported_operation",
			"infill formatting is unavailable",
		)
		return
	}
	api, ok := h.generator.(TokenizationAPI)
	if !ok {
		writeError(
			response,
			http.StatusNotImplemented,
			"unsupported_operation",
			"infill tokenization is unavailable",
		)
		return
	}
	var body infillRequest
	if !h.decodeBoundedJSON(response, request, &body) {
		return
	}
	if len(body.InputPrefix) == 0 || string(body.InputPrefix) == "null" {
		writeError(
			response,
			http.StatusBadRequest,
			"invalid_request_error",
			"input_prefix is required",
		)
		return
	}
	if len(body.InputSuffix) == 0 || string(body.InputSuffix) == "null" {
		writeError(
			response,
			http.StatusBadRequest,
			"invalid_request_error",
			"input_suffix is required",
		)
		return
	}
	prefix, err := tokenizeMixed(api, body.InputPrefix, false, false)
	if err != nil {
		writeError(
			response,
			http.StatusBadRequest,
			"invalid_request_error",
			"invalid input_prefix: "+err.Error(),
		)
		return
	}
	suffix, err := tokenizeMixed(api, body.InputSuffix, false, false)
	if err != nil {
		writeError(
			response,
			http.StatusBadRequest,
			"invalid_request_error",
			"invalid input_suffix: "+err.Error(),
		)
		return
	}
	var prompt []tokenizer.TokenID
	if len(body.Prompt) != 0 && string(body.Prompt) != "null" {
		var promptText string
		if err := json.Unmarshal(body.Prompt, &promptText); err != nil {
			writeError(
				response,
				http.StatusBadRequest,
				"invalid_request_error",
				"prompt must be a string",
			)
			return
		}
		prompt, err = api.TokenizeText(promptText, false, true)
		if err != nil {
			writeError(
				response,
				http.StatusBadRequest,
				"invalid_request_error",
				"tokenize prompt: "+err.Error(),
			)
			return
		}
	}
	var extraRequests []infillExtraRequest
	if len(body.InputExtra) != 0 {
		if err := json.Unmarshal(body.InputExtra, &extraRequests); err != nil ||
			string(body.InputExtra) == "null" {
			writeError(
				response,
				http.StatusBadRequest,
				"invalid_request_error",
				"input_extra must be an array",
			)
			return
		}
	}
	extra := make([]inference.InfillExtra, len(extraRequests))
	for index, chunk := range extraRequests {
		if chunk.Text == nil {
			writeError(
				response,
				http.StatusBadRequest,
				"invalid_request_error",
				fmt.Sprintf("input_extra %d requires string text", index),
			)
			return
		}
		extra[index].Filename = "tmp"
		if chunk.Filename != nil {
			extra[index].Filename = *chunk.Filename
		}
		extra[index].Tokens, err = api.TokenizeText(
			*chunk.Text,
			false,
			false,
		)
		if err != nil {
			writeError(
				response,
				http.StatusBadRequest,
				"invalid_request_error",
				fmt.Sprintf("tokenize input_extra %d: %v", index, err),
			)
			return
		}
	}
	maxTokens := h.config.MaxTokens
	if body.NPredict != nil && *body.NPredict != -1 {
		maxTokens = *body.NPredict
	}
	if maxTokens < 0 || maxTokens > h.config.MaxTokens {
		writeError(
			response,
			http.StatusBadRequest,
			"invalid_request_error",
			fmt.Sprintf("n_predict must be -1 or in [0,%d]", h.config.MaxTokens),
		)
		return
	}
	formatted, err := formatter.FormatInfillTokens(
		prefix,
		suffix,
		prompt,
		extra,
		inference.InfillFormatOptions{
			BatchSize:    h.config.InfillBatchSize,
			MaxNewTokens: maxTokens,
			SuffixPrefix: h.config.SPMInfill,
		},
	)
	if err != nil {
		writeError(
			response,
			http.StatusBadRequest,
			"invalid_request_error",
			err.Error(),
		)
		return
	}
	body.Prompt, err = json.Marshal(formatted)
	if err != nil {
		writeError(
			response,
			http.StatusInternalServerError,
			"internal_error",
			"encode formatted infill prompt",
		)
		return
	}
	encoded, err := json.Marshal(body.nativeCompletionRequest)
	if err != nil {
		writeError(
			response,
			http.StatusInternalServerError,
			"internal_error",
			"encode infill completion request",
		)
		return
	}
	next := request.Clone(request.Context())
	next.Body = io.NopCloser(bytes.NewReader(encoded))
	next.ContentLength = int64(len(encoded))
	h.nativeCompletions(response, next)
}

type nativeCompletionTimings struct {
	CacheN              int     `json:"cache_n"`
	PromptN             int     `json:"prompt_n"`
	PromptMS            float64 `json:"prompt_ms"`
	PromptPerTokenMS    float64 `json:"prompt_per_token_ms"`
	PromptPerSecond     float64 `json:"prompt_per_second"`
	PredictedN          int     `json:"predicted_n"`
	PredictedMS         float64 `json:"predicted_ms"`
	PredictedPerTokenMS float64 `json:"predicted_per_token_ms"`
	PredictedPerSecond  float64 `json:"predicted_per_second"`
}

type nativeCompletionResponse struct {
	Index                   int                      `json:"index"`
	Content                 string                   `json:"content"`
	Tokens                  []tokenizer.TokenID      `json:"tokens"`
	IDSlot                  int                      `json:"id_slot"`
	Stop                    bool                     `json:"stop"`
	Model                   string                   `json:"model"`
	TokensPredicted         int                      `json:"tokens_predicted"`
	TokensEvaluated         int                      `json:"tokens_evaluated"`
	GenerationSettings      map[string]any           `json:"generation_settings"`
	Prompt                  any                      `json:"prompt"`
	HasNewLine              bool                     `json:"has_new_line"`
	Truncated               bool                     `json:"truncated"`
	StopType                string                   `json:"stop_type"`
	StoppingWord            string                   `json:"stopping_word"`
	TokensCached            int                      `json:"tokens_cached"`
	Timings                 nativeCompletionTimings  `json:"timings"`
	CompletionProbabilities []nativeTokenProbability `json:"completion_probabilities,omitempty"`
}

type nativeCompletionChunk struct {
	Index                   int                      `json:"index"`
	Content                 string                   `json:"content"`
	Tokens                  []tokenizer.TokenID      `json:"tokens"`
	Stop                    bool                     `json:"stop"`
	IDSlot                  int                      `json:"id_slot"`
	TokensPredicted         int                      `json:"tokens_predicted"`
	TokensEvaluated         int                      `json:"tokens_evaluated"`
	PromptProgress          *nativePromptProgress    `json:"prompt_progress,omitempty"`
	Timings                 *nativeCompletionTimings `json:"timings,omitempty"`
	CompletionProbabilities []nativeTokenProbability `json:"completion_probabilities,omitempty"`
}

type nativeTokenLogProbability struct {
	ID      tokenizer.TokenID `json:"id"`
	Token   string            `json:"token"`
	Bytes   []int             `json:"bytes"`
	LogProb float64           `json:"logprob"`
}

type nativeTokenProbability struct {
	ID          tokenizer.TokenID             `json:"id"`
	Token       string                        `json:"token"`
	Bytes       []int                         `json:"bytes"`
	LogProb     *float64                      `json:"logprob,omitempty"`
	Prob        *float64                      `json:"prob,omitempty"`
	TopLogProbs []nativeTokenLogProbability   `json:"top_logprobs,omitempty"`
	TopProbs    []nativeTokenProbabilityValue `json:"top_probs,omitempty"`
}

type nativeTokenProbabilityValue struct {
	ID    tokenizer.TokenID `json:"id"`
	Token string            `json:"token"`
	Bytes []int             `json:"bytes"`
	Prob  float64           `json:"prob"`
}

type nativePromptProgress struct {
	Total     int   `json:"total"`
	Cache     int   `json:"cache"`
	Processed int   `json:"processed"`
	TimeMS    int64 `json:"time_ms"`
}

type nativePrompt struct {
	Text        string
	TokenIDs    []tokenizer.TokenID
	Response    any
	Image       []byte
	Images      [][]byte
	MediaText   []string
	Audio       []float32
	BeforeMedia string
	AfterMedia  string
}

type preparedPrompt struct {
	nativePrompt
	ProjectedInputs *inference.ProjectedInputs
	Multimodal      bool
}

func (h *Handler) nativeCompletions(response http.ResponseWriter, request *http.Request) {
	if request.Method != http.MethodPost {
		response.Header().Set("Allow", http.MethodPost)
		writeError(response, http.StatusMethodNotAllowed, "method_not_allowed", "POST required")
		return
	}
	var body nativeCompletionRequest
	if !h.decodeMultimodalJSON(response, request, &body) {
		return
	}
	prompts, err := h.parseNativePrompts(body.Prompt)
	if err != nil {
		writeError(response, http.StatusBadRequest, "invalid_request_error", err.Error())
		return
	}
	if body.Model != "" && body.Model != h.config.ModelID {
		writeError(response, http.StatusNotFound, "model_not_found", "requested model is not loaded")
		return
	}
	if err := validateNativeCompletionOptions(body); err != nil {
		writeError(response, http.StatusBadRequest, "invalid_request_error", err.Error())
		return
	}
	lora, loraConfigured, err := h.parseRequestLoRA(body.LoRA)
	if err != nil {
		writeError(response, http.StatusBadRequest, "invalid_request_error", err.Error())
		return
	}
	var projectedInputs *inference.ProjectedInputs
	if body.ProjectedInputs != nil {
		projected, projectedErr := body.ProjectedInputs.ProjectedInputs()
		if projectedErr != nil {
			writeError(response, http.StatusBadRequest, "invalid_request_error", projectedErr.Error())
			return
		}
		projectedInputs = &projected
	}
	if err := prepareStructuredOutput(
		&body.samplingParameters,
		body.JSONSchema,
		nil,
	); err != nil {
		writeError(response, http.StatusBadRequest, "invalid_request_error", err.Error())
		return
	}
	requestedSlot := -1
	if body.IDSlot != nil {
		requestedSlot = *body.IDSlot
	}
	if requestedSlot < -1 || requestedSlot >= h.config.MaxConcurrent {
		writeError(
			response,
			http.StatusBadRequest,
			"invalid_request_error",
			fmt.Sprintf("id_slot must be -1 or in [0,%d]", h.config.MaxConcurrent-1),
		)
		return
	}
	maxTokens := h.config.MaxTokens
	if body.NPredict != nil && *body.NPredict != -1 {
		maxTokens = *body.NPredict
	}
	if maxTokens < 0 || maxTokens > h.config.MaxTokens {
		writeError(
			response,
			http.StatusBadRequest,
			"invalid_request_error",
			fmt.Sprintf("n_predict must be -1 or in [0,%d]", h.config.MaxTokens),
		)
		return
	}
	if body.NCmpl == 0 {
		body.NCmpl = 1
	}
	if body.NCmpl < 1 || body.NCmpl > 8 {
		writeError(response, http.StatusBadRequest, "invalid_request_error", "n_cmpl must be in [1,8]")
		return
	}
	if projectedInputs != nil && (len(prompts) != 1 || body.NCmpl != 1) {
		writeError(response, http.StatusBadRequest, "invalid_request_error", "projected_inputs requires one prompt and one completion")
		return
	}
	multimodal := len(prompts) == 1 && nativePromptHasMedia(prompts[0])
	if multimodal && (projectedInputs != nil || body.NCmpl != 1) {
		writeError(response, http.StatusBadRequest, "invalid_request_error", "multimodal prompt requires one completion and no projected_inputs")
		return
	}
	if multimodal && body.CachePrompt != nil && *body.CachePrompt {
		writeError(response, http.StatusBadRequest, "invalid_request_error", "multimodal prompt cannot use cache_prompt")
		return
	}
	stops, err := parseStopSequences(body.Stop)
	if err != nil {
		writeError(response, http.StatusBadRequest, "invalid_request_error", err.Error())
		return
	}
	sampler, err := h.newSampler(body.samplingParameters)
	if err != nil {
		writeError(response, http.StatusBadRequest, "invalid_request_error", err.Error())
		return
	}
	slotID, acquired := h.acquireSlot(requestedSlot)
	if !acquired {
		response.Header().Set("Retry-After", "1")
		writeError(response, http.StatusTooManyRequests, "server_busy", "generation capacity is busy")
		return
	}
	defer h.releaseSlot(slotID)
	for index := range prompts {
		prepared, prepareErr := h.preparePrompt(request.Context(), prompts[index], false)
		if prepareErr != nil {
			writeGenerationError(response, prepareErr)
			return
		}
		prompts[index] = prepared.nativePrompt
		if prepared.ProjectedInputs != nil {
			projectedInputs = prepared.ProjectedInputs
		}
	}
	settings := h.nativeGenerationSettings(body, sampler.Config(), maxTokens, stops)
	settings["multimodal"] = multimodal
	if body.Stream {
		pingInterval := 30
		if body.SSEPingInterval != nil {
			pingInterval = int(*body.SSEPingInterval)
		}
		h.streamNativeCompletion(
			response,
			request,
			prompts,
			sampler,
			maxTokens,
			stops,
			body.NCmpl,
			settings,
			body.ResponseFields,
			slotID,
			body.NKeep,
			body.NDiscard,
			body.CachePrompt != nil && *body.CachePrompt,
			body.NCacheReuse,
			pingInterval,
			body.ReturnProgress,
			body.TimingsPerToken,
			body.TMaxPredictMS,
			body.NIndent,
			body.NProbs,
			body.PostSamplingProbs,
			lora,
			loraConfigured,
			projectedInputs,
		)
		return
	}
	results := make([]nativeCompletionResponse, 0, len(prompts)*body.NCmpl)
	resultIndex := 0
	for _, prompt := range prompts {
		for range body.NCmpl {
			choiceSampler, choiceErr := samplerForChoice(sampler, resultIndex)
			if choiceErr != nil {
				writeGenerationError(response, choiceErr)
				return
			}
			result, generationErr := h.runNativeCompletion(
				request.Context(),
				prompt,
				choiceSampler,
				maxTokens,
				stops,
				resultIndex,
				body.ReturnTokens,
				settings,
				nil,
				nil,
				body.TimingsPerToken,
				body.TMaxPredictMS,
				body.NIndent,
				body.NProbs,
				body.PostSamplingProbs,
				slotID,
				body.NKeep,
				body.NDiscard,
				body.CachePrompt != nil && *body.CachePrompt,
				body.NCacheReuse,
				lora,
				loraConfigured,
				projectedInputs,
			)
			if generationErr != nil {
				writeGenerationError(response, generationErr)
				return
			}
			results = append(results, result)
			resultIndex++
		}
	}
	if len(body.ResponseFields) > 0 {
		projected := make([]map[string]any, 0, len(results))
		for _, result := range results {
			item, projectErr := projectNativeResponse(result, body.ResponseFields)
			if projectErr != nil {
				writeError(response, http.StatusInternalServerError, "server_error", projectErr.Error())
				return
			}
			projected = append(projected, item)
		}
		if len(projected) == 1 {
			writeJSON(response, http.StatusOK, projected[0])
			return
		}
		writeJSON(response, http.StatusOK, projected)
		return
	}
	if len(results) == 1 {
		writeJSON(response, http.StatusOK, results[0])
		return
	}
	writeJSON(response, http.StatusOK, results)
}

func (h *Handler) parseNativePrompts(raw json.RawMessage) ([]nativePrompt, error) {
	if len(raw) == 0 {
		return nil, errors.New("prompt is required")
	}
	trimmedRaw := bytes.TrimSpace(raw)
	if len(trimmedRaw) > 0 && trimmedRaw[0] == '{' {
		prompt, err := h.parseNativeMultimodalPrompt(trimmedRaw)
		if err != nil {
			return nil, err
		}
		return []nativePrompt{prompt}, nil
	}
	var text string
	if json.Unmarshal(raw, &text) == nil {
		prompt, err := h.parseNativePrompt(raw)
		if err != nil {
			return nil, err
		}
		return []nativePrompt{prompt}, nil
	}
	var parts []json.RawMessage
	if err := json.Unmarshal(raw, &parts); err != nil {
		_, promptErr := h.parseNativePrompt(raw)
		return nil, promptErr
	}
	if len(parts) == 0 {
		return nil, errors.New("prompt list must not be empty")
	}
	allStrings := true
	hasNested := false
	for _, part := range parts {
		var partText string
		if json.Unmarshal(part, &partText) != nil {
			allStrings = false
		}
		trimmed := strings.TrimSpace(string(part))
		if strings.HasPrefix(trimmed, "[") || strings.HasPrefix(trimmed, "{") {
			hasNested = true
		}
	}
	if !allStrings && !hasNested {
		prompt, err := h.parseNativePrompt(raw)
		if err != nil {
			return nil, err
		}
		return []nativePrompt{prompt}, nil
	}
	if len(parts) > 64 {
		return nil, errors.New("prompt batch count exceeds 64")
	}
	result := make([]nativePrompt, 0, len(parts))
	for index, part := range parts {
		prompt, err := h.parseNativePrompt(part)
		if err != nil {
			return nil, fmt.Errorf("prompt %d: %w", index, err)
		}
		result = append(result, prompt)
	}
	return result, nil
}

func (h *Handler) parseNativeMultimodalPrompt(raw json.RawMessage) (nativePrompt, error) {
	if h.config.Qwen3VLProjector == nil && h.config.ImageProjector == nil && h.config.AudioProjector == nil {
		return nativePrompt{}, errors.New("multimodal data provided, but the server has no multimodal projector")
	}
	var document struct {
		PromptString   string   `json:"prompt_string"`
		MultimodalData []string `json:"multimodal_data"`
	}
	decoder := json.NewDecoder(bytes.NewReader(raw))
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(&document); err != nil {
		return nativePrompt{}, errors.New("prompt object must contain prompt_string and multimodal_data")
	}
	if document.PromptString == "" {
		return nativePrompt{}, errors.New("prompt_string must not be empty")
	}
	if len(document.MultimodalData) == 0 || len(document.MultimodalData) > 8 {
		return nativePrompt{}, errors.New("multimodal prompt requires one to eight media items")
	}
	const marker = "<__media__>"
	if strings.Count(document.PromptString, marker) != len(document.MultimodalData) {
		return nativePrompt{}, errors.New("prompt_string media marker count must match multimodal_data")
	}
	segments := strings.Split(document.PromptString, marker)
	before, after := segments[0], segments[1]
	if strings.HasPrefix(document.MultimodalData[0], "data:audio/") {
		if len(document.MultimodalData) != 1 {
			return nativePrompt{}, errors.New("audio cannot be combined with other media")
		}
		if h.config.AudioProjector == nil {
			return nativePrompt{}, errors.New("audio data provided, but the server has no audio projector")
		}
		audio, err := decodeNativeAudioData(document.MultimodalData[0])
		if err != nil {
			return nativePrompt{}, err
		}
		return nativePrompt{
			Text: document.PromptString, Response: document.PromptString,
			Audio: audio, BeforeMedia: before, AfterMedia: after,
		}, nil
	}
	if h.config.Qwen3VLProjector == nil && h.config.ImageProjector == nil {
		return nativePrompt{}, errors.New("image data provided, but the server has no image projector")
	}
	images := make([][]byte, len(document.MultimodalData))
	for index, encoded := range document.MultimodalData {
		if strings.HasPrefix(encoded, "data:audio/") {
			return nativePrompt{}, errors.New("audio cannot be combined with image media")
		}
		imageData, err := decodeNativeImageData(encoded)
		if err != nil {
			return nativePrompt{}, fmt.Errorf("multimodal_data image %d: %w", index, err)
		}
		images[index] = imageData
	}
	if err := validateMultimodalImages(images); err != nil {
		return nativePrompt{}, err
	}
	return nativePrompt{
		Text: document.PromptString, Response: document.PromptString,
		Image: images[0], Images: images, MediaText: segments, BeforeMedia: before, AfterMedia: after,
	}, nil
}

func decodeNativeAudioData(encoded string) ([]float32, error) {
	header, payload, ok := strings.Cut(encoded, ",")
	if !ok || !strings.HasPrefix(header, "data:audio/") || !strings.HasSuffix(header, ";base64") {
		return nil, errors.New("multimodal_data audio must use a base64 data URI")
	}
	if base64.StdEncoding.DecodedLen(len(payload)) > maxMediaBytes+2 {
		return nil, errors.New("multimodal_data audio exceeds decoded media limit")
	}
	decoded, err := base64.StdEncoding.DecodeString(payload)
	if err != nil || len(decoded) == 0 {
		return nil, errors.New("multimodal_data audio is not valid base64")
	}
	if len(decoded) > maxMediaBytes {
		return nil, errors.New("multimodal_data audio exceeds decoded media limit")
	}
	samples, sampleRate, err := projector.DecodeWAV(decoded)
	if err != nil {
		return nil, fmt.Errorf("multimodal_data audio: %w", err)
	}
	if sampleRate != 16000 {
		return nil, fmt.Errorf("multimodal_data audio sample rate %d Hz; want 16000 Hz", sampleRate)
	}
	return samples, nil
}

func decodeNativeImageData(encoded string) ([]byte, error) {
	if strings.HasPrefix(encoded, "data:") {
		header, payload, ok := strings.Cut(encoded, ",")
		if !ok || !strings.HasPrefix(header, "data:image/") || !strings.HasSuffix(header, ";base64") {
			return nil, errors.New("multimodal_data data URI must contain a base64 image")
		}
		encoded = payload
	}
	if encoded == "" {
		return nil, errors.New("multimodal_data image is empty")
	}
	if base64.StdEncoding.DecodedLen(len(encoded)) > maxImageBytes+2 {
		return nil, errors.New("multimodal_data image exceeds decoded image limit")
	}
	decoded, err := base64.StdEncoding.DecodeString(encoded)
	if err != nil {
		return nil, errors.New("multimodal_data image is not valid base64")
	}
	if len(decoded) == 0 {
		return nil, errors.New("multimodal_data image is empty")
	}
	if len(decoded) > maxImageBytes {
		return nil, errors.New("multimodal_data image exceeds decoded image limit")
	}
	return decoded, nil
}

func validateMultimodalImages(data [][]byte) error {
	if len(data) == 0 {
		return errors.New("multimodal image data is empty")
	}
	var totalBytes, totalPixels uint64
	for index, encoded := range data {
		if len(encoded) > maxImageBytes {
			return fmt.Errorf("multimodal_data image %d exceeds decoded image limit", index)
		}
		totalBytes += uint64(len(encoded))
		if totalBytes > maxMediaBytes {
			return errors.New("multimodal images exceed decoded media limit")
		}
		config, _, err := image.DecodeConfig(bytes.NewReader(encoded))
		if err != nil {
			return fmt.Errorf("multimodal_data image %d is unsupported", index)
		}
		if config.Width <= 0 || config.Height <= 0 ||
			config.Width > maxImageDimension || config.Height > maxImageDimension {
			return fmt.Errorf("multimodal_data image %d dimensions exceed limit", index)
		}
		pixels := uint64(config.Width) * uint64(config.Height)
		if pixels > maxImagePixels {
			return fmt.Errorf("multimodal_data image %d pixel count exceeds limit", index)
		}
		totalPixels += pixels
		if totalPixels > maxRequestImagePixels {
			return errors.New("multimodal images exceed aggregate pixel limit")
		}
	}
	return nil
}

func decodeMultimodalImages(data [][]byte) ([]image.Image, error) {
	if err := validateMultimodalImages(data); err != nil {
		return nil, err
	}
	images := make([]image.Image, len(data))
	for index, encoded := range data {
		input, _, err := image.Decode(bytes.NewReader(encoded))
		if err != nil {
			return nil, fmt.Errorf("multimodal_data image %d is unsupported", index)
		}
		images[index] = input
	}
	return images, nil
}

func (h *Handler) projectNativeMultimodalPrompt(
	ctx context.Context,
	prompt nativePrompt,
) (nativePrompt, inference.ProjectedInputs, error) {
	tokenizerAPI, ok := h.generator.(TokenizationAPI)
	if !ok {
		return nativePrompt{}, inference.ProjectedInputs{}, errors.New("server: generator cannot tokenize multimodal prompt")
	}
	var projected projector.MultimodalPrompt
	var err error
	if len(prompt.Audio) > 0 {
		if h.config.AudioProjector == nil {
			return nativePrompt{}, inference.ProjectedInputs{}, errors.New("server: audio projector is unavailable")
		}
		projected, err = h.config.AudioProjector.BuildAudioPrompt(
			ctx, tokenizerAPI, prompt.Audio, prompt.BeforeMedia, prompt.AfterMedia,
		)
	} else {
		imageData := prompt.Images
		if len(imageData) == 0 && len(prompt.Image) > 0 {
			imageData = [][]byte{prompt.Image}
		}
		images, decodeErr := decodeMultimodalImages(imageData)
		if decodeErr != nil {
			return nativePrompt{}, inference.ProjectedInputs{}, fmt.Errorf("server: %w", decodeErr)
		}
		if len(images) > 1 {
			multi, ok := h.config.ImageProjector.(projector.MultiImageProjector)
			if !ok {
				return nativePrompt{}, inference.ProjectedInputs{}, errors.New("server: selected projector does not support multiple images")
			}
			projected, err = multi.BuildImagesPrompt(ctx, tokenizerAPI, images, prompt.MediaText, true)
		} else if h.config.ImageProjector != nil {
			projected, err = h.config.ImageProjector.BuildImagePrompt(
				ctx, tokenizerAPI, images[0], prompt.BeforeMedia, prompt.AfterMedia, true,
			)
		} else {
			projected, err = h.config.Qwen3VLProjector.BuildQwen35ImagePrompt(
				ctx, tokenizerAPI, images[0], prompt.BeforeMedia, prompt.AfterMedia, true,
			)
		}
	}
	if err != nil {
		return nativePrompt{}, inference.ProjectedInputs{}, fmt.Errorf("server: project media: %w", err)
	}
	inputs, err := h.convertProjectedPrompt(projected)
	if err != nil {
		return nativePrompt{}, inference.ProjectedInputs{}, err
	}
	prompt.TokenIDs = projected.TokenIDs
	prompt.Image = nil
	prompt.Images = nil
	prompt.MediaText = nil
	prompt.Audio = nil
	return prompt, inputs, nil
}

func nativePromptHasMedia(prompt nativePrompt) bool {
	return len(prompt.Image) != 0 || len(prompt.Images) != 0 || len(prompt.Audio) != 0
}

func (h *Handler) preparePrompt(
	ctx context.Context,
	prompt nativePrompt,
	tokenizeText bool,
) (preparedPrompt, error) {
	result := preparedPrompt{
		nativePrompt: prompt,
		Multimodal:   nativePromptHasMedia(prompt),
	}
	if result.Multimodal {
		projectedPrompt, projected, err := h.projectNativeMultimodalPrompt(ctx, prompt)
		if err != nil {
			return preparedPrompt{}, err
		}
		result.nativePrompt = projectedPrompt
		result.ProjectedInputs = &projected
		return result, nil
	}
	if !tokenizeText || prompt.TokenIDs != nil {
		return result, nil
	}
	tokenizerAPI, ok := h.generator.(TokenizationAPI)
	if !ok {
		return preparedPrompt{}, errors.New("server: generator cannot tokenize prepared prompt")
	}
	tokens, err := tokenizerAPI.TokenizeText(prompt.Text, true, true)
	if err != nil {
		return preparedPrompt{}, err
	}
	if len(tokens) == 0 {
		return preparedPrompt{}, errors.New("server: prepared prompt produced no tokens")
	}
	result.TokenIDs = tokens
	return result, nil
}

func (h *Handler) convertProjectedPrompt(
	projected projector.MultimodalPrompt,
) (inference.ProjectedInputs, error) {
	if projected.EmbeddingWidth <= 0 || len(projected.Embeddings)%projected.EmbeddingWidth != 0 {
		return inference.ProjectedInputs{}, errors.New("server: projector returned invalid embeddings")
	}
	if properties, ok := h.generator.(ModelPropertiesAPI); ok &&
		projected.EmbeddingWidth != int(properties.ModelProperties().EmbeddingLength) {
		return inference.ProjectedInputs{}, fmt.Errorf(
			"server: projector width %d differs from model width %d",
			projected.EmbeddingWidth, properties.ModelProperties().EmbeddingLength,
		)
	}
	imageTokens := len(projected.EmbeddingTokenIndices)
	if len(projected.Embeddings) != imageTokens*projected.EmbeddingWidth {
		return inference.ProjectedInputs{}, errors.New("server: projector returned invalid embedding indices")
	}
	overrides := make([]inference.EmbeddingOverride, imageTokens)
	for index := range overrides {
		start := index * projected.EmbeddingWidth
		overrides[index] = inference.EmbeddingOverride{
			TokenIndex: projected.EmbeddingTokenIndices[index],
			Embedding:  projected.Embeddings[start : start+projected.EmbeddingWidth],
		}
	}
	inputs := inference.ProjectedInputs{EmbeddingOverrides: overrides}
	inputs.BidirectionalAttentionBlocks = make([]inference.AttentionBlock, len(projected.AttentionBlocks))
	for index, block := range projected.AttentionBlocks {
		inputs.BidirectionalAttentionBlocks[index] = inference.AttentionBlock{
			Start: block.Start, End: block.End,
		}
	}
	hasMultiAxis := false
	for _, axis := range projected.MultiAxisPositions {
		hasMultiAxis = hasMultiAxis || len(axis) > 0
	}
	if hasMultiAxis {
		positions := inference.MultiAxisPositions(projected.MultiAxisPositions)
		inputs.MultiAxisPositions = &positions
	}
	return inputs, nil
}

func (h *Handler) parseNativePrompt(raw json.RawMessage) (nativePrompt, error) {
	if len(raw) == 0 {
		return nativePrompt{}, errors.New("prompt is required")
	}
	var text string
	if err := json.Unmarshal(raw, &text); err == nil {
		if text == "" {
			return nativePrompt{}, errors.New("prompt must not be empty")
		}
		return nativePrompt{Text: text, Response: text}, nil
	}
	var parts []json.RawMessage
	if err := json.Unmarshal(raw, &parts); err != nil {
		return nativePrompt{}, errors.New("prompt must be a string or token/string sequence")
	}
	if len(parts) == 0 {
		return nativePrompt{}, errors.New("prompt token/string sequence must not be empty")
	}
	allStrings := true
	for _, part := range parts {
		var partText string
		if json.Unmarshal(part, &partText) != nil {
			allStrings = false
			break
		}
	}
	if allStrings {
		return nativePrompt{}, errors.New("multiple string prompts are not supported")
	}
	api, ok := h.generator.(TokenizationAPI)
	if !ok {
		return nativePrompt{}, errors.New("generator cannot expand mixed prompt strings")
	}
	const maxPromptTokens = 1 << 20
	tokenIDs := make([]tokenizer.TokenID, 0, len(parts))
	for index, part := range parts {
		var token int64
		if err := json.Unmarshal(part, &token); err == nil {
			if token < 0 || token >= int64(api.SamplingVocabularySize()) {
				return nativePrompt{}, fmt.Errorf("prompt token %d is out of range", token)
			}
			tokenIDs = append(tokenIDs, tokenizer.TokenID(token))
		} else {
			var partText string
			if err := json.Unmarshal(part, &partText); err != nil {
				return nativePrompt{}, fmt.Errorf(
					"prompt element %d must be an integer token ID or string",
					index,
				)
			}
			expanded, err := api.TokenizeText(partText, index == 0, false)
			if err != nil {
				return nativePrompt{}, fmt.Errorf("tokenize prompt element %d: %w", index, err)
			}
			tokenIDs = append(tokenIDs, expanded...)
		}
		if len(tokenIDs) > maxPromptTokens {
			return nativePrompt{}, errors.New("prompt token count exceeds 1048576")
		}
	}
	if len(tokenIDs) == 0 {
		return nativePrompt{}, errors.New("prompt produced no tokens")
	}
	processed := ""
	if decoder, ok := h.generator.(PromptTokenDecoder); ok {
		var err error
		processed, err = decoder.DetokenizePromptTokens(tokenIDs)
		if err != nil {
			return nativePrompt{}, fmt.Errorf("detokenize processed prompt: %w", err)
		}
	} else {
		var err error
		processed, err = api.DetokenizeTokens(tokenIDs)
		if err != nil {
			return nativePrompt{}, fmt.Errorf("detokenize processed prompt: %w", err)
		}
	}
	return nativePrompt{
		Text:     processed,
		TokenIDs: tokenIDs,
		Response: processed,
	}, nil
}

func validateNativeCompletionOptions(body nativeCompletionRequest) error {
	switch {
	case body.NIndent < 0:
		return errors.New("n_indent must be non-negative")
	case body.NKeep < -1:
		return errors.New("n_keep must be at least -1")
	case body.NDiscard < 0:
		return errors.New("n_discard must be non-negative")
	case body.NCacheReuse < 0:
		return errors.New("n_cache_reuse is negative")
	case body.NCacheReuse > 0 && (body.CachePrompt == nil || !*body.CachePrompt):
		return errors.New("n_cache_reuse requires cache_prompt")
	case body.ProjectedInputs != nil && body.CachePrompt != nil && *body.CachePrompt:
		return errors.New("projected_inputs cannot use cache_prompt")
	case body.NProbs < 0:
		return errors.New("n_probs must be non-negative")
	case body.TMaxPredictMS < -1:
		return errors.New("t_max_predict_ms must be at least -1")
	case body.SSEPingInterval != nil &&
		(math.IsNaN(*body.SSEPingInterval) ||
			math.IsInf(*body.SSEPingInterval, 0) ||
			*body.SSEPingInterval < -1 ||
			*body.SSEPingInterval > math.MaxInt32 ||
			math.Trunc(*body.SSEPingInterval) != *body.SSEPingInterval):
		return errors.New("sse_ping_interval must be an integer in [-1,2147483647]")
	case body.PostSamplingProbs && body.NProbs == 0:
		return errors.New("post_sampling_probs requires positive n_probs")
	case len(body.ResponseFields) > 64:
		return errors.New("response_fields count exceeds 64")
	case nativeJSONSchemaConfigured(body.JSONSchema) &&
		(body.Grammar != "" ||
			len(body.GrammarChoices) > 0 ||
			body.GrammarLazy ||
			len(body.GrammarTriggerPatterns) > 0 ||
			len(body.GrammarTriggerTokens) > 0):
		return errors.New("json_schema cannot be combined with grammar options")
	default:
		for index, path := range body.ResponseFields {
			if len(path) > 256 {
				return fmt.Errorf("response_fields path %d exceeds 256 bytes", index)
			}
			if strings.Count(path, "/") >= 16 {
				return fmt.Errorf("response_fields path %d exceeds 16 components", index)
			}
		}
		return nil
	}
}

func (h *Handler) parseRequestLoRA(raw json.RawMessage) ([]inference.LoRAScale, bool, error) {
	trimmed := strings.TrimSpace(string(raw))
	if trimmed == "" || trimmed == "null" {
		return nil, false, nil
	}
	controller, ok := h.generator.(LoRAControlAPI)
	if !ok {
		return nil, false, errors.New("per-request lora is unavailable")
	}
	decoder := json.NewDecoder(strings.NewReader(trimmed))
	decoder.DisallowUnknownFields()
	var requested []inference.LoRAScale
	if err := decoder.Decode(&requested); err != nil {
		return nil, false, fmt.Errorf("invalid lora: %w", err)
	}
	if err := requireEOF(decoder); err != nil {
		return nil, false, err
	}
	loaded := controller.LoRAAdapters()
	valid := make(map[int]struct{}, len(loaded))
	for _, adapter := range loaded {
		valid[adapter.ID] = struct{}{}
	}
	seen := make(map[int]struct{}, len(requested))
	for _, adapter := range requested {
		if _, exists := valid[adapter.ID]; !exists {
			return nil, false, fmt.Errorf("lora adapter ID %d is not loaded", adapter.ID)
		}
		if _, duplicate := seen[adapter.ID]; duplicate {
			return nil, false, fmt.Errorf("lora adapter ID %d is duplicated", adapter.ID)
		}
		if math.IsNaN(float64(adapter.Scale)) || math.IsInf(float64(adapter.Scale), 0) {
			return nil, false, fmt.Errorf("lora adapter ID %d scale is invalid", adapter.ID)
		}
		seen[adapter.ID] = struct{}{}
	}
	return requested, true, nil
}

func nativeJSONSchemaConfigured(value json.RawMessage) bool {
	trimmed := strings.TrimSpace(string(value))
	return trimmed != "" && trimmed != "null"
}

func grammarOptionsConfigured(parameters samplingParameters) bool {
	return parameters.Grammar != "" ||
		len(parameters.GrammarChoices) > 0 ||
		parameters.GrammarLazy ||
		len(parameters.GrammarTriggerPatterns) > 0 ||
		len(parameters.GrammarTriggerTokens) > 0
}

func prepareStructuredOutput(
	parameters *samplingParameters,
	topLevelSchema, responseFormat json.RawMessage,
) error {
	schema := topLevelSchema
	if nativeJSONSchemaConfigured(schema) && grammarOptionsConfigured(*parameters) {
		return errors.New("json_schema cannot be combined with grammar options")
	}
	if nativeJSONSchemaConfigured(responseFormat) {
		var format map[string]json.RawMessage
		if err := json.Unmarshal(responseFormat, &format); err != nil || format == nil {
			return errors.New("response_format must be an object")
		}
		var responseType string
		if rawType, ok := format["type"]; ok {
			if err := json.Unmarshal(rawType, &responseType); err != nil {
				return errors.New("response_format.type must be a string")
			}
		}
		switch responseType {
		case "", "text":
		case "json_object":
			rawSchema, hasSchema := format["schema"]
			if hasSchema || !nativeJSONSchemaConfigured(schema) {
				if hasSchema {
					schema = rawSchema
				} else {
					schema = json.RawMessage(`{}`)
				}
			}
		case "json_schema":
			schema = json.RawMessage(`{}`)
			if rawWrapper, ok := format["json_schema"]; ok {
				var wrapper map[string]json.RawMessage
				if err := json.Unmarshal(rawWrapper, &wrapper); err != nil || wrapper == nil {
					return errors.New("response_format.json_schema must be an object")
				}
				if rawSchema, ok := wrapper["schema"]; ok {
					schema = rawSchema
				}
			}
		default:
			return fmt.Errorf(
				"response_format.type must be one of %q, %q, or %q, got %q",
				"text",
				"json_object",
				"json_schema",
				responseType,
			)
		}
	}
	if !nativeJSONSchemaConfigured(schema) {
		return nil
	}
	if grammarOptionsConfigured(*parameters) {
		return errors.New("json_schema cannot be combined with grammar options")
	}
	grammar, err := sampling.JSONSchemaToGrammar(schema)
	if err != nil {
		return fmt.Errorf("json_schema: %w", err)
	}
	parameters.Grammar = grammar
	parameters.GrammarRoot = "root"
	return nil
}

func projectNativeResponse(
	response nativeCompletionResponse,
	paths []string,
) (map[string]any, error) {
	encoded, err := json.Marshal(response)
	if err != nil {
		return nil, fmt.Errorf("marshal native response for projection: %w", err)
	}
	var source map[string]any
	if err := json.Unmarshal(encoded, &source); err != nil {
		return nil, fmt.Errorf("decode native response for projection: %w", err)
	}
	result := make(map[string]any, len(paths))
	for _, path := range paths {
		var current any = source
		valid := true
		for _, component := range strings.Split(path, "/") {
			object, ok := current.(map[string]any)
			if !ok {
				valid = false
				break
			}
			current, ok = object[component]
			if !ok {
				valid = false
				break
			}
		}
		if valid {
			result[path] = current
		}
	}
	return result, nil
}

func (h *Handler) nativeGenerationSettings(
	body nativeCompletionRequest,
	config sampling.Config,
	maxTokens int,
	stops []string,
) map[string]any {
	contextLength := uint32(0)
	if api, ok := h.generator.(ModelPropertiesAPI); ok {
		contextLength = api.ModelProperties().ContextLength
	}
	return map[string]any{
		"model":                 h.config.ModelID,
		"n_ctx":                 contextLength,
		"n_predict":             maxTokens,
		"n_cmpl":                body.NCmpl,
		"seed":                  config.Seed,
		"temperature":           config.Temperature,
		"dynatemp_range":        config.DynatempRange,
		"dynatemp_exponent":     config.DynatempExponent,
		"top_k":                 config.TopK,
		"top_p":                 config.TopP,
		"min_p":                 config.MinP,
		"typical_p":             config.TypicalP,
		"top_n_sigma":           config.TopNSigma,
		"xtc_probability":       config.XTCProbability,
		"xtc_threshold":         config.XTCThreshold,
		"min_keep":              config.MinKeep,
		"repeat_last_n":         config.RepeatLastN,
		"repeat_penalty":        config.RepeatPenalty,
		"presence_penalty":      config.PresencePenalty,
		"frequency_penalty":     config.FrequencyPenalty,
		"dry_multiplier":        config.DryMultiplier,
		"dry_base":              config.DryBase,
		"dry_allowed_length":    config.DryAllowedLength,
		"dry_penalty_last_n":    config.DryPenaltyLastN,
		"dry_sequence_breakers": append([]string{}, body.DryBreakers...),
		"mirostat":              config.Mirostat,
		"mirostat_tau":          config.MirostatTau,
		"mirostat_eta":          config.MirostatEta,
		"samplers":              append([]sampling.SamplerStage(nil), config.Samplers...),
		"stop":                  append([]string{}, stops...),
		"ignore_eos":            body.IgnoreEOS,
		"stream":                body.Stream,
		"return_tokens":         body.ReturnTokens,
		"return_progress":       body.ReturnProgress,
		"timings_per_token":     body.TimingsPerToken,
		"t_max_predict_ms":      body.TMaxPredictMS,
		"n_indent":              body.NIndent,
		"n_probs":               body.NProbs,
		"post_sampling_probs":   body.PostSamplingProbs,
		"n_keep":                body.NKeep,
		"n_discard":             body.NDiscard,
		"cache_prompt":          body.CachePrompt != nil && *body.CachePrompt,
		"projected_inputs":      body.ProjectedInputs != nil,
		"sse_ping_interval":     nativeSSEPingInterval(body),
	}
}

func nativeSSEPingInterval(body nativeCompletionRequest) int {
	if body.SSEPingInterval == nil {
		return 30
	}
	return int(*body.SSEPingInterval)
}

func (h *Handler) runNativeCompletion(
	ctx context.Context,
	prompt nativePrompt,
	sampler *sampling.Sampler,
	maxTokens int,
	stops []string,
	index int,
	returnTokens bool,
	settings map[string]any,
	onChunk func(nativeCompletionChunk) error,
	onPromptProgress func(nativePromptProgress) error,
	timingsPerToken bool,
	maxPredictMS int,
	nIndent int,
	nProbs int,
	postSamplingProbabilities bool,
	slotID int,
	nKeep int,
	nDiscard int,
	cachePrompt bool,
	minCacheReuse int,
	lora []inference.LoRAScale,
	loraConfigured bool,
	projectedInputs *inference.ProjectedInputs,
) (nativeCompletionResponse, error) {
	started := time.Now()
	var output strings.Builder
	filter := newStopFilter(stops)
	generated := make([]tokenizer.TokenID, 0, maxTokens)
	probabilities := make([]nativeTokenProbability, 0, maxTokens)
	promptTokensEstimate := 0
	promptEvaluation := inference.PromptEvaluation{}
	if prompt.TokenIDs != nil {
		promptTokensEstimate = len(prompt.TokenIDs)
	} else if api, ok := h.generator.(TokenizationAPI); ok {
		if promptIDs, err := api.TokenizeText(prompt.Text, true, false); err == nil {
			promptTokensEstimate = len(promptIDs)
		}
	}
	if onPromptProgress != nil {
		if err := onPromptProgress(nativePromptProgress{
			Total: promptTokensEstimate,
		}); err != nil {
			return nativeCompletionResponse{}, err
		}
	}
	var promptProgressErr error
	var predictionStarted time.Time
	timeLimitReached := false
	indentationLimitReached := false
	ids, _, err := h.generate(
		ctx,
		slotID,
		prompt.Text,
		inference.GenerateOptions{
			MaxNewTokens:    maxTokens,
			Sampler:         sampler,
			StopSequences:   stops,
			ContextShift:    h.config.ContextShift,
			KeepTokens:      nKeep,
			DiscardTokens:   nDiscard,
			PromptTokenIDs:  prompt.TokenIDs,
			CachePrompt:     cachePrompt,
			MinCacheReuse:   minCacheReuse,
			LoRA:            lora,
			LoRAConfigured:  loraConfigured,
			ProjectedInputs: projectedInputs,
			PostSamplingProbabilities: func() int {
				if postSamplingProbabilities {
					return nProbs
				}
				return 0
			}(),
			OnPromptEvaluated: func(evaluation inference.PromptEvaluation) {
				promptEvaluation = evaluation
				if onPromptProgress != nil {
					promptProgressErr = onPromptProgress(nativePromptProgress{
						Total:     evaluation.Tokens,
						Cache:     evaluation.Cached,
						Processed: evaluation.Tokens,
						TimeMS:    evaluation.Duration.Milliseconds(),
					})
				}
			},
			OnToken: func(event inference.TokenEvent) error {
				if promptProgressErr != nil {
					return promptProgressErr
				}
				generated = append(generated, event.ID)
				var probability *nativeTokenProbability
				if nProbs > 0 {
					var item nativeTokenProbability
					var probabilityErr error
					if postSamplingProbabilities {
						item, probabilityErr =
							h.nativePostSamplingProbability(event)
					} else {
						item, probabilityErr = h.nativeTokenProbability(
							event.ID,
							event.Logits,
							nProbs,
						)
					}
					if probabilityErr != nil {
						return probabilityErr
					}
					probabilities = append(probabilities, item)
					probability = &probabilities[len(probabilities)-1]
				}
				piece := filter.Accept(event.Piece)
				previousLength := output.Len()
				output.WriteString(piece)
				if nIndent > 0 {
					if trimmed, stop := enforceNativeIndentation(
						output.String(),
						nIndent,
					); stop {
						trimmed = strings.Clone(trimmed)
						output.Reset()
						output.WriteString(trimmed)
						indentationLimitReached = true
						if previousLength < len(trimmed) {
							piece = trimmed[previousLength:]
						} else {
							piece = ""
						}
					}
				}
				if onChunk != nil && piece != "" {
					chunk := nativeCompletionChunk{
						Index:           index,
						Content:         piece,
						Tokens:          []tokenizer.TokenID{event.ID},
						Stop:            false,
						IDSlot:          -1,
						TokensPredicted: len(generated),
						TokensEvaluated: promptTokensEstimate,
					}
					if timingsPerToken {
						timings := measuredNativeTimings(
							promptTokensEstimate,
							len(generated),
							promptEvaluation,
							time.Since(started),
						)
						chunk.Timings = &timings
					}
					if probability != nil {
						chunk.CompletionProbabilities =
							[]nativeTokenProbability{*probability}
					}
					return onChunk(chunk)
				}
				return ctx.Err()
			},
			ShouldStop: func(event inference.TokenEvent) bool {
				if indentationLimitReached {
					return true
				}
				now := time.Now()
				if predictionStarted.IsZero() {
					predictionStarted = now
				}
				if maxPredictMS > 0 &&
					strings.Contains(event.Piece, "\n") &&
					now.Sub(predictionStarted) >
						time.Duration(maxPredictMS)*time.Millisecond {
					timeLimitReached = true
					return true
				}
				return false
			},
		},
	)
	if err != nil {
		return nativeCompletionResponse{}, err
	}
	if promptProgressErr != nil {
		return nativeCompletionResponse{}, promptProgressErr
	}
	promptTokens := len(ids) - len(generated)
	if pending := filter.Flush(); pending != "" {
		previousLength := output.Len()
		output.WriteString(pending)
		if nIndent > 0 {
			if trimmed, stop := enforceNativeIndentation(output.String(), nIndent); stop {
				trimmed = strings.Clone(trimmed)
				output.Reset()
				output.WriteString(trimmed)
				indentationLimitReached = true
				if previousLength < len(trimmed) {
					pending = trimmed[previousLength:]
				} else {
					pending = ""
				}
			}
		}
		if onChunk != nil && pending != "" {
			if err := onChunk(nativeCompletionChunk{
				Index:           index,
				Content:         pending,
				Tokens:          []tokenizer.TokenID{},
				Stop:            false,
				IDSlot:          -1,
				TokensPredicted: len(generated),
				TokensEvaluated: promptTokens,
			}); err != nil {
				return nativeCompletionResponse{}, err
			}
		}
	}
	stopType := "eos"
	if filter.Stopped() {
		stopType = "word"
	} else if indentationLimitReached ||
		timeLimitReached ||
		len(generated) >= maxTokens {
		stopType = "limit"
	}
	timings := measuredNativeTimings(
		promptTokens,
		len(generated),
		promptEvaluation,
		time.Since(started),
	)
	truncated := false
	if api, ok := h.generator.(ModelPropertiesAPI); ok {
		contextLength := api.ModelProperties().ContextLength
		truncated = contextLength > 0 && len(ids) > int(contextLength)
	}
	tokens := []tokenizer.TokenID{}
	if returnTokens || onChunk != nil {
		tokens = append(tokens, generated...)
	}
	content := output.String()
	return nativeCompletionResponse{
		Index:                   index,
		Content:                 content,
		Tokens:                  tokens,
		IDSlot:                  slotID,
		Stop:                    true,
		Model:                   h.config.ModelID,
		TokensPredicted:         len(generated),
		TokensEvaluated:         promptTokens,
		GenerationSettings:      settings,
		Prompt:                  prompt.Response,
		HasNewLine:              strings.Contains(content, "\n"),
		Truncated:               truncated,
		StopType:                stopType,
		StoppingWord:            filter.StoppingWord(),
		TokensCached:            promptEvaluation.Cached,
		Timings:                 timings,
		CompletionProbabilities: probabilities,
	}, nil
}

func nativeTimings(promptTokens, generatedTokens int, elapsed time.Duration) nativeCompletionTimings {
	milliseconds := float64(elapsed) / float64(time.Millisecond)
	result := nativeCompletionTimings{
		PromptN:     promptTokens,
		PredictedN:  generatedTokens,
		PredictedMS: milliseconds,
	}
	if generatedTokens > 0 && milliseconds > 0 {
		result.PredictedPerTokenMS = milliseconds / float64(generatedTokens)
		result.PredictedPerSecond = float64(generatedTokens) * 1000 / milliseconds
	}
	return result
}

func enforceNativeIndentation(content string, minimum int) (string, bool) {
	if minimum <= 0 {
		return content, false
	}
	for searchFrom := 0; ; {
		newline := strings.IndexByte(content[searchFrom:], '\n')
		if newline < 0 {
			return content, false
		}
		lineStart := searchFrom + newline + 1
		position := lineStart
		for position < len(content) &&
			(content[position] == ' ' || content[position] == '\t') {
			position++
		}
		if position == len(content) {
			return content, false
		}
		if position-lineStart < minimum {
			return content[:position], true
		}
		searchFrom = position
	}
}

func (h *Handler) nativeTokenProbability(
	selected tokenizer.TokenID,
	logits []float32,
	n int,
) (nativeTokenProbability, error) {
	if n <= 0 {
		return nativeTokenProbability{}, nil
	}
	if int(selected) < 0 || int(selected) >= len(logits) {
		return nativeTokenProbability{}, fmt.Errorf(
			"selected token %d is outside %d logits",
			selected,
			len(logits),
		)
	}
	pieces, ok := h.generator.(TokenPieceAPI)
	if !ok {
		return nativeTokenProbability{}, errors.New(
			"token probability text is unavailable",
		)
	}
	maximum := math.Inf(-1)
	for _, logit := range logits {
		if float64(logit) > maximum {
			maximum = float64(logit)
		}
	}
	total := 0.0
	for _, logit := range logits {
		total += math.Exp(float64(logit) - maximum)
	}
	if total <= 0 || math.IsNaN(total) || math.IsInf(total, 0) {
		return nativeTokenProbability{}, errors.New(
			"token probability softmax normalization is invalid",
		)
	}
	logNormalization := maximum + math.Log(total)
	indices := make([]int, len(logits))
	for index := range indices {
		indices[index] = index
	}
	sort.Slice(indices, func(left, right int) bool {
		leftLogit := logits[indices[left]]
		rightLogit := logits[indices[right]]
		if leftLogit == rightLogit {
			return indices[left] < indices[right]
		}
		return leftLogit > rightLogit
	})
	if n < len(indices) {
		indices = indices[:n]
	}
	itemFor := func(id tokenizer.TokenID) (nativeTokenLogProbability, error) {
		piece, err := pieces.TokenPiece(id)
		if err != nil {
			return nativeTokenLogProbability{}, err
		}
		bytes := []byte(piece)
		byteValues := make([]int, len(bytes))
		for index, value := range bytes {
			byteValues[index] = int(value)
		}
		return nativeTokenLogProbability{
			ID:      id,
			Token:   piece,
			Bytes:   byteValues,
			LogProb: float64(logits[int(id)]) - logNormalization,
		}, nil
	}
	selectedItem, err := itemFor(selected)
	if err != nil {
		return nativeTokenProbability{}, err
	}
	result := nativeTokenProbability{
		ID:          selectedItem.ID,
		Token:       selectedItem.Token,
		Bytes:       selectedItem.Bytes,
		TopLogProbs: make([]nativeTokenLogProbability, 0, len(indices)),
	}
	selectedLogProbability := selectedItem.LogProb
	result.LogProb = &selectedLogProbability
	for _, index := range indices {
		item, itemErr := itemFor(tokenizer.TokenID(index))
		if itemErr != nil {
			return nativeTokenProbability{}, itemErr
		}
		result.TopLogProbs = append(result.TopLogProbs, item)
	}
	return result, nil
}

func (h *Handler) nativePostSamplingProbability(
	event inference.TokenEvent,
) (nativeTokenProbability, error) {
	pieces, ok := h.generator.(TokenPieceAPI)
	if !ok {
		return nativeTokenProbability{}, errors.New(
			"token probability text is unavailable",
		)
	}
	itemFor := func(id tokenizer.TokenID) (string, []int, error) {
		piece, err := pieces.TokenPiece(id)
		if err != nil {
			return "", nil, err
		}
		raw := []byte(piece)
		bytes := make([]int, len(raw))
		for index, value := range raw {
			bytes[index] = int(value)
		}
		return piece, bytes, nil
	}
	piece, bytes, err := itemFor(event.ID)
	if err != nil {
		return nativeTokenProbability{}, err
	}
	selectedProbability := event.SelectedProbability
	result := nativeTokenProbability{
		ID:       event.ID,
		Token:    piece,
		Bytes:    bytes,
		Prob:     &selectedProbability,
		TopProbs: make([]nativeTokenProbabilityValue, 0, len(event.TopProbabilities)),
	}
	for _, item := range event.TopProbabilities {
		itemPiece, itemBytes, itemErr := itemFor(tokenizer.TokenID(item.ID))
		if itemErr != nil {
			return nativeTokenProbability{}, itemErr
		}
		result.TopProbs = append(result.TopProbs, nativeTokenProbabilityValue{
			ID:    tokenizer.TokenID(item.ID),
			Token: itemPiece,
			Bytes: itemBytes,
			Prob:  item.Probability,
		})
	}
	return result, nil
}

func measuredNativeTimings(
	promptTokens int,
	generatedTokens int,
	evaluation inference.PromptEvaluation,
	elapsed time.Duration,
) nativeCompletionTimings {
	result := nativeTimings(promptTokens, generatedTokens, elapsed)
	result.CacheN = evaluation.Cached
	result.PromptN = max(promptTokens-evaluation.Cached, 0)
	result.PromptMS = float64(evaluation.Duration) / float64(time.Millisecond)
	if result.PromptN > 0 && result.PromptMS > 0 {
		result.PromptPerTokenMS = result.PromptMS / float64(result.PromptN)
		result.PromptPerSecond = float64(result.PromptN) * 1000 / result.PromptMS
	}
	predictedDuration := elapsed - evaluation.Duration
	if predictedDuration < 0 {
		predictedDuration = 0
	}
	result.PredictedMS = float64(predictedDuration) / float64(time.Millisecond)
	if generatedTokens > 0 && result.PredictedMS > 0 {
		result.PredictedPerTokenMS = result.PredictedMS / float64(generatedTokens)
		result.PredictedPerSecond =
			float64(generatedTokens) * 1000 / result.PredictedMS
	}
	return result
}

func (h *Handler) streamNativeCompletion(
	response http.ResponseWriter,
	request *http.Request,
	prompts []nativePrompt,
	sampler *sampling.Sampler,
	maxTokens int,
	stops []string,
	n int,
	settings map[string]any,
	responseFields []string,
	slotID int,
	nKeep int,
	nDiscard int,
	cachePrompt bool,
	minCacheReuse int,
	pingIntervalSeconds int,
	returnProgress bool,
	timingsPerToken bool,
	maxPredictMS int,
	nIndent int,
	nProbs int,
	postSamplingProbabilities bool,
	lora []inference.LoRAScale,
	loraConfigured bool,
	projectedInputs *inference.ProjectedInputs,
) {
	flusher, ok := response.(http.Flusher)
	if !ok {
		writeError(response, http.StatusInternalServerError, "server_error", "streaming is unavailable")
		return
	}
	response.Header().Set("Content-Type", "text/event-stream")
	response.Header().Set("Cache-Control", "no-cache")
	response.Header().Set("X-Accel-Buffering", "no")
	response.WriteHeader(http.StatusOK)
	stream := newSynchronizedSSE(response, flusher)
	stopHeartbeat := stream.startHeartbeat(
		request.Context(),
		time.Duration(pingIntervalSeconds)*time.Second,
	)
	defer stopHeartbeat()
	resultIndex := 0
	for _, prompt := range prompts {
		for range n {
			choiceSampler, err := samplerForChoice(sampler, resultIndex)
			if err != nil {
				_ = stream.write(errorEnvelope("generation_error", err.Error()))
				return
			}
			result, err := h.runNativeCompletion(
				request.Context(),
				prompt,
				choiceSampler,
				maxTokens,
				stops,
				resultIndex,
				true,
				settings,
				func(chunk nativeCompletionChunk) error {
					if err := stream.write(chunk); err != nil {
						return err
					}
					return request.Context().Err()
				},
				func(progress nativePromptProgress) error {
					if !returnProgress {
						return nil
					}
					return stream.write(nativeCompletionChunk{
						Index:           resultIndex,
						Content:         "",
						Tokens:          []tokenizer.TokenID{},
						Stop:            false,
						IDSlot:          -1,
						TokensPredicted: 0,
						TokensEvaluated: progress.Processed,
						PromptProgress:  &progress,
					})
				},
				timingsPerToken,
				maxPredictMS,
				nIndent,
				nProbs,
				postSamplingProbabilities,
				slotID,
				nKeep,
				nDiscard,
				cachePrompt,
				minCacheReuse,
				lora,
				loraConfigured,
				projectedInputs,
			)
			if err != nil {
				_ = stream.write(errorEnvelope("generation_error", err.Error()))
				return
			}
			// native llama.cpp stream terminates with full metadata
			// envelope, but content and tokens in that final event are empty
			// because they were already delivered by partial events
			result.Content = ""
			result.Tokens = []tokenizer.TokenID{}
			var final any = result
			if len(responseFields) > 0 {
				projected, projectErr := projectNativeResponse(result, responseFields)
				if projectErr != nil {
					_ = stream.write(errorEnvelope("server_error", projectErr.Error()))
					return
				}
				final = projected
			}
			if err := stream.write(final); err != nil {
				return
			}
			resultIndex++
		}
	}
}

type synchronizedSSE struct {
	mu      sync.Mutex
	writer  io.Writer
	flusher http.Flusher
	err     error
}

func newSynchronizedSSE(writer io.Writer, flusher http.Flusher) *synchronizedSSE {
	return &synchronizedSSE{writer: writer, flusher: flusher}
}

func (stream *synchronizedSSE) write(value any) error {
	stream.mu.Lock()
	defer stream.mu.Unlock()
	if stream.err != nil {
		return stream.err
	}
	stream.err = writeSSE(stream.writer, value)
	if stream.err == nil {
		stream.flusher.Flush()
	}
	return stream.err
}

func (stream *synchronizedSSE) ping() error {
	stream.mu.Lock()
	defer stream.mu.Unlock()
	if stream.err != nil {
		return stream.err
	}
	_, stream.err = io.WriteString(stream.writer, ":\n\n")
	if stream.err == nil {
		stream.flusher.Flush()
	}
	return stream.err
}

func (stream *synchronizedSSE) startHeartbeat(
	ctx context.Context,
	interval time.Duration,
) func() {
	if interval <= 0 {
		return func() {}
	}
	stop := make(chan struct{})
	stopped := make(chan struct{})
	go func() {
		defer close(stopped)
		ticker := time.NewTicker(interval)
		defer ticker.Stop()
		for {
			select {
			case <-ctx.Done():
				return
			case <-stop:
				return
			case <-ticker.C:
				if stream.ping() != nil {
					return
				}
			}
		}
	}()
	return func() {
		close(stop)
		<-stopped
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
	request *http.Request,
	slotID int,
	prompts []nativePrompt,
	sampler *sampling.Sampler,
	maxTokens int,
	id string,
	stops []string,
	n int,
) {
	choices := make([]completionChoice, 0, len(prompts)*n)
	promptTokens := 0
	totalCompletionTokens := 0
	choiceIndex := 0
	for _, prompt := range prompts {
		for promptChoice := range n {
			choiceSampler, err := samplerForChoice(sampler, choiceIndex)
			if err != nil {
				writeGenerationError(response, err)
				return
			}
			var output strings.Builder
			completionTokens := 0
			generatedTokens := 0
			filter := newStopFilter(stops)
			ids, _, err := h.generate(
				request.Context(),
				slotID,
				prompt.Text,
				inference.GenerateOptions{
					MaxNewTokens:   maxTokens,
					Sampler:        choiceSampler,
					PromptTokenIDs: prompt.TokenIDs,
					StopSequences:  stops,
					ContextShift:   h.config.ContextShift,
					OnToken: func(event inference.TokenEvent) error {
						generatedTokens++
						if !filter.Stopped() {
							completionTokens++
						}
						output.WriteString(filter.Accept(event.Piece))
						return nil
					},
				},
			)
			if err != nil {
				writeGenerationError(response, err)
				return
			}
			if promptChoice == 0 {
				promptTokens += len(ids) - generatedTokens
			}
			output.WriteString(filter.Flush())
			finishReason := "stop"
			if !filter.Stopped() && completionTokens == maxTokens {
				finishReason = "length"
			}
			totalCompletionTokens += completionTokens
			choices = append(choices, completionChoice{
				Text:         output.String(),
				Index:        choiceIndex,
				FinishReason: finishReason,
			})
			choiceIndex++
		}
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
	slotID int,
	prompts []nativePrompt,
	sampler *sampling.Sampler,
	maxTokens int,
	id string,
	stops []string,
	n int,
) {
	flusher, ok := response.(http.Flusher)
	if !ok {
		writeError(response, http.StatusInternalServerError, "server_error", "streaming is unavailable")
		return
	}
	response.Header().Set("Content-Type", "text/event-stream")
	response.Header().Set("Cache-Control", "no-cache")
	response.Header().Set("X-Accel-Buffering", "no")
	response.WriteHeader(http.StatusOK)
	created := time.Now().Unix()
	choiceIndex := 0
	for _, prompt := range prompts {
		for range n {
			choiceSampler, err := samplerForChoice(sampler, choiceIndex)
			if err != nil {
				_ = writeSSE(response, errorEnvelope("generation_error", err.Error()))
				break
			}
			completionTokens := 0
			filter := newStopFilter(stops)
			_, _, err = h.generate(
				request.Context(),
				slotID,
				prompt.Text,
				inference.GenerateOptions{
					MaxNewTokens:   maxTokens,
					Sampler:        choiceSampler,
					PromptTokenIDs: prompt.TokenIDs,
					StopSequences:  stops,
					ContextShift:   h.config.ContextShift,
					OnToken: func(event inference.TokenEvent) error {
						if !filter.Stopped() {
							completionTokens++
						}
						piece := filter.Accept(event.Piece)
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
						if writeErr := writeSSE(response, chunk); writeErr != nil {
							return writeErr
						}
						flusher.Flush()
						return request.Context().Err()
					},
				},
			)
			if err != nil {
				_ = writeSSE(response, errorEnvelope("generation_error", err.Error()))
				break
			}
			if piece := filter.Flush(); piece != "" {
				_ = writeSSE(response, streamResponse{
					ID: id, Object: "text_completion", Created: created, Model: h.config.ModelID,
					Choices: []streamChoice{{Text: piece, Index: choiceIndex}},
				})
				flusher.Flush()
			}
			reason := "stop"
			if !filter.Stopped() && completionTokens == maxTokens {
				reason = "length"
			}
			_ = writeSSE(response, streamResponse{
				ID:      id,
				Object:  "text_completion",
				Created: created,
				Model:   h.config.ModelID,
				Choices: []streamChoice{{
					Index:        choiceIndex,
					FinishReason: &reason,
				}},
			})
			flusher.Flush()
			choiceIndex++
		}
	}
	_, _ = io.WriteString(response, "data: [DONE]\n\n")
	flusher.Flush()
}

func writeGenerationError(response http.ResponseWriter, err error) {
	if errors.Is(err, context.Canceled) || errors.Is(err, context.DeadlineExceeded) {
		writeError(response, http.StatusRequestTimeout, "request_cancelled", err.Error())
		return
	}
	writeError(response, http.StatusInternalServerError, "generation_error", err.Error())
}

func requireEOF(decoder *json.Decoder) error {
	var extra any
	if err := decoder.Decode(&extra); err == io.EOF {
		return nil
	} else if err != nil {
		return errors.New("invalid trailing JSON: " + err.Error())
	}
	return errors.New("request must contain one JSON object")
}

func writeJSON(response http.ResponseWriter, status int, value any) {
	response.Header().Set("Content-Type", "application/json")
	response.WriteHeader(status)
	_ = json.NewEncoder(response).Encode(value)
}

func writeSSE(response io.Writer, value any) error {
	data, err := json.Marshal(value)
	if err != nil {
		return err
	}
	if _, err = io.WriteString(response, "data: "); err != nil {
		return err
	}
	if _, err = response.Write(data); err != nil {
		return err
	}
	_, err = io.WriteString(response, "\n\n")
	return err
}

func writeNamedSSE(response io.Writer, event string, value any) error {
	data, err := json.Marshal(value)
	if err != nil {
		return err
	}
	if _, err = io.WriteString(response, "event: "+event+"\n"); err != nil {
		return err
	}
	if _, err = io.WriteString(response, "data: "); err != nil {
		return err
	}
	if _, err = response.Write(data); err != nil {
		return err
	}
	_, err = io.WriteString(response, "\n\n")
	return err
}

type apiError struct {
	Message string `json:"message"`
	Type    string `json:"type"`
}

type apiErrorEnvelope struct {
	Error apiError `json:"error"`
}

func errorEnvelope(kind, message string) apiErrorEnvelope {
	return apiErrorEnvelope{Error: apiError{Message: message, Type: kind}}
}

func writeError(response http.ResponseWriter, status int, kind, message string) {
	writeJSON(response, status, errorEnvelope(kind, message))
}
