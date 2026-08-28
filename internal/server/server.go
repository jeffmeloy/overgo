package server

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
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

	"overgo/internal/agentloop"
	"overgo/internal/artifact"
	"overgo/internal/capabilityruntime"
	"overgo/internal/checked"
	"overgo/internal/cuda/driver"
	"overgo/internal/dataset"
	"overgo/internal/discovery"
	"overgo/internal/inference"
	"overgo/internal/modelrecipe"
	"overgo/internal/operation"
	"overgo/internal/overgodb"
	"overgo/internal/projector"
	"overgo/internal/recipe"
	"overgo/internal/runrecord"
	"overgo/internal/sampling"
	"overgo/internal/strictjson"
	"overgo/internal/tokenizer"
	"overgo/internal/workflowruntime"
)

const (
	counterStep = 1

	errorCodeUnsupportedOperation = "unsupported_operation"
	maxRequestBytes               = 1 << 20
	maxMultimodalRequestBytes     = 32 << 20
	maxImageBytes                 = 16 << 20
	maxMediaBytes                 = 24 << 20
	maxImageDimension             = 16384
	maxImagePixels                = 16 << 20
	maxRequestImagePixels         = 32 << 20
	maxCompletionChoices          = 8
)

type Generator interface {
	Generate(
		context.Context,
		string,
		inference.GenerateOptions,
	) ([]tokenizer.TokenID, string, error)
}

type runtimePolicyProvider interface {
	RuntimePolicy() modelrecipe.RuntimePolicy
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
	Detokenize([]tokenizer.TokenID, inference.TokenRenderMode) (string, error)
	SamplingVocabularySize() int
}

type TokenPieceAPI interface {
	TokenPiece(tokenizer.TokenID) (string, error)
}

type ModelPropertiesAPI interface {
	ModelProperties() inference.ModelProperties
}

type LoRAControlAPI interface {
	LoRAAdapters() []inference.LoRAAdapterInfo
	SetLoRAScales(context.Context, []inference.LoRAScale) error
}

type DeviceMemoryAPI interface {
	DeviceMemoryStats(context.Context) (driver.MemoryStats, error)
}

type DeviceExecutionAPI interface {
	DeviceExecutionStats(context.Context) (driver.ExecutionStats, error)
}

type toolCallExecutor interface {
	ExecuteTool(context.Context, recipe.ToolCall) (recipe.ToolResult, error)
}

type capabilityBundleAPI interface {
	CapabilityBundles() []artifact.ID
	LoadCapabilityComponent(context.Context, artifact.ID, artifact.ComponentRole, string) (artifact.Content, error)
}

type Config struct {
	RuntimePolicy      modelrecipe.RuntimePolicy
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
	SPMInfill          bool
	Qwen3VLProjector   projector.Session
	ImageProjector     projector.Session
	AudioProjector     projector.Session
	RemoteMediaPolicy  *RemoteMediaPolicy
	ResponseFiles      ResponseFileResolver
	ToolProgram        recipe.Program
	ToolAdapter        func(context.Context, recipe.ToolCall) (recipe.ToolResult, error)
	MaxStoredResponses int
	ResponseStoreBytes int
	DatasetPreview     DatasetPreviewAPI
	FFmpegPath         string
	VideoFPS           float64
	VideoMaxFrames     int
	// OvergoDBPath enables read-only catalog browsing (runs, artifacts,
	// datasets) when Repository is absent; the dataset catalog itself is
	// read from the store, never from a filesystem manifest.
	OvergoDBPath string
	Repository   *overgodb.Store
	// HubEndpoint and HubToken configure Hugging Face intake; an empty
	// endpoint means the public hub. HubDownloadRoot is the only directory
	// download jobs may write under; empty disables downloads.
	HubEndpoint     string
	HubToken        string
	HubDownloadRoot string
	Environment     runrecord.Environment
	Evaluation      EvaluationWorkspaceAPI
	Analysis        AnalysisPolicy
	AgentEmbedder   dataset.AgentEmbeddingProvider
	AgentReranker   dataset.AgentRerankProvider
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

func (stats *slotRuntimeStats) snapshot(includeText bool) (string, string, slotStatusParams) {
	stats.textMu.RLock()
	defer stats.textMu.RUnlock()
	params := stats.params
	params.Stop = slices.Clone(params.Stop)
	params.Samplers = slices.Clone(params.Samplers)
	if includeText {
		return stats.prompt, stats.generated.String(), params
	}
	return "", "", params
}

type Handler struct {
	config              Config
	generator           Generator
	sessions            *capabilityruntime.ModelSessionDirector[struct{}, Generator, struct{}]
	defaultSampling     sampling.Config
	defaultOutputTokens int
	slotBusy            []atomic.Bool
	slotTasks           []atomic.Uint64
	slotStats           []slotRuntimeStats
	nextTask            atomic.Uint64
	nextID              atomic.Uint64
	started             time.Time
	requestsTotal       atomic.Uint64
	requestsActive      atomic.Int64
	generationRequests  atomic.Uint64
	generationErrors    atomic.Uint64
	downloads           downloadRegistry
	catalogMemo         *discovery.Memo
	generatedTokens     atomic.Uint64
	mediaFetcher        *remoteMediaFetcher
	responseFiles       ResponseFileResolver
	thinkingSigner      *anthropicThinkingSigner
	operations          *operation.Manager
	tools               toolCallExecutor
	issuedCalls         *issuedCallRegistry
	agentCoordinator    *agentloop.Coordinator
	agentSessions       agentSessions
	repository          *overgodb.Store
	browseRepository    *overgodb.Store
	environment         runrecord.Environment
	modelArtifact       artifact.ID
	observationErrors   atomic.Uint64
}

func New(config Config, generator Generator) (*Handler, error) {
	if generator == nil {
		return nil, errors.New("server: generator is nil")
	}
	policy := config.RuntimePolicy
	if policy.ID == (artifact.ID{}) {
		provider, ok := generator.(runtimePolicyProvider)
		if !ok {
			return nil, errors.New("server: recipe runtime policy is required")
		}
		policy = provider.RuntimePolicy()
	}
	if err := policy.ValidateIdentity(); err != nil {
		return nil, fmt.Errorf("server: runtime policy: %w", err)
	}
	config.RuntimePolicy = policy
	defaults := policy.Serving
	if config.Analysis != (AnalysisPolicy{}) {
		if err := config.Analysis.validate(); err != nil {
			return nil, err
		}
	}
	if config.ModelID == "" {
		config.ModelID = defaults.ModelID
	}
	if config.MaxTokens == 0 {
		config.MaxTokens = defaults.MaxTokens
	}
	if config.MaxTokens < 0 {
		return nil, errors.New("server: max tokens must be positive")
	}
	if config.MaxConcurrent == 0 {
		config.MaxConcurrent = defaults.MaxConcurrent
	}
	if config.MaxConcurrent <= 0 {
		return nil, errors.New("server: max concurrent requests must be positive")
	}
	if config.MaxEmbeddingInputs == 0 {
		config.MaxEmbeddingInputs = defaults.MaxEmbeddingInputs
	}
	if config.MaxEmbeddingInputs < 0 {
		return nil, errors.New("server: max embedding inputs must be positive")
	}
	if config.DefaultTemperature == 0 {
		config.DefaultTemperature = defaults.Sampling.Temperature
	}
	if config.DefaultTopP == 0 {
		config.DefaultTopP = defaults.Sampling.TopP
	}
	if config.RequestTimeout < 0 {
		return nil, errors.New("server: request timeout must be non-negative")
	}
	if config.MaxStoredResponses == 0 {
		config.MaxStoredResponses = defaults.StoredResponses
	}
	if config.MaxStoredResponses <= 0 {
		return nil, errors.New("server: stored response count must be positive")
	}
	if config.ResponseStoreBytes == 0 {
		config.ResponseStoreBytes = defaults.ResponseStoreBytes
	}
	if config.ResponseStoreBytes <= 0 {
		return nil, errors.New("server: response store bytes must be positive")
	}
	if config.VideoFPS == 0 {
		config.VideoFPS = defaults.Video.FPS
	}
	if config.VideoFPS <= 0 || math.IsNaN(config.VideoFPS) || math.IsInf(config.VideoFPS, 0) {
		return nil, errors.New("server: video FPS must be finite and positive")
	}
	if config.VideoMaxFrames == 0 {
		config.VideoMaxFrames = defaults.Video.MaxFrames
	}
	if config.VideoMaxFrames <= 0 {
		return nil, errors.New("server: video frame limit must be positive")
	}
	mediaFetcher, err := newRemoteMediaFetcher(config.RemoteMediaPolicy)
	if err != nil {
		return nil, fmt.Errorf("server remote media policy: %w", err)
	}
	thinkingSigner, err := newAnthropicThinkingSigner()
	if err != nil {
		return nil, err
	}
	defaultSampling := defaults.Sampling.Config()
	defaultSampling.Temperature = config.DefaultTemperature
	if config.DefaultTopK != 0 {
		defaultSampling.TopK = config.DefaultTopK
	}
	defaultSampling.TopP = config.DefaultTopP
	defaultSampler, err := sampling.New(defaultSampling)
	if err != nil {
		return nil, fmt.Errorf("server defaults: %w", err)
	}
	repository, environment, err := openServingRepository(config)
	if err != nil {
		return nil, err
	}
	var browseRepository *overgodb.Store
	if repository == nil && config.OvergoDBPath != "" {
		browseRepository, err = overgodb.OpenReadOnly(config.OvergoDBPath)
		if err != nil {
			return nil, fmt.Errorf("server browse repository: %w", err)
		}
	}
	generation := generator
	if config.MaxConcurrent > 1 {
		if factory, ok := generator.(ContinuousGeneratorFactory); ok {
			candidate, schedulerErr := factory.NewContinuousGenerator(
				inference.ContinuousGeneratorOptions{
					MaxSequences: config.MaxConcurrent,
					ContextShift: config.ContextShift,
				},
			)
			if schedulerErr == nil {
				generation = candidate
			}
		}
	}
	sessions, err := capabilityruntime.NewResidentModelSessionDirector(
		"inference", "compiled", config.MaxConcurrent, generation,
	)
	if err != nil {
		if browseRepository != nil {
			_ = browseRepository.Close()
		}
		return nil, err
	}
	catalogMemo := discovery.NewMemo()
	if config.Repository != nil {
		// Seed from persisted identity evidence so the first catalog view
		// of this process answers from stat checks, not re-hashing.
		catalogMemo = discovery.LoadMemo(context.Background(), config.Repository)
	}
	handler := &Handler{
		catalogMemo:         catalogMemo,
		issuedCalls:         newIssuedCallRegistry(config.MaxTokens),
		config:              config,
		generator:           generator,
		sessions:            sessions,
		defaultSampling:     defaultSampler.Config(),
		defaultOutputTokens: defaults.OutputTokens,
		slotBusy:            make([]atomic.Bool, config.MaxConcurrent),
		slotTasks:           make([]atomic.Uint64, config.MaxConcurrent),
		slotStats:           make([]slotRuntimeStats, config.MaxConcurrent),
		started:             time.Now(),
		downloads:           newDownloadRegistry(config.MaxConcurrent, config.MaxStoredResponses),
		mediaFetcher:        mediaFetcher,
		responseFiles:       config.ResponseFiles,
		thinkingSigner:      thinkingSigner,
		repository:          repository,
		browseRepository:    browseRepository,
		environment:         environment,
	}
	if identity, ok := generator.(interface{ ModelID() artifact.ID }); ok {
		handler.modelArtifact = identity.ModelID()
	}
	if err := handler.seedResponseIdentifiers(context.Background()); err != nil {
		_ = handler.Close()
		return nil, fmt.Errorf("server response identity seed: %w", err)
	}
	handler.buildAgentRuntime()
	handler.operations, err = operation.NewManager(config.MaxStoredResponses)
	if err != nil {
		_ = handler.Close()
		return nil, err
	}
	if config.ToolAdapter != nil {
		if repository == nil {
			_ = handler.Close()
			return nil, errors.New("server: tool runtime requires a repository")
		}
		handler.tools, err = workflowruntime.NewToolExecutor(
			repository, config.ToolProgram, config.MaxStoredResponses, config.ToolAdapter,
		)
		if err != nil {
			_ = handler.Close()
			return nil, fmt.Errorf("server tool runtime: %w", err)
		}
	}
	return handler, nil
}

// Close: release serving session state.
func (h *Handler) Close() error {
	if h == nil {
		return nil
	}
	if h.operations != nil {
		h.operations.Close()
	}
	if tools, ok := h.tools.(*workflowruntime.ToolExecutor); ok {
		tools.Close()
	}
	var closeErrors []error
	if h.sessions != nil {
		ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		closeErrors = append(closeErrors, h.sessions.Close(ctx))
		cancel()
	}
	if h.browseRepository != nil {
		closeErrors = append(closeErrors, h.browseRepository.Close())
	}
	return errors.Join(closeErrors...)
}

type requestSession = capabilityruntime.SessionLease[Generator]

func (h *Handler) acquireSession(requested int) (*requestSession, bool) {
	lease, ok := h.sessions.TryLease(requested)
	if !ok {
		return nil, false
	}
	h.activateSession(lease)
	return lease, true
}

func (h *Handler) activateSession(lease *requestSession) {
	id := lease.ID
	h.slotTasks[id].Store(h.nextTask.Add(counterStep))
	h.slotStats[id].reset(time.Now())
	h.slotBusy[id].Store(true)
}

func (h *Handler) acquireRequestSession(ctx context.Context, response http.ResponseWriter, requested int) (*requestSession, bool) {
	lease, err := h.sessions.Lease(ctx, requested)
	if err != nil {
		if errors.Is(err, capabilityruntime.ErrAdmissionQueueFull) || errors.Is(err, capabilityruntime.ErrSessionUnavailable) {
			writeError(response, http.StatusTooManyRequests, "server_busy", err.Error())
		} else {
			writeGenerationError(response, err)
		}
		return nil, false
	}
	h.activateSession(lease)
	return lease, true
}

func (h *Handler) releaseSession(lease *requestSession) {
	if lease == nil {
		return
	}
	id := lease.ID
	if id < 0 || id >= len(h.slotBusy) {
		return
	}
	started := h.slotStats[id].startedNanos.Load()
	if started > 0 {
		h.slotStats[id].totalNanos.Store(max(time.Now().UnixNano()-started, 0))
	}
	h.slotBusy[id].Store(false)
	lease.Release()
}

func (h *Handler) ServeHTTP(response http.ResponseWriter, request *http.Request) {
	if h.config.RequestTimeout > 0 {
		ctx, cancel := context.WithTimeout(request.Context(), h.config.RequestTimeout)
		defer cancel()
		request = request.WithContext(ctx)
	}
	h.requestsTotal.Add(counterStep)
	h.requestsActive.Add(counterStep)
	defer h.requestsActive.Add(-counterStep)
	route, routed := resolveRoute(request.URL.Path)
	if routed && route.Authentication == routeBearer && !h.authorized(request) {
		response.Header().Set("WWW-Authenticate", "Bearer")
		writeError(
			response,
			http.StatusUnauthorized,
			"invalid_api_key",
			"missing or invalid bearer token",
		)
		return
	}
	if routed && len(route.Methods) != 0 && !route.accepts(request.Method) {
		response.Header().Set("Allow", strings.Join(route.Methods, ", "))
		writeError(response, http.StatusMethodNotAllowed, "method_not_allowed", strings.Join(route.Methods, " or ")+" required")
		return
	}
	if routed {
		route.serve(h, response, request)
		return
	}
	h.serveWebUI(response, request)
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
	adaptiveTarget := h.defaultSampling.AdaptiveTarget
	if body.AdaptiveTarget != nil {
		adaptiveTarget = *body.AdaptiveTarget
	}
	adaptiveDecay := h.defaultSampling.AdaptiveDecay
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
				Bias:  sampling.BannedLogit(),
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
	if !strictjson.HasValue(raw) {
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
		return sampling.BannedLogit(), nil
	}
	var value float32
	if err := json.Unmarshal(raw, &value); err != nil || !checked.Finite32(value) {
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
