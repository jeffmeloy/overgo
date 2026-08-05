package inference

import (
	"context"
	"errors"
	"fmt"
	"math"
	"slices"
	"sync"
	"time"

	"llamacpp2go/internal/cuda/device"
	"llamacpp2go/internal/cuda/executor"
	"llamacpp2go/internal/gguf"
	"llamacpp2go/internal/model"
	"llamacpp2go/internal/sampling"
	"llamacpp2go/internal/tensor/reference"
	"llamacpp2go/internal/tokenizer"
)

type TokenEvent struct {
	ID    tokenizer.TokenID
	Piece string
	Index int
	// Logits: transient view of pre-sampling logits for this token
	// Callers must copy it if they retain it after callback returns
	Logits              []float32
	SelectedProbability float64
	TopProbabilities    []sampling.TokenProbability
}

type PromptEvaluation struct {
	Tokens   int
	Cached   int
	Duration time.Duration
}

type GenerateOptions struct {
	MaxNewTokens int
	Sampler      *sampling.Sampler
	OnToken      func(TokenEvent) error
	// ShouldStop: evaluated after OnToken and after sampled piece has
	// been appended to generated text; Returning true ends generation
	// successfully while retaining that token
	ShouldStop func(TokenEvent) bool
	// PostSamplingProbabilities: requests top N normalized candidates after
	// configured sampler chain; Zero disables this instrumentation
	PostSamplingProbabilities int
	ParseSpecial              bool
	StopSequences             []string
	// PromptTokenIDs, when non-nil, replaces text tokenization with this exact
	// caller-owned prompt sequence; No BOS/EOS token is inserted implicitly
	PromptTokenIDs []tokenizer.TokenID
	// ContextShift: permits generation to discard oldest attention KV
	// entries when active cache reaches model context length; Absolute
	// token positions and hybrid recurrent state are preserved
	ContextShift bool
	// KeepTokens: preserves this many initial prompt tokens when ContextShift
	// compacts full host cache; Minus one preserves as much of initial
	// prompt as context permits
	KeepTokens int
	// DiscardTokens: controls how many entries after KeepTokens are removed per
	// context shift; Zero uses half of discardable cache, matching
	// native server convention
	DiscardTokens int
	// CachePrompt: retains evaluated prompt state for later request whose
	// token sequence has this prompt as prefix
	CachePrompt bool
	// MinCacheReuse: requires at least this many matching prefix tokens before
	// retained prompt is reused
	MinCacheReuse     int
	OnPromptEvaluated func(PromptEvaluation)
	LoRA              []LoRAScale
	LoRAConfigured    bool
	// ProjectedInputs: prompt-only soft-token/MRoPE/deepstack payload.
	ProjectedInputs *ProjectedInputs
}

type LayerCache struct {
	Key   reference.Value
	Value reference.Value
	// States: persistent architecture-specific tensors.
	States map[string]LayerState
	// Auxiliary: transient forward-pass state.
	Auxiliary *reference.Value
}

// CacheStateMode: range-edit behavior.
type CacheStateMode uint32

const (
	CacheStateFixed CacheStateMode = 1
	CacheStateToken CacheStateMode = 2
)

// LayerState: named persistent tensor.
type LayerState struct {
	Mode  CacheStateMode
	Value reference.Value
}

type KVCache struct {
	Layers []LayerCache
	// DSATopK: transient GLM-DSA MTP handoff.
	DSATopK *reference.Value
	// Tokens: number of active attention tokens retained in Layers
	Tokens uint32
	// Position: absolute position assigned to next appended token
	// can exceed Tokens after attention-cache prefix has been removed
	Position uint32
}

// T5Session: encoder state plus decoder cache.
type T5Session struct {
	Encoder reference.Value
	Cache   *KVCache
}

// EmbeddingOverride: projected replacement for one chunk-local token.
type EmbeddingOverride struct {
	TokenIndex uint32
	Embedding  []float32
}

// MultiAxisPositions: temporal, height, width, extra MRoPE coordinates.
type MultiAxisPositions [4][]uint32

// AttentionBlock: half-open prompt range with bidirectional intra-block attention.
type AttentionBlock struct {
	Start uint32 `json:"start"`
	End   uint32 `json:"end"`
}

// ProjectedInputs: projected streams, positions, and optional attention blocks.
type ProjectedInputs struct {
	EmbeddingOverrides           []EmbeddingOverride
	MultiAxisPositions           *MultiAxisPositions
	DeepstackEmbeddings          []reference.Value
	BidirectionalAttentionBlocks []AttentionBlock
	VisualExpertBlocks           []AttentionBlock
}

// preparedModel: loaded assets; model-wide execution config.
type preparedModel struct {
	file                *gguf.File
	path                string
	spec                model.Spec
	plan                model.ModelPlan
	weights             model.Weights
	vocab               *tokenizer.Vocab
	cuda                *executor.Executor
	worker              *device.Worker
	deviceWeights       *model.DeviceF32Weights
	rawWeights          *model.DeviceWeights
	outputBias          []float32
	promptCacheCapacity int
	cachePageTokens     uint32
	modelSignature      [32]byte
	modelSignatureErr   error
	modelSignatureOnce  sync.Once
}

// runnerState: mutable LoRA and prompt-cache state.
type runnerState struct {
	closed       bool
	promptCaches []*cachedPrompt
	loraAdapters []loadedLoRA
}

// Runner: prepared assets + mutable request state.
type Runner struct {
	preparedModel
	runnerState
	mu sync.Mutex
}

type cachedPrompt struct {
	Tokens              []tokenizer.TokenID
	Hidden              reference.Value
	Cache               *KVCache
	Device              *deviceKVCache
	LoRASignature       [32]byte
	ProjectionSignature [32]byte
	HasProjection       bool
}

type OpenOptions struct {
	DeviceOrdinal           int
	PreloadDeviceWeights    bool
	PreloadQuantizedWeights bool
	// PromptCacheEntries: bounds independently reusable prompt states
	// Zero: selects default capacity of one
	PromptCacheEntries int
	// CachePageTokens: retained CUDA KV page width; zero selects the default.
	CachePageTokens uint32
	LoRAAdapters    []LoRAConfig
}

func (r *Runner) Spec() model.Spec {
	if r == nil {
		return model.Spec{}
	}
	return r.spec
}

func (r *Runner) layerPlan(layer int, recurrent bool) model.LayerPlan {
	if r != nil && layer >= 0 && layer < len(r.plan.Layers) {
		return r.plan.Layers[layer]
	}
	return r.spec.PlanLayer(uint32(layer), recurrent)
}

func (r *Runner) profile() model.ArchitectureProfile {
	if r != nil && r.plan.Profile.Name != "" {
		return r.plan.Profile
	}
	return r.spec.Profile()
}

func (r *Runner) forwardPolicy() model.ForwardPolicy {
	if r != nil && r.plan.Profile.Name != "" {
		return r.plan.Profile.Forward
	}
	profile := r.spec.Profile()
	if profile.Forward == model.ForwardCached && r.spec.NonCausalAttention {
		return model.ForwardNonCausal
	}
	return profile.Forward
}

func (r *Runner) Vocab() *tokenizer.Vocab {
	if r == nil {
		return nil
	}
	return r.vocab
}

// Forward: evaluates all layers and returns final normalized hidden states in
// ggml shape [embedding, tokens]
func (r *Runner) Forward(ctx context.Context, tokenIDs []tokenizer.TokenID) (reference.Value, error) {
	if r == nil {
		return reference.Value{}, errors.New("inference: runner is nil")
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.closed {
		return reference.Value{}, errors.New("inference: runner is closed")
	}
	return r.forwardLocked(ctx, tokenIDs)
}

// ForwardWithEmbeddingOverrides: evaluates causal decoder after replacing
// selected token lookup results with caller-provided soft-token embeddings
func (r *Runner) ForwardWithEmbeddingOverrides(
	ctx context.Context,
	tokenIDs []tokenizer.TokenID,
	overrides []EmbeddingOverride,
) (reference.Value, error) {
	if r == nil {
		return reference.Value{}, errors.New("inference: runner is nil")
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.closed {
		return reference.Value{}, errors.New("inference: runner is closed")
	}
	if r.forwardPolicy() == model.ForwardT5Encoder || r.spec.NonCausalAttention {
		return reference.Value{}, errors.New("inference: embedding overrides currently require a causal decoder")
	}
	hidden, _, err := r.forwardCachedProjectedChunkLocked(
		ctx, tokenIDs, nil, ProjectedInputs{EmbeddingOverrides: overrides},
	)
	return hidden, err
}

func (r *Runner) forwardLocked(
	ctx context.Context,
	tokenIDs []tokenizer.TokenID,
) (reference.Value, error) {
	switch r.forwardPolicy() {
	case model.ForwardDFlash:
		return reference.Value{}, errors.New("inference: DFlash requires feature fusion, cache injection, and paired target decode")
	case model.ForwardEagle3:
		return reference.Value{}, errors.New("inference: Eagle3 requires NewEagle3Session and AdvanceEagle3")
	case model.ForwardGemma4Assistant:
		return reference.Value{}, errors.New("inference: Gemma 4 assistant requires NewGemma4AssistantSession and AdvanceGemma4Assistant")
	case model.ForwardWavTokenizer:
		return r.forwardWavTokenizerLocked(ctx, tokenIDs)
	case model.ForwardT5Encoder:
		return r.forwardT5EncoderLocked(ctx, tokenIDs)
	case model.ForwardT5:
		return reference.Value{}, errors.New("inference: T5 requires NewT5Session and DecodeT5")
	case model.ForwardNonCausal:
		return r.forwardNonCausalLocked(ctx, tokenIDs)
	case model.ForwardCached:
		hidden, _, err := r.forwardCachedLocked(ctx, tokenIDs, nil)
		return hidden, err
	default:
		return reference.Value{}, errors.New("inference: unknown compiled forward policy")
	}
}

// ForwardNonCausal: evaluates entire bidirectional token sequence without
// creating or consuming decoder cache state
func (r *Runner) ForwardCached(
	ctx context.Context,
	tokenIDs []tokenizer.TokenID,
	cache *KVCache,
) (reference.Value, *KVCache, error) {
	if r == nil {
		return reference.Value{}, nil, errors.New("inference: runner is nil")
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.closed {
		return reference.Value{}, nil, errors.New("inference: runner is closed")
	}
	if r.spec.NonCausalAttention {
		return reference.Value{}, nil, errors.New("inference: non-causal models do not support KV caching")
	}
	if r.forwardPolicy() == model.ForwardT5 {
		return reference.Value{}, nil, errors.New("inference: use DecodeT5 for T5 caching")
	}
	return r.forwardCachedLocked(ctx, tokenIDs, cache)
}

// ForwardCachedWithEmbeddingOverrides: cache-producing form of
// ForwardWithEmbeddingOverrides; Override indices address only newly
// supplied chunk, allowing its returned cache to continue normal decoding
func (r *Runner) ForwardCachedWithEmbeddingOverrides(
	ctx context.Context,
	tokenIDs []tokenizer.TokenID,
	cache *KVCache,
	overrides []EmbeddingOverride,
) (reference.Value, *KVCache, error) {
	if r == nil {
		return reference.Value{}, nil, errors.New("inference: runner is nil")
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.closed {
		return reference.Value{}, nil, errors.New("inference: runner is closed")
	}
	if r.spec.NonCausalAttention {
		return reference.Value{}, nil, errors.New("inference: non-causal models do not support KV caching")
	}
	return r.forwardCachedProjectedChunkLocked(
		ctx, tokenIDs, cache, ProjectedInputs{EmbeddingOverrides: overrides},
	)
}

// ForwardCachedWithMultimodalInputs: projected embeddings plus MRoPE axes.
func (r *Runner) ForwardCachedWithMultimodalInputs(
	ctx context.Context,
	tokenIDs []tokenizer.TokenID,
	cache *KVCache,
	positions MultiAxisPositions,
	overrides []EmbeddingOverride,
) (reference.Value, *KVCache, error) {
	if r == nil {
		return reference.Value{}, nil, errors.New("inference: runner is nil")
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.closed {
		return reference.Value{}, nil, errors.New("inference: runner is closed")
	}
	if !r.spec.SupportsMultiAxisPositions() {
		return reference.Value{}, nil, errors.New("inference: model does not support multi-axis positions")
	}
	return r.forwardCachedProjectedChunkLocked(
		ctx, tokenIDs, cache, ProjectedInputs{
			EmbeddingOverrides: overrides, MultiAxisPositions: &positions,
		},
	)
}

// ForwardCachedWithProjectedInputs: projected base/deepstack decoder input.
func (r *Runner) ForwardCachedWithProjectedInputs(
	ctx context.Context,
	tokenIDs []tokenizer.TokenID,
	cache *KVCache,
	inputs ProjectedInputs,
) (reference.Value, *KVCache, error) {
	if r == nil {
		return reference.Value{}, nil, errors.New("inference: runner is nil")
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.closed {
		return reference.Value{}, nil, errors.New("inference: runner is closed")
	}
	return r.forwardCachedWithProjectedInputsLocked(ctx, tokenIDs, cache, inputs)
}

func (r *Runner) forwardCachedWithProjectedInputsLocked(
	ctx context.Context,
	tokenIDs []tokenizer.TokenID,
	cache *KVCache,
	inputs ProjectedInputs,
) (reference.Value, *KVCache, error) {
	if len(inputs.VisualExpertBlocks) == 0 {
		return r.forwardCachedProjectedChunkLocked(ctx, tokenIDs, cache, inputs)
	}
	projected, err := r.compileProjectedRequestPlan(len(tokenIDs), cache != nil, inputs)
	if err != nil {
		return reference.Value{}, nil, err
	}
	overrides := make(map[uint32][]float32, len(projected.overrides))
	for _, override := range projected.overrides {
		overrides[override.TokenIndex] = override.Embedding
	}
	var hidden reference.Value
	next := cache
	start := 0
	for _, block := range projected.visualBlocks {
		if start < int(block.Start) {
			hidden, next, err = r.forwardCachedProjectedChunkLocked(
				ctx, tokenIDs[start:int(block.Start)], next, ProjectedInputs{},
			)
			if err != nil {
				return reference.Value{}, nil, err
			}
		}
		chunk := tokenIDs[int(block.Start):int(block.End)]
		chunkOverrides := make([]EmbeddingOverride, len(chunk))
		for index := range chunk {
			chunkOverrides[index] = EmbeddingOverride{TokenIndex: uint32(index), Embedding: overrides[block.Start+uint32(index)]}
		}
		hidden, next, err = r.forwardCachedProjectedChunkLocked(
			ctx, chunk, next, ProjectedInputs{EmbeddingOverrides: chunkOverrides},
		)
		if err != nil {
			return reference.Value{}, nil, err
		}
		start = int(block.End)
	}
	if start < len(tokenIDs) {
		hidden, next, err = r.forwardCachedProjectedChunkLocked(
			ctx, tokenIDs[start:], next, ProjectedInputs{},
		)
	}
	return hidden, next, err
}

func (r *Runner) forwardCachedLocked(
	ctx context.Context,
	tokenIDs []tokenizer.TokenID,
	cache *KVCache,
) (reference.Value, *KVCache, error) {
	return r.forwardCachedProjectedChunkLocked(ctx, tokenIDs, cache, ProjectedInputs{})
}

func (r *Runner) forwardCachedProjectedChunkLocked(
	ctx context.Context,
	tokenIDs []tokenizer.TokenID,
	cache *KVCache,
	inputs ProjectedInputs,
) (reference.Value, *KVCache, error) {
	return r.forwardCachedProjectedChunkModeLocked(
		ctx, tokenIDs, cache, inputs, true, nil,
	)
}

func (r *Runner) forwardCachedPreOutputNormLocked(
	ctx context.Context,
	tokenIDs []tokenizer.TokenID,
	cache *KVCache,
) (reference.Value, *KVCache, error) {
	return r.forwardCachedProjectedChunkModeLocked(
		ctx, tokenIDs, cache, ProjectedInputs{}, false, nil,
	)
}

func (r *Runner) forwardCachedProjectedChunkModeLocked(
	ctx context.Context,
	tokenIDs []tokenizer.TokenID,
	cache *KVCache,
	inputs ProjectedInputs,
	applyOutputNorm bool,
	capture *layerInputCapture,
) (reference.Value, *KVCache, error) {
	if r.weights.Qwen35MTP != nil && r.weights.Qwen35MTP.MTPOnly {
		return reference.Value{}, nil, errors.New("inference: Qwen3.5 MTP-only model requires a paired target session")
	}
	if r.weights.Cohere2MTP != nil && r.weights.Cohere2MTP.MTPOnly {
		return reference.Value{}, nil, errors.New("inference: Cohere2-MoE MTP-only model requires a paired target session")
	}
	if r.forwardPolicy() == model.ForwardGemma4Assistant {
		return reference.Value{}, nil, errors.New("inference: Gemma 4 assistant requires shared target context")
	}
	if r.forwardPolicy() == model.ForwardT5Encoder {
		return reference.Value{}, nil, errors.New("inference: T5 encoder does not support KV caching")
	}
	projected, err := r.compileProjectedRequestPlan(len(tokenIDs), cache != nil, inputs)
	if err != nil {
		return reference.Value{}, nil, err
	}
	visualMode := projected.visualMode
	overrides := projected.overrides
	multiPositions := projected.multiPositions
	deepstackInputs := projected.deepstackInputs
	attentionBlockIDs := projected.attentionBlockIDs
	var pastTokens, nextPosition uint32
	if cache != nil {
		if err := r.validateCache(cache); err != nil {
			return reference.Value{}, nil, err
		}
		pastTokens = cache.Tokens
		nextPosition = effectiveCachePosition(cache)
	}
	if uint64(pastTokens)+uint64(len(tokenIDs)) > uint64(r.spec.ContextLength) {
		return reference.Value{}, nil, fmt.Errorf(
			"inference: cached plus new token count %d exceeds context length %d",
			uint64(pastTokens)+uint64(len(tokenIDs)),
			r.spec.ContextLength,
		)
	}
	if uint64(nextPosition)+uint64(len(tokenIDs)) > math.MaxUint32 {
		return reference.Value{}, nil, errors.New(
			"inference: absolute token position exceeds uint32",
		)
	}
	rows, err := r.tokenRows(tokenIDs)
	if err != nil {
		return reference.Value{}, nil, err
	}
	positions := tokenPositions(nextPosition, len(tokenIDs))
	if multiPositions != nil {
		positions = append(positions[:0], (*multiPositions)[0]...)
	}
	activation, err := r.loadEmbeddings(ctx, rows)
	if err != nil {
		return reference.Value{}, nil, err
	}
	embeddingScaleApplied := false
	var deepstackBase reference.Value
	graniteDeepstack := projected.deepstackBase
	if graniteDeepstack {
		deepstackBase = reference.Value{
			Shape: activation.Shape,
			Data:  slices.Clone(activation.Data),
		}
		if err := applyEmbeddingOverrides(&deepstackBase, overrides); err != nil {
			return reference.Value{}, nil, err
		}
		if scale := r.spec.InputEmbeddingScale(); scale != 1 {
			for index := range activation.Data {
				activation.Data[index] *= scale
			}
		}
		if err := applyEmbeddingOverrides(&activation, overrides); err != nil {
			return reference.Value{}, nil, err
		}
	} else if projected.overridePolicy == model.EmbeddingOverrideRawScaled && len(overrides) > 0 {
		if err := applyGemmaRawEmbeddingOverrides(&activation, overrides, r.spec.InputEmbeddingScale()); err != nil {
			return reference.Value{}, nil, err
		}
		embeddingScaleApplied = true
	} else if err := applyEmbeddingOverrides(&activation, overrides); err != nil {
		return reference.Value{}, nil, err
	}
	activation, err = r.addPositionEmbeddings(ctx, activation, positions)
	if err != nil {
		return reference.Value{}, nil, err
	}
	if scale := r.spec.InputEmbeddingScale(); !graniteDeepstack && !embeddingScaleApplied && scale != 1 {
		for index := range activation.Data {
			activation.Data[index] *= scale
		}
	}
	activation, err = r.applyTokenEmbeddingNorm(ctx, activation)
	if err != nil {
		return reference.Value{}, nil, err
	}
	var embeddingSkip reference.Value
	if r.spec.UsesUnweightedRMSNorm() {
		activation, err = r.runUnweightedRMSNorm(ctx, activation)
		if err != nil {
			return reference.Value{}, nil, err
		}
		embeddingSkip = activation
	}
	perLayerInputs, err := r.prepareGemma4PerLayerInputs(ctx, activation, rows)
	if err != nil {
		return reference.Value{}, nil, err
	}
	if r.profile().Has(model.ArchitectureAltUp) {
		if capture != nil {
			return reference.Value{}, nil, errors.New("inference: cached Gemma3n layer extraction is unsupported")
		}
		return r.forwardGemma3nCachedLocked(
			ctx, activation, perLayerInputs, positions, cache, pastTokens, nextPosition,
		)
	}
	cachePosition := nextPosition + uint32(len(tokenIDs))
	if multiPositions != nil {
		var maximum uint32
		for axis := range multiPositions {
			for _, position := range (*multiPositions)[axis] {
				maximum = max(maximum, position)
			}
		}
		if maximum == math.MaxUint32 {
			return reference.Value{}, nil, errors.New("inference: multi-axis position exceeds resumable range")
		}
		cachePosition = maximum + 1
		if cache != nil && cachePosition < nextPosition {
			return reference.Value{}, nil, errors.New("inference: multi-axis positions regress cached position")
		}
	}
	nextCache := &KVCache{
		Layers:   make([]LayerCache, len(r.weights.Layers)),
		Tokens:   pastTokens + uint32(len(tokenIDs)),
		Position: cachePosition,
	}
	if r.hasPreloadedWeights() && r.plan.CachedGraph == model.CachedGraphDense {
		return r.forwardDenseLayersPreloaded(
			ctx, activation, embeddingSkip, perLayerInputs, positions, multiPositions,
			deepstackBase, deepstackInputs, attentionBlockIDs,
			cache, nextCache, visualMode, applyOutputNorm, capture,
		)
	}
	auxiliaryValues := make(map[model.AuxiliaryFlow]*reference.Value)
	for layerIndex, layerInfo := range r.weights.Layers {
		plan := r.layerPlan(layerIndex, layerInfo.Recurrent)
		if stream := deepstackInputForLayer(plan.DeepstackBefore, deepstackBase, deepstackInputs); stream != nil {
			activation, err = addDeepstackEmbedding(activation, *stream)
			if err != nil {
				return reference.Value{}, nil, fmt.Errorf("inference layer %d deepstack input: %w", layerIndex, err)
			}
		}
		capture.set(layerIndex, activation)
		var past *LayerCache
		if plan.SharedKV {
			past = &nextCache.Layers[plan.KVSource]
		} else if cache != nil {
			past = &cache.Layers[layerIndex]
		}
		var perLayerInput *reference.Value
		if len(perLayerInputs) > 0 {
			perLayerInput = &perLayerInputs[layerIndex]
		}
		if plan.AuxiliaryInput != model.AuxiliaryNone {
			perLayerInput = auxiliaryValues[plan.AuxiliaryInput]
		}
		var layerCache LayerCache
		activation, layerCache, err = r.runLayerCached(
			ctx,
			activation,
			layerInfo,
			layerIndex,
			positions,
			rows,
			past,
			embeddingSkip,
			perLayerInput,
			multiPositions,
			visualMode,
			attentionBlockIDs,
		)
		if err != nil {
			return reference.Value{}, nil, fmt.Errorf("inference layer %d: %w", layerIndex, err)
		}
		if stream := deepstackInputForLayer(plan.DeepstackAfter, deepstackBase, deepstackInputs); stream != nil {
			activation, err = addDeepstackEmbedding(activation, *stream)
			if err != nil {
				return reference.Value{}, nil, fmt.Errorf("inference layer %d deepstack output: %w", layerIndex, err)
			}
		}
		if layerCache.Auxiliary != nil && plan.AuxiliaryOutput != model.AuxiliaryNone {
			auxiliaryValues[plan.AuxiliaryOutput] = layerCache.Auxiliary
			layerCache.Auxiliary = nil
		}
		nextCache.Layers[layerIndex] = layerCache
	}
	if topK := auxiliaryValues[model.AuxiliaryDSATopK]; topK != nil {
		value := *topK
		nextCache.DSATopK = &value
	}
	if !r.hasPreloadedWeights() && applyOutputNorm {
		activation, err = r.runOutputNorm(ctx, activation)
		if err != nil {
			return reference.Value{}, nil, err
		}
	}
	return activation, nextCache, nil
}
