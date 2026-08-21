package inference

import (
	"context"
	"errors"
	"fmt"
	"math"
	"slices"
	"sync"
	"time"

	"overgo/internal/artifact"
	"overgo/internal/cuda/device"
	"overgo/internal/cuda/executor"
	"overgo/internal/gguf"
	"overgo/internal/model"
	"overgo/internal/modelrecipe"
	"overgo/internal/recipe"
	"overgo/internal/sampling"
	"overgo/internal/tensor/reference"
	"overgo/internal/tokenizer"
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
	// DeviceGreedy: device argmax; TokenEvent.Logits omitted.
	DeviceGreedy bool
	// DeviceTopK: exact bounded sampling; TokenEvent.Logits omitted.
	DeviceTopK bool
	OnToken    func(TokenEvent) error
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
	States LayerStates
	// Auxiliary: transient forward-pass state.
	Auxiliary *reference.Value
}

// CacheStateMode: serialized range-edit behavior.
type CacheStateMode = model.CacheStateMode

const (
	CacheStateFixed = model.CacheStateFixed
	CacheStateToken = model.CacheStateToken
)

// LayerState: named persistent tensor.
type LayerState = model.CacheState[reference.Value]

// LayerStates: named persistent tensor collection.
type LayerStates = model.CacheStates[reference.Value]

type KVCache struct {
	Layers []LayerCache
	// SparseTopK: transient sparse-attention MTP handoff.
	SparseTopK *reference.Value
	// Tokens: number of active attention tokens retained in Layers
	Tokens uint32
	// Position: absolute position assigned to next appended token
	// can exceed Tokens after attention-cache prefix has been removed
	Position uint32
}

// EncoderDecoderSession: encoder state plus decoder cache.
type EncoderDecoderSession struct {
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
	program             modelrecipe.Plan
	evidenceTier        recipe.EvidenceTier
	weights             model.Weights
	vocab               *tokenizer.Vocab
	cuda                *executor.Executor
	worker              *device.Worker
	deviceWeights       *model.DeviceF32Weights
	rawWeights          *model.DeviceWeights
	decodeWeights       *model.DeviceBF16Weights
	hostWeights         *model.HostTensorStore
	outputBias          []float32
	outputExclusions    []tokenizer.TokenRange
	promptCacheCapacity int
	modelSignature      [32]byte
	modelSignatureErr   error
	modelSignatureOnce  sync.Once
	audioTables         *audioWaveformTables
	audioTablesOnce     sync.Once
}

// runnerState: mutable LoRA and prompt-cache state.
type runnerState struct {
	closed        bool
	promptCaches  []*cachedPrompt
	loraAdapters  []loadedLoRA
	decodeSession *deviceDecodeSession
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
	DeviceOrdinal int
	// PromptCacheEntries: bounds independently reusable prompt states
	// Zero: selects default capacity of one
	PromptCacheEntries int
	LoRAAdapters       []LoRAConfig
}

func (r *Runner) Spec() model.Spec {
	if r == nil {
		return model.Spec{}
	}
	return r.spec
}

func (r *Runner) EvidenceTier() recipe.EvidenceTier {
	if r == nil {
		return ""
	}
	return r.evidenceTier
}

// ModelID: exact model artifact bound by the active inference recipe.
func (r *Runner) ModelID() artifact.ID {
	if r == nil {
		return artifact.ID{}
	}
	return r.program.Identity.Model
}

// Residency: compiled weight-storage policy.
func (r *Runner) Residency() recipe.ResidencyPolicy {
	if r == nil {
		return ""
	}
	return r.program.Residency
}

func (r *Runner) RecipeRuntimeDescription(task recipe.Task) (modelrecipe.RuntimeDescription, error) {
	if r == nil || task != recipe.TaskInference {
		return modelrecipe.RuntimeDescription{}, fmt.Errorf("inference: active %s recipe is unavailable", task)
	}
	return modelrecipe.Describe(r.program)
}

// DeviceResident: all execution weights have device residency.
func (r *Runner) DeviceResident() bool { return r != nil && r.hasPreloadedWeights() }

func (r *Runner) layerProgram(layer int) model.CompiledLayerProgram {
	if r == nil {
		panic("inference: compiled layer plan is unavailable")
	}
	program, err := r.program.Model.LayerProgram(uint32(layer))
	if err != nil {
		panic("inference: compiled layer plan is unavailable")
	}
	return program
}

func (r *Runner) draftLayerProgram(offset uint32) (model.CompiledLayerProgram, error) {
	if r == nil {
		return model.CompiledLayerProgram{}, errors.New("inference: compiled draft layer is unavailable")
	}
	return r.program.Model.DraftProgram(offset)
}

func (r *Runner) forwardProgram() model.ForwardProgram {
	if r == nil || !r.program.Model.Compiled() {
		panic("inference: compiled forward program is unavailable")
	}
	return r.program.Model.Forward()
}

func (r *Runner) Vocab() *tokenizer.Vocab {
	if r == nil {
		return nil
	}
	return r.vocab
}

var (
	errRunnerNil    = errors.New("inference: runner is nil")
	errRunnerClosed = errors.New("inference: runner is closed")
)

func (r *Runner) lockOpen() error {
	if r == nil {
		return errRunnerNil
	}
	r.mu.Lock()
	if r.closed {
		r.mu.Unlock()
		return errRunnerClosed
	}
	return nil
}

// Forward: evaluates all layers and returns final normalized hidden states in
// ggml shape [embedding, tokens]
func (r *Runner) Forward(ctx context.Context, tokenIDs []tokenizer.TokenID) (reference.Value, error) {
	if err := r.lockOpen(); err != nil {
		return reference.Value{}, err
	}
	defer r.mu.Unlock()
	return r.forwardLocked(ctx, tokenIDs)
}

// ForwardWithEmbeddingOverrides: evaluates causal decoder after replacing
// selected token lookup results with caller-provided soft-token embeddings
func (r *Runner) ForwardWithEmbeddingOverrides(
	ctx context.Context,
	tokenIDs []tokenizer.TokenID,
	overrides []EmbeddingOverride,
) (reference.Value, error) {
	if err := r.lockOpen(); err != nil {
		return reference.Value{}, err
	}
	defer r.mu.Unlock()
	if r.forwardProgram().Operation == model.ForwardOperationEncoder || r.spec.NonCausalAttention {
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
	program := r.forwardProgram()
	if int(program.Operation) >= len(forwardExecutors) || forwardExecutors[program.Operation] == nil {
		return reference.Value{}, errors.New("inference: unknown compiled forward policy")
	}
	return forwardExecutors[program.Operation](r, ctx, tokenIDs)
}

type forwardExecutor func(*Runner, context.Context, []tokenizer.TokenID) (reference.Value, error)

var forwardExecutors = [...]forwardExecutor{
	model.ForwardOperationCached: func(r *Runner, ctx context.Context, ids []tokenizer.TokenID) (reference.Value, error) {
		hidden, _, err := r.forwardCachedLocked(ctx, ids, nil)
		return hidden, err
	},
	model.ForwardOperationBidirectional: func(r *Runner, ctx context.Context, ids []tokenizer.TokenID) (reference.Value, error) {
		return r.forwardNonCausalLocked(ctx, ids)
	},
	model.ForwardOperationAudioTokens: func(r *Runner, ctx context.Context, ids []tokenizer.TokenID) (reference.Value, error) {
		return r.forwardAudioTokensLocked(ctx, ids)
	},
	model.ForwardOperationEncoder: func(r *Runner, ctx context.Context, ids []tokenizer.TokenID) (reference.Value, error) {
		return r.forwardEncoderLocked(ctx, ids)
	},
	model.ForwardOperationSession: forwardSessionError,
}

var forwardSessionErrors = [...]error{
	model.ForwardSessionPairedFeatures:   errors.New("inference: model requires a paired-feature session"),
	model.ForwardSessionFeatureDraft:     errors.New("inference: model requires a feature-draft session"),
	model.ForwardSessionPairedProjection: errors.New("inference: model requires a paired projection session"),
	model.ForwardSessionEncoderDecoder:   errors.New("inference: model requires an encoder-decoder session"),
}

func forwardSessionError(r *Runner, _ context.Context, _ []tokenizer.TokenID) (reference.Value, error) {
	session := r.forwardProgram().Session
	if int(session) >= len(forwardSessionErrors) || forwardSessionErrors[session] == nil {
		return reference.Value{}, errors.New("inference: unknown compiled forward session")
	}
	return reference.Value{}, forwardSessionErrors[session]
}

// ForwardNonCausal: evaluates entire bidirectional token sequence without
// creating or consuming decoder cache state
func (r *Runner) ForwardCached(
	ctx context.Context,
	tokenIDs []tokenizer.TokenID,
	cache *KVCache,
) (reference.Value, *KVCache, error) {
	if err := r.lockOpen(); err != nil {
		return reference.Value{}, nil, err
	}
	defer r.mu.Unlock()
	if r.spec.NonCausalAttention {
		return reference.Value{}, nil, errors.New("inference: non-causal models do not support KV caching")
	}
	if r.forwardProgram().Session == model.ForwardSessionEncoderDecoder {
		return reference.Value{}, nil, errors.New("inference: use the encoder-decoder session for caching")
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
	if err := r.lockOpen(); err != nil {
		return reference.Value{}, nil, err
	}
	defer r.mu.Unlock()
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
	if err := r.lockOpen(); err != nil {
		return reference.Value{}, nil, err
	}
	defer r.mu.Unlock()
	if !r.program.Model.ProjectedInput().MultiAxis {
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
	if err := r.lockOpen(); err != nil {
		return reference.Value{}, nil, err
	}
	defer r.mu.Unlock()
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
	if _, catalog, ok := r.lookupSingleHeadMTP(); ok && catalog.MTPOnly {
		return reference.Value{}, nil, fmt.Errorf("inference: %s-only model requires a paired target session", mtpLabel)
	}
	if r.forwardProgram().Session == model.ForwardSessionPairedProjection {
		return reference.Value{}, nil, errors.New("inference: paired-projection session requires shared target context")
	}
	if r.forwardProgram().Operation == model.ForwardOperationEncoder {
		return reference.Value{}, nil, errors.New("inference: encoder-only program does not support KV caching")
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
	sequence, err := r.planForwardSequence(tokenIDs, pastTokens, nextPosition)
	if err != nil {
		return reference.Value{}, nil, err
	}
	rows, positions := sequence.rows, sequence.positions
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
		if err := applyScaledRawEmbeddingOverrides(&activation, overrides, r.spec.InputEmbeddingScale()); err != nil {
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
	if r.program.Model.Normalization().Operation == model.NormalizationUnweightedRMS {
		activation, err = r.runUnweightedRMSNorm(ctx, activation)
		if err != nil {
			return reference.Value{}, nil, err
		}
		embeddingSkip = activation
	}
	perLayerInputs, err := r.preparePerLayerInputs(ctx, activation, rows)
	if err != nil {
		return reference.Value{}, nil, err
	}
	if forward := r.forwardProgram(); forward.AlternateStates() {
		if capture != nil {
			return reference.Value{}, nil, errors.New("inference: alternate-state layer extraction is unsupported")
		}
		return r.forwardAlternatePredictionsCachedLocked(
			ctx, forward.Alternate, activation, perLayerInputs, positions, cache, pastTokens, nextPosition,
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
	if r.hasPreloadedWeights() && r.program.Model.CachedGraph() == model.CachedGraphDense {
		return r.forwardDenseLayersPreloaded(
			ctx, activation, embeddingSkip, perLayerInputs, positions, multiPositions,
			deepstackBase, deepstackInputs, attentionBlockIDs,
			cache, nextCache, visualMode, applyOutputNorm, capture,
		)
	}
	auxiliaryValues := make(map[model.AuxiliaryFlow]*reference.Value)
	for layerIndex, layerInfo := range r.weights.Layers {
		plan := r.layerProgram(layerIndex).Layer()
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
	if topK := auxiliaryValues[model.AuxiliarySparseTopK]; topK != nil {
		value := *topK
		nextCache.SparseTopK = &value
	}
	if !r.hasPreloadedWeights() && applyOutputNorm {
		activation, err = r.runOutputNorm(ctx, activation)
		if err != nil {
			return reference.Value{}, nil, err
		}
	}
	return activation, nextCache, nil
}
