package inference

import (
	"cmp"
	"context"
	"errors"
	"fmt"
	"math"
	"slices"
	"strings"
	"sync"
	"time"

	"llamacpp2go/internal/cuda/device"
	"llamacpp2go/internal/cuda/driver"
	"llamacpp2go/internal/cuda/executor"
	"llamacpp2go/internal/gguf"
	"llamacpp2go/internal/model"
	"llamacpp2go/internal/sampling"
	"llamacpp2go/internal/tensor"
	"llamacpp2go/internal/tensor/dtype"
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
	// CachePageTokens: retained CUDA KV page width; zero selects 256.
	CachePageTokens uint32
	LoRAAdapters    []LoRAConfig
}

func Open(path string, deviceOrdinal int) (*Runner, error) {
	return OpenWithOptions(path, OpenOptions{DeviceOrdinal: deviceOrdinal})
}

func OpenWithOptions(path string, options OpenOptions) (*Runner, error) {
	if options.PreloadDeviceWeights && options.PreloadQuantizedWeights {
		return nil, errors.New("inference: F32 and native-quantized preload modes are mutually exclusive")
	}
	if options.PromptCacheEntries < 0 {
		return nil, errors.New("inference: prompt cache entry count is negative")
	}
	promptCacheCapacity := options.PromptCacheEntries
	if promptCacheCapacity == 0 {
		promptCacheCapacity = 1
	}
	cachePageTokens := options.CachePageTokens
	if cachePageTokens == 0 {
		cachePageTokens = 256
	}
	file, err := gguf.Open(path)
	if err != nil {
		return nil, err
	}
	fail := func(openErr error) (*Runner, error) {
		_ = file.Close()
		return nil, openErr
	}
	spec, err := model.ReadSpec(file)
	if err != nil {
		return fail(err)
	}
	weights, err := model.ReadWeights(file, spec)
	if err != nil {
		return fail(err)
	}
	plan, err := model.CompileModelPlan(spec, weights)
	if err != nil {
		return fail(err)
	}
	vocab, err := tokenizer.Load(file)
	if err != nil {
		return fail(err)
	}
	loraAdapters := make([]loadedLoRA, len(options.LoRAAdapters))
	for index, configured := range options.LoRAAdapters {
		if math.IsNaN(float64(configured.Scale)) || math.IsInf(float64(configured.Scale), 0) {
			return fail(fmt.Errorf("inference: LoRA adapter %d scale is invalid", index))
		}
		adapter, loadErr := model.LoadLoRA(context.Background(), configured.Path, file, spec)
		if loadErr != nil {
			return fail(fmt.Errorf("inference: load LoRA adapter %d: %w", index, loadErr))
		}
		loraAdapters[index] = loadedLoRA{
			adapter: adapter, scale: configured.Scale, signature: loRAStaticSignature(adapter),
		}
	}
	var outputBias []float32
	if weights.OutputBias != nil {
		value, loadErr := model.LoadHostTensor(
			context.Background(),
			file,
			*weights.OutputBias,
		)
		if loadErr != nil {
			return fail(loadErr)
		}
		outputBias = slices.Clone(value.Data)
	}
	var cuda *executor.Executor
	var worker *device.Worker
	var deviceWeights *model.DeviceF32Weights
	var rawWeights *model.DeviceWeights
	if options.PreloadDeviceWeights || options.PreloadQuantizedWeights {
		worker, err = device.New(options.DeviceOrdinal)
		if err != nil {
			return fail(err)
		}
		cuda, err = executor.NewWithWorker(worker)
		if err != nil {
			_ = worker.Close()
			return fail(err)
		}
		deviceWeights, err = model.NewDeviceF32Weights(worker)
		if err != nil {
			_ = cuda.Close()
			_ = worker.Close()
			return fail(err)
		}
		selected := selectedModelTensors(file, weights)
		f32Tensors := selected
		if options.PreloadQuantizedWeights {
			adaptedTensors := make(map[string]struct{})
			for _, loaded := range loraAdapters {
				for name := range loaded.adapter.Weights {
					adaptedTensors[name] = struct{}{}
				}
			}
			f32Required := f32RequiredModelTensors(weights)
			f32Tensors = make([]gguf.TensorInfo, 0, len(selected))
			var quantized []gguf.TensorInfo
			for _, info := range selected {
				_, adapted := adaptedTensors[info.Name]
				_, requiresF32 := f32Required[info.Name]
				if !adapted && !requiresF32 && (info.Type == dtype.Q4_0 ||
					info.Type == dtype.Q4_1 ||
					info.Type == dtype.Q5_0 ||
					info.Type == dtype.Q5_1 ||
					info.Type == dtype.Q1_0 ||
					info.Type == dtype.Q2_0 ||
					info.Type == dtype.TQ1_0 ||
					info.Type == dtype.TQ2_0 ||
					info.Type == dtype.Q8_0 ||
					info.Type == dtype.Q8_1 ||
					info.Type == dtype.Q2K ||
					info.Type == dtype.Q3K ||
					info.Type == dtype.Q4K ||
					info.Type == dtype.Q5K ||
					info.Type == dtype.Q6K ||
					info.Type == dtype.Q8K ||
					info.Type == dtype.IQ2XXS ||
					info.Type == dtype.IQ2XS ||
					info.Type == dtype.IQ2S ||
					info.Type == dtype.IQ3XXS ||
					info.Type == dtype.IQ3S ||
					info.Type == dtype.IQ1S ||
					info.Type == dtype.IQ1M ||
					info.Type == dtype.IQ4NL ||
					info.Type == dtype.IQ4XS ||
					info.Type == dtype.MXFP4 ||
					info.Type == dtype.NVFP4) {
					quantized = append(quantized, info)
				} else {
					f32Tensors = append(f32Tensors, info)
				}
			}
			rawWeights, err = model.NewDeviceWeights(worker)
			if err != nil {
				_ = deviceWeights.Close()
				_ = cuda.Close()
				_ = worker.Close()
				return fail(err)
			}
			if err = rawWeights.Load(context.Background(), file, quantized); err != nil {
				_ = rawWeights.Close()
				_ = deviceWeights.Close()
				_ = cuda.Close()
				_ = worker.Close()
				return fail(err)
			}
		}
		if err = deviceWeights.Load(context.Background(), file, f32Tensors); err != nil {
			_ = rawWeights.Close()
			_ = deviceWeights.Close()
			_ = cuda.Close()
			_ = worker.Close()
			return fail(err)
		}
	} else {
		cuda, err = executor.New(options.DeviceOrdinal)
		if err != nil {
			return fail(err)
		}
	}
	return &Runner{preparedModel: preparedModel{
		file: file, path: path, spec: spec, plan: plan, weights: weights, vocab: vocab,
		cuda: cuda, worker: worker, deviceWeights: deviceWeights, rawWeights: rawWeights,
		outputBias:          outputBias,
		promptCacheCapacity: promptCacheCapacity, cachePageTokens: cachePageTokens,
	}, runnerState: runnerState{loraAdapters: loraAdapters}}, nil
}

func (r *Runner) Close() error {
	if r == nil {
		return nil
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.closed {
		return nil
	}
	r.closed = true
	return errors.Join(
		r.runnerState.release(context.Background()),
		r.preparedModel.close(),
	)
}

func (s *runnerState) release(ctx context.Context) error {
	var errs []error
	for _, promptCache := range s.promptCaches {
		if promptCache.Device != nil {
			errs = append(errs, promptCache.Device.Release(ctx))
			promptCache.Device = nil
		}
	}
	s.promptCaches = nil
	return errors.Join(errs...)
}

func (m *preparedModel) close() error {
	var errs []error
	if m.rawWeights != nil {
		errs = append(errs, m.rawWeights.Close())
	}
	if m.deviceWeights != nil {
		errs = append(errs, m.deviceWeights.Close())
	}
	if m.cuda != nil {
		errs = append(errs, m.cuda.Close())
	}
	if m.worker != nil {
		errs = append(errs, m.worker.Close())
	}
	if m.file != nil {
		errs = append(errs, m.file.Close())
	}
	return errors.Join(errs...)
}

func (r *Runner) hasPreloadedWeights() bool {
	return r != nil && r.deviceWeights != nil
}

func (r *Runner) deviceInput(
	builder *tensor.Builder,
	info gguf.TensorInfo,
) (*tensor.Tensor, driver.DevicePtr, error) {
	if r.rawWeights != nil {
		if _, ok := r.rawWeights.Lookup(info.Name); ok {
			return r.rawWeights.Input(builder, info.Name)
		}
	}
	if r.deviceWeights != nil {
		return r.deviceWeights.Input(builder, info.Name)
	}
	return nil, 0, fmt.Errorf("inference: device tensor %q is not preloaded", info.Name)
}

func (r *Runner) wavTokenizerGraphInputs(
	ctx context.Context,
	builder *tensor.Builder,
) (model.WavTokenizerGraphWeights, map[*tensor.Tensor]reference.Value, map[*tensor.Tensor]driver.DevicePtr, error) {
	var result model.WavTokenizerGraphWeights
	hostFeeds := make(map[*tensor.Tensor]reference.Value)
	deviceFeeds := make(map[*tensor.Tensor]driver.DevicePtr)
	if r.weights.WavTokenizer == nil {
		return result, nil, nil, errors.New("inference: WavTokenizer weights are missing")
	}
	input := func(info gguf.TensorInfo) (*tensor.Tensor, error) {
		if r.hasPreloadedWeights() {
			node, pointer, err := r.deviceInput(builder, info)
			if err != nil {
				return nil, err
			}
			deviceFeeds[node] = pointer
			return node, nil
		}
		value, err := model.LoadHostTensor(ctx, r.file, info)
		if err != nil {
			return nil, err
		}
		node := builder.Input(info.Name, dtype.F32, value.Shape)
		hostFeeds[node] = value
		return node, nil
	}
	assign := func(destination **tensor.Tensor, info gguf.TensorInfo) error {
		item, err := input(info)
		if err == nil {
			*destination = item
		}
		return err
	}
	info := r.weights.WavTokenizer
	for _, item := range []struct {
		destination **tensor.Tensor
		info        gguf.TensorInfo
	}{
		{&result.InputConv, info.InputConv}, {&result.InputConvBias, info.InputConvBias},
		{&result.TokenNorm, info.TokenNorm}, {&result.TokenNormBias, info.TokenNormBias},
		{&result.OutputNorm, info.OutputNorm}, {&result.OutputNormBias, info.OutputNormBias},
		{&result.Output, info.Output}, {&result.OutputBias, info.OutputBias},
	} {
		if err := assign(item.destination, item.info); err != nil {
			return result, nil, nil, err
		}
	}
	result.PosNet = make([]model.WavPosNetGraphWeights, len(info.PosNet))
	for block := range info.PosNet {
		source, destination := &info.PosNet[block], &result.PosNet[block]
		for _, item := range []struct {
			destination **tensor.Tensor
			info        gguf.TensorInfo
		}{
			{&destination.Norm1, source.Norm1}, {&destination.Norm1Bias, source.Norm1Bias},
			{&destination.Conv1, source.Conv1}, {&destination.Conv1Bias, source.Conv1Bias},
			{&destination.Norm2, source.Norm2}, {&destination.Norm2Bias, source.Norm2Bias},
			{&destination.Conv2, source.Conv2}, {&destination.Conv2Bias, source.Conv2Bias},
			{&destination.AttentionNorm, source.AttentionNorm}, {&destination.AttentionNormBias, source.AttentionNormBias},
			{&destination.AttentionQ, source.AttentionQ}, {&destination.AttentionQBias, source.AttentionQBias},
			{&destination.AttentionK, source.AttentionK}, {&destination.AttentionKBias, source.AttentionKBias},
			{&destination.AttentionV, source.AttentionV}, {&destination.AttentionVBias, source.AttentionVBias},
			{&destination.AttentionOutput, source.AttentionOutput}, {&destination.AttentionOutBias, source.AttentionOutBias},
		} {
			if item.info.Name != "" {
				if err := assign(item.destination, item.info); err != nil {
					return result, nil, nil, err
				}
			}
		}
	}
	result.ConvNext = make([]model.WavConvNextGraphWeights, len(info.ConvNext))
	for block := range info.ConvNext {
		source, destination := &info.ConvNext[block], &result.ConvNext[block]
		for _, item := range []struct {
			destination **tensor.Tensor
			info        gguf.TensorInfo
		}{
			{&destination.Depthwise, source.Depthwise}, {&destination.DepthwiseBias, source.DepthwiseBias},
			{&destination.Norm, source.Norm}, {&destination.NormBias, source.NormBias},
			{&destination.Pointwise1, source.Pointwise1}, {&destination.Pointwise1Bias, source.Pointwise1Bias},
			{&destination.Pointwise2, source.Pointwise2}, {&destination.Pointwise2Bias, source.Pointwise2Bias},
			{&destination.Gamma, source.Gamma},
		} {
			if err := assign(item.destination, item.info); err != nil {
				return result, nil, nil, err
			}
		}
	}
	return result, hostFeeds, deviceFeeds, nil
}

func (r *Runner) applyDeviceOutputNorm(
	builder *tensor.Builder,
	input *tensor.Tensor,
	deviceFeeds map[*tensor.Tensor]driver.DevicePtr,
) (*tensor.Tensor, error) {
	return r.buildOutputNorm(builder, input, func(info gguf.TensorInfo) (*tensor.Tensor, error) {
		node, pointer, err := r.deviceInput(builder, info)
		if err == nil {
			deviceFeeds[node] = pointer
		}
		return node, err
	})
}

func (r *Runner) buildOutputNorm(
	builder *tensor.Builder,
	input *tensor.Tensor,
	bind func(gguf.TensorInfo) (*tensor.Tensor, error),
) (*tensor.Tensor, error) {
	if r.profile().OutputNorm == model.OutputNormAbsent {
		return input, nil
	}
	if r.spec.UsesUnweightedLayerNorm() {
		return builder.LayerNorm(input, r.spec.LayerNormEpsilon), builder.Err()
	}
	if r.spec.UsesUnweightedRMSNorm() {
		return builder.RMSNorm(input, r.spec.RMSNormEpsilon), builder.Err()
	}
	weight, err := bind(r.weights.OutputNorm)
	if err != nil {
		return nil, err
	}
	var bias *tensor.Tensor
	if r.weights.OutputNormBias != nil {
		bias, err = bind(*r.weights.OutputNormBias)
		if err != nil {
			return nil, err
		}
	}
	return model.ApplyNormalization(builder, input, weight, bias, r.spec), builder.Err()
}

func (r *Runner) layerDeviceInputs(
	builder *tensor.Builder,
	info model.LayerWeights,
) (model.LayerGraphWeights, map[*tensor.Tensor]driver.DevicePtr, error) {
	if r.deviceWeights == nil {
		return model.LayerGraphWeights{}, nil, errors.New("inference: device weights are unavailable")
	}
	return r.deviceWeights.LayerGraphInputs(builder, info)
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
func (r *Runner) ForwardNonCausal(
	ctx context.Context,
	tokenIDs []tokenizer.TokenID,
) (reference.Value, error) {
	if r == nil {
		return reference.Value{}, errors.New("inference: runner is nil")
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.closed {
		return reference.Value{}, errors.New("inference: runner is closed")
	}
	if !r.spec.NonCausalAttention && r.profile().Attention != model.AttentionLFM2 {
		return reference.Value{}, errors.New("inference: model is not configured for non-causal attention")
	}
	return r.forwardNonCausalLocked(ctx, tokenIDs)
}

// DecodeWavTokenizer: decodes semantic tokens into audio-feature frames.
func (r *Runner) DecodeWavTokenizer(
	ctx context.Context,
	tokenIDs []tokenizer.TokenID,
) (reference.Value, error) {
	if r == nil {
		return reference.Value{}, errors.New("inference: runner is nil")
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.closed {
		return reference.Value{}, errors.New("inference: runner is closed")
	}
	if r.forwardPolicy() != model.ForwardWavTokenizer {
		return reference.Value{}, errors.New("inference: audio decode requires wavtokenizer-dec architecture")
	}
	return r.forwardWavTokenizerLocked(ctx, tokenIDs)
}

// ForwardNonCausalLogits: evaluates complete bidirectional sequence and
// returns vocabulary logits for every position in shape [vocabulary, tokens]
// never creates or mutates decoder cache state
func (r *Runner) ForwardNonCausalLogits(
	ctx context.Context,
	tokenIDs []tokenizer.TokenID,
) (reference.Value, error) {
	if r == nil {
		return reference.Value{}, errors.New("inference: runner is nil")
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.closed {
		return reference.Value{}, errors.New("inference: runner is closed")
	}
	if !r.spec.NonCausalAttention && r.profile().Attention != model.AttentionLFM2 {
		return reference.Value{}, errors.New("inference: model is not configured for non-causal attention")
	}
	hidden, err := r.forwardNonCausalLocked(ctx, tokenIDs)
	if err != nil {
		return reference.Value{}, err
	}
	return r.projectAllLogits(ctx, hidden)
}

func (r *Runner) forwardNonCausalLocked(
	ctx context.Context,
	tokenIDs []tokenizer.TokenID,
) (reference.Value, error) {
	if r.forwardPolicy() == model.ForwardWavTokenizer {
		return r.forwardWavTokenizerLocked(ctx, tokenIDs)
	}
	if len(tokenIDs) == 0 {
		return reference.Value{}, errors.New("inference: token sequence is empty")
	}
	if len(tokenIDs) > int(r.spec.ContextLength) {
		return reference.Value{}, fmt.Errorf(
			"inference: token count %d exceeds context length %d",
			len(tokenIDs), r.spec.ContextLength,
		)
	}
	rows := make([]uint32, len(tokenIDs))
	positions := make([]uint32, len(tokenIDs))
	for index, id := range tokenIDs {
		if id < 0 || int(id) >= r.vocab.Len() {
			return reference.Value{}, fmt.Errorf("inference: token ID %d is out of range", id)
		}
		rows[index] = uint32(id)
		positions[index] = uint32(index)
	}
	activation, err := r.loadEmbeddings(ctx, rows)
	if err != nil {
		return reference.Value{}, err
	}
	activation, err = r.addTokenTypeEmbedding(ctx, activation)
	if err != nil {
		return reference.Value{}, err
	}
	activation, err = r.addPositionEmbeddings(ctx, activation, positions)
	if err != nil {
		return reference.Value{}, err
	}
	if scale := r.spec.InputEmbeddingScale(); scale != 1 {
		for index := range activation.Data {
			activation.Data[index] *= scale
		}
	}
	activation, err = r.applyTokenEmbeddingNorm(ctx, activation)
	if err != nil {
		return reference.Value{}, err
	}
	if r.profile().Attention == model.AttentionLFM2 {
		for layerIndex, layerInfo := range r.weights.Layers {
			activation, err = r.runLFM2LayerNonCausal(
				ctx, activation, layerInfo, layerIndex, positions,
			)
			if err != nil {
				return reference.Value{}, fmt.Errorf("inference layer %d: %w", layerIndex, err)
			}
		}
		return r.runOutputNorm(ctx, activation)
	}
	if r.hasPreloadedWeights() {
		return r.forwardDenseLayersNoCachePreloaded(ctx, activation, positions)
	}
	for layerIndex, layerInfo := range r.weights.Layers {
		activation, err = r.runDenseLayerNoCache(
			ctx, activation, layerInfo, layerIndex, positions,
		)
		if err != nil {
			return reference.Value{}, fmt.Errorf("inference layer %d: %w", layerIndex, err)
		}
	}
	return r.runOutputNorm(ctx, activation)
}

func (r *Runner) forwardWavTokenizerLocked(
	ctx context.Context,
	tokenIDs []tokenizer.TokenID,
) (reference.Value, error) {
	if len(tokenIDs) == 0 {
		return reference.Value{}, errors.New("inference: token sequence is empty")
	}
	if len(tokenIDs) > int(r.spec.ContextLength) {
		return reference.Value{}, fmt.Errorf(
			"inference: token count %d exceeds context length %d",
			len(tokenIDs), r.spec.ContextLength,
		)
	}
	rows := make([]uint32, len(tokenIDs))
	for index, id := range tokenIDs {
		if id < 0 || int(id) >= r.vocab.Len() {
			return reference.Value{}, fmt.Errorf("inference: token ID %d is out of range", id)
		}
		rows[index] = uint32(id)
	}
	embeddings, err := r.loadEmbeddings(ctx, rows)
	if err != nil {
		return reference.Value{}, err
	}
	runtime := r.newInferenceGraphRuntime(ctx)
	input := runtime.input("wavtokenizer.embeddings", embeddings)
	graphWeights, hostFeeds, deviceFeeds, err := r.wavTokenizerGraphInputs(ctx, runtime.builder)
	if err != nil {
		return reference.Value{}, err
	}
	runtime.addHostFeeds(hostFeeds)
	runtime.addDeviceFeeds(deviceFeeds)
	output, err := model.BuildWavTokenizerDecoder(runtime.builder, input, r.spec, graphWeights)
	if err != nil {
		return reference.Value{}, err
	}
	results, err := runtime.execute(output)
	if err != nil {
		return reference.Value{}, err
	}
	return results[output], nil
}

func (r *Runner) projectAllLogits(
	ctx context.Context,
	hidden reference.Value,
) (reference.Value, error) {
	if r.spec.IsEncoderOnly() {
		return reference.Value{}, errors.New("inference: encoder exposes hidden states, not vocabulary logits")
	}
	if hidden.Shape.Rank != 2 || hidden.Shape.Dims[0] != uint64(r.spec.EmbeddingLength) {
		return reference.Value{}, errors.New("inference: non-causal hidden-state shape is incompatible")
	}
	outputInfo := r.weights.TokenEmbedding
	if r.weights.Output != nil {
		outputInfo = *r.weights.Output
	}
	shape := tensor.MustShape(uint64(r.spec.VocabularySize), hidden.Shape.Dims[1])
	if !r.hasPreloadedWeights() {
		elements, err := shape.Elements()
		if err != nil || elements > uint64(math.MaxInt) {
			return reference.Value{}, errors.New("inference: non-causal logits shape is too large")
		}
		result := reference.Value{Shape: shape, Data: make([]float32, 0, int(elements))}
		width := int(hidden.Shape.Dims[0])
		for token := 0; token < int(hidden.Shape.Dims[1]); token++ {
			start := token * width
			logits, err := r.logits(ctx, outputInfo, hidden.Data[start:start+width])
			if err != nil {
				return reference.Value{}, err
			}
			result.Data = append(result.Data, logits...)
		}
		return result, nil
	}
	runtime := r.newInferenceGraphRuntime(ctx)
	input := runtime.input("non_causal.hidden", hidden)
	table, err := runtime.weight(outputInfo)
	if err != nil {
		return reference.Value{}, err
	}
	output := runtime.builder.MulMat(table, input)
	if r.weights.OutputBias != nil {
		bias, biasErr := runtime.weight(*r.weights.OutputBias)
		if biasErr != nil {
			return reference.Value{}, biasErr
		}
		output = runtime.builder.Add(output, bias)
	}
	if scale := r.spec.OutputLogitMultiplier(); scale != 1 {
		output = runtime.builder.Scale(output, scale)
	}
	if err := runtime.builder.Err(); err != nil {
		return reference.Value{}, err
	}
	results, err := runtime.execute(output)
	if err != nil {
		return reference.Value{}, err
	}
	result := results[output]
	result.Data = r.finalizeLogits(result.Data)
	return result, nil
}

func (r *Runner) forwardT5EncoderLocked(
	ctx context.Context,
	tokenIDs []tokenizer.TokenID,
) (reference.Value, error) {
	if len(tokenIDs) == 0 {
		return reference.Value{}, errors.New("inference: token sequence is empty")
	}
	if len(tokenIDs) > int(r.spec.ContextLength) {
		return reference.Value{}, fmt.Errorf(
			"inference: token count %d exceeds context length %d",
			len(tokenIDs),
			r.spec.ContextLength,
		)
	}
	rows := make([]uint32, len(tokenIDs))
	for index, id := range tokenIDs {
		if id < 0 || int(id) >= r.vocab.Len() {
			return reference.Value{}, fmt.Errorf("inference: token ID %d is out of range", id)
		}
		rows[index] = uint32(id)
	}
	activation, err := r.loadEmbeddings(ctx, rows)
	if err != nil {
		return reference.Value{}, err
	}
	layers := r.weights.Layers
	if r.forwardPolicy() == model.ForwardT5 {
		layers = r.weights.EncoderLayers
	}
	for layerIndex, layerInfo := range layers {
		activation, err = r.runT5EncoderLayer(ctx, activation, layerInfo, layerIndex)
		if err != nil {
			return reference.Value{}, fmt.Errorf("inference encoder layer %d: %w", layerIndex, err)
		}
	}
	if r.forwardPolicy() == model.ForwardT5 {
		return r.runT5EncoderOutputNorm(ctx, activation)
	}
	return r.runOutputNorm(ctx, activation)
}

// NewT5Session: encodes one source sequence.
func (r *Runner) NewT5Session(ctx context.Context, sourceIDs []tokenizer.TokenID) (*T5Session, error) {
	if r == nil {
		return nil, errors.New("inference: runner is nil")
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.closed {
		return nil, errors.New("inference: runner is closed")
	}
	if r.forwardPolicy() != model.ForwardT5 {
		return nil, errors.New("inference: T5 session requires T5 architecture")
	}
	encoder, err := r.forwardT5EncoderLocked(ctx, sourceIDs)
	if err != nil {
		return nil, err
	}
	return &T5Session{Encoder: encoder}, nil
}

// DecodeT5: appends decoder tokens and returns all chunk logits.
func (r *Runner) DecodeT5(
	ctx context.Context,
	session *T5Session,
	decoderIDs []tokenizer.TokenID,
) (reference.Value, *T5Session, error) {
	if r == nil {
		return reference.Value{}, nil, errors.New("inference: runner is nil")
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.closed {
		return reference.Value{}, nil, errors.New("inference: runner is closed")
	}
	if r.forwardPolicy() != model.ForwardT5 {
		return reference.Value{}, nil, errors.New("inference: T5 decode requires T5 architecture")
	}
	return r.decodeT5Locked(ctx, session, decoderIDs)
}

func (r *Runner) decodeT5Locked(
	ctx context.Context,
	session *T5Session,
	decoderIDs []tokenizer.TokenID,
) (reference.Value, *T5Session, error) {
	if session == nil {
		return reference.Value{}, nil, errors.New("inference: T5 session is nil")
	}
	if session.Encoder.Shape.Rank != 2 ||
		session.Encoder.Shape.Dims[0] != uint64(r.spec.EmbeddingLength) ||
		session.Encoder.Shape.Dims[1] == 0 {
		return reference.Value{}, nil, errors.New("inference: T5 encoder state shape is incompatible")
	}
	if len(decoderIDs) == 0 {
		return reference.Value{}, nil, errors.New("inference: T5 decoder token sequence is empty")
	}
	var pastTokens, nextPosition uint32
	if session.Cache != nil {
		if err := r.validateT5Cache(session.Cache, session.Encoder.Shape.Dims[1]); err != nil {
			return reference.Value{}, nil, err
		}
		pastTokens = session.Cache.Tokens
		nextPosition = effectiveCachePosition(session.Cache)
	}
	if uint64(pastTokens)+uint64(len(decoderIDs)) > uint64(r.spec.ContextLength) {
		return reference.Value{}, nil, errors.New("inference: T5 decoder sequence exceeds context length")
	}
	rows := make([]uint32, len(decoderIDs))
	for index, id := range decoderIDs {
		if id < 0 || int(id) >= r.vocab.Len() {
			return reference.Value{}, nil, fmt.Errorf("inference: token ID %d is out of range", id)
		}
		rows[index] = uint32(id)
	}
	activation, err := r.loadEmbeddings(ctx, rows)
	if err != nil {
		return reference.Value{}, nil, err
	}
	nextCache := &KVCache{
		Layers:   make([]LayerCache, len(r.weights.Layers)),
		Tokens:   pastTokens + uint32(len(decoderIDs)),
		Position: nextPosition + uint32(len(decoderIDs)),
	}
	for layerIndex, layerInfo := range r.weights.Layers {
		var past *LayerCache
		if session.Cache != nil {
			past = &session.Cache.Layers[layerIndex]
		}
		activation, nextCache.Layers[layerIndex], err = r.runT5DecoderLayer(
			ctx, activation, session.Encoder, layerInfo, layerIndex, past,
		)
		if err != nil {
			return reference.Value{}, nil, fmt.Errorf("inference decoder layer %d: %w", layerIndex, err)
		}
	}
	activation, err = r.runOutputNorm(ctx, activation)
	if err != nil {
		return reference.Value{}, nil, err
	}
	logits, err := r.projectAllLogits(ctx, activation)
	if err != nil {
		return reference.Value{}, nil, err
	}
	return logits, &T5Session{Encoder: session.Encoder, Cache: nextCache}, nil
}

// ForwardCached: evaluates prompt chunk and returns host KV/recurrent cache
// suitable for later incremental call or ShiftCache edit
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
	if !supportsMultiAxisPositions(r.spec) {
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

func validateVisualExpertBlocks(tokenCount int, blocks []AttentionBlock, overrides []EmbeddingOverride) ([]AttentionBlock, error) {
	ordered := slices.Clone(blocks)
	slices.SortFunc(ordered, func(a, b AttentionBlock) int { return cmp.Compare(a.Start, b.Start) })
	expected := 0
	previousEnd := uint32(0)
	for index, block := range ordered {
		if block.Start >= block.End || block.End > uint32(tokenCount) || index > 0 && block.Start < previousEnd {
			return nil, fmt.Errorf("inference: invalid CogVLM visual expert block [%d,%d)", block.Start, block.End)
		}
		expected += int(block.End - block.Start)
		previousEnd = block.End
	}
	if len(overrides) != expected {
		return nil, fmt.Errorf("inference: CogVLM visual blocks cover %d tokens but have %d embeddings", expected, len(overrides))
	}
	seen := make(map[uint32]struct{}, len(overrides))
	for _, override := range overrides {
		inside := false
		for _, block := range ordered {
			if override.TokenIndex >= block.Start && override.TokenIndex < block.End {
				inside = true
				break
			}
		}
		if !inside {
			return nil, fmt.Errorf("inference: CogVLM visual embedding token %d is outside visual blocks", override.TokenIndex)
		}
		if _, ok := seen[override.TokenIndex]; ok {
			return nil, fmt.Errorf("inference: duplicate CogVLM visual embedding for token index %d", override.TokenIndex)
		}
		seen[override.TokenIndex] = struct{}{}
	}
	return ordered, nil
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
	rows := make([]uint32, len(tokenIDs))
	positions := make([]uint32, len(tokenIDs))
	for index, id := range tokenIDs {
		if id < 0 || int(id) >= r.vocab.Len() {
			return reference.Value{}, nil, fmt.Errorf("inference: token ID %d is out of range", id)
		}
		rows[index] = uint32(id)
		positions[index] = nextPosition + uint32(index)
	}
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

func applyGemmaRawEmbeddingOverrides(activation *reference.Value, overrides []EmbeddingOverride, scale float32) error {
	if activation == nil {
		return errors.New("inference: embedding activation is nil")
	}
	if scale != 1 {
		for index := range activation.Data {
			activation.Data[index] *= scale
		}
	}
	return applyEmbeddingOverrides(activation, overrides)
}

func applyEmbeddingOverrides(activation *reference.Value, overrides []EmbeddingOverride) error {
	if len(overrides) == 0 {
		return nil
	}
	if activation == nil || activation.Shape.Rank != 2 {
		return errors.New("inference: token embeddings have invalid shape")
	}
	width := int(activation.Shape.Dims[0])
	tokens := activation.Shape.Dims[1]
	if width <= 0 || len(activation.Data) != width*int(tokens) {
		return errors.New("inference: token embeddings have invalid storage")
	}
	seen := make(map[uint32]struct{}, len(overrides))
	for overrideIndex, override := range overrides {
		if uint64(override.TokenIndex) >= tokens {
			return fmt.Errorf(
				"inference: embedding override %d token index %d is out of range for %d tokens",
				overrideIndex, override.TokenIndex, tokens,
			)
		}
		if _, duplicate := seen[override.TokenIndex]; duplicate {
			return fmt.Errorf("inference: duplicate embedding override for token index %d", override.TokenIndex)
		}
		seen[override.TokenIndex] = struct{}{}
		if len(override.Embedding) != width {
			return fmt.Errorf(
				"inference: embedding override %d width %d differs from model width %d",
				overrideIndex, len(override.Embedding), width,
			)
		}
		for valueIndex, value := range override.Embedding {
			if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) {
				return fmt.Errorf(
					"inference: embedding override %d contains non-finite value at %d",
					overrideIndex, valueIndex,
				)
			}
		}
		start := int(override.TokenIndex) * width
		copy(activation.Data[start:start+width], override.Embedding)
	}
	return nil
}

func validateCogVLMVisualOverrides(tokenCount int, overrides []EmbeddingOverride) error {
	if len(overrides) != tokenCount {
		return fmt.Errorf(
			"inference: CogVLM visual mode requires one projected embedding per token; got %d for %d tokens",
			len(overrides), tokenCount,
		)
	}
	seen := make([]bool, tokenCount)
	for _, override := range overrides {
		if int(override.TokenIndex) >= tokenCount {
			return fmt.Errorf(
				"inference: CogVLM visual embedding token index %d is out of range for %d tokens",
				override.TokenIndex, tokenCount,
			)
		}
		if seen[override.TokenIndex] {
			return fmt.Errorf(
				"inference: duplicate CogVLM visual embedding for token index %d",
				override.TokenIndex,
			)
		}
		seen[override.TokenIndex] = true
	}
	return nil
}

func (r *Runner) applyCogVLMVisualWeights(
	ctx context.Context,
	builder *tensor.Builder,
	info model.LayerWeights,
	weights *model.LayerGraphWeights,
	hostFeeds map[*tensor.Tensor]reference.Value,
	deviceFeeds map[*tensor.Tensor]driver.DevicePtr,
) error {
	if r.spec.Architecture != "cogvlm" {
		return errors.New("inference: visual expert weights require CogVLM architecture")
	}
	if weights == nil {
		return errors.New("inference: CogVLM graph weights are nil")
	}
	infos := []*gguf.TensorInfo{
		info.VisualAttentionQKV,
		info.VisualAttentionOutput,
		info.VisualFeedForwardGate,
		info.VisualFeedForwardUp,
		info.VisualFeedForwardDown,
	}
	nodes := make([]*tensor.Tensor, len(infos))
	for index, tensorInfo := range infos {
		if tensorInfo == nil {
			return errors.New("inference: CogVLM visual expert catalog is incomplete")
		}
		if r.hasPreloadedWeights() {
			node, pointer, err := r.deviceInput(builder, *tensorInfo)
			if err != nil {
				return err
			}
			nodes[index] = node
			deviceFeeds[node] = pointer
			continue
		}
		value, err := model.LoadHostTensor(ctx, r.file, *tensorInfo)
		if err != nil {
			return err
		}
		node := builder.Input(tensorInfo.Name, dtype.F32, value.Shape)
		nodes[index] = node
		hostFeeds[node] = value
	}
	if err := selectCogVLMVisualGraphWeights(weights, nodes); err != nil {
		return err
	}
	return builder.Err()
}

func selectCogVLMVisualGraphWeights(
	weights *model.LayerGraphWeights,
	nodes []*tensor.Tensor,
) error {
	if weights == nil {
		return errors.New("inference: CogVLM graph weights are nil")
	}
	if len(nodes) != 5 {
		return errors.New("inference: CogVLM visual graph weight set is incomplete")
	}
	for _, node := range nodes {
		if node == nil {
			return errors.New("inference: CogVLM visual graph weight is nil")
		}
	}
	weights.AttentionQ, weights.AttentionK, weights.AttentionV = nil, nil, nil
	weights.AttentionQKV = nodes[0]
	weights.AttentionOutput = nodes[1]
	weights.FeedForwardGate = nodes[2]
	weights.FeedForwardUp = nodes[3]
	weights.FeedForwardDown = nodes[4]
	return nil
}

func (r *Runner) forwardDenseLayersPreloaded(
	ctx context.Context,
	activation reference.Value,
	embeddingSkip reference.Value,
	perLayerInputs []reference.Value,
	positions []uint32,
	multiPositions *MultiAxisPositions,
	deepstackBase reference.Value,
	deepstackInputs []reference.Value,
	attentionBlockIDs []float32,
	cache *KVCache,
	nextCache *KVCache,
	visualMode bool,
	applyOutputNorm bool,
	capture *layerInputCapture,
) (reference.Value, *KVCache, error) {
	builder := r.newGraphBuilder()
	input := builder.Input("model.input", dtype.F32, activation.Shape)
	current := input
	hostFeeds := map[*tensor.Tensor]reference.Value{input: activation}
	deviceFeeds := make(map[*tensor.Tensor]driver.DevicePtr)
	var attentionBlockInput *tensor.Tensor
	if len(attentionBlockIDs) > 0 {
		shape := tensor.MustShape(uint64(len(attentionBlockIDs)))
		attentionBlockInput = builder.Input("model.attention_block_ids", dtype.F32, shape)
		hostFeeds[attentionBlockInput] = reference.Value{Shape: shape, Data: attentionBlockIDs}
	}
	keys := make([]*tensor.Tensor, len(r.weights.Layers))
	values := make([]*tensor.Tensor, len(r.weights.Layers))
	captured := make(map[int32]*tensor.Tensor)
	for layerIndex, info := range r.weights.Layers {
		plan := r.layerPlan(layerIndex, info.Recurrent)
		if stream := deepstackInputForLayer(plan.DeepstackBefore, deepstackBase, deepstackInputs); stream != nil {
			deepstack := builder.Input(
				fmt.Sprintf("blk.%d.deepstack_input", layerIndex), dtype.F32, stream.Shape,
			)
			hostFeeds[deepstack] = *stream
			current = builder.Add(current, deepstack)
		}
		if capture.wants(layerIndex) {
			captured[int32(layerIndex)] = current
		}
		graphWeights, layerFeeds, err := r.layerDeviceInputs(builder, info)
		if err != nil {
			return reference.Value{}, nil, err
		}
		if visualMode {
			if err := r.applyCogVLMVisualWeights(
				ctx, builder, info, &graphWeights, nil, layerFeeds,
			); err != nil {
				return reference.Value{}, nil, err
			}
		}
		sideInputs := layerSideInputs{embeddingSkip: input, attentionBlock: attentionBlockInput}
		if len(perLayerInputs) > 0 {
			perLayer := builder.Input(
				fmt.Sprintf("blk.%d.per_layer_input", layerIndex),
				dtype.F32,
				perLayerInputs[layerIndex].Shape,
			)
			hostFeeds[perLayer] = perLayerInputs[layerIndex]
			sideInputs.perLayerInput = perLayer
		}
		for node, pointer := range layerFeeds {
			deviceFeeds[node] = pointer
		}
		if _, err := bindLayerSideInputs(
			builder, r.spec, positions, plan, hostFeeds, &graphWeights, sideInputs,
		); err != nil {
			return reference.Value{}, nil, err
		}
		var pastKey, pastValue *tensor.Tensor
		if plan.SharedKV {
			pastKey, pastValue = keys[plan.KVSource], values[plan.KVSource]
		} else if cache != nil {
			past := cache.Layers[layerIndex]
			pastKey = builder.Input(
				fmt.Sprintf("blk.%d.cache_key", layerIndex),
				dtype.F32,
				past.Key.Shape,
			)
			pastValue = builder.Input(
				fmt.Sprintf("blk.%d.cache_value", layerIndex),
				dtype.F32,
				past.Value.Shape,
			)
			hostFeeds[pastKey] = past.Key
			hostFeeds[pastValue] = past.Value
		}
		var result model.DenseBlockResult
		if multiPositions != nil {
			result, err = model.BuildDenseBlockCachedForLayerWithMultiPositions(
				builder, current, r.spec, graphWeights, [4][]uint32(*multiPositions),
				pastKey, pastValue, uint32(layerIndex),
			)
		} else {
			result, err = model.BuildDenseBlockCachedForLayer(
				builder, current, r.spec, graphWeights, positions,
				pastKey, pastValue, uint32(layerIndex),
			)
		}
		if err != nil {
			return reference.Value{}, nil, err
		}
		current = result.Output
		if stream := deepstackInputForLayer(plan.DeepstackAfter, deepstackBase, deepstackInputs); stream != nil {
			deepstack := builder.Input(
				fmt.Sprintf("blk.%d.deepstack_output", layerIndex), dtype.F32, stream.Shape,
			)
			hostFeeds[deepstack] = *stream
			current = builder.Add(current, deepstack)
		}
		keys[layerIndex] = result.Key
		values[layerIndex] = result.Value
	}
	if applyOutputNorm {
		normalized, normErr := r.applyDeviceOutputNorm(builder, current, deviceFeeds)
		if normErr != nil {
			return reference.Value{}, nil, normErr
		}
		current = normalized
	}
	if err := builder.Err(); err != nil {
		return reference.Value{}, nil, err
	}
	outputs := make([]*tensor.Tensor, 1, 1+2*len(keys))
	outputs[0] = current
	for layerIndex := range keys {
		outputs = append(outputs, keys[layerIndex], values[layerIndex])
	}
	if capture != nil {
		for _, layer := range capture.order {
			if node := captured[layer]; node != nil {
				outputs = append(outputs, node)
			}
		}
	}
	results, err := r.cuda.ExecuteWithDeviceFeeds(ctx, outputs, hostFeeds, deviceFeeds)
	if err != nil {
		return reference.Value{}, nil, err
	}
	for layerIndex := range keys {
		nextCache.Layers[layerIndex] = LayerCache{
			Key:   results[keys[layerIndex]],
			Value: results[values[layerIndex]],
		}
	}
	for layer, node := range captured {
		capture.set(int(layer), results[node])
	}
	return results[current], nextCache, nil
}

func (r *Runner) forwardDenseLayersNoCachePreloaded(
	ctx context.Context,
	activation reference.Value,
	positions []uint32,
) (reference.Value, error) {
	builder := r.newGraphBuilder()
	input := builder.Input("model.input", dtype.F32, activation.Shape)
	current := input
	hostFeeds := map[*tensor.Tensor]reference.Value{input: activation}
	deviceFeeds := make(map[*tensor.Tensor]driver.DevicePtr)
	for layerIndex, info := range r.weights.Layers {
		plan := r.layerPlan(layerIndex, info.Recurrent)
		graphWeights, layerFeeds, err := r.layerDeviceInputs(builder, info)
		if err != nil {
			return reference.Value{}, err
		}
		for node, pointer := range layerFeeds {
			deviceFeeds[node] = pointer
		}
		if _, err := bindLayerSideInputs(
			builder, r.spec, positions, plan, hostFeeds, &graphWeights, layerSideInputs{},
		); err != nil {
			return reference.Value{}, err
		}
		result, err := model.BuildDenseBlockCachedForLayer(
			builder,
			current,
			r.spec,
			graphWeights,
			positions,
			nil,
			nil,
			uint32(layerIndex),
		)
		if err != nil {
			return reference.Value{}, err
		}
		current = result.Output
	}
	var err error
	current, err = r.applyDeviceOutputNorm(builder, current, deviceFeeds)
	if err != nil {
		return reference.Value{}, err
	}
	if err := builder.Err(); err != nil {
		return reference.Value{}, err
	}
	results, err := r.cuda.ExecuteWithDeviceFeeds(
		ctx,
		[]*tensor.Tensor{current},
		hostFeeds,
		deviceFeeds,
	)
	if err != nil {
		return reference.Value{}, err
	}
	return results[current], nil
}

func (r *Runner) runT5EncoderLayer(
	ctx context.Context,
	activation reference.Value,
	info model.LayerWeights,
	layerIndex int,
) (reference.Value, error) {
	runtime := r.newInferenceGraphRuntime(ctx)
	input := runtime.input("input", activation)
	graphWeights, err := runtime.layer(info, fmt.Sprintf("enc.blk.%d.", layerIndex))
	if err != nil {
		return reference.Value{}, err
	}
	output, err := model.BuildT5EncoderBlock(runtime.builder, input, r.spec, graphWeights)
	if err != nil {
		return reference.Value{}, err
	}
	results, err := runtime.execute(output)
	if err != nil {
		return reference.Value{}, err
	}
	return results[output], nil
}

func (r *Runner) runT5DecoderLayer(
	ctx context.Context,
	activation, encoder reference.Value,
	info model.LayerWeights,
	layerIndex int,
	past *LayerCache,
) (reference.Value, LayerCache, error) {
	runtime := r.newInferenceGraphRuntime(ctx)
	input := runtime.input("input", activation)
	var encoderInput *tensor.Tensor
	var pastSelfKey, pastSelfValue, pastCrossKey, pastCrossValue *tensor.Tensor
	if past == nil {
		encoderInput = runtime.input("encoder", encoder)
	} else {
		pastSelfKey = runtime.input("past_self_key", past.Key)
		pastSelfValue = runtime.input("past_self_value", past.Value)
		crossKey, hasKey := past.States["cross_key"]
		crossValue, hasValue := past.States["cross_value"]
		if !hasKey || !hasValue {
			return reference.Value{}, LayerCache{}, errors.New("T5 decoder cross cache is missing")
		}
		pastCrossKey = runtime.input("past_cross_key", crossKey.Value)
		pastCrossValue = runtime.input("past_cross_value", crossValue.Value)
	}
	graphWeights, err := runtime.layer(info, fmt.Sprintf("dec.blk.%d.", layerIndex))
	if err != nil {
		return reference.Value{}, LayerCache{}, err
	}
	result, err := model.BuildT5DecoderBlockCached(
		runtime.builder, input, encoderInput, r.spec, graphWeights,
		pastSelfKey, pastSelfValue, pastCrossKey, pastCrossValue,
	)
	if err != nil {
		return reference.Value{}, LayerCache{}, err
	}
	outputs := []*tensor.Tensor{result.Output, result.Key, result.Value}
	for _, state := range result.FixedStates {
		outputs = append(outputs, state)
	}
	results, err := runtime.execute(outputs...)
	if err != nil {
		return reference.Value{}, LayerCache{}, err
	}
	layerCache := LayerCache{
		Key: results[result.Key], Value: results[result.Value],
		States: make(map[string]LayerState, len(result.FixedStates)),
	}
	for name, state := range result.FixedStates {
		layerCache.States[name] = LayerState{Mode: CacheStateFixed, Value: results[state]}
	}
	return results[result.Output], layerCache, nil
}

func (r *Runner) runT5EncoderOutputNorm(
	ctx context.Context,
	activation reference.Value,
) (reference.Value, error) {
	if r.weights.EncoderOutputNorm == nil {
		return reference.Value{}, errors.New("inference: T5 encoder output norm is missing")
	}
	runtime := r.newInferenceGraphRuntime(ctx)
	input := runtime.input("enc.output_norm.input", activation)
	weight, err := runtime.weight(*r.weights.EncoderOutputNorm)
	if err != nil {
		return reference.Value{}, err
	}
	output := runtime.builder.WeightedRMSNorm(input, weight, r.spec.RMSNormEpsilon)
	if err := runtime.builder.Err(); err != nil {
		return reference.Value{}, err
	}
	results, err := runtime.execute(output)
	if err != nil {
		return reference.Value{}, err
	}
	return results[output], nil
}

func (r *Runner) runLayerCached(
	ctx context.Context,
	activation reference.Value,
	info model.LayerWeights,
	layerIndex int,
	positions []uint32,
	tokenRows []uint32,
	past *LayerCache,
	embeddingSkip reference.Value,
	perLayerInput *reference.Value,
	multiPositions *MultiAxisPositions,
	visualMode bool,
	attentionBlockIDs []float32,
) (reference.Value, LayerCache, error) {
	plan := r.layerPlan(layerIndex, info.Recurrent)
	if plan.Attention == model.AttentionQwenGDN {
		return r.runQwen35LayerCached(
			ctx,
			activation,
			info,
			layerIndex,
			positions,
			past,
			multiPositions,
		)
	}
	if plan.Attention == model.AttentionLFM2 {
		return r.runLFM2LayerCached(ctx, activation, info, layerIndex, positions, past)
	}
	builder := r.newGraphBuilder()
	input := builder.Input("input", dtype.F32, activation.Shape)
	hostFeeds := map[*tensor.Tensor]reference.Value{input: activation}
	deviceFeeds := make(map[*tensor.Tensor]driver.DevicePtr)
	var graphWeights model.LayerGraphWeights
	if r.hasPreloadedWeights() {
		var err error
		graphWeights, deviceFeeds, err = r.layerDeviceInputs(builder, info)
		if err != nil {
			return reference.Value{}, LayerCache{}, err
		}
	} else {
		hostLayer, err := model.LoadHostLayer(ctx, r.file, info)
		if err != nil {
			return reference.Value{}, LayerCache{}, err
		}
		var layerFeeds map[*tensor.Tensor]reference.Value
		graphWeights, layerFeeds, err = hostLayer.GraphInputs(builder, fmt.Sprintf("blk.%d.", layerIndex))
		if err != nil {
			return reference.Value{}, LayerCache{}, err
		}
		for node, value := range layerFeeds {
			hostFeeds[node] = value
		}
	}
	if visualMode {
		if err := r.applyCogVLMVisualWeights(
			ctx, builder, info, &graphWeights, hostFeeds, deviceFeeds,
		); err != nil {
			return reference.Value{}, LayerCache{}, err
		}
	}
	sideInputs := layerSideInputs{}
	if plan.EmbeddingSkip {
		skip := builder.Input("embedding_skip", dtype.F32, embeddingSkip.Shape)
		hostFeeds[skip] = embeddingSkip
		sideInputs.embeddingSkip = skip
	}
	if perLayerInput != nil {
		perLayer := builder.Input("per_layer_input", dtype.F32, perLayerInput.Shape)
		hostFeeds[perLayer] = *perLayerInput
		sideInputs.perLayerInput = perLayer
	}
	if len(attentionBlockIDs) > 0 {
		shape := tensor.MustShape(uint64(len(attentionBlockIDs)))
		blockInput := builder.Input("attention_block_ids", dtype.F32, shape)
		hostFeeds[blockInput] = reference.Value{Shape: shape, Data: attentionBlockIDs}
		sideInputs.attentionBlock = blockInput
	}
	boundSideInputs, sideErr := bindLayerSideInputs(
		builder, r.spec, positions, plan, hostFeeds, &graphWeights, sideInputs,
	)
	if sideErr != nil {
		return reference.Value{}, LayerCache{}, sideErr
	}
	cacheInputs, cacheErr := r.hostLayerCacheInputs(builder, layerIndex, info, plan, past, hostFeeds)
	if cacheErr != nil {
		return reference.Value{}, LayerCache{}, cacheErr
	}
	var (
		result model.DenseBlockResult
		err    error
	)
	var dispatchMultiPositions *[4][]uint32
	if multiPositions != nil {
		converted := [4][]uint32(*multiPositions)
		dispatchMultiPositions = &converted
	}
	result, err = model.BuildArchitectureBlockCached(model.BlockDispatchOptions{
		Builder:          builder,
		Input:            input,
		Spec:             r.spec,
		Weights:          graphWeights,
		Positions:        positions,
		MultiPositions:   dispatchMultiPositions,
		TokenRows:        tokenRows,
		PastKey:          cacheInputs.key,
		PastValue:        cacheInputs.value,
		PastIndexerKey:   cacheInputs.indexerKey,
		PastConvState:    cacheInputs.convState,
		PastSSMState:     cacheInputs.ssmState,
		PastStates:       cacheInputs.states,
		CurrentPositions: boundSideInputs.currentPositions,
		PerLayerInput:    graphWeights.PerLayerInput,
		Layer:            uint32(layerIndex),
		Recurrent:        info.Recurrent,
		Plan:             &plan,
	})
	if err != nil {
		return reference.Value{}, LayerCache{}, err
	}
	outputTensor := result.Output
	if r.hasPreloadedWeights() && layerIndex == len(r.weights.Layers)-1 {
		outputTensor, err = r.applyDeviceOutputNorm(builder, result.Output, deviceFeeds)
		if err != nil {
			return reference.Value{}, LayerCache{}, err
		}
	}
	outputs := []*tensor.Tensor{outputTensor, result.Key, result.Value}
	if result.Auxiliary != nil {
		outputs = append(outputs, result.Auxiliary)
	}
	for _, state := range result.States {
		outputs = append(outputs, state)
	}
	for _, state := range result.FixedStates {
		outputs = append(outputs, state)
	}
	var results map[*tensor.Tensor]reference.Value
	if r.hasPreloadedWeights() {
		results, err = r.cuda.ExecuteWithDeviceFeeds(ctx, outputs, hostFeeds, deviceFeeds)
	} else {
		results, err = r.cuda.Execute(ctx, outputs, hostFeeds)
	}
	if err != nil {
		return reference.Value{}, LayerCache{}, err
	}
	layerCache := LayerCache{
		Key:   results[result.Key],
		Value: results[result.Value],
	}
	if result.Auxiliary != nil {
		auxiliary := results[result.Auxiliary]
		layerCache.Auxiliary = &auxiliary
	}
	if len(result.States)+len(result.FixedStates) > 0 {
		layerCache.States = make(map[string]LayerState, len(result.States)+len(result.FixedStates))
		for name, state := range result.States {
			layerCache.States[name] = LayerState{Mode: CacheStateToken, Value: results[state]}
		}
		for name, state := range result.FixedStates {
			layerCache.States[name] = LayerState{Mode: CacheStateFixed, Value: results[state]}
		}
	}
	return results[outputTensor], layerCache, nil
}

func (r *Runner) runLFM2LayerCached(
	ctx context.Context,
	activation reference.Value,
	info model.LayerWeights,
	layerIndex int,
	positions []uint32,
	past *LayerCache,
) (reference.Value, LayerCache, error) {
	builder := r.newGraphBuilder()
	input := builder.Input("input", dtype.F32, activation.Shape)
	hostFeeds := map[*tensor.Tensor]reference.Value{input: activation}
	deviceFeeds := make(map[*tensor.Tensor]driver.DevicePtr)
	var graphWeights model.LayerGraphWeights
	var err error
	if r.hasPreloadedWeights() {
		graphWeights, deviceFeeds, err = r.layerDeviceInputs(builder, info)
	} else {
		var hostLayer model.HostLayer
		hostLayer, err = model.LoadHostLayer(ctx, r.file, info)
		if err == nil {
			var layerFeeds map[*tensor.Tensor]reference.Value
			graphWeights, layerFeeds, err = hostLayer.GraphInputs(builder, fmt.Sprintf("blk.%d.", layerIndex))
			for node, value := range layerFeeds {
				hostFeeds[node] = value
			}
		}
	}
	if err != nil {
		return reference.Value{}, LayerCache{}, err
	}
	var pastKey, pastValue *tensor.Tensor
	if info.Recurrent {
		keyValue := reference.Value{}
		valueValue := reference.Value{}
		if past == nil {
			shape := tensor.MustShape(
				uint64(r.spec.ShortConvCacheLength-1), uint64(r.spec.EmbeddingLength),
			)
			elements, _ := shape.Elements()
			keyValue = reference.Value{Shape: shape, Data: make([]float32, int(elements))}
			valueValue = reference.Value{Shape: tensor.MustShape(1), Data: []float32{0}}
		} else {
			keyValue, valueValue = past.Key, past.Value
		}
		pastKey = builder.Input(fmt.Sprintf("blk.%d.conv_state", layerIndex), dtype.F32, keyValue.Shape)
		pastValue = builder.Input(fmt.Sprintf("blk.%d.reserved_state", layerIndex), dtype.F32, valueValue.Shape)
		hostFeeds[pastKey], hostFeeds[pastValue] = keyValue, valueValue
	} else if past != nil {
		pastKey = builder.Input(fmt.Sprintf("blk.%d.cache_key", layerIndex), dtype.F32, past.Key.Shape)
		pastValue = builder.Input(fmt.Sprintf("blk.%d.cache_value", layerIndex), dtype.F32, past.Value.Shape)
		hostFeeds[pastKey], hostFeeds[pastValue] = past.Key, past.Value
	}
	result, err := model.BuildLFM2BlockCached(
		builder, input, r.spec, graphWeights, positions, info.Recurrent, pastKey, pastValue,
		uint32(layerIndex),
	)
	if err != nil {
		return reference.Value{}, LayerCache{}, err
	}
	output := result.Output
	if r.hasPreloadedWeights() && layerIndex == len(r.weights.Layers)-1 {
		output, err = r.applyDeviceOutputNorm(builder, output, deviceFeeds)
		if err != nil {
			return reference.Value{}, LayerCache{}, err
		}
	}
	outputs := []*tensor.Tensor{output, result.Key, result.Value}
	var results map[*tensor.Tensor]reference.Value
	if r.hasPreloadedWeights() {
		results, err = r.cuda.ExecuteWithDeviceFeeds(ctx, outputs, hostFeeds, deviceFeeds)
	} else {
		results, err = r.cuda.Execute(ctx, outputs, hostFeeds)
	}
	if err != nil {
		return reference.Value{}, LayerCache{}, err
	}
	return results[output], LayerCache{Key: results[result.Key], Value: results[result.Value]}, nil
}

func (r *Runner) runLFM2LayerNonCausal(
	ctx context.Context,
	activation reference.Value,
	info model.LayerWeights,
	layerIndex int,
	positions []uint32,
) (reference.Value, error) {
	builder := r.newGraphBuilder()
	input := builder.Input("input", dtype.F32, activation.Shape)
	hostFeeds := map[*tensor.Tensor]reference.Value{input: activation}
	deviceFeeds := make(map[*tensor.Tensor]driver.DevicePtr)
	var graphWeights model.LayerGraphWeights
	var err error
	if r.hasPreloadedWeights() {
		graphWeights, deviceFeeds, err = r.layerDeviceInputs(builder, info)
	} else {
		var hostLayer model.HostLayer
		hostLayer, err = model.LoadHostLayer(ctx, r.file, info)
		if err == nil {
			var layerFeeds map[*tensor.Tensor]reference.Value
			graphWeights, layerFeeds, err = hostLayer.GraphInputs(
				builder, fmt.Sprintf("blk.%d.", layerIndex),
			)
			for node, value := range layerFeeds {
				hostFeeds[node] = value
			}
		}
	}
	if err != nil {
		return reference.Value{}, err
	}
	var state, reserved *tensor.Tensor
	if info.Recurrent {
		stateShape := tensor.MustShape(
			uint64(r.spec.ShortConvCacheLength-1), uint64(r.spec.EmbeddingLength),
		)
		stateElements, _ := stateShape.Elements()
		stateValue := reference.Value{
			Shape: stateShape, Data: make([]float32, int(stateElements)),
		}
		reservedValue := reference.Value{
			Shape: tensor.MustShape(1), Data: []float32{0},
		}
		state = builder.Input(fmt.Sprintf("blk.%d.conv_state", layerIndex), dtype.F32, stateShape)
		reserved = builder.Input(
			fmt.Sprintf("blk.%d.reserved_state", layerIndex), dtype.F32, reservedValue.Shape,
		)
		hostFeeds[state], hostFeeds[reserved] = stateValue, reservedValue
	}
	spec := r.spec
	spec.NonCausalAttention = true
	result, err := model.BuildLFM2BlockCached(
		builder, input, spec, graphWeights, positions, info.Recurrent, state, reserved,
		uint32(layerIndex),
	)
	if err != nil {
		return reference.Value{}, err
	}
	var results map[*tensor.Tensor]reference.Value
	if r.hasPreloadedWeights() {
		results, err = r.cuda.ExecuteWithDeviceFeeds(
			ctx, []*tensor.Tensor{result.Output}, hostFeeds, deviceFeeds,
		)
	} else {
		results, err = r.cuda.Execute(ctx, []*tensor.Tensor{result.Output}, hostFeeds)
	}
	if err != nil {
		return reference.Value{}, err
	}
	return results[result.Output], nil
}

func (r *Runner) runDenseLayerNoCache(
	ctx context.Context,
	activation reference.Value,
	info model.LayerWeights,
	layerIndex int,
	positions []uint32,
) (reference.Value, error) {
	plan := r.layerPlan(layerIndex, info.Recurrent)
	builder := r.newGraphBuilder()
	input := builder.Input("input", dtype.F32, activation.Shape)
	hostFeeds := map[*tensor.Tensor]reference.Value{input: activation}
	hostLayer, err := model.LoadHostLayer(ctx, r.file, info)
	if err != nil {
		return reference.Value{}, err
	}
	graphWeights, layerFeeds, err := hostLayer.GraphInputs(
		builder, fmt.Sprintf("blk.%d.", layerIndex),
	)
	if err != nil {
		return reference.Value{}, err
	}
	for node, value := range layerFeeds {
		hostFeeds[node] = value
	}
	if _, err := bindLayerSideInputs(
		builder, r.spec, positions, plan, hostFeeds, &graphWeights, layerSideInputs{},
	); err != nil {
		return reference.Value{}, err
	}
	result, err := model.BuildDenseBlockCachedForLayer(
		builder,
		input,
		r.spec,
		graphWeights,
		positions,
		nil,
		nil,
		uint32(layerIndex),
	)
	if err != nil {
		return reference.Value{}, err
	}
	results, err := r.cuda.Execute(ctx, []*tensor.Tensor{result.Output}, hostFeeds)
	if err != nil {
		return reference.Value{}, err
	}
	return results[result.Output], nil
}

func (r *Runner) runQwen35LayerCached(
	ctx context.Context,
	activation reference.Value,
	info model.LayerWeights,
	layerIndex int,
	positions []uint32,
	past *LayerCache,
	multiPositions *MultiAxisPositions,
) (reference.Value, LayerCache, error) {
	builder := r.newGraphBuilder()
	input := builder.Input("input", dtype.F32, activation.Shape)
	hostFeeds := map[*tensor.Tensor]reference.Value{input: activation}
	deviceFeeds := make(map[*tensor.Tensor]driver.DevicePtr)
	var graphWeights model.LayerGraphWeights
	if r.hasPreloadedWeights() {
		var err error
		graphWeights, deviceFeeds, err = r.layerDeviceInputs(builder, info)
		if err != nil {
			return reference.Value{}, LayerCache{}, err
		}
	} else {
		hostLayer, err := model.LoadHostLayer(ctx, r.file, info)
		if err != nil {
			return reference.Value{}, LayerCache{}, err
		}
		var layerFeeds map[*tensor.Tensor]reference.Value
		graphWeights, layerFeeds, err = hostLayer.GraphInputs(builder, fmt.Sprintf("blk.%d.", layerIndex))
		if err != nil {
			return reference.Value{}, LayerCache{}, err
		}
		for node, value := range layerFeeds {
			hostFeeds[node] = value
		}
	}

	var pastKey, pastValue, convState, ssmState *tensor.Tensor
	if info.Recurrent {
		var convValue, ssmValue reference.Value
		if past == nil {
			convChannels := uint64(r.spec.SSMInnerSize) +
				2*uint64(r.spec.SSMStateSize)*uint64(r.spec.SSMGroupCount)
			convShape := tensor.MustShape(uint64(r.spec.SSMConvKernel-1), convChannels)
			ssmShape := tensor.MustShape(
				uint64(r.spec.SSMStateSize),
				uint64(r.spec.SSMStateSize),
				uint64(r.spec.SSMTimeStepRank),
				1,
			)
			convElements, _ := convShape.Elements()
			ssmElements, _ := ssmShape.Elements()
			convValue = reference.Value{Shape: convShape, Data: make([]float32, int(convElements))}
			ssmValue = reference.Value{Shape: ssmShape, Data: make([]float32, int(ssmElements))}
		} else {
			convValue = past.Key
			ssmValue = past.Value
		}
		convState = builder.Input(
			fmt.Sprintf("blk.%d.conv_state", layerIndex),
			dtype.F32,
			convValue.Shape,
		)
		ssmState = builder.Input(
			fmt.Sprintf("blk.%d.ssm_state", layerIndex),
			dtype.F32,
			ssmValue.Shape,
		)
		hostFeeds[convState] = convValue
		hostFeeds[ssmState] = ssmValue
	} else if past != nil {
		pastKey = builder.Input(
			fmt.Sprintf("blk.%d.cache_key", layerIndex),
			dtype.F32,
			past.Key.Shape,
		)
		pastValue = builder.Input(
			fmt.Sprintf("blk.%d.cache_value", layerIndex),
			dtype.F32,
			past.Value.Shape,
		)
		hostFeeds[pastKey] = past.Key
		hostFeeds[pastValue] = past.Value
	}
	var (
		result model.Qwen35BlockResult
		err    error
	)
	if multiPositions != nil {
		result, err = model.BuildQwen35BlockCachedWithMultiPositions(
			builder, input, r.spec, graphWeights, [4][]uint32(*multiPositions),
			info.Recurrent,
			pastKey, pastValue, convState, ssmState,
		)
	} else {
		result, err = model.BuildQwen35BlockCached(
			builder, input, r.spec, graphWeights, positions, info.Recurrent,
			pastKey, pastValue, convState, ssmState,
		)
	}
	if err != nil {
		return reference.Value{}, LayerCache{}, err
	}
	outputTensor := result.Output
	if r.hasPreloadedWeights() && layerIndex == len(r.weights.Layers)-1 {
		outputTensor, err = r.applyDeviceOutputNorm(builder, result.Output, deviceFeeds)
		if err != nil {
			return reference.Value{}, LayerCache{}, err
		}
	}
	outputs := []*tensor.Tensor{outputTensor}
	if info.Recurrent {
		outputs = append(outputs, result.ConvState, result.SSMState)
	} else {
		outputs = append(outputs, result.Key, result.Value)
	}
	var results map[*tensor.Tensor]reference.Value
	if r.hasPreloadedWeights() {
		results, err = r.cuda.ExecuteWithDeviceFeeds(ctx, outputs, hostFeeds, deviceFeeds)
	} else {
		results, err = r.cuda.Execute(ctx, outputs, hostFeeds)
	}
	if err != nil {
		return reference.Value{}, LayerCache{}, err
	}
	if info.Recurrent {
		return results[outputTensor], LayerCache{
			Key:   results[result.ConvState],
			Value: results[result.SSMState],
		}, nil
	}
	return results[outputTensor], LayerCache{
		Key:   results[result.Key],
		Value: results[result.Value],
	}, nil
}

func (r *Runner) runOutputNorm(ctx context.Context, activation reference.Value) (reference.Value, error) {
	if r.profile().OutputNorm == model.OutputNormAbsent {
		return activation, nil
	}
	runtime := r.newInferenceGraphRuntime(ctx)
	input := runtime.input("output_norm.input", activation)
	output, err := r.buildOutputNorm(runtime.builder, input, runtime.weight)
	if err != nil {
		return reference.Value{}, err
	}
	results, err := runtime.execute(output)
	if err != nil {
		return reference.Value{}, err
	}
	return results[output], nil
}

func (r *Runner) runUnweightedRMSNorm(
	ctx context.Context,
	activation reference.Value,
) (reference.Value, error) {
	runtime := r.newInferenceGraphRuntime(ctx)
	input := runtime.input("rms_norm.input", activation)
	output := runtime.builder.RMSNorm(input, r.spec.RMSNormEpsilon)
	results, err := runtime.execute(output)
	if err != nil {
		return reference.Value{}, err
	}
	return results[output], nil
}

// Greedy: tokenizes prompt and appends up to maxNewTokens argmax tokens
func (r *Runner) Greedy(
	ctx context.Context,
	prompt string,
	maxNewTokens int,
) ([]tokenizer.TokenID, string, error) {
	sampler, err := sampling.New(sampling.Config{})
	if err != nil {
		return nil, "", err
	}
	return r.Generate(ctx, prompt, GenerateOptions{
		MaxNewTokens: maxNewTokens,
		Sampler:      sampler,
	})
}

// Generate: performs correctness-first token generation and optionally reports
// each new token synchronously through OnToken
func (r *Runner) Generate(
	ctx context.Context,
	prompt string,
	options GenerateOptions,
) ([]tokenizer.TokenID, string, error) {
	if r == nil || r.vocab == nil {
		return nil, "", errors.New("inference: runner is nil")
	}
	if r.forwardPolicy() == model.ForwardT5Encoder {
		return nil, "", errors.New("inference: T5 encoder models do not generate tokens")
	}
	if r.forwardPolicy() == model.ForwardT5 {
		generated, _, _, err := r.GenerateT5(ctx, prompt, options)
		if err != nil {
			return nil, "", err
		}
		sourceIDs, err := r.promptTokenIDs(prompt, options)
		if err != nil {
			return nil, "", err
		}
		ids := append(slices.Clone(sourceIDs), generated...)
		text, err := r.vocab.Decode(ids, false)
		return ids, text, err
	}
	if r.spec.NonCausalAttention {
		return nil, "", errors.New("inference: non-causal models require diffusion generation")
	}
	if options.MaxNewTokens < 0 {
		return nil, "", errors.New("inference: max new tokens is negative")
	}
	if options.MinCacheReuse < 0 {
		return nil, "", errors.New("inference: minimum cache reuse is negative")
	}
	if options.KeepTokens < -1 {
		return nil, "", errors.New("inference: keep token count must be at least -1")
	}
	if options.DiscardTokens < 0 {
		return nil, "", errors.New("inference: discard token count is negative")
	}
	if options.PostSamplingProbabilities < 0 {
		return nil, "", errors.New(
			"inference: post-sampling probability count is negative",
		)
	}
	if err := validateStopSequences(options.StopSequences); err != nil {
		return nil, "", err
	}
	if options.Sampler == nil {
		var err error
		options.Sampler, err = sampling.New(sampling.Config{})
		if err != nil {
			return nil, "", err
		}
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.closed {
		return nil, "", errors.New("inference: runner is closed")
	}
	if options.LoRAConfigured {
		next, err := validatedLoRAScales(len(r.loraAdapters), options.LoRA)
		if err != nil {
			return nil, "", err
		}
		previous := make([]float32, len(r.loraAdapters))
		for index := range r.loraAdapters {
			previous[index] = r.loraAdapters[index].scale
			r.loraAdapters[index].scale = next[index]
		}
		defer func() {
			for index := range r.loraAdapters {
				r.loraAdapters[index].scale = previous[index]
			}
		}()
	}
	ids, err := r.promptTokenIDs(prompt, options)
	if err != nil {
		return nil, "", err
	}
	aloraID, aloraStart, err := r.activeALoRA(ids)
	if err != nil {
		return nil, "", err
	}
	if options.ProjectedInputs != nil && aloraID >= 0 {
		return nil, "", errors.New("inference: projected inputs cannot use invocation-activated LoRA")
	}
	var aloraScale float32
	if aloraID >= 0 {
		aloraScale = r.loraAdapters[aloraID].scale
		defer func() { r.loraAdapters[aloraID].scale = aloraScale }()
		options.CachePrompt = false
		if aloraStart < 0 {
			r.loraAdapters[aloraID].scale = 0
		}
	}
	keepTokens := effectiveKeepTokens(
		options.KeepTokens,
		len(ids),
		r.spec.ContextLength,
	)
	outputTable := r.weights.TokenEmbedding
	if r.weights.Output != nil {
		outputTable = *r.weights.Output
	}
	var hidden reference.Value
	var cache *KVCache
	var deviceCache *deviceKVCache
	var selectedPromptCache *cachedPrompt
	var projectionSignature [32]byte
	if options.ProjectedInputs != nil {
		projectionSignature = projectedInputsSignature(*options.ProjectedInputs)
	}
	useDeviceCache := options.ProjectedInputs == nil && aloraID < 0 &&
		r.hasPreloadedWeights() && supportsPersistentDeviceCache(r.spec)
	defer func() {
		if deviceCache != nil &&
			!r.ownsDevicePromptCache(deviceCache) {
			_ = deviceCache.Release(context.Background())
		}
	}()
	if options.MaxNewTokens > 0 {
		promptStarted := time.Now()
		cached := 0
		if useDeviceCache {
			var retainedPrefix *deviceKVCache
			if options.CachePrompt {
				selectedPromptCache, cached = r.selectPromptCache(
					ids,
					options.MinCacheReuse,
					true,
				)
			}
			if selectedPromptCache != nil {
				if cached > 0 &&
					cached < len(selectedPromptCache.Tokens) &&
					r.promptCacheCapacity > 1 {
					// device suffix view would mutate selected entry
					// Preserve independent multi-entry caches and evaluate
					// divergent prompt from scratch
					selectedPromptCache = nil
					cached = 0
				}
				if cached > 0 &&
					cached < len(selectedPromptCache.Tokens) &&
					r.hasRecurrentCache() {
					cached = 0
				}
				if cached > 0 {
					retainedPrefix = selectedPromptCache.Device
					if cached < len(selectedPromptCache.Tokens) {
						base := cached
						// retained logits describe old final token
						// For exact shorter prompt, reevaluate its final
						// token from preceding cache entry
						if cached == len(ids) {
							base--
						}
						if base == 0 {
							cached = 0
							retainedPrefix = nil
						} else if trimErr := trimDeviceCacheSuffix(
							retainedPrefix,
							uint32(base),
						); trimErr != nil {
							return nil, "", trimErr
						} else {
							selectedPromptCache.Tokens = append(
								[]tokenizer.TokenID(nil),
								ids[:base]...,
							)
							cached = base
						}
					}
				}
			}
			if cached == len(ids) &&
				retainedPrefix != nil &&
				len(retainedPrefix.Logits) == 0 {
				if cached <= 1 {
					cached = 0
					retainedPrefix = nil
				} else if trimErr := trimDeviceCacheSuffix(
					retainedPrefix,
					uint32(cached-1),
				); trimErr != nil {
					return nil, "", trimErr
				} else {
					cached--
					selectedPromptCache.Tokens = append(
						[]tokenizer.TokenID(nil),
						ids[:cached]...,
					)
				}
			}
			if cached == len(ids) && len(retainedPrefix.Logits) > 0 {
				deviceCache = retainedPrefix
			} else {
				deviceCache = retainedPrefix
				var nextDevice *deviceKVCache
				hidden, nextDevice, err = r.forwardDeviceCachedLocked(
					ctx,
					ids[cached:],
					retainedPrefix,
				)
				if err == nil {
					deviceCache = nextDevice
				}
			}
		} else if aloraID >= 0 && aloraStart >= 0 {
			r.loraAdapters[aloraID].scale = 0
			if aloraStart > 0 {
				_, cache, err = r.forwardCachedLocked(ctx, ids[:aloraStart], nil)
			}
			r.loraAdapters[aloraID].scale = aloraScale
			if err == nil {
				hidden, cache, err = r.forwardCachedLocked(ctx, ids[aloraStart:], cache)
			}
		} else if options.ProjectedInputs != nil {
			if options.CachePrompt {
				selectedPromptCache, cached = r.selectProjectedPromptCache(
					ids, projectionSignature, options.MinCacheReuse,
				)
			}
			if selectedPromptCache != nil {
				hidden = selectedPromptCache.Hidden
				cache = selectedPromptCache.Cache
			} else {
				inputs := *options.ProjectedInputs
				hidden, cache, err = r.forwardCachedWithProjectedInputsLocked(ctx, ids, nil, inputs)
			}
		} else if options.CachePrompt {
			selectedPromptCache, cached = r.selectPromptCache(
				ids,
				options.MinCacheReuse,
				false,
			)
			if selectedPromptCache == nil {
				hidden, cache, err = r.forwardCachedLocked(ctx, ids, nil)
			} else {
				if cached > 0 &&
					cached < len(selectedPromptCache.Tokens) &&
					r.hasRecurrentCache() {
					cached = 0
				}
				if cached > 0 {
					hidden = selectedPromptCache.Hidden
					cache = selectedPromptCache.Cache
					if cached < len(selectedPromptCache.Tokens) {
						hidden, cache, err = r.trimHostPromptCache(
							hidden,
							cache,
							uint32(cached),
						)
					}
					if err == nil && cached < len(ids) {
						hidden, cache, err = r.forwardCachedLocked(ctx, ids[cached:], cache)
					}
				} else {
					hidden, cache, err = r.forwardCachedLocked(ctx, ids, nil)
				}
			}
		} else if !useDeviceCache {
			hidden, cache, err = r.forwardCachedLocked(ctx, ids, nil)
		}
		if err != nil {
			return nil, "", err
		}
		if options.CachePrompt {
			nextPromptCache := &cachedPrompt{
				Tokens:              slices.Clone(ids),
				Hidden:              hidden,
				Cache:               cache,
				Device:              deviceCache,
				LoRASignature:       r.currentLoRASignature(),
				ProjectionSignature: projectionSignature,
				HasProjection:       options.ProjectedInputs != nil,
			}
			if storeErr := r.storePromptCache(
				ctx,
				nextPromptCache,
			); storeErr != nil {
				return nil, "", storeErr
			}
			selectedPromptCache = nextPromptCache
		}
		if options.OnPromptEvaluated != nil {
			options.OnPromptEvaluated(PromptEvaluation{
				Tokens:   len(ids),
				Cached:   cached,
				Duration: time.Since(promptStarted),
			})
		}
	}
	var generatedText strings.Builder
	for generatedIndex := range options.MaxNewTokens {
		if generatedIndex > 0 {
			if useDeviceCache {
				if options.ContextShift {
					var shiftedDeviceCache *deviceKVCache
					shiftedDeviceCache, err = r.compactDeviceCacheForAppend(
						ctx,
						deviceCache,
						1,
						keepTokens,
						options.DiscardTokens,
						r.ownsDevicePromptCache(deviceCache),
					)
					if err != nil {
						return nil, "", err
					}
					if shiftedDeviceCache != deviceCache {
						oldDeviceCache := deviceCache
						deviceCache = shiftedDeviceCache
						if !r.ownsDevicePromptCache(oldDeviceCache) {
							if releaseErr := oldDeviceCache.Release(ctx); releaseErr != nil {
								return nil, "", releaseErr
							}
						}
					}
				}
				var nextDeviceCache *deviceKVCache
				hidden, nextDeviceCache, err = r.forwardDeviceCachedLocked(
					ctx,
					[]tokenizer.TokenID{ids[len(ids)-1]},
					deviceCache,
				)
				if err == nil {
					oldDeviceCache := deviceCache
					deviceCache = nextDeviceCache
					if !r.ownsDevicePromptCache(oldDeviceCache) {
						err = oldDeviceCache.Release(ctx)
					}
				} else if nextDeviceCache != nil {
					_ = nextDeviceCache.Release(context.Background())
				}
			} else {
				cache, err = r.cacheForAppendKeeping(
					cache,
					1,
					options.ContextShift,
					keepTokens,
					options.DiscardTokens,
				)
				if err != nil {
					return nil, "", err
				}
				hidden, cache, err = r.forwardCachedLocked(
					ctx,
					[]tokenizer.TokenID{ids[len(ids)-1]},
					cache,
				)
			}
			if err != nil {
				return nil, "", err
			}
		}
		var logits []float32
		var logitsErr error
		if useDeviceCache {
			logits = deviceCache.Logits
		} else {
			width := int(hidden.Shape.Dims[0])
			last := hidden.Data[len(hidden.Data)-width:]
			logits, logitsErr = r.logits(ctx, outputTable, last)
		}
		if logitsErr != nil {
			return nil, "", logitsErr
		}
		history := make([]int, len(ids))
		for index, id := range ids {
			history[index] = int(id)
		}
		next := 0
		var topProbabilities []sampling.TokenProbability
		selectedProbability := 0.0
		var sampleErr error
		if options.PostSamplingProbabilities > 0 {
			var probabilityResult sampling.SampleProbabilityResult
			probabilityResult, sampleErr =
				options.Sampler.SampleWithHistoryProbabilities(
					logits,
					history,
					options.PostSamplingProbabilities,
				)
			next = probabilityResult.Token
			selectedProbability = probabilityResult.SelectedProbability
			topProbabilities = probabilityResult.Top
		} else {
			next, sampleErr = options.Sampler.SampleWithHistory(logits, history)
		}
		if sampleErr != nil {
			return nil, "", sampleErr
		}
		nextID := tokenizer.TokenID(next)
		ids = append(ids, nextID)
		piece := ""
		event := TokenEvent{
			ID:                  nextID,
			Index:               generatedIndex,
			Logits:              logits,
			SelectedProbability: selectedProbability,
			TopProbabilities:    topProbabilities,
		}
		if options.OnToken != nil ||
			options.ShouldStop != nil ||
			len(options.StopSequences) > 0 {
			var decodeErr error
			piece, decodeErr = r.vocab.DecodePiece(nextID, false)
			if decodeErr != nil {
				return nil, "", decodeErr
			}
			event.Piece = piece
			if options.OnToken != nil {
				if callbackErr := options.OnToken(event); callbackErr != nil {
					return nil, "", callbackErr
				}
			}
		}
		generatedText.WriteString(piece)
		if options.ShouldStop != nil && options.ShouldStop(event) {
			break
		}
		if r.vocab.IsEOG(nextID) ||
			matchesStopSequence(generatedText.String(), options.StopSequences) {
			break
		}
	}
	text, err := r.vocab.Decode(ids, false)
	if err != nil {
		return nil, "", err
	}
	return ids, text, nil
}

func reusablePromptPrefix(cached, requested []tokenizer.TokenID, minimum int) int {
	if len(cached) == 0 || len(requested) == 0 {
		return 0
	}
	common := min(len(cached), len(requested))
	for index := 0; index < common; index++ {
		if cached[index] != requested[index] {
			common = index
			break
		}
	}
	if common < minimum {
		return 0
	}
	return common
}

func (r *Runner) hasRecurrentCache() bool {
	if r == nil {
		return false
	}
	for _, layer := range r.weights.Layers {
		if layer.Recurrent {
			return true
		}
	}
	return false
}

func (r *Runner) promptTokenIDs(prompt string, options GenerateOptions) ([]tokenizer.TokenID, error) {
	if options.PromptTokenIDs == nil {
		ids, err := r.vocab.Encode(prompt, tokenizer.EncodeOptions{
			AddSpecial:   true,
			ParseSpecial: options.ParseSpecial,
		})
		if err != nil {
			return nil, err
		}
		if len(ids) == 0 {
			return nil, errors.New("inference: prompt produced no tokens")
		}
		return ids, nil
	}
	if len(options.PromptTokenIDs) == 0 {
		return nil, errors.New("inference: exact prompt token list is empty")
	}
	ids := slices.Clone(options.PromptTokenIDs)
	for index, id := range ids {
		if _, ok := r.vocab.Token(id); !ok {
			return nil, fmt.Errorf(
				"inference: prompt token %d has out-of-range ID %d",
				index,
				id,
			)
		}
	}
	return ids, nil
}

func validateStopSequences(stops []string) error {
	if len(stops) > 256 {
		return errors.New("inference: stop sequence count exceeds 256")
	}
	for _, stop := range stops {
		if stop == "" {
			return errors.New("inference: stop sequence is empty")
		}
	}
	return nil
}

func matchesStopSequence(text string, stops []string) bool {
	for _, stop := range stops {
		if strings.Contains(text, stop) {
			return true
		}
	}
	return false
}

func (r *Runner) loadEmbeddings(ctx context.Context, rows []uint32) (reference.Value, error) {
	return r.loadRows(ctx, r.weights.TokenEmbedding, rows)
}

func (r *Runner) prepareGemma4PerLayerInputs(
	ctx context.Context,
	activation reference.Value,
	rows []uint32,
) ([]reference.Value, error) {
	if !r.profile().Has(model.ArchitecturePerLayerEmbeddings) ||
		r.spec.EmbeddingPerLayer == 0 {
		return nil, nil
	}
	if r.weights.PerLayerTokenEmbedding == nil || r.weights.PerLayerModelProjection == nil ||
		r.weights.PerLayerProjectionNorm == nil {
		return nil, errors.New("inference: Gemma per-layer weights are incomplete")
	}
	selected, err := r.loadRows(ctx, *r.weights.PerLayerTokenEmbedding, rows)
	if err != nil {
		return nil, fmt.Errorf("inference: load Gemma per-layer embeddings: %w", err)
	}
	runtime := r.newInferenceGraphRuntime(ctx)
	input := runtime.input("gemma4.per_layer.input", activation)
	selectedInput := runtime.input("gemma4.per_layer.selected", selected)
	projection, err := runtime.weight(*r.weights.PerLayerModelProjection)
	if err != nil {
		return nil, err
	}
	norm, err := runtime.weight(*r.weights.PerLayerProjectionNorm)
	if err != nil {
		return nil, err
	}
	outputs, err := model.BuildGemma4PerLayerInputs(
		runtime.builder, input, selectedInput, projection, norm, r.spec,
	)
	if err != nil {
		return nil, err
	}
	results, err := runtime.execute(outputs...)
	if err != nil {
		return nil, err
	}
	values := make([]reference.Value, len(outputs))
	for index, output := range outputs {
		values[index] = results[output]
	}
	return values, nil
}

func (r *Runner) loadRows(
	ctx context.Context,
	info gguf.TensorInfo,
	rows []uint32,
) (reference.Value, error) {
	if !r.hasPreloadedWeights() {
		value, err := model.LoadHostRows(ctx, r.file, info, rows)
		if err != nil {
			return reference.Value{}, err
		}
		return r.applyLoRAEmbeddingRows(info.Name, rows, value)
	}
	runtime := r.newInferenceGraphRuntime(ctx)
	table, err := runtime.weight(info)
	if err != nil {
		return reference.Value{}, err
	}
	output := runtime.builder.GetRows(table, rows)
	if err := runtime.builder.Err(); err != nil {
		return reference.Value{}, err
	}
	results, err := runtime.execute(output)
	if err != nil {
		return reference.Value{}, err
	}
	return results[output], nil
}

func (r *Runner) addPositionEmbeddings(
	ctx context.Context,
	activation reference.Value,
	positions []uint32,
) (reference.Value, error) {
	if r.weights.PositionEmbedding == nil {
		return activation, nil
	}
	for _, position := range positions {
		if position >= r.spec.ContextLength {
			return reference.Value{}, fmt.Errorf(
				"inference: learned position %d exceeds context length %d",
				position,
				r.spec.ContextLength,
			)
		}
	}
	positionRows, err := r.loadRows(ctx, *r.weights.PositionEmbedding, positions)
	if err != nil {
		return reference.Value{}, fmt.Errorf("inference: load position embeddings: %w", err)
	}
	if positionRows.Shape != activation.Shape || len(positionRows.Data) != len(activation.Data) {
		return reference.Value{}, errors.New("inference: position embedding shape differs from token embeddings")
	}
	for index := range activation.Data {
		activation.Data[index] += positionRows.Data[index]
	}
	return activation, nil
}

func (r *Runner) addTokenTypeEmbedding(
	ctx context.Context,
	activation reference.Value,
) (reference.Value, error) {
	if r.weights.TokenTypeEmbedding == nil {
		return activation, nil
	}
	typeRow, err := r.loadRows(ctx, *r.weights.TokenTypeEmbedding, []uint32{0})
	if err != nil {
		return reference.Value{}, fmt.Errorf("inference: load token-type embedding: %w", err)
	}
	width := int(activation.Shape.Dims[0])
	if typeRow.Shape.Rank != 2 || typeRow.Shape.Dims[0] != uint64(width) ||
		typeRow.Shape.Dims[1] != 1 || len(typeRow.Data) != width {
		return reference.Value{}, errors.New("inference: token-type embedding shape is incompatible")
	}
	for token := 0; token < int(activation.Shape.Dims[1]); token++ {
		start := token * width
		for index, value := range typeRow.Data {
			activation.Data[start+index] += value
		}
	}
	return activation, nil
}

func (r *Runner) applyTokenEmbeddingNorm(
	ctx context.Context,
	activation reference.Value,
) (reference.Value, error) {
	if r.weights.TokenEmbeddingNorm == nil {
		return activation, nil
	}
	runtime := r.newInferenceGraphRuntime(ctx)
	input := runtime.input("token_embd_norm.input", activation)
	weightInput, err := runtime.weight(*r.weights.TokenEmbeddingNorm)
	if err != nil {
		return reference.Value{}, err
	}
	var biasInput *tensor.Tensor
	if r.weights.TokenEmbeddingNormBias != nil {
		biasInput, err = runtime.weight(*r.weights.TokenEmbeddingNormBias)
		if err != nil {
			return reference.Value{}, err
		}
	}
	output := model.ApplyNormalization(runtime.builder, input, weightInput, biasInput, r.spec)
	if err := runtime.builder.Err(); err != nil {
		return reference.Value{}, err
	}
	results, err := runtime.execute(output)
	if err != nil {
		return reference.Value{}, err
	}
	return results[output], nil
}

func (r *Runner) logits(
	ctx context.Context,
	outputInfo gguf.TensorInfo,
	hidden []float32,
) ([]float32, error) {
	if !r.hasPreloadedWeights() {
		logits, err := model.DotRows(
			ctx,
			r.file,
			outputInfo,
			hidden,
			1024,
		)
		if err != nil {
			return nil, err
		}
		if err := r.applyLoRALogits(outputInfo.Name, hidden, logits); err != nil {
			return nil, err
		}
		if err := addOutputBias(logits, r.outputBias); err != nil {
			return nil, err
		}
		scaleLogits(logits, r.spec.OutputLogitMultiplier())
		return r.finalizeLogits(logits), nil
	}
	runtime := r.newInferenceGraphRuntime(ctx)
	table, err := runtime.weight(outputInfo)
	if err != nil {
		return nil, err
	}
	inputShape := tensor.MustShape(uint64(len(hidden)), 1)
	inputValue, err := reference.NewValue(inputShape, hidden)
	if err != nil {
		return nil, err
	}
	input := runtime.input("logits.input", inputValue)
	output := runtime.builder.MulMat(table, input)
	if r.weights.OutputBias != nil {
		bias, biasErr := runtime.weight(*r.weights.OutputBias)
		if biasErr != nil {
			return nil, biasErr
		}
		output = runtime.builder.Add(output, bias)
	}
	if scale := r.spec.OutputLogitMultiplier(); scale != 1 {
		output = runtime.builder.Scale(output, scale)
	}
	if err := runtime.builder.Err(); err != nil {
		return nil, err
	}
	results, err := runtime.execute(output)
	if err != nil {
		return nil, err
	}
	return r.finalizeLogits(results[output].Data), nil
}

func scaleLogits(logits []float32, scale float32) {
	if scale == 1 {
		return
	}
	for index := range logits {
		logits[index] *= scale
	}
}

func applyLogitSoftcap(logits []float32, cap float32) []float32 {
	if cap <= 0 {
		return logits
	}
	for index, value := range logits {
		logits[index] = cap * float32(math.Tanh(float64(value/cap)))
	}
	return logits
}

func (r *Runner) finalizeLogits(logits []float32) []float32 {
	logits = applyLogitSoftcap(logits, r.spec.FinalLogitSoftcap)
	if r.spec.Architecture != "chameleon" || r.spec.VocabularySize == 0 {
		return logits
	}
	vocabulary := int(r.spec.VocabularySize)
	if len(logits)%vocabulary != 0 {
		return logits
	}
	end := 8196
	if end > vocabulary {
		end = vocabulary
	}
	for base := 0; base < len(logits); base += vocabulary {
		for token := 4; token < end; token++ {
			logits[base+token] = -math.MaxFloat32
		}
	}
	return logits
}

func addOutputBias(logits, bias []float32) error {
	if len(bias) == 0 {
		return nil
	}
	if len(logits)%len(bias) != 0 {
		return fmt.Errorf(
			"inference: %d logits are not divisible by output bias length %d",
			len(logits),
			len(bias),
		)
	}
	for index := range logits {
		logits[index] += bias[index%len(bias)]
	}
	return nil
}

func supportsMultiAxisPositions(spec model.Spec) bool {
	sections := false
	for _, count := range spec.RopeSections {
		sections = sections || count > 0
	}
	if !sections {
		return false
	}
	profile := spec.Profile()
	if !profile.Has(model.ArchitectureMultiAxisPositions) {
		return false
	}
	switch spec.Architecture {
	case "glm4", "glm4moe", "hunyuan-dense", "hunyuan_vl":
		return spec.RopeSections[0] > 0 && spec.RopeSections[1] > 0
	default:
		return true
	}
}

func supportsDeepstackInputs(spec model.Spec) bool {
	return spec.DeepstackLayerCount > 0 && spec.Profile().Deepstack != model.DeepstackNone
}

func validateDeepstackInputs(spec model.Spec, tokens int, inputs []reference.Value) error {
	if len(inputs) == 0 {
		return nil
	}
	if !supportsDeepstackInputs(spec) {
		return errors.New("inference: model does not support deepstack embeddings")
	}
	if len(inputs) != int(spec.DeepstackLayerCount) {
		return fmt.Errorf(
			"inference: received %d deepstack streams, need %d",
			len(inputs), spec.DeepstackLayerCount,
		)
	}
	want := tensor.MustShape(uint64(spec.EmbeddingLength), uint64(tokens))
	for streamIndex, stream := range inputs {
		if !stream.Shape.Equal(want) || len(stream.Data) != int(spec.EmbeddingLength)*tokens {
			return fmt.Errorf("inference: deepstack stream %d has invalid shape", streamIndex)
		}
		for _, value := range stream.Data {
			if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) {
				return fmt.Errorf("inference: deepstack stream %d contains non-finite value", streamIndex)
			}
		}
	}
	return nil
}

func projectedAttentionBlockIDs(
	spec model.Spec,
	tokens int,
	hasCache bool,
	blocks []AttentionBlock,
) ([]float32, error) {
	if len(blocks) == 0 {
		return nil, nil
	}
	if spec.Profile().AttentionBlocks != model.AttentionBlocksUncached {
		return nil, errors.New("inference: model does not support bidirectional attention blocks")
	}
	if hasCache {
		return nil, errors.New("inference: bidirectional attention blocks require an uncached prompt prefill")
	}
	ids := make([]float32, tokens)
	for index := range ids {
		ids[index] = -1
	}
	var previousEnd uint32
	for index, block := range blocks {
		if block.Start >= block.End || uint64(block.End) > uint64(tokens) {
			return nil, fmt.Errorf("inference: attention block %d range [%d,%d) is invalid for %d tokens", index, block.Start, block.End, tokens)
		}
		if index > 0 && block.Start < previousEnd {
			return nil, fmt.Errorf("inference: attention block %d overlaps or precedes the prior block", index)
		}
		for token := block.Start; token < block.End; token++ {
			ids[token] = float32(index)
		}
		previousEnd = block.End
	}
	return ids, nil
}

func deepstackInputForLayer(
	source model.DeepstackSource,
	base reference.Value,
	inputs []reference.Value,
) *reference.Value {
	if len(inputs) == 0 || source == model.DeepstackSourceNone {
		return nil
	}
	if source == model.DeepstackSourceBase {
		return &base
	}
	index := int(source)
	if index < 0 || index >= len(inputs) {
		return nil
	}
	return &inputs[index]
}

func addDeepstackEmbedding(activation, deepstack reference.Value) (reference.Value, error) {
	if !activation.Shape.Equal(deepstack.Shape) || len(activation.Data) != len(deepstack.Data) {
		return reference.Value{}, errors.New("activation and deepstack shapes differ")
	}
	output := reference.Value{
		Shape: activation.Shape,
		Data:  slices.Clone(activation.Data),
	}
	for index, value := range deepstack.Data {
		output.Data[index] += value
	}
	return output, nil
}

func f32RequiredModelTensors(weights model.Weights) map[string]struct{} {
	result := make(map[string]struct{})
	layers := slices.Clone(weights.Layers)
	for _, draft := range weights.DraftCatalogs() {
		layers = append(layers, draft.Layer)
	}
	for _, layer := range layers {
		for _, info := range []*gguf.TensorInfo{
			layer.SSMConv1D,
			layer.SSMQueryConv,
			layer.SSMKeyConv,
			layer.SSMValueConv,
			layer.ShortConvKernel,
		} {
			if info != nil {
				result[info.Name] = struct{}{}
			}
		}
	}
	return result
}

func selectedModelTensors(file *gguf.File, weights model.Weights) []gguf.TensorInfo {
	names := map[string]struct{}{weights.TokenEmbedding.Name: {}}
	for _, mtp := range weights.DraftCatalogs() {
		for _, info := range []gguf.TensorInfo{mtp.EHProjection, mtp.EmbeddingNorm, mtp.HiddenNorm} {
			names[info.Name] = struct{}{}
		}
		for _, info := range []*gguf.TensorInfo{mtp.TokenEmbedding, mtp.LayerOutputNorm, mtp.OutputNorm, mtp.Output} {
			if info != nil {
				names[info.Name] = struct{}{}
			}
		}
	}
	if weights.PositionEmbedding != nil {
		names[weights.PositionEmbedding.Name] = struct{}{}
	}
	if weights.TokenTypeEmbedding != nil {
		names[weights.TokenTypeEmbedding.Name] = struct{}{}
	}
	if weights.TokenEmbeddingNorm != nil {
		names[weights.TokenEmbeddingNorm.Name] = struct{}{}
	}
	if weights.TokenEmbeddingNormBias != nil {
		names[weights.TokenEmbeddingNormBias.Name] = struct{}{}
	}
	if weights.OutputNorm.Name != "" {
		names[weights.OutputNorm.Name] = struct{}{}
	}
	if weights.EncoderOutputNorm != nil {
		names[weights.EncoderOutputNorm.Name] = struct{}{}
	}
	if weights.OutputNormBias != nil {
		names[weights.OutputNormBias.Name] = struct{}{}
	}
	if weights.Output != nil {
		names[weights.Output.Name] = struct{}{}
	}
	if weights.OutputBias != nil {
		names[weights.OutputBias.Name] = struct{}{}
	}
	if weights.Dense2Output != nil {
		names[weights.Dense2Output.Name] = struct{}{}
	}
	if weights.Dense3Output != nil {
		names[weights.Dense3Output.Name] = struct{}{}
	}
	if weights.ClassifierOutput != nil {
		names[weights.ClassifierOutput.Name] = struct{}{}
	}
	for _, pointer := range []*gguf.TensorInfo{
		weights.PerLayerTokenEmbedding,
		weights.PerLayerModelProjection,
		weights.PerLayerProjectionNorm,
		weights.AltUpProjection,
		weights.AltUpUnembedding,
		weights.FeatureProjection,
		weights.FeatureProjectionPost,
		weights.DraftToTarget,
	} {
		if pointer != nil {
			names[pointer.Name] = struct{}{}
		}
	}
	if wav := weights.WavTokenizer; wav != nil {
		infos := []gguf.TensorInfo{
			wav.InputConv, wav.InputConvBias, wav.TokenNorm, wav.TokenNormBias,
			wav.OutputNorm, wav.OutputNormBias, wav.Output, wav.OutputBias,
		}
		for _, layer := range wav.PosNet {
			infos = append(infos,
				layer.Norm1, layer.Norm1Bias, layer.Conv1, layer.Conv1Bias,
				layer.Norm2, layer.Norm2Bias, layer.Conv2, layer.Conv2Bias,
				layer.AttentionNorm, layer.AttentionNormBias,
				layer.AttentionQ, layer.AttentionQBias, layer.AttentionK, layer.AttentionKBias,
				layer.AttentionV, layer.AttentionVBias, layer.AttentionOutput, layer.AttentionOutBias,
			)
		}
		for _, layer := range wav.ConvNext {
			infos = append(infos,
				layer.Depthwise, layer.DepthwiseBias, layer.Norm, layer.NormBias,
				layer.Pointwise1, layer.Pointwise1Bias, layer.Pointwise2, layer.Pointwise2Bias, layer.Gamma,
			)
		}
		for _, info := range infos {
			if info.Name != "" {
				names[info.Name] = struct{}{}
			}
		}
	}
	allLayers := weights.LayerCatalog()
	for _, layer := range allLayers {
		infos := []gguf.TensorInfo{
			layer.AttentionNorm,
			layer.FeedForwardNorm,
			layer.FeedForwardGate,
			layer.FeedForwardUp,
			layer.FeedForwardDown,
		}
		if layer.Recurrent {
			if layer.SSMQueryConv != nil {
				infos = append(infos, layer.AttentionQ, layer.AttentionK, layer.AttentionV, layer.AttentionOutput)
			}
			for _, pointer := range []*gguf.TensorInfo{
				layer.SSMInput,
				layer.AttentionQKV,
				layer.AttentionGate,
				layer.SSMConv1D,
				layer.SSMConv1DBias,
				layer.SSMX,
				layer.SSMTimeStepWeight,
				layer.SSMTimeStep,
				layer.SSMTimeStepNorm,
				layer.SSMA,
				layer.SSMD,
				layer.SSMBNorm,
				layer.SSMCNorm,
				layer.SSMBeta,
				layer.SSMAlpha,
				layer.SSMBetaAlpha,
				layer.SSMNorm,
				layer.SSMOutput,
				layer.SSMQueryConv,
				layer.SSMKeyConv,
				layer.SSMValueConv,
				layer.SSMForgetA,
				layer.SSMForgetB,
				layer.SSMOutputGateA,
				layer.SSMOutputGateB,
				layer.ShortConvKernel,
				layer.ShortConvInput,
				layer.ShortConvOutput,
				layer.TimeMixW1,
				layer.TimeMixW2,
				layer.TimeMixW0,
				layer.TimeMixA0,
				layer.TimeMixA1,
				layer.TimeMixA2,
				layer.TimeMixV0,
				layer.TimeMixV1,
				layer.TimeMixV2,
				layer.TimeMixG1,
				layer.TimeMixG2,
				layer.TimeMixKK,
				layer.TimeMixKA,
				layer.TimeMixRK,
				layer.TimeMixLerpX,
				layer.TimeMixLerpFused,
				layer.TimeMixLerpW,
				layer.TimeMixLerpK,
				layer.TimeMixLerpV,
				layer.TimeMixLerpR,
				layer.TimeMixLerpG,
				layer.TimeMixFirst,
				layer.TimeMixDecay,
				layer.TimeMixDecayW1,
				layer.TimeMixDecayW2,
				layer.TimeMixKey,
				layer.TimeMixValue,
				layer.TimeMixReceptance,
				layer.TimeMixGate,
				layer.TimeMixLN,
				layer.TimeMixLNBias,
				layer.TimeMixOutput,
				layer.ChannelMixLerpK,
				layer.ChannelMixLerpR,
				layer.ChannelMixKey,
				layer.ChannelMixValue,
				layer.ChannelMixReceptance,
			} {
				if pointer != nil {
					infos = append(infos, *pointer)
				}
			}
		} else {
			if layer.AttentionQKV != nil {
				infos = append(infos, *layer.AttentionQKV)
			} else {
				infos = append(infos, layer.AttentionQ, layer.AttentionK, layer.AttentionV)
			}
			infos = append(infos, layer.AttentionOutput)
		}
		for _, info := range infos {
			if info.Name != "" {
				names[info.Name] = struct{}{}
			}
		}
		if layer.AttentionQNorm != nil {
			names[layer.AttentionQNorm.Name] = struct{}{}
		}
		for _, pointer := range []*gguf.TensorInfo{
			layer.AttentionQB,
			layer.AttentionNormBias,
			layer.AttentionNorm2,
			layer.AttentionNorm2Bias,
			layer.AttentionQScale,
			layer.AttentionKScale,
			layer.AttentionVScale,
			layer.AttentionOutputScale,
			layer.AttentionSubNorm,
			layer.AttentionOutputGate,
			layer.AttentionSinks,
			layer.AttentionQKVBias,
			layer.AttentionQNormBias,
			layer.AttentionKNormBias,
			layer.AttentionQBias,
			layer.AttentionKBias,
			layer.AttentionVBias,
			layer.AttentionOutputBias,
			layer.FeedForwardGateBias,
			layer.FeedForwardUpBias,
			layer.FeedForwardDownBias,
			layer.FeedForwardNormBias,
			layer.FeedForwardExpertNorm,
			layer.FeedForwardGateScale,
			layer.FeedForwardUpScale,
			layer.FeedForwardDownScale,
			layer.FeedForwardActivationScale,
			layer.FeedForwardSubNorm,
			layer.FeedForwardRouter,
			layer.FeedForwardRouterBias,
			layer.FeedForwardGateUpExperts,
			layer.FeedForwardGateExperts,
			layer.FeedForwardUpExperts,
			layer.FeedForwardDownExperts,
			layer.FeedForwardDownExpertsScale,
			layer.FeedForwardGateChunkExperts,
			layer.FeedForwardUpChunkExperts,
			layer.FeedForwardDownChunkExperts,
			layer.FeedForwardExpertBias,
			layer.FeedForwardLatentDown,
			layer.FeedForwardLatentUp,
			layer.FeedForwardSharedGate,
			layer.FeedForwardSharedUp,
			layer.FeedForwardSharedDown,
			layer.FeedForwardSharedRouter,
			layer.LayerOutputScale,
			layer.FeedForwardPreNorm2,
			layer.FeedForwardPostNorm1,
			layer.FeedForwardPostNorm2,
			layer.FeedForwardRouterScale,
			layer.PerLayerInputGate,
			layer.PerLayerProjection,
			layer.PerLayerPostNorm,
			layer.AltUpCorrectCoefficient,
			layer.AltUpCorrectScale,
			layer.AltUpPredictCoefficient,
			layer.AltUpRouter,
			layer.AltUpRouterNorm,
			layer.LaurelLeft,
			layer.LaurelRight,
			layer.LaurelPostNorm,
			layer.AttentionKVAMQA,
			layer.AttentionKVANorm,
			layer.AttentionKVB,
			layer.AttentionKB,
			layer.AttentionVB,
			layer.CrossAttentionNorm,
			layer.CrossAttentionQ,
			layer.CrossAttentionK,
			layer.CrossAttentionV,
			layer.CrossAttentionOutput,
			layer.IndexerKNorm,
			layer.IndexerKNormBias,
			layer.IndexerProjection,
			layer.IndexerAttentionK,
			layer.IndexerAttentionQB,
			layer.AttentionOutputA,
			layer.AttentionCompressorKV,
			layer.AttentionCompressorGate,
			layer.AttentionCompressorAPE,
			layer.AttentionCompressorNorm,
			layer.IndexerCompressorKV,
			layer.IndexerCompressorGate,
			layer.IndexerCompressorAPE,
			layer.IndexerCompressorNorm,
			layer.HyperAttentionFN,
			layer.HyperAttentionBase,
			layer.HyperAttentionScale,
			layer.HyperFeedForwardFN,
			layer.HyperFeedForwardBase,
			layer.HyperFeedForwardScale,
			layer.HyperHeadFN,
			layer.HyperHeadBase,
			layer.HyperHeadScale,
			layer.FeedForwardHashExperts,
			layer.VisualAttentionQKV,
			layer.VisualAttentionOutput,
			layer.VisualFeedForwardGate,
			layer.VisualFeedForwardUp,
			layer.VisualFeedForwardDown,
			layer.SSMInput,
			layer.SSMConv1D,
			layer.SSMConv1DBias,
			layer.SSMTimeStep,
			layer.SSMA,
			layer.SSMD,
			layer.SSMNorm,
			layer.SSMOutput,
		} {
			if pointer != nil {
				names[pointer.Name] = struct{}{}
			}
		}
		if layer.AttentionKNorm != nil {
			names[layer.AttentionKNorm.Name] = struct{}{}
		}
		if layer.AttentionPostNorm != nil {
			names[layer.AttentionPostNorm.Name] = struct{}{}
		}
		if layer.AttentionPostNormBias != nil {
			names[layer.AttentionPostNormBias.Name] = struct{}{}
		}
		if layer.AttentionRelativeBias != nil {
			names[layer.AttentionRelativeBias.Name] = struct{}{}
		}
		if layer.RopeFactors != nil {
			names[layer.RopeFactors.Name] = struct{}{}
		}
		if layer.FeedForwardPostNorm != nil {
			names[layer.FeedForwardPostNorm.Name] = struct{}{}
		}
		if layer.FeedForwardPostNormBias != nil {
			names[layer.FeedForwardPostNormBias.Name] = struct{}{}
		}
	}
	result := make([]gguf.TensorInfo, 0, len(names))
	for _, info := range file.Tensors {
		if _, ok := names[info.Name]; ok {
			result = append(result, info)
		}
	}
	return result
}
