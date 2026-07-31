package inference

import (
	"context"
	"errors"
	"fmt"
	"math"
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
	// Logits is a transient view of the pre-sampling logits for this token.
	// Callers must copy it if they retain it after the callback returns.
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
	// ShouldStop is evaluated after OnToken and after the sampled piece has
	// been appended to generated text. Returning true ends generation
	// successfully while retaining that token.
	ShouldStop func(TokenEvent) bool
	// PostSamplingProbabilities requests the top N normalized candidates after
	// the configured sampler chain. Zero disables this instrumentation.
	PostSamplingProbabilities int
	ParseSpecial              bool
	StopSequences             []string
	// PromptTokenIDs, when non-nil, replaces text tokenization with this exact
	// caller-owned prompt sequence. No BOS/EOS token is inserted implicitly.
	PromptTokenIDs []tokenizer.TokenID
	// ContextShift permits generation to discard the oldest attention KV
	// entries when the active cache reaches the model context length. Absolute
	// token positions and hybrid recurrent state are preserved.
	ContextShift bool
	// KeepTokens preserves this many initial prompt tokens when ContextShift
	// compacts a full host cache. Minus one preserves as much of the initial
	// prompt as the context permits.
	KeepTokens int
	// DiscardTokens controls how many entries after KeepTokens are removed per
	// context shift. Zero uses half of the discardable cache, matching the
	// native server convention.
	DiscardTokens int
	// CachePrompt retains the evaluated prompt state for a later request whose
	// token sequence has this prompt as a prefix.
	CachePrompt bool
	// MinCacheReuse requires at least this many matching prefix tokens before
	// a retained prompt is reused.
	MinCacheReuse     int
	OnPromptEvaluated func(PromptEvaluation)
}

type LayerCache struct {
	Key   reference.Value
	Value reference.Value
}

type KVCache struct {
	Layers []LayerCache
	// Tokens is the number of active attention tokens retained in Layers.
	Tokens uint32
	// Position is the absolute position assigned to the next appended token.
	// It can exceed Tokens after an attention-cache prefix has been removed.
	Position uint32
}

// Runner is a correctness-first Llama/Qwen inference runtime with an editable
// host attention/recurrent cache and optional persistent F32 or
// native-quantized weights.
type Runner struct {
	file          *gguf.File
	path          string
	spec          model.Spec
	weights       model.Weights
	vocab         *tokenizer.Vocab
	cuda          *executor.Executor
	worker        *device.Worker
	deviceWeights *model.DeviceF32Weights
	rawWeights    *model.DeviceWeights
	outputBias    []float32

	mu                  sync.Mutex
	closed              bool
	modelSignature      [32]byte
	modelSignatureErr   error
	modelSignatureOnce  sync.Once
	promptCaches        []*cachedPrompt
	promptCacheCapacity int
}

type cachedPrompt struct {
	Tokens []tokenizer.TokenID
	Hidden reference.Value
	Cache  *KVCache
	Device *deviceKVCache
}

type OpenOptions struct {
	DeviceOrdinal           int
	PreloadDeviceWeights    bool
	PreloadQuantizedWeights bool
	// PromptCacheEntries bounds independently reusable prompt states.
	// Zero selects the default capacity of one.
	PromptCacheEntries int
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
	vocab, err := tokenizer.Load(file)
	if err != nil {
		return fail(err)
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
		outputBias = append([]float32(nil), value.Data...)
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
			f32Tensors = make([]gguf.TensorInfo, 0, len(selected))
			var quantized []gguf.TensorInfo
			for _, info := range selected {
				if info.Type == dtype.Q4_0 ||
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
					info.Type == dtype.NVFP4 {
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
	return &Runner{
		file:                file,
		path:                path,
		spec:                spec,
		weights:             weights,
		vocab:               vocab,
		cuda:                cuda,
		worker:              worker,
		deviceWeights:       deviceWeights,
		rawWeights:          rawWeights,
		outputBias:          outputBias,
		promptCacheCapacity: promptCacheCapacity,
	}, nil
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
	var errs []error
	for _, promptCache := range r.promptCaches {
		if promptCache.Device != nil {
			errs = append(errs, promptCache.Device.Release(context.Background()))
			promptCache.Device = nil
		}
	}
	r.promptCaches = nil
	if r.rawWeights != nil {
		errs = append(errs, r.rawWeights.Close())
	}
	if r.deviceWeights != nil {
		errs = append(errs, r.deviceWeights.Close())
	}
	if r.cuda != nil {
		errs = append(errs, r.cuda.Close())
	}
	if r.worker != nil {
		errs = append(errs, r.worker.Close())
	}
	if r.file != nil {
		errs = append(errs, r.file.Close())
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

func (r *Runner) applyDeviceOutputNorm(
	builder *tensor.Builder,
	input *tensor.Tensor,
	deviceFeeds map[*tensor.Tensor]driver.DevicePtr,
) (*tensor.Tensor, error) {
	if r.spec.UsesUnweightedLayerNorm() {
		return builder.LayerNorm(input, r.spec.LayerNormEpsilon), builder.Err()
	}
	weight, pointer, err := r.deviceInput(builder, r.weights.OutputNorm)
	if err != nil {
		return nil, err
	}
	deviceFeeds[weight] = pointer
	var bias *tensor.Tensor
	if r.weights.OutputNormBias != nil {
		bias, pointer, err = r.deviceInput(builder, *r.weights.OutputNormBias)
		if err != nil {
			return nil, err
		}
		deviceFeeds[bias] = pointer
	}
	return model.ApplyNormalization(builder, input, weight, bias, r.spec), builder.Err()
}

func (r *Runner) layerDeviceInputs(
	builder *tensor.Builder,
	info model.LayerWeights,
) (model.LayerGraphWeights, map[*tensor.Tensor]driver.DevicePtr, error) {
	feeds := make(map[*tensor.Tensor]driver.DevicePtr, 11)
	input := func(tensorInfo gguf.TensorInfo) (*tensor.Tensor, error) {
		node, pointer, err := r.deviceInput(builder, tensorInfo)
		if err != nil {
			return nil, err
		}
		feeds[node] = pointer
		return node, nil
	}
	var result model.LayerGraphWeights
	var err error
	if info.AttentionNorm.Name != "" {
		if result.AttentionNorm, err = input(info.AttentionNorm); err != nil {
			return result, nil, err
		}
	}
	if info.AttentionNormBias != nil {
		if result.AttentionNormBias, err = input(*info.AttentionNormBias); err != nil {
			return result, nil, err
		}
	}
	if info.AttentionNorm2 != nil {
		if result.AttentionNorm2, err = input(*info.AttentionNorm2); err != nil {
			return result, nil, err
		}
	}
	if info.AttentionNorm2Bias != nil {
		if result.AttentionNorm2Bias, err = input(*info.AttentionNorm2Bias); err != nil {
			return result, nil, err
		}
	}
	if info.Recurrent {
		recurrent := []struct {
			info        *gguf.TensorInfo
			destination **tensor.Tensor
		}{
			{info.AttentionQKV, &result.AttentionQKV},
			{info.AttentionGate, &result.AttentionGate},
			{info.SSMConv1D, &result.SSMConv1D},
			{info.SSMTimeStep, &result.SSMTimeStep},
			{info.SSMA, &result.SSMA},
			{info.SSMBeta, &result.SSMBeta},
			{info.SSMAlpha, &result.SSMAlpha},
			{info.SSMNorm, &result.SSMNorm},
			{info.SSMOutput, &result.SSMOutput},
		}
		for _, item := range recurrent {
			if item.info == nil {
				return result, nil, errors.New("inference: recurrent layer catalog is incomplete")
			}
			if *item.destination, err = input(*item.info); err != nil {
				return result, nil, err
			}
		}
	} else {
		if info.AttentionQKV != nil {
			if result.AttentionQKV, err = input(*info.AttentionQKV); err != nil {
				return result, nil, err
			}
		} else {
			if result.AttentionQ, err = input(info.AttentionQ); err != nil {
				return result, nil, err
			}
			if result.AttentionK, err = input(info.AttentionK); err != nil {
				return result, nil, err
			}
			if result.AttentionV, err = input(info.AttentionV); err != nil {
				return result, nil, err
			}
		}
		if result.AttentionOutput, err = input(info.AttentionOutput); err != nil {
			return result, nil, err
		}
	}
	if info.AttentionQNorm != nil {
		if result.AttentionQNorm, err = input(*info.AttentionQNorm); err != nil {
			return result, nil, err
		}
	}
	for _, item := range []struct {
		info        *gguf.TensorInfo
		destination **tensor.Tensor
	}{
		{info.AttentionQScale, &result.AttentionQScale},
		{info.AttentionKScale, &result.AttentionKScale},
		{info.AttentionVScale, &result.AttentionVScale},
		{info.AttentionOutputScale, &result.AttentionOutputScale},
		{info.AttentionSubNorm, &result.AttentionSubNorm},
		{info.AttentionQKVBias, &result.AttentionQKVBias},
		{info.AttentionQBias, &result.AttentionQBias},
		{info.AttentionKBias, &result.AttentionKBias},
		{info.AttentionVBias, &result.AttentionVBias},
		{info.AttentionOutputBias, &result.AttentionOutputBias},
		{info.FeedForwardNormBias, &result.FeedForwardNormBias},
		{info.FeedForwardGateBias, &result.FeedForwardGateBias},
		{info.FeedForwardUpBias, &result.FeedForwardUpBias},
		{info.FeedForwardDownBias, &result.FeedForwardDownBias},
		{info.FeedForwardGateScale, &result.FeedForwardGateScale},
		{info.FeedForwardUpScale, &result.FeedForwardUpScale},
		{info.FeedForwardDownScale, &result.FeedForwardDownScale},
		{info.FeedForwardSubNorm, &result.FeedForwardSubNorm},
	} {
		if item.info != nil {
			if *item.destination, err = input(*item.info); err != nil {
				return result, nil, err
			}
		}
	}
	if info.AttentionKNorm != nil {
		if result.AttentionKNorm, err = input(*info.AttentionKNorm); err != nil {
			return result, nil, err
		}
	}
	if info.AttentionPostNorm != nil {
		if result.AttentionPostNorm, err = input(*info.AttentionPostNorm); err != nil {
			return result, nil, err
		}
	}
	if info.AttentionRelativeBias != nil {
		if result.AttentionRelativeBias, err = input(*info.AttentionRelativeBias); err != nil {
			return result, nil, err
		}
	}
	if info.RopeFactors != nil {
		if result.RopeFactors, err = input(*info.RopeFactors); err != nil {
			return result, nil, err
		}
	}
	if info.FeedForwardPostNorm != nil {
		if result.FeedForwardPostNorm, err = input(*info.FeedForwardPostNorm); err != nil {
			return result, nil, err
		}
	}
	if info.FeedForwardNorm.Name != "" {
		if result.FeedForwardNorm, err = input(info.FeedForwardNorm); err != nil {
			return result, nil, err
		}
	}
	if info.FeedForwardGate.Name != "" {
		if result.FeedForwardGate, err = input(info.FeedForwardGate); err != nil {
			return result, nil, err
		}
	}
	if result.FeedForwardUp, err = input(info.FeedForwardUp); err != nil {
		return result, nil, err
	}
	if result.FeedForwardDown, err = input(info.FeedForwardDown); err != nil {
		return result, nil, err
	}
	return result, feeds, builder.Err()
}

func (r *Runner) Spec() model.Spec {
	if r == nil {
		return model.Spec{}
	}
	return r.spec
}

func (r *Runner) Vocab() *tokenizer.Vocab {
	if r == nil {
		return nil
	}
	return r.vocab
}

// Forward evaluates all layers and returns final normalized hidden states in
// ggml shape [embedding, tokens].
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

func (r *Runner) forwardLocked(
	ctx context.Context,
	tokenIDs []tokenizer.TokenID,
) (reference.Value, error) {
	if r.spec.Architecture == "t5encoder" {
		return r.forwardT5EncoderLocked(ctx, tokenIDs)
	}
	hidden, _, err := r.forwardCachedLocked(ctx, tokenIDs, nil)
	return hidden, err
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
	for layerIndex, layerInfo := range r.weights.Layers {
		activation, err = r.runT5EncoderLayer(ctx, activation, layerInfo, layerIndex)
		if err != nil {
			return reference.Value{}, fmt.Errorf("inference encoder layer %d: %w", layerIndex, err)
		}
	}
	return r.runOutputNorm(ctx, activation)
}

// ForwardCached evaluates a prompt chunk and returns a host KV/recurrent cache
// suitable for a later incremental call or ShiftCache edit.
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
	return r.forwardCachedLocked(ctx, tokenIDs, cache)
}

func (r *Runner) forwardCachedLocked(
	ctx context.Context,
	tokenIDs []tokenizer.TokenID,
	cache *KVCache,
) (reference.Value, *KVCache, error) {
	if r.spec.Architecture == "t5encoder" {
		return reference.Value{}, nil, errors.New("inference: T5 encoder does not support KV caching")
	}
	if len(tokenIDs) == 0 {
		return reference.Value{}, nil, errors.New("inference: token sequence is empty")
	}
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
	activation, err := r.loadEmbeddings(ctx, rows)
	if err != nil {
		return reference.Value{}, nil, err
	}
	if scale := r.spec.InputEmbeddingScale(); scale != 1 {
		for index := range activation.Data {
			activation.Data[index] *= scale
		}
	}
	nextCache := &KVCache{
		Layers:   make([]LayerCache, len(r.weights.Layers)),
		Tokens:   pastTokens + uint32(len(tokenIDs)),
		Position: nextPosition + uint32(len(tokenIDs)),
	}
	if r.hasPreloadedWeights() && r.spec.Architecture != "qwen35" {
		return r.forwardDenseLayersPreloaded(ctx, activation, positions, cache, nextCache)
	}
	for layerIndex, layerInfo := range r.weights.Layers {
		var past *LayerCache
		if cache != nil {
			past = &cache.Layers[layerIndex]
		}
		var layerCache LayerCache
		activation, layerCache, err = r.runLayerCached(
			ctx,
			activation,
			layerInfo,
			layerIndex,
			positions,
			past,
		)
		if err != nil {
			return reference.Value{}, nil, fmt.Errorf("inference layer %d: %w", layerIndex, err)
		}
		nextCache.Layers[layerIndex] = layerCache
	}
	if !r.hasPreloadedWeights() {
		activation, err = r.runOutputNorm(ctx, activation)
		if err != nil {
			return reference.Value{}, nil, err
		}
	}
	return activation, nextCache, nil
}

func (r *Runner) forwardDenseLayersPreloaded(
	ctx context.Context,
	activation reference.Value,
	positions []uint32,
	cache *KVCache,
	nextCache *KVCache,
) (reference.Value, *KVCache, error) {
	builder := tensor.NewBuilder()
	input := builder.Input("model.input", dtype.F32, activation.Shape)
	current := input
	hostFeeds := map[*tensor.Tensor]reference.Value{input: activation}
	deviceFeeds := make(map[*tensor.Tensor]driver.DevicePtr)
	keys := make([]*tensor.Tensor, len(r.weights.Layers))
	values := make([]*tensor.Tensor, len(r.weights.Layers))
	for layerIndex, info := range r.weights.Layers {
		graphWeights, layerFeeds, err := r.layerDeviceInputs(builder, info)
		if err != nil {
			return reference.Value{}, nil, err
		}
		for node, pointer := range layerFeeds {
			deviceFeeds[node] = pointer
		}
		var pastKey, pastValue *tensor.Tensor
		if cache != nil {
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
		result, err := model.BuildDenseBlockCachedForLayer(
			builder,
			current,
			r.spec,
			graphWeights,
			positions,
			pastKey,
			pastValue,
			uint32(layerIndex),
		)
		if err != nil {
			return reference.Value{}, nil, err
		}
		current = result.Output
		keys[layerIndex] = result.Key
		values[layerIndex] = result.Value
	}
	current, err := r.applyDeviceOutputNorm(builder, current, deviceFeeds)
	if err != nil {
		return reference.Value{}, nil, err
	}
	if err := builder.Err(); err != nil {
		return reference.Value{}, nil, err
	}
	outputs := make([]*tensor.Tensor, 1, 1+2*len(keys))
	outputs[0] = current
	for layerIndex := range keys {
		outputs = append(outputs, keys[layerIndex], values[layerIndex])
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
	return results[current], nextCache, nil
}

func (r *Runner) runT5EncoderLayer(
	ctx context.Context,
	activation reference.Value,
	info model.LayerWeights,
	layerIndex int,
) (reference.Value, error) {
	builder := tensor.NewBuilder()
	input := builder.Input("input", dtype.F32, activation.Shape)
	hostFeeds := map[*tensor.Tensor]reference.Value{input: activation}
	deviceFeeds := make(map[*tensor.Tensor]driver.DevicePtr)
	var graphWeights model.LayerGraphWeights
	if r.hasPreloadedWeights() {
		var err error
		graphWeights, deviceFeeds, err = r.layerDeviceInputs(builder, info)
		if err != nil {
			return reference.Value{}, err
		}
	} else {
		hostLayer, err := model.LoadHostLayer(ctx, r.file, info)
		if err != nil {
			return reference.Value{}, err
		}
		var layerFeeds map[*tensor.Tensor]reference.Value
		graphWeights, layerFeeds, err = hostLayer.GraphInputs(
			builder,
			fmt.Sprintf("enc.blk.%d.", layerIndex),
		)
		if err != nil {
			return reference.Value{}, err
		}
		for node, value := range layerFeeds {
			hostFeeds[node] = value
		}
	}
	output, err := model.BuildT5EncoderBlock(builder, input, r.spec, graphWeights)
	if err != nil {
		return reference.Value{}, err
	}
	var results map[*tensor.Tensor]reference.Value
	if r.hasPreloadedWeights() {
		results, err = r.cuda.ExecuteWithDeviceFeeds(
			ctx,
			[]*tensor.Tensor{output},
			hostFeeds,
			deviceFeeds,
		)
	} else {
		results, err = r.cuda.Execute(ctx, []*tensor.Tensor{output}, hostFeeds)
	}
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
	past *LayerCache,
) (reference.Value, LayerCache, error) {
	if r.spec.Architecture == "qwen35" {
		return r.runQwen35LayerCached(
			ctx,
			activation,
			info,
			layerIndex,
			positions,
			past,
		)
	}
	builder := tensor.NewBuilder()
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
	var pastKey, pastValue *tensor.Tensor
	if past != nil {
		pastKey = builder.Input(fmt.Sprintf("blk.%d.cache_key", layerIndex), dtype.F32, past.Key.Shape)
		pastValue = builder.Input(fmt.Sprintf("blk.%d.cache_value", layerIndex), dtype.F32, past.Value.Shape)
		hostFeeds[pastKey] = past.Key
		hostFeeds[pastValue] = past.Value
	}
	result, err := model.BuildDenseBlockCachedForLayer(
		builder,
		input,
		r.spec,
		graphWeights,
		positions,
		pastKey,
		pastValue,
		uint32(layerIndex),
	)
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
	var results map[*tensor.Tensor]reference.Value
	if r.hasPreloadedWeights() {
		results, err = r.cuda.ExecuteWithDeviceFeeds(ctx, outputs, hostFeeds, deviceFeeds)
	} else {
		results, err = r.cuda.Execute(ctx, outputs, hostFeeds)
	}
	if err != nil {
		return reference.Value{}, LayerCache{}, err
	}
	return results[outputTensor], LayerCache{
		Key:   results[result.Key],
		Value: results[result.Value],
	}, nil
}

func (r *Runner) runQwen35LayerCached(
	ctx context.Context,
	activation reference.Value,
	info model.LayerWeights,
	layerIndex int,
	positions []uint32,
	past *LayerCache,
) (reference.Value, LayerCache, error) {
	builder := tensor.NewBuilder()
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
	result, err := model.BuildQwen35BlockCached(
		builder,
		input,
		r.spec,
		graphWeights,
		positions,
		info.Recurrent,
		pastKey,
		pastValue,
		convState,
		ssmState,
	)
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
	if r.spec.UsesUnweightedLayerNorm() {
		builder := tensor.NewBuilder()
		input := builder.Input("output_norm.input", dtype.F32, activation.Shape)
		output := builder.LayerNorm(input, r.spec.LayerNormEpsilon)
		feeds := map[*tensor.Tensor]reference.Value{input: activation}
		var (
			results map[*tensor.Tensor]reference.Value
			err     error
		)
		if r.hasPreloadedWeights() {
			results, err = r.cuda.ExecuteWithDeviceFeeds(
				ctx, []*tensor.Tensor{output}, feeds, nil,
			)
		} else {
			results, err = r.cuda.Execute(ctx, []*tensor.Tensor{output}, feeds)
		}
		if err != nil {
			return reference.Value{}, err
		}
		return results[output], nil
	}
	if r.hasPreloadedWeights() {
		builder := tensor.NewBuilder()
		input := builder.Input("output_norm.input", dtype.F32, activation.Shape)
		weightInput, pointer, err := r.deviceInput(builder, r.weights.OutputNorm)
		if err != nil {
			return reference.Value{}, err
		}
		deviceFeeds := map[*tensor.Tensor]driver.DevicePtr{weightInput: pointer}
		var biasInput *tensor.Tensor
		if r.weights.OutputNormBias != nil {
			biasInput, pointer, err = r.deviceInput(builder, *r.weights.OutputNormBias)
			if err != nil {
				return reference.Value{}, err
			}
			deviceFeeds[biasInput] = pointer
		}
		output := model.ApplyNormalization(builder, input, weightInput, biasInput, r.spec)
		if err := builder.Err(); err != nil {
			return reference.Value{}, err
		}
		results, err := r.cuda.ExecuteWithDeviceFeeds(
			ctx,
			[]*tensor.Tensor{output},
			map[*tensor.Tensor]reference.Value{input: activation},
			deviceFeeds,
		)
		if err != nil {
			return reference.Value{}, err
		}
		return results[output], nil
	}
	weight, err := model.LoadHostTensor(ctx, r.file, r.weights.OutputNorm)
	if err != nil {
		return reference.Value{}, err
	}
	builder := tensor.NewBuilder()
	input := builder.Input("output_norm.input", dtype.F32, activation.Shape)
	weightInput := builder.Input("output_norm.weight", dtype.F32, weight.Shape)
	feeds := map[*tensor.Tensor]reference.Value{
		input:       activation,
		weightInput: weight,
	}
	var biasInput *tensor.Tensor
	if r.weights.OutputNormBias != nil {
		bias, biasErr := model.LoadHostTensor(ctx, r.file, *r.weights.OutputNormBias)
		if biasErr != nil {
			return reference.Value{}, biasErr
		}
		biasInput = builder.Input("output_norm.bias", dtype.F32, bias.Shape)
		feeds[biasInput] = bias
	}
	output := model.ApplyNormalization(builder, input, weightInput, biasInput, r.spec)
	if err := builder.Err(); err != nil {
		return reference.Value{}, err
	}
	results, err := r.cuda.Execute(ctx, []*tensor.Tensor{output}, feeds)
	if err != nil {
		return reference.Value{}, err
	}
	return results[output], nil
}

// Greedy tokenizes prompt and appends up to maxNewTokens argmax tokens.
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

// Generate performs correctness-first token generation and optionally reports
// each new token synchronously through OnToken.
func (r *Runner) Generate(
	ctx context.Context,
	prompt string,
	options GenerateOptions,
) ([]tokenizer.TokenID, string, error) {
	if r == nil || r.vocab == nil {
		return nil, "", errors.New("inference: runner is nil")
	}
	if r.spec.Architecture == "t5encoder" {
		return nil, "", errors.New("inference: T5 encoder models do not generate tokens")
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
	ids, err := r.promptTokenIDs(prompt, options)
	if err != nil {
		return nil, "", err
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
	useDeviceCache := r.hasPreloadedWeights()
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
					// A device suffix view would mutate the selected entry.
					// Preserve independent multi-entry caches and evaluate
					// this divergent prompt from scratch.
					selectedPromptCache = nil
					cached = 0
				}
				if cached > 0 &&
					cached < len(selectedPromptCache.Tokens) &&
					r.spec.Architecture == "qwen35" {
					cached = 0
				}
				if cached > 0 {
					retainedPrefix = selectedPromptCache.Device
					if cached < len(selectedPromptCache.Tokens) {
						base := cached
						// The retained logits describe the old final token.
						// For an exact shorter prompt, reevaluate its final
						// token from the preceding cache entry.
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
					r.spec.Architecture == "qwen35" {
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
				Tokens: append([]tokenizer.TokenID(nil), ids...),
				Hidden: hidden,
				Cache:  cache,
				Device: deviceCache,
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
	ids := append([]tokenizer.TokenID(nil), options.PromptTokenIDs...)
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
	if !r.hasPreloadedWeights() {
		return model.LoadHostRows(ctx, r.file, r.weights.TokenEmbedding, rows)
	}
	builder := tensor.NewBuilder()
	table, pointer, err := r.deviceInput(builder, r.weights.TokenEmbedding)
	if err != nil {
		return reference.Value{}, err
	}
	output := builder.GetRows(table, rows)
	if err := builder.Err(); err != nil {
		return reference.Value{}, err
	}
	results, err := r.cuda.ExecuteWithDeviceFeeds(
		ctx,
		[]*tensor.Tensor{output},
		nil,
		map[*tensor.Tensor]driver.DevicePtr{table: pointer},
	)
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
		if err := addOutputBias(logits, r.outputBias); err != nil {
			return nil, err
		}
		scaleLogits(logits, r.spec.OutputLogitMultiplier())
		return applyLogitSoftcap(logits, r.spec.FinalLogitSoftcap), nil
	}
	builder := tensor.NewBuilder()
	table, pointer, err := r.deviceInput(builder, outputInfo)
	if err != nil {
		return nil, err
	}
	inputShape := tensor.MustShape(uint64(len(hidden)), 1)
	input := builder.Input("logits.input", dtype.F32, inputShape)
	output := builder.MulMat(table, input)
	deviceFeeds := map[*tensor.Tensor]driver.DevicePtr{table: pointer}
	if r.weights.OutputBias != nil {
		bias, biasPointer, biasErr := r.deviceInput(builder, *r.weights.OutputBias)
		if biasErr != nil {
			return nil, biasErr
		}
		deviceFeeds[bias] = biasPointer
		output = builder.Add(output, bias)
	}
	if scale := r.spec.OutputLogitMultiplier(); scale != 1 {
		output = builder.Scale(output, scale)
	}
	if err := builder.Err(); err != nil {
		return nil, err
	}
	inputValue, err := reference.NewValue(inputShape, hidden)
	if err != nil {
		return nil, err
	}
	results, err := r.cuda.ExecuteWithDeviceFeeds(
		ctx,
		[]*tensor.Tensor{output},
		map[*tensor.Tensor]reference.Value{input: inputValue},
		deviceFeeds,
	)
	if err != nil {
		return nil, err
	}
	return applyLogitSoftcap(
		results[output].Data,
		r.spec.FinalLogitSoftcap,
	), nil
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

func selectedModelTensors(file *gguf.File, weights model.Weights) []gguf.TensorInfo {
	names := map[string]struct{}{weights.TokenEmbedding.Name: {}}
	if weights.OutputNorm.Name != "" {
		names[weights.OutputNorm.Name] = struct{}{}
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
	for _, layer := range weights.Layers {
		infos := []gguf.TensorInfo{
			layer.AttentionNorm,
			layer.FeedForwardNorm,
			layer.FeedForwardGate,
			layer.FeedForwardUp,
			layer.FeedForwardDown,
		}
		if layer.Recurrent {
			for _, pointer := range []*gguf.TensorInfo{
				layer.AttentionQKV,
				layer.AttentionGate,
				layer.SSMConv1D,
				layer.SSMTimeStep,
				layer.SSMA,
				layer.SSMBeta,
				layer.SSMAlpha,
				layer.SSMNorm,
				layer.SSMOutput,
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
			layer.AttentionNormBias,
			layer.AttentionNorm2,
			layer.AttentionNorm2Bias,
			layer.AttentionQScale,
			layer.AttentionKScale,
			layer.AttentionVScale,
			layer.AttentionOutputScale,
			layer.AttentionSubNorm,
			layer.AttentionQKVBias,
			layer.AttentionQBias,
			layer.AttentionKBias,
			layer.AttentionVBias,
			layer.AttentionOutputBias,
			layer.FeedForwardGateBias,
			layer.FeedForwardUpBias,
			layer.FeedForwardDownBias,
			layer.FeedForwardNormBias,
			layer.FeedForwardGateScale,
			layer.FeedForwardUpScale,
			layer.FeedForwardDownScale,
			layer.FeedForwardSubNorm,
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
		if layer.AttentionRelativeBias != nil {
			names[layer.AttentionRelativeBias.Name] = struct{}{}
		}
		if layer.RopeFactors != nil {
			names[layer.RopeFactors.Name] = struct{}{}
		}
		if layer.FeedForwardPostNorm != nil {
			names[layer.FeedForwardPostNorm.Name] = struct{}{}
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
