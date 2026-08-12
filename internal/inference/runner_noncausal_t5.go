package inference

import (
	"context"
	"errors"
	"fmt"
	"math"

	"overgo/internal/model"
	"overgo/internal/tensor"
	"overgo/internal/tensor/reference"
	"overgo/internal/tokenizer"
)

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
	rows, err := r.tokenRows(tokenIDs)
	if err != nil {
		return reference.Value{}, err
	}
	positions := tokenPositions(0, len(tokenIDs))
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
	rows, err := r.tokenRows(tokenIDs)
	if err != nil {
		return reference.Value{}, err
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
	outputInfo := r.outputTensor()
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
	rows, err := r.tokenRows(tokenIDs)
	if err != nil {
		return reference.Value{}, err
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
	rows, err := r.tokenRows(decoderIDs)
	if err != nil {
		return reference.Value{}, nil, err
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

func (r *Runner) runT5EncoderLayer(
	ctx context.Context,
	activation reference.Value,
	info model.LayerWeights,
	layerIndex int,
) (reference.Value, error) {
	program, err := r.program.Model.EncoderProgram(r.spec, layerIndex)
	if err != nil {
		return reference.Value{}, err
	}
	runtime := r.newInferenceGraphRuntime(ctx)
	input := runtime.input("input", activation)
	graphWeights, err := runtime.layer(info, fmt.Sprintf("enc.blk.%d.", layerIndex))
	if err != nil {
		return reference.Value{}, err
	}
	result, err := program.Build(model.CachedBlockContext{
		Builder: runtime.builder, Input: input,
	}, graphWeights)
	if err != nil {
		return reference.Value{}, err
	}
	results, err := runtime.execute(result.Output)
	if err != nil {
		return reference.Value{}, err
	}
	return results[result.Output], nil
}

func (r *Runner) runT5DecoderLayer(
	ctx context.Context,
	activation, encoder reference.Value,
	info model.LayerWeights,
	layerIndex int,
	past *LayerCache,
) (reference.Value, LayerCache, error) {
	program, err := r.program.Model.DecoderProgram(r.spec, layerIndex)
	if err != nil {
		return reference.Value{}, LayerCache{}, err
	}
	runtime := r.newInferenceGraphRuntime(ctx)
	input := runtime.input("input", activation)
	var encoderInput *tensor.Tensor
	var pastSelfKey, pastSelfValue, pastCrossKey, pastCrossValue *tensor.Tensor
	if past == nil {
		encoderInput = runtime.input("encoder", encoder)
	} else {
		pastSelfKey = runtime.input("past_self_key", past.Key)
		pastSelfValue = runtime.input("past_self_value", past.Value)
		crossKey, hasKey := past.States[model.CacheStateCrossKey]
		crossValue, hasValue := past.States[model.CacheStateCrossValue]
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
	result, err := program.Build(model.CachedBlockContext{
		Builder: runtime.builder, Input: input, Encoder: encoderInput,
		PastKey: pastSelfKey, PastValue: pastSelfValue,
		CrossKey: pastCrossKey, CrossValue: pastCrossValue,
	}, graphWeights)
	if err != nil {
		return reference.Value{}, LayerCache{}, err
	}
	outputs := []*tensor.Tensor{result.Output, result.Key, result.Value}
	outputs = result.States.AppendValues(outputs)
	results, err := runtime.execute(outputs...)
	if err != nil {
		return reference.Value{}, LayerCache{}, err
	}
	layerCache := LayerCache{
		Key: results[result.Key], Value: results[result.Value],
		States: model.MapCacheStateValues(
			result.States, func(value *tensor.Tensor) reference.Value { return results[value] },
		),
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
