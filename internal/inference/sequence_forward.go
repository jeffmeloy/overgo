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

// DecodeAudioTokens: semantic tokens to audio-feature frames.
func (r *Runner) DecodeAudioTokens(
	ctx context.Context,
	tokenIDs []tokenizer.TokenID,
) (reference.Value, error) {
	if r == nil {
		return reference.Value{}, errRunnerNil
	}
	if err := r.lockOpen(); err != nil {
		return reference.Value{}, err
	}
	defer r.mu.Unlock()
	if r.forwardProgram().Operation != model.ForwardOperationAudioTokens {
		return reference.Value{}, errors.New("inference: model has no audio-token decoder")
	}
	return r.forwardAudioTokensLocked(ctx, tokenIDs)
}

func (r *Runner) forwardNonCausalLocked(
	ctx context.Context,
	tokenIDs []tokenizer.TokenID,
) (reference.Value, error) {
	if r.forwardProgram().Operation == model.ForwardOperationAudioTokens {
		return r.forwardAudioTokensLocked(ctx, tokenIDs)
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
	rows, err := r.vocab.TensorIndices(tokenIDs)
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
	if r.forwardProgram().NonCausalRecurrent() {
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

func (r *Runner) forwardAudioTokensLocked(
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
	rows, err := r.vocab.TensorIndices(tokenIDs)
	if err != nil {
		return reference.Value{}, err
	}
	embeddings, err := r.loadEmbeddings(ctx, rows)
	if err != nil {
		return reference.Value{}, err
	}
	runtime := r.newInferenceGraphRuntime(ctx)
	input := runtime.input("audio_tokens.embeddings", embeddings)
	graphWeights, err := model.BindSequenceOutputGraphWeights(r.weights.AudioDecoder, runtime.weight)
	if err != nil {
		return reference.Value{}, err
	}
	output, err := r.program.Model.SequenceOutput().Build(runtime.builder, input, graphWeights)
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

func (r *Runner) forwardEncoderLocked(
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
	rows, err := r.vocab.TensorIndices(tokenIDs)
	if err != nil {
		return reference.Value{}, err
	}
	activation, err := r.loadEmbeddings(ctx, rows)
	if err != nil {
		return reference.Value{}, err
	}
	layers := r.weights.Layers
	if r.forwardProgram().Session == model.ForwardSessionEncoderDecoder {
		layers = r.weights.EncoderLayers
	}
	for layerIndex, layerInfo := range layers {
		activation, err = r.runEncoderLayer(ctx, activation, layerInfo, layerIndex)
		if err != nil {
			return reference.Value{}, fmt.Errorf("inference encoder layer %d: %w", layerIndex, err)
		}
	}
	if r.forwardProgram().Session == model.ForwardSessionEncoderDecoder {
		return r.runEncoderOutputNorm(ctx, activation)
	}
	return r.runOutputNorm(ctx, activation)
}

// NewEncoderDecoderSession: encodes one source sequence.
func (r *Runner) NewEncoderDecoderSession(ctx context.Context, sourceIDs []tokenizer.TokenID) (*EncoderDecoderSession, error) {
	if r == nil {
		return nil, errRunnerNil
	}
	if err := r.lockOpen(); err != nil {
		return nil, err
	}
	defer r.mu.Unlock()
	if r.forwardProgram().Session != model.ForwardSessionEncoderDecoder {
		return nil, errors.New("inference: encoder-decoder session requires a compiled session program")
	}
	encoder, err := r.forwardEncoderLocked(ctx, sourceIDs)
	if err != nil {
		return nil, err
	}
	return &EncoderDecoderSession{Encoder: encoder}, nil
}

// DecodeEncoderDecoder: appends decoder tokens and returns all chunk logits.
func (r *Runner) DecodeEncoderDecoder(
	ctx context.Context,
	session *EncoderDecoderSession, decoderIDs []tokenizer.TokenID,
) (reference.Value, *EncoderDecoderSession, error) {
	if r == nil {
		return reference.Value{}, nil, errRunnerNil
	}
	if err := r.lockOpen(); err != nil {
		return reference.Value{}, nil, err
	}
	defer r.mu.Unlock()
	if r.forwardProgram().Session != model.ForwardSessionEncoderDecoder {
		return reference.Value{}, nil, errors.New("inference: decoder requires a compiled encoder-decoder program")
	}
	return r.decodeEncoderDecoderLocked(ctx, session, decoderIDs)
}

func (r *Runner) decodeEncoderDecoderLocked(
	ctx context.Context,
	session *EncoderDecoderSession, decoderIDs []tokenizer.TokenID,
) (reference.Value, *EncoderDecoderSession, error) {
	if session == nil {
		return reference.Value{}, nil, errors.New("inference: encoder-decoder session is nil")
	}
	if session.Encoder.Shape.Rank != 2 ||
		session.Encoder.Shape.Dims[0] != uint64(r.spec.EmbeddingLength) ||
		session.Encoder.Shape.Dims[1] == 0 {
		return reference.Value{}, nil, errors.New("inference: encoder state shape is incompatible")
	}
	if len(decoderIDs) == 0 {
		return reference.Value{}, nil, errors.New("inference: decoder token sequence is empty")
	}
	var pastTokens, nextPosition uint32
	if session.Cache != nil {
		if err := r.validateEncoderDecoderCache(session.Cache, session.Encoder.Shape.Dims[1]); err != nil {
			return reference.Value{}, nil, err
		}
		pastTokens = session.Cache.Tokens
		nextPosition = effectiveCachePosition(session.Cache)
	}
	if uint64(pastTokens)+uint64(len(decoderIDs)) > uint64(r.spec.ContextLength) {
		return reference.Value{}, nil, errors.New("inference: decoder sequence exceeds context length")
	}
	rows, err := r.vocab.TensorIndices(decoderIDs)
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
		activation, nextCache.Layers[layerIndex], err = r.runDecoderLayer(
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
	return logits, &EncoderDecoderSession{Encoder: session.Encoder, Cache: nextCache}, nil
}

// ForwardCached: evaluates prompt chunk and returns host KV/recurrent cache
// suitable for later incremental call or ShiftCache edit

func (r *Runner) runEncoderLayer(
	ctx context.Context,
	activation reference.Value,
	info model.LayerWeights,
	layerIndex int,
) (reference.Value, error) {
	program, err := r.program.Model.EncoderProgram(uint32(layerIndex))
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

func (r *Runner) runDecoderLayer(
	ctx context.Context,
	activation, encoder reference.Value,
	info model.LayerWeights,
	layerIndex int,
	past *LayerCache,
) (reference.Value, LayerCache, error) {
	program, err := r.program.Model.DecoderProgram(uint32(layerIndex))
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
			return reference.Value{}, LayerCache{}, errors.New("encoder-decoder cross cache is missing")
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

func (r *Runner) runEncoderOutputNorm(
	ctx context.Context,
	activation reference.Value,
) (reference.Value, error) {
	if r.weights.EncoderOutputNorm == nil {
		return reference.Value{}, errors.New("inference: encoder output norm is missing")
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
