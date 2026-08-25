package model

import (
	"errors"

	"overgo/internal/hostmath"
	"overgo/internal/tensor"
)

func buildBidirectionalEncoderAttentionMix(
	builder *tensor.Builder,
	input *tensor.Tensor,
	spec Spec,
	weights LayerGraphWeights,
	positions []uint32,
	pastKey, pastValue *tensor.Tensor,
	plan LayerPlan,
) (DenseBlockResult, error) {
	layerIndex, encoder := plan.Layer, plan.EncoderOperator
	if !encoder.bidirectionalProjection() {
		return DenseBlockResult{}, errors.New("bidirectional projection requires a compatible encoder policy")
	}
	tokens, validInput := tensor.MatrixRows(input.Shape, uint64(spec.EmbeddingLength))
	if !validInput {
		return DenseBlockResult{}, errors.New("bidirectional projection input shape is incompatible")
	}
	if uint64(len(positions)) != tokens {
		return DenseBlockResult{}, errors.New("bidirectional projection position count is incompatible")
	}
	if pastKey != nil || pastValue != nil {
		return DenseBlockResult{}, errors.New("bidirectional projection does not support a KV cache")
	}
	required := graphWeights{
		weights.AttentionOutput,
	}
	if err := required.validate("bidirectional encoder attention"); err != nil {
		return DenseBlockResult{}, err
	}
	if weights.AttentionQKV == nil &&
		(weights.AttentionQ == nil || weights.AttentionK == nil || weights.AttentionV == nil) {
		return DenseBlockResult{}, errors.New("post-normalized encoder attention Q/K/V weights are incomplete")
	}
	runtime := denseBlockRuntime{
		builder: builder, spec: spec, weights: weights,
		layer: layerIndex, tokens: tokens,
	}
	query, key, value := runtime.projectAttention(input)
	if encoder.usesALiBiQKNorm() {
		if weights.AttentionQNorm != nil {
			query = plan.Normalization.Apply(builder, query, weights.AttentionQNorm, weights.AttentionQNormBias)
		}
		if weights.AttentionKNorm != nil {
			key = plan.Normalization.Apply(builder, key, weights.AttentionKNorm, weights.AttentionKNormBias)
		}
	}
	query = builder.Reshape(query, uint64(spec.KeyLength), uint64(spec.HeadCount), tokens)
	key = builder.Reshape(key, uint64(spec.KeyLength), uint64(spec.HeadCountKV), tokens)
	value = builder.Reshape(value, uint64(spec.ValueLength), uint64(spec.HeadCountKV), tokens)
	if encoder.usesRoPE() {
		query, key = applyRoPEPairWithOptions(builder, query, key, tensor.RoPEOptions{
			Layout: tensor.RoPELayoutNeoX, Positions: positions,
			RotaryDimensions: spec.RopeDimensionCount,
			FrequencyBase:    spec.RopeFrequencyBase, FrequencyScale: spec.ropeFrequencyScale(),
		})
	}
	attentionScale := hostmath.InvSqrt32(uint64(spec.KeyLength))
	attentionOptions := tensor.AttentionOptions{Scale: attentionScale}
	if encoder.usesALiBiQKNorm() {
		attentionOptions.MaxALiBiBias = spec.MaxALiBiBias
	}
	attention := builder.AttentionWithOptions(query, key, value, attentionOptions)
	attention = builder.Reshape(attention, uint64(spec.HeadCount)*uint64(spec.ValueLength), tokens)
	attention = builder.MulMat(weights.AttentionOutput, attention)
	if weights.AttentionOutputBias != nil {
		attention = builder.Add(attention, weights.AttentionOutputBias)
	}
	if err := builder.Err(); err != nil {
		return DenseBlockResult{}, err
	}
	return DenseBlockResult{Output: attention, Key: key, Value: value}, nil
}

func buildEncoderFeedForwardMix(
	builder *tensor.Builder,
	input *tensor.Tensor,
	spec Spec,
	weights LayerGraphWeights,
	plan LayerPlan,
) (*tensor.Tensor, error) {
	encoder := plan.EncoderOperator
	if !encoder.bidirectionalProjection() {
		return nil, errors.New("encoder feed-forward requires a bidirectional projection policy")
	}
	if input == nil {
		return nil, errors.New("post-normalized encoder feed-forward input shape is incompatible")
	}
	tokens, validInput := tensor.MatrixRows(input.Shape, uint64(spec.EmbeddingLength))
	if !validInput {
		return nil, errors.New("post-normalized encoder feed-forward input shape is incompatible")
	}
	required := graphWeights{}
	usesExperts := encoder.usesExperts() && weights.FeedForwardRouter != nil
	if usesExperts {
		required = append(required, weights.FeedForwardRouter)
		required = append(required, weights.FeedForwardUpExperts)
		required = append(required, weights.FeedForwardDownExperts)
	} else {
		required = append(required, weights.FeedForwardUp)
		required = append(required, weights.FeedForwardDown)
	}
	if err := required.validate("encoder feed-forward"); err != nil {
		return nil, err
	}
	if encoder == encoderOperatorPreNormRoPEGated && weights.FeedForwardGate == nil {
		return nil, errors.New("gated encoder feed-forward weight is nil")
	}
	var feedForward *tensor.Tensor
	if usesExperts {
		experts := plan.Experts
		experts.Activation = tensor.MoEActivationGELU
		experts.NormalizeTopKProb = true
		feedForward = experts.BuildLayer(builder, input, nil, weights)
	} else {
		feedForward = builder.MulMat(weights.FeedForwardUp, input)
		if weights.FeedForwardUpBias != nil {
			feedForward = builder.Add(feedForward, weights.FeedForwardUpBias)
		}
	}
	if !usesExperts {
		if encoder.usesALiBiQKNorm() {
			width := uint64(spec.FeedForwardLength)
			if weights.FeedForwardGate != nil {
				gate := builder.MulMat(weights.FeedForwardGate, input)
				feedForward = builder.GEGLU(gate, feedForward)
			} else if tensor.IsMatrix(weights.FeedForwardUp.Shape, uint64(spec.EmbeddingLength), 2*width) {
				stride := tensor.PairedExtent * width
				gate := builder.Reshape(builder.GroupSlice(
					feedForward, tensor.FirstOffset, width, tensor.SingletonExtent, stride,
				), width, tokens)
				up := builder.Reshape(builder.GroupSlice(
					feedForward, width, width, tensor.SingletonExtent, stride,
				), width, tokens)
				feedForward = builder.GEGLU(gate, up)
			} else {
				feedForward = builder.GELU(feedForward)
			}
		} else if encoder == encoderOperatorPreNormRoPEGated {
			gate := builder.MulMat(weights.FeedForwardGate, input)
			feedForward = builder.SwiGLU(gate, feedForward)
		} else {
			feedForward = builder.GELU(feedForward)
		}
		feedForward = builder.MulMat(weights.FeedForwardDown, feedForward)
		if weights.FeedForwardDownBias != nil {
			feedForward = builder.Add(feedForward, weights.FeedForwardDownBias)
		}
	}
	if err := builder.Err(); err != nil {
		return nil, err
	}
	return feedForward, nil
}

func buildBidirectionalFusedQKVMix(
	builder *tensor.Builder,
	input *tensor.Tensor,
	spec Spec,
	weights LayerGraphWeights,
	positions []uint32,
	pastKey, pastValue *tensor.Tensor,
	plan LayerPlan,
) (DenseBlockResult, error) {
	layerIndex := plan.Layer
	tokens, validInput := tensor.MatrixRows(input.Shape, uint64(spec.EmbeddingLength))
	if !validInput {
		return DenseBlockResult{}, errors.New("fused-QKV sliding attention input shape is incompatible")
	}
	if uint64(len(positions)) != tokens {
		return DenseBlockResult{}, errors.New("fused-QKV sliding attention position count is incompatible")
	}
	if pastKey != nil || pastValue != nil {
		return DenseBlockResult{}, errors.New("fused-QKV sliding attention does not support a KV cache")
	}
	required := graphWeights{
		weights.AttentionQKV,
		weights.AttentionOutput,
	}
	if layerIndex > tensor.FirstOffset {
		required = append(required, weights.AttentionNorm)
	}
	if err := required.validate("fused-QKV sliding attention"); err != nil {
		return DenseBlockResult{}, err
	}
	normalized := input
	if weights.AttentionNorm != nil {
		normalized = plan.Normalization.Apply(builder, input, weights.AttentionNorm, nil)
	}
	mixed := builder.MulMat(weights.AttentionQKV, normalized)
	if weights.AttentionQKVBias != nil {
		mixed = builder.Add(mixed, weights.AttentionQKVBias)
	}
	width := uint64(spec.EmbeddingLength)
	stride := tensor.TripleExtent * width
	query := builder.Reshape(builder.GroupSlice(
		mixed, tensor.FirstOffset, width, tensor.SingletonExtent, stride,
	), width, tokens)
	key := builder.Reshape(builder.GroupSlice(
		mixed, width, width, tensor.SingletonExtent, stride,
	), width, tokens)
	value := builder.Reshape(builder.GroupSlice(
		mixed, tensor.PairedExtent*width, width, tensor.SingletonExtent, stride,
	), width, tokens)
	query = builder.Reshape(query, uint64(spec.KeyLength), uint64(spec.HeadCount), tokens)
	key = builder.Reshape(key, uint64(spec.KeyLength), uint64(spec.HeadCountKV), tokens)
	value = builder.Reshape(value, uint64(spec.ValueLength), uint64(spec.HeadCountKV), tokens)
	frequencyBase := spec.RopeFrequencyBase
	if plan.Sliding {
		frequencyBase = spec.RopeFrequencySWA
	}
	query, key = applyRoPEPairWithOptions(builder, query, key, tensor.RoPEOptions{
		Layout: tensor.RoPELayoutNeoX, Positions: positions,
		RotaryDimensions: spec.RopeDimensionCount,
		FrequencyBase:    frequencyBase, FrequencyScale: spec.ropeFrequencyScale(),
	})
	attentionScale := hostmath.InvSqrt32(uint64(spec.KeyLength))
	attentionOptions := tensor.AttentionOptions{Scale: attentionScale}
	if plan.Sliding {
		attentionOptions.SymmetricWindow = true
		attentionOptions.Window = spec.SlidingWindow
	}
	attention := builder.AttentionWithOptions(query, key, value, attentionOptions)
	attention = builder.Reshape(attention, width, tokens)
	attention = builder.MulMat(weights.AttentionOutput, attention)
	if err := builder.Err(); err != nil {
		return DenseBlockResult{}, err
	}
	return DenseBlockResult{Output: attention, Key: key, Value: value}, nil
}

func buildBidirectionalQKNormMix(
	builder *tensor.Builder,
	input *tensor.Tensor,
	spec Spec,
	weights LayerGraphWeights,
	positions []uint32,
	pastKey, pastValue *tensor.Tensor,
	plan LayerPlan,
) (DenseBlockResult, error) {
	layerIndex := plan.Layer
	tokens, validInput := tensor.MatrixRows(input.Shape, uint64(spec.EmbeddingLength))
	if !validInput {
		return DenseBlockResult{}, errors.New("bidirectional Q/K-normalized attention input shape is incompatible")
	}
	if uint64(len(positions)) != tokens {
		return DenseBlockResult{}, errors.New("bidirectional Q/K-normalized attention position count is incompatible")
	}
	if pastKey != nil || pastValue != nil {
		return DenseBlockResult{}, errors.New("bidirectional Q/K-normalized attention does not support a KV cache")
	}
	if weights.AttentionQKV != nil &&
		(weights.AttentionQBias != nil || weights.AttentionKBias != nil || weights.AttentionVBias != nil) {
		return DenseBlockResult{}, errors.New("fused QKV cannot use separate projection biases")
	}
	if weights.AttentionQKV == nil && weights.AttentionQKVBias != nil {
		return DenseBlockResult{}, errors.New("fused QKV bias has no fused projection")
	}
	required := graphWeights{
		weights.AttentionOutput,
		weights.AttentionQNorm,
		weights.AttentionKNorm,
	}
	if weights.AttentionQKV != nil {
		required = append(required, weights.AttentionQKV)
	} else {
		required = append(required, weights.AttentionQ)
		required = append(required, weights.AttentionK)
		required = append(required, weights.AttentionV)
	}
	if err := required.validate("bidirectional Q/K-normalized attention"); err != nil {
		return DenseBlockResult{}, err
	}
	shapes := spec.TensorShapes(layerIndex)
	headCount, kvHeadCount := uint64(shapes.QueryHeads), uint64(shapes.KVHeads)
	runtime := denseBlockRuntime{
		builder: builder, spec: spec, weights: weights,
		layer: layerIndex, tokens: tokens,
	}
	query, key, value := runtime.projectAttention(input)
	query = builder.Reshape(query, uint64(spec.KeyLength), headCount, tokens)
	key = builder.Reshape(key, uint64(spec.KeyLength), kvHeadCount, tokens)
	value = builder.Reshape(value, uint64(spec.ValueLength), kvHeadCount, tokens)
	query = builder.WeightedRMSNorm(query, weights.AttentionQNorm, spec.RMSNormEpsilon)
	key = builder.WeightedRMSNorm(key, weights.AttentionKNorm, spec.RMSNormEpsilon)
	frequencyBase := spec.RopeFrequencyBase
	if plan.Sliding {
		frequencyBase = spec.RopeFrequencySWA
	}
	query, key = applyRoPEPairWithOptions(builder, query, key, tensor.RoPEOptions{
		Layout: tensor.RoPELayoutNeoX, Positions: positions,
		RotaryDimensions: spec.RopeDimensionCount,
		FrequencyBase:    frequencyBase, FrequencyScale: spec.ropeFrequencyScale(),
	})
	query = builder.Scale(query, hostmath.InvSqrt32(uint64(spec.KeyLength)))
	attentionOptions := tensor.AttentionOptions{Scale: tensor.UnitScale}
	if plan.Sliding {
		attentionOptions.SymmetricWindow = true
		attentionOptions.Window = spec.SlidingWindow
	}
	attention := builder.AttentionWithOptions(query, key, value, attentionOptions)
	attention = builder.Reshape(attention, headCount*uint64(spec.ValueLength), tokens)
	attention = builder.MulMat(weights.AttentionOutput, attention)
	if err := builder.Err(); err != nil {
		return DenseBlockResult{}, err
	}
	return DenseBlockResult{Output: attention, Key: key, Value: value}, nil
}

func buildCausalPostQKNormMixCached(
	builder *tensor.Builder,
	input *tensor.Tensor,
	spec Spec,
	weights LayerGraphWeights,
	positions []uint32,
	pastKey, pastValue *tensor.Tensor,
	layerIndex uint32,
) (DenseBlockResult, error) {
	tokens, validInput := tensor.MatrixRows(input.Shape, uint64(spec.EmbeddingLength))
	if !validInput {
		return DenseBlockResult{}, errors.New("causal post-Q/K-normalized attention input shape is incompatible")
	}
	if uint64(len(positions)) != tokens {
		return DenseBlockResult{}, errors.New("causal post-Q/K-normalized attention position count is incompatible")
	}
	if err := requireTensorPair(pastKey, pastValue, "causal post-Q/K-normalized attention cache pair is incomplete"); err != nil {
		return DenseBlockResult{}, err
	}
	if weights.AttentionQKV != nil &&
		(weights.AttentionQBias != nil || weights.AttentionKBias != nil || weights.AttentionVBias != nil) {
		return DenseBlockResult{}, errors.New("Talkie fused QKV cannot use separate projection biases")
	}
	if weights.AttentionQKV == nil && weights.AttentionQKVBias != nil {
		return DenseBlockResult{}, errors.New("Talkie fused QKV bias has no fused projection")
	}
	required := graphWeights{
		weights.AttentionOutput,
		weights.AttentionQNorm,
	}
	if weights.AttentionQKV != nil {
		required = append(required, weights.AttentionQKV)
	} else {
		required = append(required, weights.AttentionQ)
		required = append(required, weights.AttentionK)
		required = append(required, weights.AttentionV)
	}
	if err := required.validate("causal post-Q/K-normalized attention"); err != nil {
		return DenseBlockResult{}, err
	}

	shapes := spec.TensorShapes(layerIndex)
	headCount, kvHeadCount := uint64(shapes.QueryHeads), uint64(shapes.KVHeads)
	runtime := denseBlockRuntime{
		builder: builder, spec: spec, weights: weights,
		layer: layerIndex, tokens: tokens,
	}
	query, key, value := runtime.projectAttention(input)
	query = builder.Reshape(query, uint64(spec.KeyLength), headCount, tokens)
	key = builder.Reshape(key, uint64(spec.KeyLength), kvHeadCount, tokens)
	value = builder.Reshape(value, uint64(spec.ValueLength), kvHeadCount, tokens)
	query = builder.RoPEWithOptions(query, tensor.RoPEOptions{Layout: tensor.RoPELayoutNeoX, Positions: positions, RotaryDimensions: spec.RopeDimensionCount, FrequencyBase: spec.RopeFrequencyBase, FrequencyScale: tensor.UnitFrequencyScale})
	key = builder.RoPEWithOptions(key, tensor.RoPEOptions{Layout: tensor.RoPELayoutNeoX, Positions: positions, RotaryDimensions: spec.RopeDimensionCount, FrequencyBase: spec.RopeFrequencyBase, FrequencyScale: tensor.UnitFrequencyScale})
	query = builder.WeightedRMSNorm(query, weights.AttentionQNorm, spec.RMSNormEpsilon)
	key = builder.RMSNorm(key, spec.RMSNormEpsilon)

	cacheKey := key
	cacheValue := value
	var queryStart uint32
	if pastKey != nil {
		pastTokens, keyAxis, validKey := tensor.TrailingExtent32(pastKey.Shape, shapes.Key, shapes.KVHeads)
		valueTokens, valueAxis, validValue := tensor.TrailingExtent32(pastValue.Shape, shapes.Value, shapes.KVHeads)
		if !validKey || !validValue || pastTokens != valueTokens {
			return DenseBlockResult{}, errors.New("causal post-Q/K-normalized attention cache shape is invalid")
		}
		queryStart = pastTokens
		cacheKey = builder.Concat(pastKey, key, keyAxis)
		cacheValue = builder.Concat(pastValue, value, valueAxis)
	}
	attention := builder.AttentionWithOptions(
		query, cacheKey, cacheValue, tensor.AttentionOptions{Scale: hostmath.InvSqrt32(uint64(spec.KeyLength)), Causal: true, QueryStart: queryStart})

	attention = builder.Reshape(attention, headCount*uint64(spec.ValueLength), tokens)
	attention = builder.MulMat(weights.AttentionOutput, attention)
	if weights.AttentionOutputBias != nil {
		attention = builder.Add(attention, weights.AttentionOutputBias)
	}
	if err := builder.Err(); err != nil {
		return DenseBlockResult{}, err
	}
	return DenseBlockResult{Output: attention, Key: cacheKey, Value: cacheValue}, nil
}

func headedProjection(
	builder *tensor.Builder,
	input, weight *tensor.Tensor,
	width uint32,
	heads uint32,
	tokens uint64,
) *tensor.Tensor {
	return builder.Reshape(builder.MulMat(weight, input), uint64(width), uint64(heads), tokens)
}

func buildRelativeFeedForwardMix(
	builder *tensor.Builder,
	residual *tensor.Tensor,
	spec Spec,
	weights LayerGraphWeights,
) (*tensor.Tensor, error) {
	required := graphWeights{
		weights.FeedForwardNorm,
		weights.FeedForwardUp,
		weights.FeedForwardDown,
	}
	if err := required.validate("relative feed-forward"); err != nil {
		return nil, err
	}
	normalized := builder.WeightedRMSNorm(residual, weights.FeedForwardNorm, spec.RMSNormEpsilon)
	up := builder.MulMat(weights.FeedForwardUp, normalized)
	activated := builder.ReLU(up)
	if weights.FeedForwardGate != nil {
		activated = builder.GEGLU(builder.MulMat(weights.FeedForwardGate, normalized), up)
	}
	return builder.MulMat(weights.FeedForwardDown, activated), builder.Err()
}

func buildRelativeSelfAttentionMix(
	builder *tensor.Builder,
	input *tensor.Tensor,
	spec Spec,
	weights LayerGraphWeights,
	pastKey, pastValue *tensor.Tensor,
	causal bool,
) (DenseBlockResult, error) {
	if builder == nil || input == nil {
		return DenseBlockResult{}, errors.New("relative self-attention input is invalid")
	}
	tokens, validInput := tensor.MatrixRows(input.Shape, uint64(spec.EmbeddingLength))
	if !validInput {
		return DenseBlockResult{}, errors.New("relative self-attention input is invalid")
	}
	if err := requireTensorPair(pastKey, pastValue, "relative self-attention cache is incomplete"); err != nil {
		return DenseBlockResult{}, err
	}
	if !causal && pastKey != nil {
		return DenseBlockResult{}, errors.New("bidirectional relative attention does not support a cache")
	}
	required := graphWeights{
		weights.AttentionQ,
		weights.AttentionK,
		weights.AttentionV,
		weights.AttentionOutput,
		weights.AttentionRelativeBias,
	}
	if err := required.validate("relative self-attention"); err != nil {
		return DenseBlockResult{}, err
	}
	query := headedProjection(builder, input, weights.AttentionQ, spec.KeyLength, spec.HeadCount, tokens)
	key := headedProjection(builder, input, weights.AttentionK, spec.KeyLength, spec.HeadCountKV, tokens)
	value := headedProjection(builder, input, weights.AttentionV, spec.ValueLength, spec.HeadCountKV, tokens)
	cacheKey, cacheValue := key, value
	var queryStart uint32
	if pastKey != nil {
		pastTokens, keyAxis, validKey := tensor.TrailingExtent32(
			pastKey.Shape, uint64(spec.KeyLength), uint64(spec.HeadCountKV),
		)
		valueTokens, valueAxis, validValue := tensor.TrailingExtent32(
			pastValue.Shape, uint64(spec.ValueLength), uint64(spec.HeadCountKV),
		)
		if !validKey || !validValue || pastTokens != valueTokens {
			return DenseBlockResult{}, errors.New("relative self-attention cache shape is incompatible")
		}
		queryStart = pastTokens
		cacheKey = builder.Concat(pastKey, key, keyAxis)
		cacheValue = builder.Concat(pastValue, value, valueAxis)
	}
	var attention *tensor.Tensor
	if causal {
		attention = builder.AttentionWithOptions(
			query, cacheKey, cacheValue, tensor.AttentionOptions{Bias: weights.AttentionRelativeBias, Scale: tensor.UnitScale, Causal: true, QueryStart: queryStart})

	} else {
		attention = builder.AttentionWithOptions(query, cacheKey, cacheValue, tensor.AttentionOptions{Bias: weights.AttentionRelativeBias, Scale: tensor.UnitScale, RelativeBidirectional: true})
	}
	attention = builder.Reshape(attention, uint64(spec.HeadCount)*uint64(spec.ValueLength), tokens)
	result := DenseBlockResult{Output: builder.MulMat(weights.AttentionOutput, attention)}
	if causal {
		result.Key, result.Value = cacheKey, cacheValue
	}
	return result, builder.Err()
}

func buildCrossAttentionMix(
	builder *tensor.Builder,
	input, encoderState *tensor.Tensor,
	spec Spec,
	weights LayerGraphWeights,
	pastCrossKey, pastCrossValue *tensor.Tensor,
) (DenseBlockResult, error) {
	if builder == nil || input == nil {
		return DenseBlockResult{}, errors.New("cross-attention input is invalid")
	}
	tokens, validInput := tensor.MatrixRows(input.Shape, uint64(spec.EmbeddingLength))
	if !validInput {
		return DenseBlockResult{}, errors.New("cross-attention input is invalid")
	}
	if err := requireTensorPair(pastCrossKey, pastCrossValue, "cross-attention cache is incomplete"); err != nil {
		return DenseBlockResult{}, err
	}
	if pastCrossKey == nil && encoderState == nil {
		return DenseBlockResult{}, errors.New("cross-attention encoder state is nil")
	}
	required := graphWeights{
		weights.CrossAttentionQ,
		weights.CrossAttentionK,
		weights.CrossAttentionV,
		weights.CrossAttentionOutput,
	}
	if err := required.validate("cross-attention"); err != nil {
		return DenseBlockResult{}, err
	}
	crossQuery := headedProjection(builder, input, weights.CrossAttentionQ, spec.KeyLength, spec.HeadCount, tokens)
	crossKey, crossValue := pastCrossKey, pastCrossValue
	if crossKey == nil {
		encoderTokens, validEncoder := tensor.MatrixRows(encoderState.Shape, uint64(spec.EmbeddingLength))
		if !validEncoder {
			return DenseBlockResult{}, errors.New("decoder encoder state shape is incompatible")
		}
		crossKey = headedProjection(builder, encoderState, weights.CrossAttentionK, spec.KeyLength, spec.HeadCountKV, encoderTokens)
		crossValue = headedProjection(builder, encoderState, weights.CrossAttentionV, spec.ValueLength, spec.HeadCountKV, encoderTokens)
	}
	crossAttention := builder.AttentionWithOptions(
		crossQuery, crossKey, crossValue,
		tensor.AttentionOptions{Scale: tensor.UnitScale, QueryStart: tensor.FirstOffset},
	)
	crossAttention = builder.Reshape(crossAttention, uint64(spec.HeadCount)*uint64(spec.ValueLength), tokens)
	return DenseBlockResult{
		Output: builder.MulMat(weights.CrossAttentionOutput, crossAttention),
		States: CacheStates[*tensor.Tensor]{
			CacheStateCrossKey:   {Mode: CacheStateFixed, Value: crossKey},
			CacheStateCrossValue: {Mode: CacheStateFixed, Value: crossValue},
		},
	}, builder.Err()
}
