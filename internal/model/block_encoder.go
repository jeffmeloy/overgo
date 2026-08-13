package model

import (
	"errors"
	"math"

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
	if input.Shape.Rank != 2 || input.Shape.Dims[0] != uint64(spec.EmbeddingLength) {
		return DenseBlockResult{}, errors.New("bidirectional projection input shape is incompatible")
	}
	if len(positions) == 0 || uint64(len(positions)) != input.Shape.Dims[1] {
		return DenseBlockResult{}, errors.New("bidirectional projection position count is incompatible")
	}
	if pastKey != nil || pastValue != nil {
		return DenseBlockResult{}, errors.New("bidirectional projection does not support a KV cache")
	}
	required := graphWeights{
		requireGraphWeight("attention output", weights.AttentionOutput),
	}
	if err := required.validate("bidirectional encoder attention"); err != nil {
		return DenseBlockResult{}, err
	}
	tokens := uint64(len(positions))
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
		frequencyScale := float32(1)
		if spec.RopeScalingType == "linear" {
			frequencyScale = 1 / spec.RopeScalingFactor
		}
		query, key = applyRoPEPairWithOptions(builder, query, key, tensor.RoPEOptions{
			Layout: tensor.RoPELayoutNeoX, Positions: positions,
			RotaryDimensions: spec.RopeDimensionCount,
			FrequencyBase:    spec.RopeFrequencyBase, FrequencyScale: frequencyScale,
		})
	}
	attentionScale := float32(1 / math.Sqrt(float64(spec.KeyLength)))
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
	if input == nil || input.Shape.Rank != 2 || input.Shape.Dims[0] != uint64(spec.EmbeddingLength) {
		return nil, errors.New("post-normalized encoder feed-forward input shape is incompatible")
	}
	required := graphWeights{}
	usesExperts := encoder.usesExperts() && weights.FeedForwardRouter != nil
	if usesExperts {
		required.add("feed-forward router", weights.FeedForwardRouter)
		required.add("feed-forward expert up", weights.FeedForwardUpExperts)
		required.add("feed-forward expert down", weights.FeedForwardDownExperts)
	} else {
		required.add("feed-forward up", weights.FeedForwardUp)
		required.add("feed-forward down", weights.FeedForwardDown)
	}
	if err := required.validate("encoder feed-forward"); err != nil {
		return nil, err
	}
	if encoder == encoderOperatorPreNormRoPEGated && weights.FeedForwardGate == nil {
		return nil, errors.New("gated encoder feed-forward weight is nil")
	}
	tokens := input.Shape.Dims[1]
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
			} else if weights.FeedForwardUp.Shape.Dims[1] == 2*width {
				gate := builder.Reshape(builder.GroupSlice(feedForward, 0, width, 1, 2*width), width, tokens)
				up := builder.Reshape(builder.GroupSlice(feedForward, width, width, 1, 2*width), width, tokens)
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
	if input.Shape.Rank != 2 || input.Shape.Dims[0] != uint64(spec.EmbeddingLength) {
		return DenseBlockResult{}, errors.New("fused-QKV sliding attention input shape is incompatible")
	}
	if len(positions) == 0 || uint64(len(positions)) != input.Shape.Dims[1] {
		return DenseBlockResult{}, errors.New("fused-QKV sliding attention position count is incompatible")
	}
	if pastKey != nil || pastValue != nil {
		return DenseBlockResult{}, errors.New("fused-QKV sliding attention does not support a KV cache")
	}
	required := graphWeights{
		requireGraphWeight("attention QKV", weights.AttentionQKV),
		requireGraphWeight("attention output", weights.AttentionOutput),
	}
	if layerIndex > 0 {
		required.add("attention norm", weights.AttentionNorm)
	}
	if err := required.validate("fused-QKV sliding attention"); err != nil {
		return DenseBlockResult{}, err
	}
	tokens := uint64(len(positions))
	normalized := input
	if weights.AttentionNorm != nil {
		normalized = plan.Normalization.Apply(builder, input, weights.AttentionNorm, nil)
	}
	mixed := builder.MulMat(weights.AttentionQKV, normalized)
	if weights.AttentionQKVBias != nil {
		mixed = builder.Add(mixed, weights.AttentionQKVBias)
	}
	width := uint64(spec.EmbeddingLength)
	query := builder.Reshape(builder.GroupSlice(mixed, 0, width, 1, 3*width), width, tokens)
	key := builder.Reshape(builder.GroupSlice(mixed, width, width, 1, 3*width), width, tokens)
	value := builder.Reshape(builder.GroupSlice(mixed, 2*width, width, 1, 3*width), width, tokens)
	query = builder.Reshape(query, uint64(spec.KeyLength), uint64(spec.HeadCount), tokens)
	key = builder.Reshape(key, uint64(spec.KeyLength), uint64(spec.HeadCountKV), tokens)
	value = builder.Reshape(value, uint64(spec.ValueLength), uint64(spec.HeadCountKV), tokens)
	frequencyBase := spec.RopeFrequencyBase
	if plan.Sliding {
		frequencyBase = spec.RopeFrequencySWA
	}
	frequencyScale := float32(1)
	if spec.RopeScalingType == "linear" {
		frequencyScale = 1 / spec.RopeScalingFactor
	}
	query, key = applyRoPEPairWithOptions(builder, query, key, tensor.RoPEOptions{
		Layout: tensor.RoPELayoutNeoX, Positions: positions,
		RotaryDimensions: spec.RopeDimensionCount,
		FrequencyBase:    frequencyBase, FrequencyScale: frequencyScale,
	})
	attentionScale := float32(1 / math.Sqrt(float64(spec.KeyLength)))
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
	if input.Shape.Rank != 2 || input.Shape.Dims[0] != uint64(spec.EmbeddingLength) {
		return DenseBlockResult{}, errors.New("bidirectional Q/K-normalized attention input shape is incompatible")
	}
	if len(positions) == 0 || uint64(len(positions)) != input.Shape.Dims[1] {
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
		requireGraphWeight("attention output", weights.AttentionOutput),
		requireGraphWeight("attention Q norm", weights.AttentionQNorm),
		requireGraphWeight("attention K norm", weights.AttentionKNorm),
	}
	if weights.AttentionQKV != nil {
		required.add("attention QKV", weights.AttentionQKV)
	} else {
		required.add("attention Q", weights.AttentionQ)
		required.add("attention K", weights.AttentionK)
		required.add("attention V", weights.AttentionV)
	}
	if err := required.validate("bidirectional Q/K-normalized attention"); err != nil {
		return DenseBlockResult{}, err
	}
	tokens := uint64(len(positions))
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
	frequencyScale := float32(1)
	if spec.RopeScalingType == "linear" {
		frequencyScale = 1 / spec.RopeScalingFactor
	}
	query, key = applyRoPEPairWithOptions(builder, query, key, tensor.RoPEOptions{
		Layout: tensor.RoPELayoutNeoX, Positions: positions,
		RotaryDimensions: spec.RopeDimensionCount,
		FrequencyBase:    frequencyBase, FrequencyScale: frequencyScale,
	})
	query = builder.Scale(query, float32(1/math.Sqrt(float64(spec.KeyLength))))
	attentionOptions := tensor.AttentionOptions{Scale: 1}
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
	if input.Shape.Rank != 2 || input.Shape.Dims[0] != uint64(spec.EmbeddingLength) {
		return DenseBlockResult{}, errors.New("causal post-Q/K-normalized attention input shape is incompatible")
	}
	if len(positions) == 0 || uint64(len(positions)) != input.Shape.Dims[1] {
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
		requireGraphWeight("attention output", weights.AttentionOutput),
		requireGraphWeight("attention Q norm", weights.AttentionQNorm),
	}
	if weights.AttentionQKV != nil {
		required.add("attention QKV", weights.AttentionQKV)
	} else {
		required.add("attention Q", weights.AttentionQ)
		required.add("attention K", weights.AttentionK)
		required.add("attention V", weights.AttentionV)
	}
	if err := required.validate("causal post-Q/K-normalized attention"); err != nil {
		return DenseBlockResult{}, err
	}

	tokens := uint64(len(positions))
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
	query = builder.RoPENeoXScaled(query, positions, spec.RopeDimensionCount, spec.RopeFrequencyBase, 1)
	key = builder.RoPENeoXScaled(key, positions, spec.RopeDimensionCount, spec.RopeFrequencyBase, 1)
	query = builder.WeightedRMSNorm(query, weights.AttentionQNorm, spec.RMSNormEpsilon)
	key = builder.RMSNorm(key, spec.RMSNormEpsilon)

	cacheKey := key
	cacheValue := value
	var queryStart uint32
	if pastKey != nil {
		if pastKey.Shape.Dims[2] > math.MaxUint32 {
			return DenseBlockResult{}, errors.New("Talkie KV cache token count exceeds uint32")
		}
		queryStart = uint32(pastKey.Shape.Dims[2])
		cacheKey = builder.Concat(pastKey, key, 2)
		cacheValue = builder.Concat(pastValue, value, 2)
	}
	attention := builder.AttentionWithOffset(
		query, cacheKey, cacheValue,
		float32(1/math.Sqrt(float64(spec.KeyLength))), true, queryStart,
	)
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

func buildEncoderDecoderFeedForward(
	builder *tensor.Builder,
	residual *tensor.Tensor,
	spec Spec,
	weights LayerGraphWeights,
) *tensor.Tensor {
	normalized := builder.WeightedRMSNorm(residual, weights.FeedForwardNorm, spec.RMSNormEpsilon)
	up := builder.MulMat(weights.FeedForwardUp, normalized)
	activated := builder.ReLU(up)
	if weights.FeedForwardGate != nil {
		activated = builder.GEGLU(builder.MulMat(weights.FeedForwardGate, normalized), up)
	}
	return builder.Add(residual, builder.MulMat(weights.FeedForwardDown, activated))
}

func validateEncoderDecoderWeights(weights LayerGraphWeights, cross bool) error {
	required := graphWeights{
		requireGraphWeight("attention norm", weights.AttentionNorm),
		requireGraphWeight("attention Q", weights.AttentionQ),
		requireGraphWeight("attention K", weights.AttentionK),
		requireGraphWeight("attention V", weights.AttentionV),
		requireGraphWeight("attention output", weights.AttentionOutput),
		requireGraphWeight("attention relative bias", weights.AttentionRelativeBias),
		requireGraphWeight("feed-forward norm", weights.FeedForwardNorm),
		requireGraphWeight("feed-forward up", weights.FeedForwardUp),
		requireGraphWeight("feed-forward down", weights.FeedForwardDown),
	}
	if cross {
		required = append(required,
			requireGraphWeight("cross-attention norm", weights.CrossAttentionNorm),
			requireGraphWeight("cross-attention Q", weights.CrossAttentionQ),
			requireGraphWeight("cross-attention K", weights.CrossAttentionK),
			requireGraphWeight("cross-attention V", weights.CrossAttentionV),
			requireGraphWeight("cross-attention output", weights.CrossAttentionOutput),
		)
	}
	return required.validate("relative-attention block")
}

// buildEncoderBlock: full bidirectional relative-attention block.
func buildEncoderBlock(
	builder *tensor.Builder,
	input *tensor.Tensor,
	spec Spec,
	weights LayerGraphWeights,
	encoder EncoderOperatorPolicy,
) (*tensor.Tensor, error) {
	if builder == nil || input == nil {
		return nil, errors.New("encoder block input is nil")
	}
	if encoder != encoderOperatorRelativeEncoderDecoder && encoder != encoderOperatorRelativeEncoder {
		return nil, errors.New("encoder block requires a compiled relative-attention program")
	}
	if err := validateEncoderDecoderWeights(weights, false); err != nil {
		return nil, err
	}
	if input.Shape.Rank != 2 || input.Shape.Dims[0] != uint64(spec.EmbeddingLength) {
		return nil, errors.New("encoder block input shape is incompatible")
	}
	tokens := input.Shape.Dims[1]
	normalized := builder.WeightedRMSNorm(input, weights.AttentionNorm, spec.RMSNormEpsilon)
	query := headedProjection(builder, normalized, weights.AttentionQ, spec.KeyLength, spec.HeadCount, tokens)
	key := headedProjection(builder, normalized, weights.AttentionK, spec.KeyLength, spec.HeadCountKV, tokens)
	value := headedProjection(builder, normalized, weights.AttentionV, spec.ValueLength, spec.HeadCountKV, tokens)
	attention := builder.AttentionWithRelativeBias(
		query,
		key,
		value,
		weights.AttentionRelativeBias,
		1,
	)
	attention = builder.Reshape(
		attention,
		uint64(spec.HeadCount)*uint64(spec.ValueLength),
		tokens,
	)
	attention = builder.MulMat(weights.AttentionOutput, attention)
	residual := builder.Add(input, attention)
	output := buildEncoderDecoderFeedForward(builder, residual, spec, weights)
	if err := builder.Err(); err != nil {
		return nil, err
	}
	return output, nil
}

// buildDecoderBlockCached: causal self-attention plus fixed cross-attention.
func buildDecoderBlockCached(
	builder *tensor.Builder,
	input, encoderState *tensor.Tensor,
	spec Spec,
	weights LayerGraphWeights,
	pastSelfKey, pastSelfValue, pastCrossKey, pastCrossValue *tensor.Tensor,
	encoderPolicy EncoderOperatorPolicy,
) (DenseBlockResult, error) {
	if builder == nil || input == nil {
		return DenseBlockResult{}, errors.New("decoder block input is nil")
	}
	if encoderPolicy != encoderOperatorRelativeEncoderDecoder {
		return DenseBlockResult{}, errors.New("decoder block requires a compiled relative-attention program")
	}
	if err := validateEncoderDecoderWeights(weights, true); err != nil {
		return DenseBlockResult{}, err
	}
	if input.Shape.Rank != 2 || input.Shape.Dims[0] != uint64(spec.EmbeddingLength) {
		return DenseBlockResult{}, errors.New("decoder block input shape is incompatible")
	}
	if err := requireTensorPair(pastSelfKey, pastSelfValue, "decoder self cache is incomplete"); err != nil {
		return DenseBlockResult{}, err
	}
	if err := requireTensorPair(pastCrossKey, pastCrossValue, "decoder cross cache is incomplete"); err != nil {
		return DenseBlockResult{}, err
	}
	if pastCrossKey == nil && encoderState == nil {
		return DenseBlockResult{}, errors.New("decoder encoder state is nil")
	}
	tokens := input.Shape.Dims[1]
	normalized := builder.WeightedRMSNorm(input, weights.AttentionNorm, spec.RMSNormEpsilon)
	query := headedProjection(builder, normalized, weights.AttentionQ, spec.KeyLength, spec.HeadCount, tokens)
	key := headedProjection(builder, normalized, weights.AttentionK, spec.KeyLength, spec.HeadCountKV, tokens)
	value := headedProjection(builder, normalized, weights.AttentionV, spec.ValueLength, spec.HeadCountKV, tokens)
	cacheKey, cacheValue := key, value
	var queryStart uint32
	if pastSelfKey != nil {
		if pastSelfKey.Shape.Rank != 3 || pastSelfValue.Shape.Rank != 3 || pastSelfKey.Shape.Dims[2] > math.MaxUint32 {
			return DenseBlockResult{}, errors.New("decoder self cache shape is incompatible")
		}
		queryStart = uint32(pastSelfKey.Shape.Dims[2])
		cacheKey = builder.Concat(pastSelfKey, key, 2)
		cacheValue = builder.Concat(pastSelfValue, value, 2)
	}
	attention := builder.AttentionWithRelativeBiasAndOffset(
		query, cacheKey, cacheValue, weights.AttentionRelativeBias, 1, queryStart,
	)
	attention = builder.Reshape(attention, uint64(spec.HeadCount)*uint64(spec.ValueLength), tokens)
	residual := builder.Add(input, builder.MulMat(weights.AttentionOutput, attention))

	normalized = builder.WeightedRMSNorm(residual, weights.CrossAttentionNorm, spec.RMSNormEpsilon)
	crossQuery := headedProjection(builder, normalized, weights.CrossAttentionQ, spec.KeyLength, spec.HeadCount, tokens)
	crossKey, crossValue := pastCrossKey, pastCrossValue
	if crossKey == nil {
		if encoderState.Shape.Rank != 2 || encoderState.Shape.Dims[0] != uint64(spec.EmbeddingLength) {
			return DenseBlockResult{}, errors.New("decoder encoder state shape is incompatible")
		}
		encoderTokens := encoderState.Shape.Dims[1]
		crossKey = headedProjection(builder, encoderState, weights.CrossAttentionK, spec.KeyLength, spec.HeadCountKV, encoderTokens)
		crossValue = headedProjection(builder, encoderState, weights.CrossAttentionV, spec.ValueLength, spec.HeadCountKV, encoderTokens)
	}
	crossAttention := builder.AttentionWithOffset(crossQuery, crossKey, crossValue, 1, false, 0)
	crossAttention = builder.Reshape(crossAttention, uint64(spec.HeadCount)*uint64(spec.ValueLength), tokens)
	residual = builder.Add(residual, builder.MulMat(weights.CrossAttentionOutput, crossAttention))

	output := buildEncoderDecoderFeedForward(builder, residual, spec, weights)
	if err := builder.Err(); err != nil {
		return DenseBlockResult{}, err
	}
	return DenseBlockResult{
		Output: output,
		Key:    cacheKey,
		Value:  cacheValue,
		States: CacheStates[*tensor.Tensor]{
			CacheStateCrossKey:   {Mode: CacheStateFixed, Value: crossKey},
			CacheStateCrossValue: {Mode: CacheStateFixed, Value: crossValue},
		},
	}, nil
}
