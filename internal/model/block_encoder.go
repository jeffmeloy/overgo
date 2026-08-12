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
	layerIndex uint32,
) (DenseBlockResult, error) {
	encoder := spec.Profile().EncoderGraph
	if !encoder.bertFamily() {
		return DenseBlockResult{}, errors.New("BERT-family block requires a supported encoder architecture")
	}
	if input.Shape.Rank != 2 || input.Shape.Dims[0] != uint64(spec.EmbeddingLength) {
		return DenseBlockResult{}, errors.New("BERT-family block input shape is incompatible")
	}
	if len(positions) == 0 || uint64(len(positions)) != input.Shape.Dims[1] {
		return DenseBlockResult{}, errors.New("BERT-family block position count is incompatible")
	}
	if pastKey != nil || pastValue != nil {
		return DenseBlockResult{}, errors.New("BERT-family block does not support a KV cache")
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
		builder: builder, spec: spec, weights: weights, profile: spec.Profile(),
		layer: layerIndex, tokens: tokens,
	}
	query, key, value := runtime.projectAttention(input)
	if encoder.Kind == encoderGraphJinaV2 {
		if weights.AttentionQNorm != nil {
			query = ApplyNormalization(builder, query, weights.AttentionQNorm, weights.AttentionQNormBias, spec)
		}
		if weights.AttentionKNorm != nil {
			key = ApplyNormalization(builder, key, weights.AttentionKNorm, weights.AttentionKNormBias, spec)
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
	if encoder.Kind == encoderGraphJinaV2 {
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
	layerIndex uint32,
) (*tensor.Tensor, error) {
	encoder := spec.Profile().EncoderGraph
	if !encoder.bertFamily() {
		return nil, errors.New("post-normalized encoder feed-forward requires a BERT-family architecture")
	}
	if input == nil || input.Shape.Rank != 2 || input.Shape.Dims[0] != uint64(spec.EmbeddingLength) {
		return nil, errors.New("post-normalized encoder feed-forward input shape is incompatible")
	}
	required := graphWeights{}
	usesExperts := encoder.usesExperts() && spec.IsInterleavedMoELayer(layerIndex)
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
	if encoder.Kind == encoderGraphNomic && weights.FeedForwardGate == nil {
		return nil, errors.New("NomicBERT feed-forward gate weight is nil")
	}
	tokens := input.Shape.Dims[1]
	var feedForward *tensor.Tensor
	if usesExperts {
		plan := spec.moeGraphPlan(layerIndex)
		plan.Activation = tensor.MoEActivationGELU
		plan.NormalizeTopKProb = true
		feedForward = plan.BuildLayer(builder, input, nil, weights)
	} else {
		feedForward = builder.MulMat(weights.FeedForwardUp, input)
		if weights.FeedForwardUpBias != nil {
			feedForward = builder.Add(feedForward, weights.FeedForwardUpBias)
		}
	}
	if !usesExperts {
		if encoder.Kind == encoderGraphJinaV2 {
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
		} else if encoder.Kind == encoderGraphNomic {
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
	layerIndex uint32,
) (DenseBlockResult, error) {
	if spec.Profile().EncoderGraph.Kind != encoderGraphModernBERT {
		return DenseBlockResult{}, errors.New("ModernBERT block requires ModernBERT architecture")
	}
	if input.Shape.Rank != 2 || input.Shape.Dims[0] != uint64(spec.EmbeddingLength) {
		return DenseBlockResult{}, errors.New("ModernBERT block input shape is incompatible")
	}
	if len(positions) == 0 || uint64(len(positions)) != input.Shape.Dims[1] {
		return DenseBlockResult{}, errors.New("ModernBERT block position count is incompatible")
	}
	if pastKey != nil || pastValue != nil {
		return DenseBlockResult{}, errors.New("ModernBERT block does not support a KV cache")
	}
	required := graphWeights{
		requireGraphWeight("attention QKV", weights.AttentionQKV),
		requireGraphWeight("attention output", weights.AttentionOutput),
	}
	if layerIndex > 0 {
		required.add("attention norm", weights.AttentionNorm)
	}
	if err := required.validate("ModernBERT block"); err != nil {
		return DenseBlockResult{}, err
	}
	tokens := uint64(len(positions))
	normalized := input
	if weights.AttentionNorm != nil {
		normalized = ApplyNormalization(builder, input, weights.AttentionNorm, nil, spec)
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
	if spec.IsSlidingLayer(layerIndex) {
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
	if spec.IsSlidingLayer(layerIndex) {
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
	layerIndex uint32,
) (DenseBlockResult, error) {
	if spec.Profile().EncoderGraph.Kind != encoderGraphGemmaEmbedding {
		return DenseBlockResult{}, errors.New("bidirectional Q/K-normalized attention requires gemma-embedding architecture")
	}
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
		builder: builder, spec: spec, weights: weights, profile: spec.Profile(),
		layer: layerIndex, tokens: tokens,
	}
	query, key, value := runtime.projectAttention(input)
	query = builder.Reshape(query, uint64(spec.KeyLength), headCount, tokens)
	key = builder.Reshape(key, uint64(spec.KeyLength), kvHeadCount, tokens)
	value = builder.Reshape(value, uint64(spec.ValueLength), kvHeadCount, tokens)
	query = builder.WeightedRMSNorm(query, weights.AttentionQNorm, spec.RMSNormEpsilon)
	key = builder.WeightedRMSNorm(key, weights.AttentionKNorm, spec.RMSNormEpsilon)
	frequencyBase := spec.RopeFrequencyBase
	if spec.IsSlidingLayer(layerIndex) {
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
	if spec.IsSlidingLayer(layerIndex) {
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
	if spec.Profile().DenseGraph != DenseGraphTalkie {
		return DenseBlockResult{}, errors.New("causal post-Q/K-normalized attention requires Talkie policy")
	}
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
		builder: builder, spec: spec, weights: weights, profile: spec.Profile(),
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

func t5Projection(
	builder *tensor.Builder,
	input, weight *tensor.Tensor,
	width uint32,
	heads uint32,
	tokens uint64,
) *tensor.Tensor {
	return builder.Reshape(builder.MulMat(weight, input), uint64(width), uint64(heads), tokens)
}

func buildT5FeedForward(
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

func validateT5Weights(weights LayerGraphWeights, cross bool) error {
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
	return required.validate("T5 block")
}

// buildT5EncoderBlock: full bidirectional encoder block.
func buildT5EncoderBlock(
	builder *tensor.Builder,
	input *tensor.Tensor,
	spec Spec,
	weights LayerGraphWeights,
) (*tensor.Tensor, error) {
	if builder == nil || input == nil {
		return nil, errors.New("T5 encoder block input is nil")
	}
	encoder := spec.Profile().EncoderGraph.Kind
	if encoder != encoderGraphT5 && encoder != encoderGraphT5Encoder {
		return nil, errors.New("T5 encoder block requires T5 architecture")
	}
	if err := validateT5Weights(weights, false); err != nil {
		return nil, err
	}
	if input.Shape.Rank != 2 || input.Shape.Dims[0] != uint64(spec.EmbeddingLength) {
		return nil, errors.New("T5 encoder block input shape is incompatible")
	}
	tokens := input.Shape.Dims[1]
	normalized := builder.WeightedRMSNorm(input, weights.AttentionNorm, spec.RMSNormEpsilon)
	query := t5Projection(builder, normalized, weights.AttentionQ, spec.KeyLength, spec.HeadCount, tokens)
	key := t5Projection(builder, normalized, weights.AttentionK, spec.KeyLength, spec.HeadCountKV, tokens)
	value := t5Projection(builder, normalized, weights.AttentionV, spec.ValueLength, spec.HeadCountKV, tokens)
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
	output := buildT5FeedForward(builder, residual, spec, weights)
	if err := builder.Err(); err != nil {
		return nil, err
	}
	return output, nil
}

// buildT5DecoderBlockCached: causal self-attention plus fixed cross-attention.
func buildT5DecoderBlockCached(
	builder *tensor.Builder,
	input, encoder *tensor.Tensor,
	spec Spec,
	weights LayerGraphWeights,
	pastSelfKey, pastSelfValue, pastCrossKey, pastCrossValue *tensor.Tensor,
) (DenseBlockResult, error) {
	if builder == nil || input == nil {
		return DenseBlockResult{}, errors.New("T5 decoder block input is nil")
	}
	if spec.Profile().EncoderGraph.Kind != encoderGraphT5 {
		return DenseBlockResult{}, errors.New("T5 decoder block requires T5 architecture")
	}
	if err := validateT5Weights(weights, true); err != nil {
		return DenseBlockResult{}, err
	}
	if input.Shape.Rank != 2 || input.Shape.Dims[0] != uint64(spec.EmbeddingLength) {
		return DenseBlockResult{}, errors.New("T5 decoder block input shape is incompatible")
	}
	if err := requireTensorPair(pastSelfKey, pastSelfValue, "T5 decoder self cache is incomplete"); err != nil {
		return DenseBlockResult{}, err
	}
	if err := requireTensorPair(pastCrossKey, pastCrossValue, "T5 decoder cross cache is incomplete"); err != nil {
		return DenseBlockResult{}, err
	}
	if pastCrossKey == nil && encoder == nil {
		return DenseBlockResult{}, errors.New("T5 decoder encoder state is nil")
	}
	tokens := input.Shape.Dims[1]
	normalized := builder.WeightedRMSNorm(input, weights.AttentionNorm, spec.RMSNormEpsilon)
	query := t5Projection(builder, normalized, weights.AttentionQ, spec.KeyLength, spec.HeadCount, tokens)
	key := t5Projection(builder, normalized, weights.AttentionK, spec.KeyLength, spec.HeadCountKV, tokens)
	value := t5Projection(builder, normalized, weights.AttentionV, spec.ValueLength, spec.HeadCountKV, tokens)
	cacheKey, cacheValue := key, value
	var queryStart uint32
	if pastSelfKey != nil {
		if pastSelfKey.Shape.Rank != 3 || pastSelfValue.Shape.Rank != 3 || pastSelfKey.Shape.Dims[2] > math.MaxUint32 {
			return DenseBlockResult{}, errors.New("T5 decoder self cache shape is incompatible")
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
	crossQuery := t5Projection(builder, normalized, weights.CrossAttentionQ, spec.KeyLength, spec.HeadCount, tokens)
	crossKey, crossValue := pastCrossKey, pastCrossValue
	if crossKey == nil {
		if encoder.Shape.Rank != 2 || encoder.Shape.Dims[0] != uint64(spec.EmbeddingLength) {
			return DenseBlockResult{}, errors.New("T5 decoder encoder state shape is incompatible")
		}
		encoderTokens := encoder.Shape.Dims[1]
		crossKey = t5Projection(builder, encoder, weights.CrossAttentionK, spec.KeyLength, spec.HeadCountKV, encoderTokens)
		crossValue = t5Projection(builder, encoder, weights.CrossAttentionV, spec.ValueLength, spec.HeadCountKV, encoderTokens)
	}
	crossAttention := builder.AttentionWithOffset(crossQuery, crossKey, crossValue, 1, false, 0)
	crossAttention = builder.Reshape(crossAttention, uint64(spec.HeadCount)*uint64(spec.ValueLength), tokens)
	residual = builder.Add(residual, builder.MulMat(weights.CrossAttentionOutput, crossAttention))

	output := buildT5FeedForward(builder, residual, spec, weights)
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
