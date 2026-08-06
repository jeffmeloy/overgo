package model

import (
	"errors"
	"math"

	"llamacpp2go/internal/tensor"
)

func buildBERTEncoderBlock(
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
		requireGraphWeight("attention post norm", weights.AttentionPostNorm),
		requireGraphWeight("attention post norm bias", weights.AttentionPostNormBias),
		requireGraphWeight("feed-forward post norm", weights.FeedForwardPostNorm),
		requireGraphWeight("feed-forward post norm bias", weights.FeedForwardPostNormBias),
	}
	usesExperts := encoder.usesExperts() &&
		spec.IsInterleavedMoELayer(layerIndex)
	if usesExperts {
		required.add("feed-forward router", weights.FeedForwardRouter)
		required.add("feed-forward expert up", weights.FeedForwardUpExperts)
		required.add("feed-forward expert down", weights.FeedForwardDownExperts)
	} else {
		required.add("feed-forward up", weights.FeedForwardUp)
		required.add("feed-forward down", weights.FeedForwardDown)
	}
	if err := required.validate("BERT-family block"); err != nil {
		return DenseBlockResult{}, err
	}
	if encoder.Kind == encoderGraphNomic && weights.FeedForwardGate == nil {
		return DenseBlockResult{}, errors.New("NomicBERT block feed-forward gate weight is nil")
	}
	tokens := uint64(len(positions))
	queryLength := uint64(spec.HeadCount) * uint64(spec.KeyLength)
	keyLength := uint64(spec.HeadCountKV) * uint64(spec.KeyLength)
	valueLength := uint64(spec.HeadCountKV) * uint64(spec.ValueLength)
	var query, key, value *tensor.Tensor
	if weights.AttentionQKV != nil {
		mixed := builder.MulMat(weights.AttentionQKV, input)
		if weights.AttentionQKVBias != nil {
			mixed = builder.Add(mixed, weights.AttentionQKVBias)
		}
		stride := queryLength + keyLength + valueLength
		query = builder.Reshape(builder.GroupSlice(mixed, 0, queryLength, 1, stride), queryLength, tokens)
		key = builder.Reshape(builder.GroupSlice(mixed, queryLength, keyLength, 1, stride), keyLength, tokens)
		value = builder.Reshape(builder.GroupSlice(mixed, queryLength+keyLength, valueLength, 1, stride), valueLength, tokens)
	} else {
		if weights.AttentionQ == nil || weights.AttentionK == nil || weights.AttentionV == nil {
			return DenseBlockResult{}, errors.New("BERT-family block Q/K/V weights are incomplete")
		}
		query = builder.MulMat(weights.AttentionQ, input)
		key = builder.MulMat(weights.AttentionK, input)
		value = builder.MulMat(weights.AttentionV, input)
		if weights.AttentionQBias != nil {
			query = builder.Add(query, weights.AttentionQBias)
		}
		if weights.AttentionKBias != nil {
			key = builder.Add(key, weights.AttentionKBias)
		}
		if weights.AttentionVBias != nil {
			value = builder.Add(value, weights.AttentionVBias)
		}
	}
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
	attention = builder.AffineLayerNorm(
		builder.Add(input, attention), weights.AttentionPostNorm,
		weights.AttentionPostNormBias, spec.LayerNormEpsilon,
	)
	if encoder.Kind == encoderGraphJinaV2 && weights.AttentionNorm2 != nil {
		attention = ApplyNormalization(
			builder, builder.Add(attention, input),
			weights.AttentionNorm2, weights.AttentionNorm2Bias, spec,
		)
	}
	var feedForward *tensor.Tensor
	if usesExperts {
		plan := spec.moeGraphPlan(layerIndex)
		plan.Activation = tensor.MoEActivationGELU
		plan.NormalizeTopKProb = true
		feedForward = plan.BuildLayer(builder, attention, nil, weights)
	} else {
		feedForward = builder.MulMat(weights.FeedForwardUp, attention)
		if weights.FeedForwardUpBias != nil {
			feedForward = builder.Add(feedForward, weights.FeedForwardUpBias)
		}
	}
	if !usesExperts {
		if encoder.Kind == encoderGraphJinaV2 {
			width := uint64(spec.FeedForwardLength)
			if weights.FeedForwardGate != nil {
				gate := builder.MulMat(weights.FeedForwardGate, attention)
				feedForward = builder.GEGLU(gate, feedForward)
			} else if weights.FeedForwardUp.Shape.Dims[1] == 2*width {
				gate := builder.Reshape(builder.GroupSlice(feedForward, 0, width, 1, 2*width), width, tokens)
				up := builder.Reshape(builder.GroupSlice(feedForward, width, width, 1, 2*width), width, tokens)
				feedForward = builder.GEGLU(gate, up)
			} else {
				feedForward = builder.GELU(feedForward)
			}
		} else if encoder.Kind == encoderGraphNomic {
			gate := builder.MulMat(weights.FeedForwardGate, attention)
			feedForward = builder.SwiGLU(gate, feedForward)
		} else {
			feedForward = builder.GELU(feedForward)
		}
		feedForward = builder.MulMat(weights.FeedForwardDown, feedForward)
		if weights.FeedForwardDownBias != nil {
			feedForward = builder.Add(feedForward, weights.FeedForwardDownBias)
		}
	}
	output := builder.AffineLayerNorm(
		builder.Add(attention, feedForward), weights.FeedForwardPostNorm,
		weights.FeedForwardPostNormBias, spec.LayerNormEpsilon,
	)
	if err := builder.Err(); err != nil {
		return DenseBlockResult{}, err
	}
	return DenseBlockResult{Output: output, Key: key, Value: value}, nil
}

func buildModernBERTBlock(
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
		requireGraphWeight("feed-forward norm", weights.FeedForwardNorm),
		requireGraphWeight("feed-forward up", weights.FeedForwardUp),
		requireGraphWeight("feed-forward down", weights.FeedForwardDown),
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
	residual := builder.Add(input, attention)
	normalized = ApplyNormalization(builder, residual, weights.FeedForwardNorm, nil, spec)
	fused := builder.MulMat(weights.FeedForwardUp, normalized)
	feedForwardWidth := uint64(spec.FeedForwardLength)
	gate := builder.Reshape(
		builder.GroupSlice(fused, 0, feedForwardWidth, 1, 2*feedForwardWidth),
		feedForwardWidth, tokens,
	)
	up := builder.Reshape(
		builder.GroupSlice(fused, feedForwardWidth, feedForwardWidth, 1, 2*feedForwardWidth),
		feedForwardWidth, tokens,
	)
	switch spec.HiddenActivation {
	case "swiglu":
		fused = builder.SwiGLU(gate, up)
	case "reglu":
		fused = builder.ReGLU(gate, up)
	default:
		fused = builder.GEGLU(gate, up)
	}
	output := builder.Add(residual, builder.MulMat(weights.FeedForwardDown, fused))
	if err := builder.Err(); err != nil {
		return DenseBlockResult{}, err
	}
	return DenseBlockResult{Output: output, Key: key, Value: value}, nil
}

func buildGemmaEmbeddingBlock(
	builder *tensor.Builder,
	input *tensor.Tensor,
	spec Spec,
	weights LayerGraphWeights,
	positions []uint32,
	pastKey, pastValue *tensor.Tensor,
	layerIndex uint32,
) (DenseBlockResult, error) {
	if spec.Profile().EncoderGraph.Kind != encoderGraphGemmaEmbedding {
		return DenseBlockResult{}, errors.New("Gemma embedding block requires gemma-embedding architecture")
	}
	if input.Shape.Rank != 2 || input.Shape.Dims[0] != uint64(spec.EmbeddingLength) {
		return DenseBlockResult{}, errors.New("Gemma embedding block input shape is incompatible")
	}
	if len(positions) == 0 || uint64(len(positions)) != input.Shape.Dims[1] {
		return DenseBlockResult{}, errors.New("Gemma embedding block position count is incompatible")
	}
	if pastKey != nil || pastValue != nil {
		return DenseBlockResult{}, errors.New("Gemma embedding block does not support a KV cache")
	}
	if weights.AttentionQKV != nil &&
		(weights.AttentionQBias != nil || weights.AttentionKBias != nil || weights.AttentionVBias != nil) {
		return DenseBlockResult{}, errors.New("Gemma embedding fused QKV cannot use separate projection biases")
	}
	if weights.AttentionQKV == nil && weights.AttentionQKVBias != nil {
		return DenseBlockResult{}, errors.New("Gemma embedding fused QKV bias has no fused projection")
	}
	required := graphWeights{
		requireGraphWeight("attention norm", weights.AttentionNorm),
		requireGraphWeight("attention output", weights.AttentionOutput),
		requireGraphWeight("attention Q norm", weights.AttentionQNorm),
		requireGraphWeight("attention K norm", weights.AttentionKNorm),
		requireGraphWeight("attention post norm", weights.AttentionPostNorm),
		requireGraphWeight("feed-forward norm", weights.FeedForwardNorm),
		requireGraphWeight("feed-forward gate", weights.FeedForwardGate),
		requireGraphWeight("feed-forward up", weights.FeedForwardUp),
		requireGraphWeight("feed-forward down", weights.FeedForwardDown),
		requireGraphWeight("feed-forward post norm", weights.FeedForwardPostNorm),
	}
	if weights.AttentionQKV != nil {
		required.add("attention QKV", weights.AttentionQKV)
	} else {
		required.add("attention Q", weights.AttentionQ)
		required.add("attention K", weights.AttentionK)
		required.add("attention V", weights.AttentionV)
	}
	if err := required.validate("Gemma embedding block"); err != nil {
		return DenseBlockResult{}, err
	}
	tokens := uint64(len(positions))
	headCount := uint64(spec.HeadCount)
	kvHeadCount := uint64(spec.HeadCountKV)
	queryLength := headCount * uint64(spec.KeyLength)
	keyLength := kvHeadCount * uint64(spec.KeyLength)
	valueLength := kvHeadCount * uint64(spec.ValueLength)
	normalized := builder.WeightedRMSNorm(input, weights.AttentionNorm, spec.RMSNormEpsilon)
	var query, key, value *tensor.Tensor
	if weights.AttentionQKV != nil {
		mixed := builder.MulMat(weights.AttentionQKV, normalized)
		if weights.AttentionQKVBias != nil {
			mixed = builder.Add(mixed, weights.AttentionQKVBias)
		}
		stride := queryLength + keyLength + valueLength
		query = builder.Reshape(builder.GroupSlice(mixed, 0, queryLength, 1, stride), queryLength, tokens)
		key = builder.Reshape(builder.GroupSlice(mixed, queryLength, keyLength, 1, stride), keyLength, tokens)
		value = builder.Reshape(builder.GroupSlice(mixed, queryLength+keyLength, valueLength, 1, stride), valueLength, tokens)
	} else {
		query = builder.MulMat(weights.AttentionQ, normalized)
		key = builder.MulMat(weights.AttentionK, normalized)
		value = builder.MulMat(weights.AttentionV, normalized)
		if weights.AttentionQBias != nil {
			query = builder.Add(query, weights.AttentionQBias)
		}
		if weights.AttentionKBias != nil {
			key = builder.Add(key, weights.AttentionKBias)
		}
		if weights.AttentionVBias != nil {
			value = builder.Add(value, weights.AttentionVBias)
		}
	}
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
	attention = builder.WeightedRMSNorm(attention, weights.AttentionPostNorm, spec.RMSNormEpsilon)
	residual := builder.Add(input, attention)
	normalized = builder.WeightedRMSNorm(residual, weights.FeedForwardNorm, spec.RMSNormEpsilon)
	gate := builder.MulMat(weights.FeedForwardGate, normalized)
	up := builder.MulMat(weights.FeedForwardUp, normalized)
	feedForward := builder.MulMat(weights.FeedForwardDown, builder.GEGLU(gate, up))
	feedForward = builder.WeightedRMSNorm(feedForward, weights.FeedForwardPostNorm, spec.RMSNormEpsilon)
	output := builder.Add(residual, feedForward)
	if err := builder.Err(); err != nil {
		return DenseBlockResult{}, err
	}
	return DenseBlockResult{Output: output, Key: key, Value: value}, nil
}

func buildTalkieBlock(
	builder *tensor.Builder,
	input *tensor.Tensor,
	spec Spec,
	weights LayerGraphWeights,
	positions []uint32,
	pastKey, pastValue *tensor.Tensor,
) (DenseBlockResult, error) {
	if input.Shape.Rank != 2 || input.Shape.Dims[0] != uint64(spec.EmbeddingLength) {
		return DenseBlockResult{}, errors.New("Talkie block input shape is incompatible")
	}
	if weights.EmbeddingSkip == nil || !weights.EmbeddingSkip.Shape.Equal(input.Shape) {
		return DenseBlockResult{}, errors.New("Talkie block embedding skip is missing or incompatible")
	}
	if len(positions) == 0 || uint64(len(positions)) != input.Shape.Dims[1] {
		return DenseBlockResult{}, errors.New("Talkie block position count is incompatible")
	}
	if err := requireTensorPair(pastKey, pastValue, "Talkie block past key/value cache must both be present"); err != nil {
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
		requireGraphWeight("feed-forward gate", weights.FeedForwardGate),
		requireGraphWeight("feed-forward up", weights.FeedForwardUp),
		requireGraphWeight("feed-forward down", weights.FeedForwardDown),
		requireGraphWeight("layer output scale", weights.LayerOutputScale),
	}
	if weights.AttentionQKV != nil {
		required.add("attention QKV", weights.AttentionQKV)
	} else {
		required.add("attention Q", weights.AttentionQ)
		required.add("attention K", weights.AttentionK)
		required.add("attention V", weights.AttentionV)
	}
	if err := required.validate("Talkie block"); err != nil {
		return DenseBlockResult{}, err
	}

	tokens := uint64(len(positions))
	headCount := uint64(spec.HeadCount)
	kvHeadCount := uint64(spec.HeadCountKV)
	queryLength := headCount * uint64(spec.KeyLength)
	keyLength := kvHeadCount * uint64(spec.KeyLength)
	valueLength := kvHeadCount * uint64(spec.ValueLength)
	normalized := builder.RMSNorm(input, spec.RMSNormEpsilon)
	var query, key, value *tensor.Tensor
	if weights.AttentionQKV != nil {
		mixed := builder.MulMat(weights.AttentionQKV, normalized)
		if weights.AttentionQKVBias != nil {
			mixed = builder.Add(mixed, weights.AttentionQKVBias)
		}
		stride := queryLength + keyLength + valueLength
		query = builder.Reshape(builder.GroupSlice(mixed, 0, queryLength, 1, stride), queryLength, tokens)
		key = builder.Reshape(builder.GroupSlice(mixed, queryLength, keyLength, 1, stride), keyLength, tokens)
		value = builder.Reshape(builder.GroupSlice(mixed, queryLength+keyLength, valueLength, 1, stride), valueLength, tokens)
	} else {
		query = builder.MulMat(weights.AttentionQ, normalized)
		key = builder.MulMat(weights.AttentionK, normalized)
		value = builder.MulMat(weights.AttentionV, normalized)
		if weights.AttentionQBias != nil {
			query = builder.Add(query, weights.AttentionQBias)
		}
		if weights.AttentionKBias != nil {
			key = builder.Add(key, weights.AttentionKBias)
		}
		if weights.AttentionVBias != nil {
			value = builder.Add(value, weights.AttentionVBias)
		}
	}
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
	residual := builder.Add(input, attention)
	normalized = builder.RMSNorm(residual, spec.RMSNormEpsilon)
	gate := builder.MulMat(weights.FeedForwardGate, normalized)
	up := builder.MulMat(weights.FeedForwardUp, normalized)
	feedForward := builder.MulMat(weights.FeedForwardDown, builder.SwiGLU(gate, up))
	output := builder.Add(residual, feedForward)
	output = builder.Add(output, builder.Multiply(weights.EmbeddingSkip, weights.LayerOutputScale))
	if err := builder.Err(); err != nil {
		return DenseBlockResult{}, err
	}
	return DenseBlockResult{Output: output, Key: cacheKey, Value: cacheValue}, nil
}

// BuildT5EncoderBlock: full bidirectional encoder block.
func BuildT5EncoderBlock(
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
	if err := required.validate("T5 encoder block"); err != nil {
		return nil, err
	}
	if input.Shape.Rank != 2 || input.Shape.Dims[0] != uint64(spec.EmbeddingLength) {
		return nil, errors.New("T5 encoder block input shape is incompatible")
	}
	tokens := input.Shape.Dims[1]
	normalized := builder.WeightedRMSNorm(input, weights.AttentionNorm, spec.RMSNormEpsilon)
	query := builder.Reshape(
		builder.MulMat(weights.AttentionQ, normalized),
		uint64(spec.KeyLength),
		uint64(spec.HeadCount),
		tokens,
	)
	key := builder.Reshape(
		builder.MulMat(weights.AttentionK, normalized),
		uint64(spec.KeyLength),
		uint64(spec.HeadCountKV),
		tokens,
	)
	value := builder.Reshape(
		builder.MulMat(weights.AttentionV, normalized),
		uint64(spec.ValueLength),
		uint64(spec.HeadCountKV),
		tokens,
	)
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
	normalized = builder.WeightedRMSNorm(
		residual,
		weights.FeedForwardNorm,
		spec.RMSNormEpsilon,
	)
	up := builder.MulMat(weights.FeedForwardUp, normalized)
	var activated *tensor.Tensor
	if weights.FeedForwardGate != nil {
		gate := builder.MulMat(weights.FeedForwardGate, normalized)
		activated = builder.GEGLU(gate, up)
	} else {
		activated = builder.ReLU(up)
	}
	feedForward := builder.MulMat(weights.FeedForwardDown, activated)
	output := builder.Add(residual, feedForward)
	if err := builder.Err(); err != nil {
		return nil, err
	}
	return output, nil
}

// BuildT5DecoderBlockCached: causal self-attention plus fixed cross-attention.
func BuildT5DecoderBlockCached(
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
	required := graphWeights{
		requireGraphWeight("attention norm", weights.AttentionNorm),
		requireGraphWeight("attention Q", weights.AttentionQ),
		requireGraphWeight("attention K", weights.AttentionK),
		requireGraphWeight("attention V", weights.AttentionV),
		requireGraphWeight("attention output", weights.AttentionOutput),
		requireGraphWeight("attention relative bias", weights.AttentionRelativeBias),
		requireGraphWeight("cross-attention norm", weights.CrossAttentionNorm),
		requireGraphWeight("cross-attention Q", weights.CrossAttentionQ),
		requireGraphWeight("cross-attention K", weights.CrossAttentionK),
		requireGraphWeight("cross-attention V", weights.CrossAttentionV),
		requireGraphWeight("cross-attention output", weights.CrossAttentionOutput),
		requireGraphWeight("feed-forward norm", weights.FeedForwardNorm),
		requireGraphWeight("feed-forward up", weights.FeedForwardUp),
		requireGraphWeight("feed-forward down", weights.FeedForwardDown),
	}
	if err := required.validate("T5 decoder block"); err != nil {
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
	query := builder.Reshape(
		builder.MulMat(weights.AttentionQ, normalized),
		uint64(spec.KeyLength), uint64(spec.HeadCount), tokens,
	)
	key := builder.Reshape(
		builder.MulMat(weights.AttentionK, normalized),
		uint64(spec.KeyLength), uint64(spec.HeadCountKV), tokens,
	)
	value := builder.Reshape(
		builder.MulMat(weights.AttentionV, normalized),
		uint64(spec.ValueLength), uint64(spec.HeadCountKV), tokens,
	)
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
	crossQuery := builder.Reshape(
		builder.MulMat(weights.CrossAttentionQ, normalized),
		uint64(spec.KeyLength), uint64(spec.HeadCount), tokens,
	)
	crossKey, crossValue := pastCrossKey, pastCrossValue
	if crossKey == nil {
		if encoder.Shape.Rank != 2 || encoder.Shape.Dims[0] != uint64(spec.EmbeddingLength) {
			return DenseBlockResult{}, errors.New("T5 decoder encoder state shape is incompatible")
		}
		encoderTokens := encoder.Shape.Dims[1]
		crossKey = builder.Reshape(
			builder.MulMat(weights.CrossAttentionK, encoder),
			uint64(spec.KeyLength), uint64(spec.HeadCountKV), encoderTokens,
		)
		crossValue = builder.Reshape(
			builder.MulMat(weights.CrossAttentionV, encoder),
			uint64(spec.ValueLength), uint64(spec.HeadCountKV), encoderTokens,
		)
	}
	crossAttention := builder.AttentionWithOffset(crossQuery, crossKey, crossValue, 1, false, 0)
	crossAttention = builder.Reshape(crossAttention, uint64(spec.HeadCount)*uint64(spec.ValueLength), tokens)
	residual = builder.Add(residual, builder.MulMat(weights.CrossAttentionOutput, crossAttention))

	normalized = builder.WeightedRMSNorm(residual, weights.FeedForwardNorm, spec.RMSNormEpsilon)
	up := builder.MulMat(weights.FeedForwardUp, normalized)
	var activated *tensor.Tensor
	if weights.FeedForwardGate != nil {
		gate := builder.MulMat(weights.FeedForwardGate, normalized)
		activated = builder.GEGLU(gate, up)
	} else {
		activated = builder.ReLU(up)
	}
	output := builder.Add(residual, builder.MulMat(weights.FeedForwardDown, activated))
	if err := builder.Err(); err != nil {
		return DenseBlockResult{}, err
	}
	return DenseBlockResult{
		Output: output,
		Key:    cacheKey,
		Value:  cacheValue,
		FixedStates: map[CacheStateName]*tensor.Tensor{
			CacheStateCrossKey: crossKey, CacheStateCrossValue: crossValue,
		},
	}, nil
}
