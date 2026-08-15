package model

import (
	"errors"
	"math"

	"overgo/internal/tensor"
)

func buildSelectiveScanMixCached(
	builder *tensor.Builder,
	input *tensor.Tensor,
	spec Spec,
	weights LayerGraphWeights,
	convState, ssmState *tensor.Tensor,
	mixer RecurrentMixerPolicy,
) (DenseBlockResult, error) {
	useWeightedStateNorm := mixer == recurrentMixerWeightedSelectiveScan
	if builder == nil || input == nil || convState == nil || ssmState == nil {
		return DenseBlockResult{}, errors.New("selective-scan input/state is nil")
	}
	if input.Shape.Rank != 2 ||
		input.Shape.Dims[0] != uint64(spec.EmbeddingLength) {
		return DenseBlockResult{}, errors.New("selective-scan input shape is invalid")
	}
	required := graphWeights{
		requireGraphWeight("SSM input", weights.SSMInput),
		requireGraphWeight("SSM convolution", weights.SSMConv1D),
		requireGraphWeight("SSM convolution bias", weights.SSMConv1DBias),
		requireGraphWeight("SSM X", weights.SSMX),
		requireGraphWeight("SSM time-step weight", weights.SSMTimeStepWeight),
		requireGraphWeight("SSM time-step bias", weights.SSMTimeStep),
		requireGraphWeight("SSM A", weights.SSMA),
		requireGraphWeight("SSM D", weights.SSMD),
		requireGraphWeight("SSM output", weights.SSMOutput),
	}
	if err := required.validate("selective-scan mixer"); err != nil {
		return DenseBlockResult{}, err
	}
	if useWeightedStateNorm {
		if err := (graphWeights{
			requireGraphWeight("SSM time-step norm", weights.SSMTimeStepNorm),
			requireGraphWeight("SSM B norm", weights.SSMBNorm),
			requireGraphWeight("SSM C norm", weights.SSMCNorm),
		}).validate("weighted selective-scan mixer"); err != nil {
			return DenseBlockResult{}, err
		}
	}
	convShape := tensor.MustShape(uint64(spec.SSMConvKernel-1), uint64(spec.SSMInnerSize))
	ssmShape := tensor.MustShape(uint64(spec.SSMStateSize), uint64(spec.SSMInnerSize))
	if !convState.Shape.Equal(convShape) || !ssmState.Shape.Equal(ssmShape) {
		return DenseBlockResult{}, errors.New("selective-scan recurrent cache shape is invalid")
	}
	tokens := input.Shape.Dims[1]
	inner := uint64(spec.SSMInnerSize)
	stateWidth := uint64(spec.SSMStateSize)
	rank := uint64(spec.SSMTimeStepRank)
	xz := builder.MulMat(weights.SSMInput, input)
	x := builder.Reshape(builder.GroupSlice(xz, 0, inner, 1, inner), inner, tokens)
	z := builder.Reshape(builder.GroupSlice(xz, inner, inner, 1, inner), inner, tokens)
	convInput := builder.Concat(convState, builder.Transpose2D(x), 0)
	nextConvState := builder.Reshape(
		builder.GroupSlice(convInput, tokens, uint64(spec.SSMConvKernel-1), 1, uint64(spec.SSMConvKernel-1)),
		uint64(spec.SSMConvKernel-1), inner,
	)
	x = builder.SiLU(builder.Add(builder.SSMConv(convInput, weights.SSMConv1D), weights.SSMConv1DBias))
	xdb := builder.MulMat(weights.SSMX, x)
	dt := builder.Reshape(builder.GroupSlice(xdb, 0, rank, 1, rank), rank, tokens)
	beta := builder.Reshape(
		builder.GroupSlice(xdb, rank, stateWidth, 1, stateWidth),
		stateWidth, 1, tokens, 1,
	)
	c := builder.Reshape(
		builder.GroupSlice(xdb, rank+stateWidth, stateWidth, 1, stateWidth),
		stateWidth, 1, tokens, 1,
	)
	if spec.SSMDtBCNorm {
		dt = builder.RMSNorm(dt, spec.RMSNormEpsilon)
		beta = builder.RMSNorm(beta, spec.RMSNormEpsilon)
		c = builder.RMSNorm(c, spec.RMSNormEpsilon)
	} else if useWeightedStateNorm {
		dt = builder.WeightedRMSNorm(dt, weights.SSMTimeStepNorm, spec.RMSNormEpsilon)
		beta = builder.WeightedRMSNorm(beta, weights.SSMBNorm, spec.RMSNormEpsilon)
		c = builder.WeightedRMSNorm(c, weights.SSMCNorm, spec.RMSNormEpsilon)
	}
	dt = builder.Add(builder.MulMat(weights.SSMTimeStepWeight, dt), weights.SSMTimeStep)
	packed := builder.SSMScan(
		builder.Reshape(ssmState, stateWidth, 1, inner, 1),
		builder.Reshape(x, 1, inner, tokens, 1),
		builder.Reshape(dt, inner, tokens, 1),
		weights.SSMA,
		beta,
		c,
	)
	attentionElements := inner * tokens
	attention := builder.FlatSlice(packed, 0, inner, tokens)
	nextSSMState := builder.FlatSlice(packed, attentionElements, stateWidth, inner)
	attention = builder.Add(attention, builder.Multiply(x, weights.SSMD))
	attention = builder.Multiply(attention, builder.SiLU(z))
	attention = builder.MulMat(weights.SSMOutput, attention)
	if err := builder.Err(); err != nil {
		return DenseBlockResult{}, err
	}
	return DenseBlockResult{Output: attention, Key: nextConvState, Value: nextSSMState}, nil
}

func buildGroupedSelectiveScanMixCached(
	builder *tensor.Builder,
	input *tensor.Tensor,
	spec Spec,
	weights LayerGraphWeights,
	convState, ssmState *tensor.Tensor,
	mixer RecurrentMixerPolicy,
) (DenseBlockResult, error) {
	if builder == nil || input == nil || convState == nil || ssmState == nil {
		return DenseBlockResult{}, errors.New("grouped selective-scan input/state is nil")
	}
	if input.Shape.Rank != 2 ||
		input.Shape.Dims[0] != uint64(spec.EmbeddingLength) {
		return DenseBlockResult{}, errors.New("grouped selective-scan input shape is invalid")
	}
	required := graphWeights{
		requireGraphWeight("SSM input", weights.SSMInput),
		requireGraphWeight("SSM convolution", weights.SSMConv1D),
		requireGraphWeight("SSM time-step bias", weights.SSMTimeStep),
		requireGraphWeight("SSM A", weights.SSMA),
		requireGraphWeight("SSM D", weights.SSMD),
		requireGraphWeight("SSM output", weights.SSMOutput),
	}
	if mixer != recurrentMixerAttentionGroupedSelectiveScan {
		required.add("SSM norm", weights.SSMNorm)
	}
	if err := required.validate("grouped selective-scan mixer"); err != nil {
		return DenseBlockResult{}, err
	}
	inner := uint64(spec.SSMInnerSize)
	stateWidth := uint64(spec.SSMStateSize)
	heads := uint64(spec.SSMTimeStepRank)
	groups := uint64(spec.SSMGroupCount)
	headWidth := inner / heads
	convWidth := inner + 2*groups*stateWidth
	convShape := tensor.MustShape(uint64(spec.SSMConvKernel-1), convWidth)
	ssmShape := tensor.MustShape(stateWidth, inner)
	if !convState.Shape.Equal(convShape) || !ssmState.Shape.Equal(ssmShape) {
		return DenseBlockResult{}, errors.New("grouped selective-scan cache shape is invalid")
	}
	tokens := input.Shape.Dims[1]
	zxBCdt := builder.MulMat(weights.SSMInput, input)
	z := builder.Reshape(builder.GroupSlice(zxBCdt, 0, headWidth, heads, headWidth), headWidth, heads, tokens, 1)
	xBC := builder.Reshape(builder.GroupSlice(zxBCdt, inner, convWidth, 1, convWidth), convWidth, tokens)
	dt := builder.Reshape(
		builder.GroupSlice(zxBCdt, inner+convWidth, heads, 1, heads), heads, tokens, 1,
	)
	convInput := builder.Concat(convState, builder.Transpose2D(xBC), 0)
	nextConvState := builder.Reshape(
		builder.GroupSlice(convInput, tokens, uint64(spec.SSMConvKernel-1), 1, uint64(spec.SSMConvKernel-1)),
		uint64(spec.SSMConvKernel-1), convWidth,
	)
	xBC = builder.SSMConv(convInput, weights.SSMConv1D)
	if weights.SSMConv1DBias != nil {
		xBC = builder.Add(xBC, weights.SSMConv1DBias)
	}
	xBC = builder.SiLU(xBC)
	x := builder.Reshape(builder.GroupSlice(xBC, 0, headWidth, heads, headWidth), headWidth, heads, tokens, 1)
	beta := builder.Reshape(
		builder.GroupSlice(xBC, inner, stateWidth, groups, stateWidth), stateWidth, groups, tokens, 1,
	)
	c := builder.Reshape(
		builder.GroupSlice(xBC, inner+groups*stateWidth, stateWidth, groups, stateWidth),
		stateWidth, groups, tokens, 1,
	)
	dt = builder.Add(dt, weights.SSMTimeStep)
	packed := builder.SSMScan(
		builder.Reshape(ssmState, stateWidth, headWidth, heads, 1),
		x,
		dt,
		weights.SSMA,
		beta,
		c,
	)
	attentionElements := inner * tokens
	attention := builder.FlatSlice(packed, 0, headWidth, heads, tokens, 1)
	nextSSMState := builder.FlatSlice(packed, attentionElements, stateWidth, inner)
	attention = builder.Add(attention, builder.Multiply(x, weights.SSMD))
	attention = builder.Multiply(attention, builder.SiLU(z))
	attention = builder.Reshape(attention, inner/groups, groups, tokens, 1)
	if weights.SSMNorm != nil {
		attention = builder.WeightedRMSNorm(attention, weights.SSMNorm, spec.RMSNormEpsilon)
	}
	attention = builder.Reshape(attention, inner, tokens)
	attention = builder.MulMat(weights.SSMOutput, attention)
	if err := builder.Err(); err != nil {
		return DenseBlockResult{}, err
	}
	return DenseBlockResult{Output: attention, Key: nextConvState, Value: nextSSMState}, nil
}

func buildAttentionSSMHybridMixCached(
	builder *tensor.Builder,
	normalized *tensor.Tensor,
	spec Spec,
	weights LayerGraphWeights,
	positions []uint32,
	pastKey, pastValue, convState, ssmState *tensor.Tensor,
	plan LayerPlan,
) (DenseBlockResult, error) {
	if builder == nil || normalized == nil || convState == nil || ssmState == nil {
		return DenseBlockResult{}, errors.New("hybrid attention-scan input/state is nil")
	}
	if normalized.Shape.Rank != 2 || normalized.Shape.Dims[0] != uint64(spec.EmbeddingLength) {
		return DenseBlockResult{}, errors.New("hybrid attention-scan input is invalid")
	}
	if len(positions) == 0 || uint64(len(positions)) != normalized.Shape.Dims[1] {
		return DenseBlockResult{}, errors.New("hybrid attention-scan position count is invalid")
	}
	if err := requireTensorPair(pastKey, pastValue, "hybrid attention-scan KV cache is incomplete"); err != nil {
		return DenseBlockResult{}, err
	}
	if err := (graphWeights{
		requireGraphWeight("attention output", weights.AttentionOutput),
	}).validate("hybrid attention-scan"); err != nil {
		return DenseBlockResult{}, err
	}
	if weights.AttentionQKV == nil && (weights.AttentionQ == nil || weights.AttentionK == nil || weights.AttentionV == nil) {
		return DenseBlockResult{}, errors.New("hybrid attention projection catalog is incomplete")
	}
	tokens := uint64(len(positions))
	heads := uint64(spec.HeadCount)
	kvHeads := uint64(spec.HeadCountKV)
	query, key, value := (denseBlockRuntime{
		builder: builder, spec: spec, weights: weights, plan: plan,
		layer: plan.Layer, tokens: tokens,
	}).projectAttention(normalized)
	query = builder.Reshape(query, uint64(spec.KeyLength), heads, tokens)
	key = builder.Reshape(key, uint64(spec.KeyLength), kvHeads, tokens)
	value = builder.Reshape(value, uint64(spec.ValueLength), kvHeads, tokens)
	if weights.RopeFactors != nil {
		query = builder.RoPEWithOptions(query, tensor.RoPEOptions{Layout: tensor.RoPELayoutNeoX, Positions: positions, FrequencyFactors: weights.RopeFactors, RotaryDimensions: spec.RopeDimensionCount, FrequencyBase: spec.RopeFrequencyBase, FrequencyScale: 1})
		key = builder.RoPEWithOptions(key, tensor.RoPEOptions{Layout: tensor.RoPELayoutNeoX, Positions: positions, FrequencyFactors: weights.RopeFactors, RotaryDimensions: spec.RopeDimensionCount, FrequencyBase: spec.RopeFrequencyBase, FrequencyScale: 1})
	} else {
		query = builder.RoPEWithOptions(query, tensor.RoPEOptions{Layout: tensor.RoPELayoutNeoX, Positions: positions, RotaryDimensions: spec.RopeDimensionCount, FrequencyBase: spec.RopeFrequencyBase, FrequencyScale: 1})
		key = builder.RoPEWithOptions(key, tensor.RoPEOptions{Layout: tensor.RoPELayoutNeoX, Positions: positions, RotaryDimensions: spec.RopeDimensionCount, FrequencyBase: spec.RopeFrequencyBase, FrequencyScale: 1})
	}
	cacheKey, cacheValue := key, value
	var queryStart uint32
	if pastKey != nil {
		if pastKey.Shape.Rank != 3 || pastValue.Shape.Rank != 3 || pastKey.Shape.Dims[2] > math.MaxUint32 {
			return DenseBlockResult{}, errors.New("hybrid attention-scan KV cache shape is invalid")
		}
		queryStart = uint32(pastKey.Shape.Dims[2])
		cacheKey = builder.Concat(pastKey, key, 2)
		cacheValue = builder.Concat(pastValue, value, 2)
	}
	attentionScale := float32(1 / math.Sqrt(float64(spec.KeyLength)))
	if spec.AttentionScale > 0 {
		attentionScale = spec.AttentionScale
	}
	attention := builder.AttentionWithOffset(query, cacheKey, cacheValue, attentionScale, true, queryStart)
	attention = builder.Reshape(attention, heads*uint64(spec.ValueLength), tokens)
	attention = builder.MulMat(weights.AttentionOutput, attention)
	if weights.AttentionOutputBias != nil {
		attention = builder.Add(attention, weights.AttentionOutputBias)
	}

	ssm, err := buildGroupedSelectiveScanMixCached(
		builder, normalized, spec, weights, convState, ssmState, plan.Mixer,
	)
	if err != nil {
		return DenseBlockResult{}, err
	}
	hybrid := builder.Add(attention, ssm.Output)
	if err := builder.Err(); err != nil {
		return DenseBlockResult{}, err
	}
	return DenseBlockResult{
		Output: hybrid, Key: cacheKey, Value: cacheValue,
		States: CacheStates[*tensor.Tensor]{
			CacheStateConvolution: {Mode: CacheStateFixed, Value: ssm.Key},
			CacheStateSSM:         {Mode: CacheStateFixed, Value: ssm.Value},
		},
	}, nil
}

func buildNormalizedSelectiveScanMixCached(
	builder *tensor.Builder,
	normalized *tensor.Tensor,
	spec Spec,
	weights LayerGraphWeights,
	convState, ssmState *tensor.Tensor,
) (DenseBlockResult, error) {
	if builder == nil || normalized == nil || convState == nil || ssmState == nil {
		return DenseBlockResult{}, errors.New("normalized selective-scan input/state is nil")
	}
	if normalized.Shape.Rank != 2 || normalized.Shape.Dims[0] != uint64(spec.EmbeddingLength) {
		return DenseBlockResult{}, errors.New("normalized selective-scan input is invalid")
	}
	required := graphWeights{
		requireGraphWeight("SSM input", weights.SSMInput),
		requireGraphWeight("SSM convolution", weights.SSMConv1D),
		requireGraphWeight("SSM X", weights.SSMX),
		requireGraphWeight("SSM time-step weight", weights.SSMTimeStepWeight),
		requireGraphWeight("SSM time-step bias", weights.SSMTimeStep),
		requireGraphWeight("SSM time-step norm", weights.SSMTimeStepNorm),
		requireGraphWeight("SSM A", weights.SSMA),
		requireGraphWeight("SSM D", weights.SSMD),
		requireGraphWeight("SSM B norm", weights.SSMBNorm),
		requireGraphWeight("SSM C norm", weights.SSMCNorm),
		requireGraphWeight("SSM output", weights.SSMOutput),
	}
	if err := required.validate("normalized selective-scan mixer"); err != nil {
		return DenseBlockResult{}, err
	}
	inner := uint64(spec.SSMInnerSize)
	stateWidth := uint64(spec.SSMStateSize)
	heads := uint64(spec.SSMTimeStepRank)
	headWidth := inner / heads
	dtWidth := reducedTimeStepWidth(spec.EmbeddingLength)
	convShape := tensor.MustShape(uint64(spec.SSMConvKernel-1), inner)
	ssmShape := tensor.MustShape(stateWidth, inner)
	if !convState.Shape.Equal(convShape) || !ssmState.Shape.Equal(ssmShape) {
		return DenseBlockResult{}, errors.New("normalized selective-scan cache shape is invalid")
	}
	tokens := normalized.Shape.Dims[1]
	zx := builder.MulMat(weights.SSMInput, normalized)
	z := builder.Reshape(builder.GroupSlice(zx, 0, headWidth, heads, 2*headWidth), headWidth, heads, tokens, 1)
	x := builder.Reshape(builder.GroupSlice(zx, headWidth, headWidth, heads, 2*headWidth), inner, tokens)
	convInput := builder.Concat(convState, builder.Transpose2D(x), 0)
	nextConvState := builder.Reshape(
		builder.GroupSlice(convInput, tokens, uint64(spec.SSMConvKernel-1), 1, uint64(spec.SSMConvKernel-1)),
		uint64(spec.SSMConvKernel-1), inner,
	)
	x = builder.SiLU(builder.SSMConv(convInput, weights.SSMConv1D))
	bcdt := builder.MulMat(weights.SSMX, x)
	beta := builder.Reshape(builder.GroupSlice(bcdt, 0, stateWidth, 1, stateWidth), stateWidth, 1, tokens, 1)
	c := builder.Reshape(builder.GroupSlice(bcdt, stateWidth, stateWidth, 1, stateWidth), stateWidth, 1, tokens, 1)
	dt := builder.Reshape(builder.GroupSlice(bcdt, 2*stateWidth, dtWidth, 1, dtWidth), dtWidth, tokens)
	beta = builder.WeightedRMSNorm(beta, weights.SSMBNorm, spec.RMSNormEpsilon)
	c = builder.WeightedRMSNorm(c, weights.SSMCNorm, spec.RMSNormEpsilon)
	dt = builder.WeightedRMSNorm(dt, weights.SSMTimeStepNorm, spec.RMSNormEpsilon)
	dt = builder.Add(builder.MulMat(weights.SSMTimeStepWeight, dt), weights.SSMTimeStep)
	x = builder.Reshape(x, headWidth, heads, tokens, 1)
	packed := builder.SSMScan(
		builder.Reshape(ssmState, stateWidth, headWidth, heads, 1),
		x,
		builder.Reshape(dt, heads, tokens, 1),
		builder.Reshape(weights.SSMA, 1, heads),
		beta,
		c,
	)
	attentionElements := inner * tokens
	mixer := builder.FlatSlice(packed, 0, headWidth, heads, tokens, 1)
	nextSSMState := builder.FlatSlice(packed, attentionElements, stateWidth, inner)
	mixer = builder.Add(mixer, builder.Multiply(x, builder.Reshape(weights.SSMD, 1, heads)))
	mixer = builder.Multiply(mixer, builder.SiLU(z))
	mixer = builder.MulMat(weights.SSMOutput, builder.Reshape(mixer, inner, tokens))
	if err := builder.Err(); err != nil {
		return DenseBlockResult{}, err
	}
	return DenseBlockResult{Output: mixer, Key: nextConvState, Value: nextSSMState}, nil
}

func buildCausalProjectionMixCached(
	builder *tensor.Builder,
	normalized *tensor.Tensor,
	spec Spec,
	weights LayerGraphWeights,
	positions []uint32,
	pastKey, pastValue *tensor.Tensor,
	layerIndex uint32,
) (DenseBlockResult, error) {
	if builder == nil || normalized == nil || normalized.Shape.Rank != 2 {
		return DenseBlockResult{}, errors.New("sparse grouped attention input is invalid")
	}
	if len(positions) == 0 || uint64(len(positions)) != normalized.Shape.Dims[1] {
		return DenseBlockResult{}, errors.New("sparse grouped position count is invalid")
	}
	if err := requireTensorPair(pastKey, pastValue, "sparse grouped cache must contain both tensors"); err != nil {
		return DenseBlockResult{}, err
	}
	if err := (graphWeights{
		requireGraphWeight("attention Q", weights.AttentionQ),
		requireGraphWeight("attention K", weights.AttentionK),
		requireGraphWeight("attention V", weights.AttentionV),
		requireGraphWeight("attention output", weights.AttentionOutput),
	}).validate("sparse grouped attention"); err != nil {
		return DenseBlockResult{}, err
	}
	tokens := normalized.Shape.Dims[1]
	query := builder.MulMat(weights.AttentionQ, normalized)
	key := builder.MulMat(weights.AttentionK, normalized)
	value := builder.MulMat(weights.AttentionV, normalized)
	if weights.AttentionQBias != nil {
		query = builder.Add(query, weights.AttentionQBias)
	}
	if weights.AttentionKBias != nil {
		key = builder.Add(key, weights.AttentionKBias)
	}
	if weights.AttentionVBias != nil {
		value = builder.Add(value, weights.AttentionVBias)
	}
	heads := uint64(spec.LayerHeadCount(layerIndex))
	kvHeads := uint64(spec.LayerKVHeadCount(layerIndex))
	query = builder.Reshape(query, uint64(spec.KeyLength), heads, tokens)
	key = builder.Reshape(key, uint64(spec.KeyLength), kvHeads, tokens)
	value = builder.Reshape(value, uint64(spec.ValueLength), kvHeads, tokens)
	cacheKey, cacheValue := key, value
	var queryStart uint32
	if pastKey != nil {
		if pastKey.Shape.Dims[2] > math.MaxUint32 {
			return DenseBlockResult{}, errors.New("sparse grouped cache token count exceeds uint32")
		}
		queryStart = uint32(pastKey.Shape.Dims[2])
		cacheKey = builder.Concat(pastKey, key, 2)
		cacheValue = builder.Concat(pastValue, value, 2)
	}
	scale := float32(1 / math.Sqrt(float64(spec.KeyLength)))
	if spec.AttentionScale > 0 {
		scale = spec.AttentionScale
	}
	attention := builder.AttentionWithOffset(query, cacheKey, cacheValue, scale, true, queryStart)
	attention = builder.Reshape(attention, heads*uint64(spec.ValueLength), tokens)
	attention = builder.MulMat(weights.AttentionOutput, attention)
	if weights.AttentionOutputBias != nil {
		attention = builder.Add(attention, weights.AttentionOutputBias)
	}
	return DenseBlockResult{Output: attention, Key: cacheKey, Value: cacheValue}, builder.Err()
}

func buildRoutedSquaredReLUFeedForwardMix(
	builder *tensor.Builder,
	normalized *tensor.Tensor,
	weights LayerGraphWeights,
	plan MoEGraphPlan,
) (*tensor.Tensor, error) {
	if builder == nil || normalized == nil || normalized.Shape.Rank != 2 {
		return nil, errors.New("sparse grouped feed-forward input is invalid")
	}
	var feedForward *tensor.Tensor
	if weights.FeedForwardRouter != nil {
		if err := (graphWeights{
			requireGraphWeight("router", weights.FeedForwardRouter),
			requireGraphWeight("expert bias", weights.FeedForwardExpertBias),
			requireGraphWeight("expert up", weights.FeedForwardUpExperts),
			requireGraphWeight("expert down", weights.FeedForwardDownExperts),
			requireGraphWeight("shared up", weights.FeedForwardSharedUp),
			requireGraphWeight("shared down", weights.FeedForwardSharedDown),
		}).validate("sparse grouped MoE"); err != nil {
			return nil, err
		}
		expertInput := normalized
		if weights.FeedForwardLatentDown != nil {
			if weights.FeedForwardLatentUp == nil {
				return nil, errors.New("sparse grouped MoE latent projection is incomplete")
			}
			expertInput = builder.MulMat(weights.FeedForwardLatentDown, normalized)
		} else if weights.FeedForwardLatentUp != nil {
			return nil, errors.New("sparse grouped MoE latent projection is incomplete")
		}
		feedForward = plan.BuildLayer(builder, expertInput, normalized, weights)
		if weights.FeedForwardLatentUp != nil {
			feedForward = builder.MulMat(weights.FeedForwardLatentUp, feedForward)
		}
		shared := builder.MulMat(weights.FeedForwardSharedUp, normalized)
		shared = builder.MulMat(weights.FeedForwardSharedDown, builder.ReLUSquared(shared))
		feedForward = builder.Add(feedForward, shared)
	} else {
		if weights.FeedForwardUp == nil || weights.FeedForwardDown == nil {
			return nil, errors.New("sparse grouped dense FFN catalog is incomplete")
		}
		feedForward = builder.MulMat(weights.FeedForwardUp, normalized)
		if weights.FeedForwardUpBias != nil {
			feedForward = builder.Add(feedForward, weights.FeedForwardUpBias)
		}
		feedForward = builder.MulMat(weights.FeedForwardDown, builder.ReLUSquared(feedForward))
		if weights.FeedForwardDownBias != nil {
			feedForward = builder.Add(feedForward, weights.FeedForwardDownBias)
		}
	}
	return feedForward, builder.Err()
}
