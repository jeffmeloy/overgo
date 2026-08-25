package model

import (
	"errors"

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
	tokens, validInput := tensor.MatrixRows(input.Shape, uint64(spec.EmbeddingLength))
	if !validInput {
		return DenseBlockResult{}, errors.New("selective-scan input shape is invalid")
	}
	required := graphWeights{
		weights.SSMInput,
		weights.SSMConv1D,
		weights.SSMConv1DBias,
		weights.SSMX,
		weights.SSMTimeStepWeight,
		weights.SSMTimeStep,
		weights.SSMA,
		weights.SSMD,
		weights.SSMOutput,
	}
	if err := required.validate("selective-scan mixer"); err != nil {
		return DenseBlockResult{}, err
	}
	if useWeightedStateNorm {
		if err := (graphWeights{
			weights.SSMTimeStepNorm,
			weights.SSMBNorm,
			weights.SSMCNorm,
		}).validate("weighted selective-scan mixer"); err != nil {
			return DenseBlockResult{}, err
		}
	}
	window := spec.ssmConvolutionWindow()
	if !tensor.HasDimensions(convState.Shape, window, uint64(spec.SSMInnerSize)) ||
		!tensor.HasDimensions(ssmState.Shape, uint64(spec.SSMStateSize), uint64(spec.SSMInnerSize)) {
		return DenseBlockResult{}, errors.New("selective-scan recurrent cache shape is invalid")
	}
	inner := uint64(spec.SSMInnerSize)
	stateWidth := uint64(spec.SSMStateSize)
	rank := uint64(spec.SSMTimeStepRank)
	xz := builder.MulMat(weights.SSMInput, input)
	x := builder.Reshape(builder.GroupSlice(
		xz, tensor.FirstOffset, inner, tensor.SingletonExtent, inner,
	), inner, tokens)
	z := builder.Reshape(builder.GroupSlice(
		xz, inner, inner, tensor.SingletonExtent, inner,
	), inner, tokens)
	convInput := builder.Concat(convState, builder.Transpose2D(x), tensor.FirstOffset)
	nextConvState := builder.Reshape(
		builder.GroupSlice(convInput, tokens, window, tensor.SingletonExtent, window),
		window, inner,
	)
	x = builder.SiLU(builder.Add(builder.SSMConv(convInput, weights.SSMConv1D), weights.SSMConv1DBias))
	xdb := builder.MulMat(weights.SSMX, x)
	dt := builder.Reshape(builder.GroupSlice(
		xdb, tensor.FirstOffset, rank, tensor.SingletonExtent, rank,
	), rank, tokens)
	beta := builder.Reshape(
		builder.GroupSlice(xdb, rank, stateWidth, tensor.SingletonExtent, stateWidth),
		stateWidth, tensor.SingletonExtent, tokens, tensor.SingletonExtent,
	)
	c := builder.Reshape(
		builder.GroupSlice(xdb, rank+stateWidth, stateWidth, tensor.SingletonExtent, stateWidth),
		stateWidth, tensor.SingletonExtent, tokens, tensor.SingletonExtent,
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
		builder.Reshape(ssmState, stateWidth, tensor.SingletonExtent, inner, tensor.SingletonExtent),
		builder.Reshape(x, tensor.SingletonExtent, inner, tokens, tensor.SingletonExtent),
		builder.Reshape(dt, inner, tokens, tensor.SingletonExtent),
		weights.SSMA,
		beta,
		c,
	)
	attentionElements := inner * tokens
	attention := builder.FlatSlice(packed, tensor.FirstOffset, inner, tokens)
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
	tokens, validInput := tensor.MatrixRows(input.Shape, uint64(spec.EmbeddingLength))
	if !validInput {
		return DenseBlockResult{}, errors.New("grouped selective-scan input shape is invalid")
	}
	required := graphWeights{
		weights.SSMInput,
		weights.SSMConv1D,
		weights.SSMTimeStep,
		weights.SSMA,
		weights.SSMD,
		weights.SSMOutput,
	}
	if mixer != recurrentMixerAttentionGroupedSelectiveScan {
		required = append(required, weights.SSMNorm)
	}
	if err := required.validate("grouped selective-scan mixer"); err != nil {
		return DenseBlockResult{}, err
	}
	inner := uint64(spec.SSMInnerSize)
	stateWidth := uint64(spec.SSMStateSize)
	heads := uint64(spec.SSMTimeStepRank)
	groups := uint64(spec.SSMGroupCount)
	headWidth := inner / heads
	convWidth := inner + tensor.PairedExtent*groups*stateWidth
	window := spec.ssmConvolutionWindow()
	if !tensor.HasDimensions(convState.Shape, window, convWidth) ||
		!tensor.HasDimensions(ssmState.Shape, stateWidth, inner) {
		return DenseBlockResult{}, errors.New("grouped selective-scan cache shape is invalid")
	}
	zxBCdt := builder.MulMat(weights.SSMInput, input)
	z := builder.Reshape(builder.GroupSlice(
		zxBCdt, tensor.FirstOffset, headWidth, heads, headWidth,
	), headWidth, heads, tokens, tensor.SingletonExtent)
	xBC := builder.Reshape(builder.GroupSlice(
		zxBCdt, inner, convWidth, tensor.SingletonExtent, convWidth,
	), convWidth, tokens)
	dt := builder.Reshape(
		builder.GroupSlice(zxBCdt, inner+convWidth, heads, tensor.SingletonExtent, heads),
		heads, tokens, tensor.SingletonExtent,
	)
	convInput := builder.Concat(convState, builder.Transpose2D(xBC), tensor.FirstOffset)
	nextConvState := builder.Reshape(
		builder.GroupSlice(convInput, tokens, window, tensor.SingletonExtent, window),
		window, convWidth,
	)
	xBC = builder.SSMConv(convInput, weights.SSMConv1D)
	if weights.SSMConv1DBias != nil {
		xBC = builder.Add(xBC, weights.SSMConv1DBias)
	}
	xBC = builder.SiLU(xBC)
	x := builder.Reshape(builder.GroupSlice(
		xBC, tensor.FirstOffset, headWidth, heads, headWidth,
	), headWidth, heads, tokens, tensor.SingletonExtent)
	beta := builder.Reshape(
		builder.GroupSlice(xBC, inner, stateWidth, groups, stateWidth),
		stateWidth, groups, tokens, tensor.SingletonExtent,
	)
	c := builder.Reshape(
		builder.GroupSlice(xBC, inner+groups*stateWidth, stateWidth, groups, stateWidth),
		stateWidth, groups, tokens, tensor.SingletonExtent,
	)
	dt = builder.Add(dt, weights.SSMTimeStep)
	packed := builder.SSMScan(
		builder.Reshape(ssmState, stateWidth, headWidth, heads, tensor.SingletonExtent),
		x,
		dt,
		weights.SSMA,
		beta,
		c,
	)
	attentionElements := inner * tokens
	attention := builder.FlatSlice(packed, tensor.FirstOffset, headWidth, heads, tokens, tensor.SingletonExtent)
	nextSSMState := builder.FlatSlice(packed, attentionElements, stateWidth, inner)
	attention = builder.Add(attention, builder.Multiply(x, weights.SSMD))
	attention = builder.Multiply(attention, builder.SiLU(z))
	attention = builder.Reshape(attention, inner/groups, groups, tokens, tensor.SingletonExtent)
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
	tokens, validInput := tensor.MatrixRows(normalized.Shape, uint64(spec.EmbeddingLength))
	if !validInput {
		return DenseBlockResult{}, errors.New("hybrid attention-scan input is invalid")
	}
	if uint64(len(positions)) != tokens {
		return DenseBlockResult{}, errors.New("hybrid attention-scan position count is invalid")
	}
	if err := requireTensorPair(pastKey, pastValue, "hybrid attention-scan KV cache is incomplete"); err != nil {
		return DenseBlockResult{}, err
	}
	if err := (graphWeights{
		weights.AttentionOutput,
	}).validate("hybrid attention-scan"); err != nil {
		return DenseBlockResult{}, err
	}
	if weights.AttentionQKV == nil && (weights.AttentionQ == nil || weights.AttentionK == nil || weights.AttentionV == nil) {
		return DenseBlockResult{}, errors.New("hybrid attention projection catalog is incomplete")
	}
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
		query = builder.RoPEWithOptions(query, tensor.RoPEOptions{Layout: tensor.RoPELayoutNeoX, Positions: positions, FrequencyFactors: weights.RopeFactors, RotaryDimensions: spec.RopeDimensionCount, FrequencyBase: spec.RopeFrequencyBase, FrequencyScale: tensor.UnitFrequencyScale})
		key = builder.RoPEWithOptions(key, tensor.RoPEOptions{Layout: tensor.RoPELayoutNeoX, Positions: positions, FrequencyFactors: weights.RopeFactors, RotaryDimensions: spec.RopeDimensionCount, FrequencyBase: spec.RopeFrequencyBase, FrequencyScale: tensor.UnitFrequencyScale})
	} else {
		query = builder.RoPEWithOptions(query, tensor.RoPEOptions{Layout: tensor.RoPELayoutNeoX, Positions: positions, RotaryDimensions: spec.RopeDimensionCount, FrequencyBase: spec.RopeFrequencyBase, FrequencyScale: tensor.UnitFrequencyScale})
		key = builder.RoPEWithOptions(key, tensor.RoPEOptions{Layout: tensor.RoPELayoutNeoX, Positions: positions, RotaryDimensions: spec.RopeDimensionCount, FrequencyBase: spec.RopeFrequencyBase, FrequencyScale: tensor.UnitFrequencyScale})
	}
	cacheKey, cacheValue := key, value
	var queryStart uint32
	if pastKey != nil {
		pastTokens, keyAxis, validKey := tensor.TrailingExtent32(pastKey.Shape, uint64(spec.KeyLength), kvHeads)
		valueTokens, valueAxis, validValue := tensor.TrailingExtent32(pastValue.Shape, uint64(spec.ValueLength), kvHeads)
		if !validKey || !validValue || pastTokens != valueTokens {
			return DenseBlockResult{}, errors.New("hybrid attention-scan KV cache shape is invalid")
		}
		queryStart = pastTokens
		cacheKey = builder.Concat(pastKey, key, keyAxis)
		cacheValue = builder.Concat(pastValue, value, valueAxis)
	}
	attentionScale := spec.resolvedAttentionScale(uint64(spec.KeyLength))
	attention := builder.AttentionWithOptions(query, cacheKey, cacheValue, tensor.AttentionOptions{Scale: attentionScale, Causal: true, QueryStart: queryStart})
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
	tokens, validInput := tensor.MatrixRows(normalized.Shape, uint64(spec.EmbeddingLength))
	if !validInput {
		return DenseBlockResult{}, errors.New("normalized selective-scan input is invalid")
	}
	required := graphWeights{
		weights.SSMInput,
		weights.SSMConv1D,
		weights.SSMX,
		weights.SSMTimeStepWeight,
		weights.SSMTimeStep,
		weights.SSMTimeStepNorm,
		weights.SSMA,
		weights.SSMD,
		weights.SSMBNorm,
		weights.SSMCNorm,
		weights.SSMOutput,
	}
	if err := required.validate("normalized selective-scan mixer"); err != nil {
		return DenseBlockResult{}, err
	}
	inner := uint64(spec.SSMInnerSize)
	stateWidth := uint64(spec.SSMStateSize)
	heads := uint64(spec.SSMTimeStepRank)
	headWidth := inner / heads
	dtWidth, validTimeStep := tensor.VectorWidth(weights.SSMTimeStepNorm.Shape)
	if !validTimeStep {
		return DenseBlockResult{}, errors.New("normalized selective-scan time-step norm is not a vector")
	}
	window := spec.ssmConvolutionWindow()
	if !tensor.HasDimensions(convState.Shape, window, inner) ||
		!tensor.HasDimensions(ssmState.Shape, stateWidth, inner) {
		return DenseBlockResult{}, errors.New("normalized selective-scan cache shape is invalid")
	}
	zx := builder.MulMat(weights.SSMInput, normalized)
	stride := tensor.PairedExtent * headWidth
	z := builder.Reshape(builder.GroupSlice(
		zx, tensor.FirstOffset, headWidth, heads, stride,
	), headWidth, heads, tokens, tensor.SingletonExtent)
	x := builder.Reshape(builder.GroupSlice(
		zx, headWidth, headWidth, heads, stride,
	), inner, tokens)
	convInput := builder.Concat(convState, builder.Transpose2D(x), tensor.FirstOffset)
	nextConvState := builder.Reshape(
		builder.GroupSlice(convInput, tokens, window, tensor.SingletonExtent, window),
		window, inner,
	)
	x = builder.SiLU(builder.SSMConv(convInput, weights.SSMConv1D))
	bcdt := builder.MulMat(weights.SSMX, x)
	beta := builder.Reshape(builder.GroupSlice(
		bcdt, tensor.FirstOffset, stateWidth, tensor.SingletonExtent, stateWidth,
	), stateWidth, tensor.SingletonExtent, tokens, tensor.SingletonExtent)
	c := builder.Reshape(builder.GroupSlice(
		bcdt, stateWidth, stateWidth, tensor.SingletonExtent, stateWidth,
	), stateWidth, tensor.SingletonExtent, tokens, tensor.SingletonExtent)
	dt := builder.Reshape(builder.GroupSlice(
		bcdt, tensor.PairedExtent*stateWidth, dtWidth, tensor.SingletonExtent, dtWidth,
	), dtWidth, tokens)
	beta = builder.WeightedRMSNorm(beta, weights.SSMBNorm, spec.RMSNormEpsilon)
	c = builder.WeightedRMSNorm(c, weights.SSMCNorm, spec.RMSNormEpsilon)
	dt = builder.WeightedRMSNorm(dt, weights.SSMTimeStepNorm, spec.RMSNormEpsilon)
	dt = builder.Add(builder.MulMat(weights.SSMTimeStepWeight, dt), weights.SSMTimeStep)
	x = builder.Reshape(x, headWidth, heads, tokens, tensor.SingletonExtent)
	packed := builder.SSMScan(
		builder.Reshape(ssmState, stateWidth, headWidth, heads, tensor.SingletonExtent),
		x,
		builder.Reshape(dt, heads, tokens, tensor.SingletonExtent),
		builder.Reshape(weights.SSMA, tensor.SingletonExtent, heads),
		beta,
		c,
	)
	attentionElements := inner * tokens
	mixer := builder.FlatSlice(packed, tensor.FirstOffset, headWidth, heads, tokens, tensor.SingletonExtent)
	nextSSMState := builder.FlatSlice(packed, attentionElements, stateWidth, inner)
	mixer = builder.Add(mixer, builder.Multiply(
		x, builder.Reshape(weights.SSMD, tensor.SingletonExtent, heads),
	))
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
	if builder == nil || normalized == nil {
		return DenseBlockResult{}, errors.New("sparse grouped attention input is invalid")
	}
	tokens, validInput := tensor.MatrixRows(normalized.Shape, uint64(spec.EmbeddingLength))
	if !validInput || uint64(len(positions)) != tokens {
		return DenseBlockResult{}, errors.New("sparse grouped position count is invalid")
	}
	if err := requireTensorPair(pastKey, pastValue, "sparse grouped cache must contain both tensors"); err != nil {
		return DenseBlockResult{}, err
	}
	if err := (graphWeights{
		weights.AttentionQ,
		weights.AttentionK,
		weights.AttentionV,
		weights.AttentionOutput,
	}).validate("sparse grouped attention"); err != nil {
		return DenseBlockResult{}, err
	}
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
		pastTokens, keyAxis, validKey := tensor.TrailingExtent32(pastKey.Shape, uint64(spec.KeyLength), kvHeads)
		valueTokens, valueAxis, validValue := tensor.TrailingExtent32(pastValue.Shape, uint64(spec.ValueLength), kvHeads)
		if !validKey || !validValue || pastTokens != valueTokens {
			return DenseBlockResult{}, errors.New("sparse grouped cache shape is invalid")
		}
		queryStart = pastTokens
		cacheKey = builder.Concat(pastKey, key, keyAxis)
		cacheValue = builder.Concat(pastValue, value, valueAxis)
	}
	scale := spec.resolvedAttentionScale(uint64(spec.KeyLength))
	attention := builder.AttentionWithOptions(query, cacheKey, cacheValue, tensor.AttentionOptions{Scale: scale, Causal: true, QueryStart: queryStart})
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
	if builder == nil || normalized == nil {
		return nil, errors.New("sparse grouped feed-forward input is invalid")
	}
	if _, _, validInput := tensor.MatrixExtents(normalized.Shape); !validInput {
		return nil, errors.New("sparse grouped feed-forward input is invalid")
	}
	var feedForward *tensor.Tensor
	if weights.FeedForwardRouter != nil {
		if err := (graphWeights{
			weights.FeedForwardRouter,
			weights.FeedForwardExpertBias,
			weights.FeedForwardUpExperts,
			weights.FeedForwardDownExperts,
			weights.FeedForwardSharedUp,
			weights.FeedForwardSharedDown,
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
