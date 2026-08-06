package model

import (
	"errors"
	"math"

	"llamacpp2go/internal/tensor"
)

func BuildMambaBlockCached(
	builder *tensor.Builder,
	input *tensor.Tensor,
	spec Spec,
	weights LayerGraphWeights,
	convState, ssmState *tensor.Tensor,
) (DenseBlockResult, error) {
	if builder == nil || input == nil || convState == nil || ssmState == nil {
		return DenseBlockResult{}, errors.New("Mamba block input/state is nil")
	}
	profile := spec.Profile()
	if (profile.Block != BlockMamba && profile.RecurrentBlock != BlockJamba) || input.Shape.Rank != 2 ||
		input.Shape.Dims[0] != uint64(spec.EmbeddingLength) {
		return DenseBlockResult{}, errors.New("Mamba block architecture/input is invalid")
	}
	required := graphWeights{
		requireGraphWeight("attention norm", weights.AttentionNorm),
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
	if err := required.validate("Mamba block"); err != nil {
		return DenseBlockResult{}, err
	}
	if profile.RecurrentBlock == BlockJamba {
		if err := (graphWeights{
			requireGraphWeight("SSM time-step norm", weights.SSMTimeStepNorm),
			requireGraphWeight("SSM B norm", weights.SSMBNorm),
			requireGraphWeight("SSM C norm", weights.SSMCNorm),
		}).validate("Jamba block"); err != nil {
			return DenseBlockResult{}, err
		}
	}
	convShape := tensor.MustShape(uint64(spec.SSMConvKernel-1), uint64(spec.SSMInnerSize))
	ssmShape := tensor.MustShape(uint64(spec.SSMStateSize), uint64(spec.SSMInnerSize))
	if !convState.Shape.Equal(convShape) || !ssmState.Shape.Equal(ssmShape) {
		return DenseBlockResult{}, errors.New("Mamba recurrent cache shape is invalid")
	}
	tokens := input.Shape.Dims[1]
	inner := uint64(spec.SSMInnerSize)
	stateWidth := uint64(spec.SSMStateSize)
	rank := uint64(spec.SSMTimeStepRank)
	normalized := builder.WeightedRMSNorm(input, weights.AttentionNorm, spec.RMSNormEpsilon)
	xz := builder.MulMat(weights.SSMInput, normalized)
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
	} else if profile.RecurrentBlock == BlockJamba {
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
	output := builder.Add(input, attention)
	if err := builder.Err(); err != nil {
		return DenseBlockResult{}, err
	}
	return DenseBlockResult{Output: output, Key: nextConvState, Value: nextSSMState}, nil
}

func BuildJambaRecurrentBlockCached(
	builder *tensor.Builder,
	input *tensor.Tensor,
	spec Spec,
	weights LayerGraphWeights,
	convState, ssmState *tensor.Tensor,
) (DenseBlockResult, error) {
	if spec.Profile().RecurrentBlock != BlockJamba {
		return DenseBlockResult{}, errors.New("Jamba recurrent block architecture is invalid")
	}
	result, err := BuildMambaBlockCached(builder, input, spec, weights, convState, ssmState)
	if err != nil {
		return DenseBlockResult{}, err
	}
	if weights.FeedForwardNorm == nil {
		return DenseBlockResult{}, errors.New("Jamba feed-forward norm is nil")
	}
	normalized := builder.WeightedRMSNorm(result.Output, weights.FeedForwardNorm, spec.RMSNormEpsilon)
	var feedForward *tensor.Tensor
	if weights.FeedForwardRouter != nil {
		if weights.FeedForwardGateExperts == nil || weights.FeedForwardUpExperts == nil ||
			weights.FeedForwardDownExperts == nil {
			return DenseBlockResult{}, errors.New("Jamba expert catalog is incomplete")
		}
		plan := spec.moeGraphPlan(0)
		plan.NormalizeTopKProb = false
		feedForward = plan.BuildLayer(builder, normalized, nil, weights)
	} else {
		if weights.FeedForwardGate == nil || weights.FeedForwardUp == nil || weights.FeedForwardDown == nil {
			return DenseBlockResult{}, errors.New("Jamba dense FFN catalog is incomplete")
		}
		gate := builder.MulMat(weights.FeedForwardGate, normalized)
		up := builder.MulMat(weights.FeedForwardUp, normalized)
		feedForward = builder.MulMat(weights.FeedForwardDown, builder.SwiGLU(gate, up))
	}
	result.Output = builder.Add(result.Output, feedForward)
	if err := builder.Err(); err != nil {
		return DenseBlockResult{}, err
	}
	return result, nil
}

func buildMamba2MixerCached(
	builder *tensor.Builder,
	input *tensor.Tensor,
	spec Spec,
	weights LayerGraphWeights,
	convState, ssmState *tensor.Tensor,
) (DenseBlockResult, error) {
	if builder == nil || input == nil || convState == nil || ssmState == nil {
		return DenseBlockResult{}, errors.New("Mamba2 block input/state is nil")
	}
	profile := spec.Profile()
	if (profile.Block != BlockMamba2 && profile.RecurrentBlock != BlockGraniteHybrid &&
		profile.Block != BlockFalconH1 && profile.Block != BlockNemotronH) || input.Shape.Rank != 2 ||
		input.Shape.Dims[0] != uint64(spec.EmbeddingLength) {
		return DenseBlockResult{}, errors.New("Mamba2 mixer architecture/input is invalid")
	}
	required := graphWeights{
		requireGraphWeight("attention norm", weights.AttentionNorm),
		requireGraphWeight("SSM input", weights.SSMInput),
		requireGraphWeight("SSM convolution", weights.SSMConv1D),
		requireGraphWeight("SSM time-step bias", weights.SSMTimeStep),
		requireGraphWeight("SSM A", weights.SSMA),
		requireGraphWeight("SSM D", weights.SSMD),
		requireGraphWeight("SSM output", weights.SSMOutput),
	}
	if profile.Block != BlockFalconH1 {
		required.add("SSM norm", weights.SSMNorm)
	}
	if err := required.validate("Mamba2 block"); err != nil {
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
		return DenseBlockResult{}, errors.New("Mamba2 recurrent cache shape is invalid")
	}
	tokens := input.Shape.Dims[1]
	normalized := builder.WeightedRMSNorm(input, weights.AttentionNorm, spec.RMSNormEpsilon)
	zxBCdt := builder.MulMat(weights.SSMInput, normalized)
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

func BuildFalconH1BlockCached(
	builder *tensor.Builder,
	input *tensor.Tensor,
	spec Spec,
	weights LayerGraphWeights,
	positions []uint32,
	pastKey, pastValue, convState, ssmState *tensor.Tensor,
) (DenseBlockResult, error) {
	if builder == nil || input == nil || convState == nil || ssmState == nil {
		return DenseBlockResult{}, errors.New("Falcon-H1 block input/state is nil")
	}
	if spec.Profile().Block != BlockFalconH1 || input.Shape.Rank != 2 ||
		input.Shape.Dims[0] != uint64(spec.EmbeddingLength) {
		return DenseBlockResult{}, errors.New("Falcon-H1 block architecture/input is invalid")
	}
	if len(positions) == 0 || uint64(len(positions)) != input.Shape.Dims[1] {
		return DenseBlockResult{}, errors.New("Falcon-H1 block position count is invalid")
	}
	if err := requireTensorPair(pastKey, pastValue, "Falcon-H1 KV cache is incomplete"); err != nil {
		return DenseBlockResult{}, err
	}
	if err := (graphWeights{
		requireGraphWeight("attention norm", weights.AttentionNorm),
		requireGraphWeight("attention output", weights.AttentionOutput),
		requireGraphWeight("feed-forward norm", weights.FeedForwardNorm),
		requireGraphWeight("feed-forward gate", weights.FeedForwardGate),
		requireGraphWeight("feed-forward up", weights.FeedForwardUp),
		requireGraphWeight("feed-forward down", weights.FeedForwardDown),
	}).validate("Falcon-H1 block"); err != nil {
		return DenseBlockResult{}, err
	}
	if weights.AttentionQKV == nil && (weights.AttentionQ == nil || weights.AttentionK == nil || weights.AttentionV == nil) {
		return DenseBlockResult{}, errors.New("Falcon-H1 attention projection catalog is incomplete")
	}

	tokens := uint64(len(positions))
	heads := uint64(spec.HeadCount)
	kvHeads := uint64(spec.HeadCountKV)
	queryWidth := heads * uint64(spec.KeyLength)
	keyWidth := kvHeads * uint64(spec.KeyLength)
	valueWidth := kvHeads * uint64(spec.ValueLength)
	normalized := builder.WeightedRMSNorm(input, weights.AttentionNorm, spec.RMSNormEpsilon)
	var query, key, value *tensor.Tensor
	if weights.AttentionQKV != nil {
		mixed := builder.MulMat(weights.AttentionQKV, normalized)
		if weights.AttentionQKVBias != nil {
			mixed = builder.Add(mixed, weights.AttentionQKVBias)
		}
		stride := queryWidth + keyWidth + valueWidth
		query = builder.Reshape(builder.GroupSlice(mixed, 0, queryWidth, 1, stride), queryWidth, tokens)
		key = builder.Reshape(builder.GroupSlice(mixed, queryWidth, keyWidth, 1, stride), keyWidth, tokens)
		value = builder.Reshape(builder.GroupSlice(mixed, queryWidth+keyWidth, valueWidth, 1, stride), valueWidth, tokens)
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
	query = builder.Reshape(query, uint64(spec.KeyLength), heads, tokens)
	key = builder.Reshape(key, uint64(spec.KeyLength), kvHeads, tokens)
	value = builder.Reshape(value, uint64(spec.ValueLength), kvHeads, tokens)
	if weights.RopeFactors != nil {
		query = builder.RoPENeoXScaledWithFactors(query, positions, spec.RopeDimensionCount, spec.RopeFrequencyBase, 1, weights.RopeFactors)
		key = builder.RoPENeoXScaledWithFactors(key, positions, spec.RopeDimensionCount, spec.RopeFrequencyBase, 1, weights.RopeFactors)
	} else {
		query = builder.RoPENeoXScaled(query, positions, spec.RopeDimensionCount, spec.RopeFrequencyBase, 1)
		key = builder.RoPENeoXScaled(key, positions, spec.RopeDimensionCount, spec.RopeFrequencyBase, 1)
	}
	cacheKey, cacheValue := key, value
	var queryStart uint32
	if pastKey != nil {
		if pastKey.Shape.Rank != 3 || pastValue.Shape.Rank != 3 || pastKey.Shape.Dims[2] > math.MaxUint32 {
			return DenseBlockResult{}, errors.New("Falcon-H1 KV cache shape is invalid")
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

	ssm, err := buildMamba2MixerCached(builder, input, spec, weights, convState, ssmState)
	if err != nil {
		return DenseBlockResult{}, err
	}
	residual := builder.Add(input, builder.Add(attention, ssm.Output))
	ffnInput := builder.WeightedRMSNorm(residual, weights.FeedForwardNorm, spec.RMSNormEpsilon)
	gate := builder.MulMat(weights.FeedForwardGate, ffnInput)
	up := builder.MulMat(weights.FeedForwardUp, ffnInput)
	if weights.FeedForwardGateBias != nil {
		gate = builder.Add(gate, weights.FeedForwardGateBias)
	}
	if weights.FeedForwardUpBias != nil {
		up = builder.Add(up, weights.FeedForwardUpBias)
	}
	feedForward := builder.MulMat(weights.FeedForwardDown, builder.SwiGLU(gate, up))
	if weights.FeedForwardDownBias != nil {
		feedForward = builder.Add(feedForward, weights.FeedForwardDownBias)
	}
	output := builder.Add(residual, feedForward)
	if err := builder.Err(); err != nil {
		return DenseBlockResult{}, err
	}
	return DenseBlockResult{
		Output: output, Key: cacheKey, Value: cacheValue,
		States: CacheStates[*tensor.Tensor]{
			CacheStateConvolution: {Mode: CacheStateFixed, Value: ssm.Key},
			CacheStateSSM:         {Mode: CacheStateFixed, Value: ssm.Value},
		},
	}, nil
}

func BuildMamba2BlockCached(
	builder *tensor.Builder,
	input *tensor.Tensor,
	spec Spec,
	weights LayerGraphWeights,
	convState, ssmState *tensor.Tensor,
) (DenseBlockResult, error) {
	if spec.Profile().Block != BlockMamba2 {
		return DenseBlockResult{}, errors.New("Mamba2 block architecture is invalid")
	}
	if weights.SSMConv1DBias == nil {
		return DenseBlockResult{}, errors.New("Mamba2 block SSM convolution bias is nil")
	}
	result, err := buildMamba2MixerCached(builder, input, spec, weights, convState, ssmState)
	if err != nil {
		return DenseBlockResult{}, err
	}
	result.Output = builder.Add(input, result.Output)
	if err := builder.Err(); err != nil {
		return DenseBlockResult{}, err
	}
	return result, nil
}

func BuildGraniteHybridRecurrentBlockCached(
	builder *tensor.Builder,
	input *tensor.Tensor,
	spec Spec,
	weights LayerGraphWeights,
	convState, ssmState *tensor.Tensor,
) (DenseBlockResult, error) {
	if spec.Profile().RecurrentBlock != BlockGraniteHybrid {
		return DenseBlockResult{}, errors.New("Granite Hybrid recurrent block architecture is invalid")
	}
	result, err := buildMamba2MixerCached(builder, input, spec, weights, convState, ssmState)
	if err != nil {
		return DenseBlockResult{}, err
	}
	mixer := result.Output
	if spec.ResidualScale > 0 {
		mixer = builder.Scale(mixer, spec.ResidualScale)
	}
	residual := builder.Add(input, mixer)
	if weights.FeedForwardNorm == nil {
		return DenseBlockResult{}, errors.New("Granite Hybrid feed-forward norm is nil")
	}
	normalized := builder.WeightedRMSNorm(residual, weights.FeedForwardNorm, spec.RMSNormEpsilon)
	var feedForward *tensor.Tensor
	if weights.FeedForwardRouter != nil {
		if weights.FeedForwardUpExperts == nil || weights.FeedForwardDownExperts == nil {
			return DenseBlockResult{}, errors.New("Granite Hybrid expert catalog is incomplete")
		}
		feedForward = spec.moeGraphPlan(0).BuildLayer(builder, normalized, nil, weights)
		if spec.SharedExpertFF > 0 {
			if weights.FeedForwardSharedGate == nil || weights.FeedForwardSharedUp == nil ||
				weights.FeedForwardSharedDown == nil {
				return DenseBlockResult{}, errors.New("Granite Hybrid shared expert catalog is incomplete")
			}
			shared := buildSharedSwiGLU(builder, normalized, weights)
			feedForward = builder.Add(feedForward, shared)
		}
	} else {
		if weights.FeedForwardGate == nil || weights.FeedForwardUp == nil || weights.FeedForwardDown == nil {
			return DenseBlockResult{}, errors.New("Granite Hybrid dense FFN catalog is incomplete")
		}
		gate := builder.MulMat(weights.FeedForwardGate, normalized)
		up := builder.MulMat(weights.FeedForwardUp, normalized)
		if weights.FeedForwardGateBias != nil {
			gate = builder.Add(gate, weights.FeedForwardGateBias)
		}
		if weights.FeedForwardUpBias != nil {
			up = builder.Add(up, weights.FeedForwardUpBias)
		}
		feedForward = builder.MulMat(weights.FeedForwardDown, builder.SwiGLU(gate, up))
		if weights.FeedForwardDownBias != nil {
			feedForward = builder.Add(feedForward, weights.FeedForwardDownBias)
		}
	}
	if spec.ResidualScale > 0 {
		feedForward = builder.Scale(feedForward, spec.ResidualScale)
	}
	result.Output = builder.Add(residual, feedForward)
	if err := builder.Err(); err != nil {
		return DenseBlockResult{}, err
	}
	return result, nil
}

func BuildPLaMo2RecurrentBlockCached(
	builder *tensor.Builder,
	input *tensor.Tensor,
	spec Spec,
	weights LayerGraphWeights,
	convState, ssmState *tensor.Tensor,
) (DenseBlockResult, error) {
	if builder == nil || input == nil || convState == nil || ssmState == nil {
		return DenseBlockResult{}, errors.New("PLaMo2 block input/state is nil")
	}
	if spec.Profile().RecurrentBlock != BlockPLaMo2 || input.Shape.Rank != 2 ||
		input.Shape.Dims[0] != uint64(spec.EmbeddingLength) {
		return DenseBlockResult{}, errors.New("PLaMo2 block architecture/input is invalid")
	}
	required := graphWeights{
		requireGraphWeight("attention norm", weights.AttentionNorm),
		requireGraphWeight("attention post norm", weights.AttentionPostNorm),
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
		requireGraphWeight("feed-forward norm", weights.FeedForwardNorm),
		requireGraphWeight("feed-forward up", weights.FeedForwardUp),
		requireGraphWeight("feed-forward down", weights.FeedForwardDown),
		requireGraphWeight("feed-forward post norm", weights.FeedForwardPostNorm),
	}
	if err := required.validate("PLaMo2 block"); err != nil {
		return DenseBlockResult{}, err
	}
	inner := uint64(spec.SSMInnerSize)
	stateWidth := uint64(spec.SSMStateSize)
	heads := uint64(spec.SSMTimeStepRank)
	headWidth := inner / heads
	dtWidth := plamo2TimeStepWidth(spec.EmbeddingLength)
	convShape := tensor.MustShape(uint64(spec.SSMConvKernel-1), inner)
	ssmShape := tensor.MustShape(stateWidth, inner)
	if !convState.Shape.Equal(convShape) || !ssmState.Shape.Equal(ssmShape) {
		return DenseBlockResult{}, errors.New("PLaMo2 recurrent cache shape is invalid")
	}
	tokens := input.Shape.Dims[1]
	normalized := builder.WeightedRMSNorm(input, weights.AttentionNorm, spec.RMSNormEpsilon)
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
	mixer = builder.WeightedRMSNorm(mixer, weights.AttentionPostNorm, spec.RMSNormEpsilon)
	residual := builder.Add(input, mixer)
	normalized = builder.WeightedRMSNorm(residual, weights.FeedForwardNorm, spec.RMSNormEpsilon)
	fused := builder.MulMat(weights.FeedForwardUp, normalized)
	gate := builder.Reshape(builder.GroupSlice(fused, 0, uint64(spec.FeedForwardLength), 1, 2*uint64(spec.FeedForwardLength)), uint64(spec.FeedForwardLength), tokens)
	up := builder.Reshape(builder.GroupSlice(fused, uint64(spec.FeedForwardLength), uint64(spec.FeedForwardLength), 1, 2*uint64(spec.FeedForwardLength)), uint64(spec.FeedForwardLength), tokens)
	feedForward := builder.MulMat(weights.FeedForwardDown, builder.SwiGLU(gate, up))
	feedForward = builder.WeightedRMSNorm(feedForward, weights.FeedForwardPostNorm, spec.RMSNormEpsilon)
	output := builder.Add(residual, feedForward)
	if err := builder.Err(); err != nil {
		return DenseBlockResult{}, err
	}
	return DenseBlockResult{Output: output, Key: nextConvState, Value: nextSSMState}, nil
}

// BuildNemotronHBlockCached: attention, Mamba2, or FFN mixer.
func BuildNemotronHBlockCached(
	builder *tensor.Builder,
	input *tensor.Tensor,
	spec Spec,
	weights LayerGraphWeights,
	positions []uint32,
	pastKey, pastValue *tensor.Tensor,
	layerIndex uint32,
) (DenseBlockResult, error) {
	if builder == nil || input == nil || input.Shape.Rank != 2 ||
		spec.Profile().Block != BlockNemotronH {
		return DenseBlockResult{}, errors.New("Nemotron-H block architecture/input is invalid")
	}
	if weights.AttentionNorm == nil {
		return DenseBlockResult{}, errors.New("Nemotron-H block norm is nil")
	}
	if len(positions) == 0 || uint64(len(positions)) != input.Shape.Dims[1] {
		return DenseBlockResult{}, errors.New("Nemotron-H position count is invalid")
	}
	if err := requireTensorPair(pastKey, pastValue, "Nemotron-H cache must contain both tensors"); err != nil {
		return DenseBlockResult{}, err
	}
	if spec.IsRecurrentLayer(layerIndex) {
		result, err := buildMamba2MixerCached(builder, input, spec, weights, pastKey, pastValue)
		if err != nil {
			return DenseBlockResult{}, err
		}
		result.Output = builder.Add(input, result.Output)
		return result, builder.Err()
	}
	normalized := builder.WeightedRMSNorm(input, weights.AttentionNorm, spec.RMSNormEpsilon)
	tokens := input.Shape.Dims[1]
	if spec.LayerFeedForwardLength(layerIndex) == 0 {
		if err := (graphWeights{
			requireGraphWeight("attention Q", weights.AttentionQ),
			requireGraphWeight("attention K", weights.AttentionK),
			requireGraphWeight("attention V", weights.AttentionV),
			requireGraphWeight("attention output", weights.AttentionOutput),
		}).validate("Nemotron-H"); err != nil {
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
			if pastKey.Shape.Dims[2] > math.MaxUint32 {
				return DenseBlockResult{}, errors.New("Nemotron-H cache token count exceeds uint32")
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
		return DenseBlockResult{Output: builder.Add(input, attention), Key: cacheKey, Value: cacheValue}, builder.Err()
	}
	sentinel := builder.GroupSlice(input, 0, 1, 1, input.Shape.Dims[0])
	cacheKey, cacheValue := sentinel, sentinel
	if pastKey != nil {
		wantPrefix := tensor.MustShape(1, 1, pastKey.Shape.Dims[2])
		if !pastKey.Shape.Equal(wantPrefix) || !pastValue.Shape.Equal(wantPrefix) {
			return DenseBlockResult{}, errors.New("Nemotron-H FFN sentinel cache shape is invalid")
		}
		cacheKey = builder.Concat(pastKey, sentinel, 2)
		cacheValue = builder.Concat(pastValue, sentinel, 2)
	}
	var feedForward *tensor.Tensor
	if spec.Profile().Has(ArchitectureMoE) {
		if err := (graphWeights{
			requireGraphWeight("router", weights.FeedForwardRouter),
			requireGraphWeight("expert bias", weights.FeedForwardExpertBias),
			requireGraphWeight("expert up", weights.FeedForwardUpExperts),
			requireGraphWeight("expert down", weights.FeedForwardDownExperts),
			requireGraphWeight("shared up", weights.FeedForwardSharedUp),
			requireGraphWeight("shared down", weights.FeedForwardSharedDown),
		}).validate("Nemotron-H MoE"); err != nil {
			return DenseBlockResult{}, err
		}
		expertInput := normalized
		if weights.FeedForwardLatentDown != nil {
			if weights.FeedForwardLatentUp == nil {
				return DenseBlockResult{}, errors.New("Nemotron-H MoE latent projection is incomplete")
			}
			expertInput = builder.MulMat(weights.FeedForwardLatentDown, normalized)
		} else if weights.FeedForwardLatentUp != nil {
			return DenseBlockResult{}, errors.New("Nemotron-H MoE latent projection is incomplete")
		}
		plan := spec.moeGraphPlan(layerIndex)
		plan.Routing = tensor.MoERoutingSigmoid
		plan.Activation = tensor.MoEActivationReLUSquared
		plan.SelectionBias = true
		feedForward = plan.BuildLayer(builder, expertInput, normalized, weights)
		if weights.FeedForwardLatentUp != nil {
			feedForward = builder.MulMat(weights.FeedForwardLatentUp, feedForward)
		}
		shared := builder.MulMat(weights.FeedForwardSharedUp, normalized)
		shared = builder.MulMat(weights.FeedForwardSharedDown, builder.ReLUSquared(shared))
		feedForward = builder.Add(feedForward, shared)
	} else {
		if weights.FeedForwardUp == nil || weights.FeedForwardDown == nil {
			return DenseBlockResult{}, errors.New("Nemotron-H dense FFN catalog is incomplete")
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
	return DenseBlockResult{Output: builder.Add(input, feedForward), Key: cacheKey, Value: cacheValue}, builder.Err()
}
