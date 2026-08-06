package model

import (
	"errors"
	"math"

	"llamacpp2go/internal/tensor"
)

const (
	rwkvLayerRescale     = float32(0.5)
	rwkv6TimeMixStreams  = uint64(5)
	rwkv7TimeMixStreams  = uint64(5)
	rwkv7GatedMixStreams = uint64(6)
	rwkv7DecayScale      = float32(-0.606531)
)

// BuildDeepSeek4BlockCached: HC compressed-attention block.
func BuildDeepSeek4BlockCached(
	builder *tensor.Builder,
	input *tensor.Tensor,
	spec Spec,
	weights LayerGraphWeights,
	positions, tokenRows []uint32,
	pastKV *tensor.Tensor,
	pastStates CacheStates[*tensor.Tensor],
	currentPositions *tensor.Tensor,
	layerIndex uint32,
) (DenseBlockResult, error) {
	if spec.Profile().Block != BlockDeepSeek4 || builder == nil || input == nil ||
		(input.Shape.Rank != 2 && input.Shape.Rank != 3) || layerIndex >= spec.BlockCount || len(positions) == 0 ||
		uint64(len(positions)) != input.Shape.Dims[input.Shape.Rank-1] ||
		currentPositions == nil || currentPositions.Shape != tensor.MustShape(1, 1, uint64(len(positions))) {
		return DenseBlockResult{}, errors.New("DeepSeek 4 block input is invalid")
	}
	if layerIndex == 0 {
		if input.Shape.Rank != 2 {
			return DenseBlockResult{}, errors.New("DeepSeek 4 initial input must be rank 2")
		}
		input = builder.DeepSeek4HCInit(input, spec.HyperConnectionCount)
	} else if input.Shape.Rank != 3 || input.Shape.Dims[1] != uint64(spec.HyperConnectionCount) {
		return DenseBlockResult{}, errors.New("DeepSeek 4 HC input is invalid")
	}
	required := graphWeights{
		requireGraphWeight("attention norm", weights.AttentionNorm),
		requireGraphWeight("attention Q-A", weights.AttentionQ),
		requireGraphWeight("attention Q-A norm", weights.AttentionQNorm),
		requireGraphWeight("attention Q-B", weights.AttentionQB),
		requireGraphWeight("attention KV", weights.AttentionK),
		requireGraphWeight("attention KV norm", weights.AttentionKNorm),
		requireGraphWeight("attention sinks", weights.AttentionSinks),
		requireGraphWeight("attention output A", weights.AttentionOutputA),
		requireGraphWeight("attention output B", weights.AttentionOutput),
		requireGraphWeight("attention HC function", weights.HyperAttentionFN),
		requireGraphWeight("attention HC base", weights.HyperAttentionBase),
		requireGraphWeight("attention HC scale", weights.HyperAttentionScale),
		requireGraphWeight("feed-forward norm", weights.FeedForwardNorm),
		requireGraphWeight("feed-forward router", weights.FeedForwardRouter),
		requireGraphWeight("expert gate", weights.FeedForwardGateExperts),
		requireGraphWeight("expert up", weights.FeedForwardUpExperts),
		requireGraphWeight("expert down", weights.FeedForwardDownExperts),
		requireGraphWeight("shared gate", weights.FeedForwardSharedGate),
		requireGraphWeight("shared up", weights.FeedForwardSharedUp),
		requireGraphWeight("shared down", weights.FeedForwardSharedDown),
		requireGraphWeight("feed-forward HC function", weights.HyperFeedForwardFN),
		requireGraphWeight("feed-forward HC base", weights.HyperFeedForwardBase),
		requireGraphWeight("feed-forward HC scale", weights.HyperFeedForwardScale),
	}
	if layerIndex < spec.HashLayerCount {
		required.add("hash routing table", weights.FeedForwardHashExperts)
		if len(tokenRows) != len(positions) {
			return DenseBlockResult{}, errors.New("DeepSeek 4 hash routing rows are missing")
		}
	} else {
		required.add("router bias", weights.FeedForwardRouterBias)
	}
	ratio := spec.CompressRatios[layerIndex]
	if ratio != 0 {
		required.add("compressor KV", weights.AttentionCompressorKV)
		required.add("compressor gate", weights.AttentionCompressorGate)
		required.add("compressor APE", weights.AttentionCompressorAPE)
		required.add("compressor norm", weights.AttentionCompressorNorm)
	}
	if ratio == 4 {
		required.add("indexer projection", weights.IndexerProjection)
		required.add("indexer Q-B", weights.IndexerAttentionQB)
		required.add("indexer compressor KV", weights.IndexerCompressorKV)
		required.add("indexer compressor gate", weights.IndexerCompressorGate)
		required.add("indexer compressor APE", weights.IndexerCompressorAPE)
		required.add("indexer compressor norm", weights.IndexerCompressorNorm)
	}
	if layerIndex+1 == spec.BlockCount {
		required.add("output HC function", weights.HyperHeadFN)
		required.add("output HC base", weights.HyperHeadBase)
		required.add("output HC scale", weights.HyperHeadScale)
	}
	if err := required.validate("DeepSeek 4"); err != nil {
		return DenseBlockResult{}, err
	}
	tokens := uint64(len(positions))
	heads := uint64(spec.HeadCount)
	headWidth := uint64(spec.KeyLength)
	hc := spec.HyperConnectionCount
	residual := input
	current := builder.DeepSeek4HCPre(input, weights.HyperAttentionFN, weights.HyperAttentionScale,
		weights.HyperAttentionBase, hc, spec.HyperSinkhornIters, spec.RMSNormEpsilon, spec.HyperConnectionEps)
	current = builder.WeightedRMSNorm(current, weights.AttentionNorm, spec.RMSNormEpsilon)
	queryRank := builder.WeightedRMSNorm(builder.MulMat(weights.AttentionQ, current), weights.AttentionQNorm, spec.RMSNormEpsilon)
	query := builder.RMSNorm(builder.Reshape(builder.MulMat(weights.AttentionQB, queryRank), headWidth, heads, tokens), spec.RMSNormEpsilon)
	kv := builder.WeightedRMSNorm(builder.MulMat(weights.AttentionK, current), weights.AttentionKNorm, spec.RMSNormEpsilon)
	kv = builder.Reshape(kv, headWidth, 1, tokens)
	frequencyBase := spec.RopeFrequencyBase
	frequencyScale := float32(1)
	extFactor := float32(0)
	attentionFactor := float32(1)
	originalContext := uint32(0)
	betaFast, betaSlow := float32(0), float32(0)
	if ratio != 0 {
		frequencyBase = spec.CompressRopeBase
		if spec.RopeScalingFactor > 0 {
			frequencyScale = 1 / spec.RopeScalingFactor
		}
		extFactor = spec.YaRNExtFactor
		attentionFactor = spec.YaRNAttentionFactor
		if attentionFactor <= 0 {
			attentionFactor = 1
		}
		originalContext = spec.OriginalContextLength
		betaFast, betaSlow = spec.YaRNBetaFast, spec.YaRNBetaSlow
	}
	cacheKV := kv
	if pastKV != nil {
		if pastKV.Shape.Rank != 3 || pastKV.Shape.Dims[0] != headWidth || pastKV.Shape.Dims[1] != 1 {
			return DenseBlockResult{}, errors.New("DeepSeek 4 raw cache shape is invalid")
		}
		cacheKV = builder.Concat(pastKV, kv, 2)
	}
	states := make(CacheStates[*tensor.Tensor])
	cachePositions := currentPositions
	if previous := pastStates[CacheStatePositions].Value; previous != nil {
		cachePositions = builder.Concat(previous, currentPositions, 2)
	}
	states[CacheStatePositions] = CacheState[*tensor.Tensor]{Mode: CacheStateToken, Value: cachePositions}
	appendState := func(name CacheStateName, current *tensor.Tensor) *tensor.Tensor {
		if current == nil {
			return nil
		}
		current = builder.Reshape(current, current.Shape.Dims[0], 1, current.Shape.Dims[1])
		if previous := pastStates[name].Value; previous != nil {
			current = builder.Concat(previous, current, 2)
		}
		states[name] = CacheState[*tensor.Tensor]{Mode: CacheStateToken, Value: current}
		return current
	}
	var compressorKV, compressorScore, indexerQuery, indexerWeights, indexerKV, indexerScore *tensor.Tensor
	if ratio != 0 {
		rows := make([]uint32, len(positions))
		for index, position := range positions {
			rows[index] = position % ratio
		}
		compressorKV = appendState(CacheStateCompressorKV, builder.MulMat(weights.AttentionCompressorKV, current))
		compressorScore = appendState(CacheStateCompressorScore, builder.Add(
			builder.MulMat(weights.AttentionCompressorGate, current), builder.GetRows(weights.AttentionCompressorAPE, rows),
		))
	}
	if ratio == 4 {
		indexerWidth := uint64(spec.IndexerKeyLength)
		indexerHeads := uint64(spec.IndexerHeadCount)
		indexerQuery = builder.Reshape(builder.MulMat(weights.IndexerAttentionQB, queryRank), indexerWidth, indexerHeads, tokens)
		indexerWeights = builder.Scale(builder.MulMat(weights.IndexerProjection, current),
			float32(1/math.Sqrt(float64(spec.IndexerKeyLength*spec.IndexerHeadCount))))
		rows := make([]uint32, len(positions))
		for index, position := range positions {
			rows[index] = position % 4
		}
		indexerKV = appendState(CacheStateIndexerCompressorKV, builder.MulMat(weights.IndexerCompressorKV, current))
		indexerScore = appendState(CacheStateIndexerCompressorScore, builder.Add(
			builder.MulMat(weights.IndexerCompressorGate, current), builder.GetRows(weights.IndexerCompressorAPE, rows),
		))
	}
	attributes := tensor.DeepSeek4AttentionAttributes{
		Positions: positions, Ratio: tensor.DeepSeek4CompressionRatio(ratio), Window: spec.SlidingWindow, Heads: spec.HeadCount,
		IndexerHeads: spec.IndexerHeadCount, IndexerTopK: spec.IndexerTopK,
		RotaryDimensions: spec.RopeDimensionCount, FrequencyBase: frequencyBase, FrequencyScale: frequencyScale,
		OriginalContext: originalContext, ExtFactor: extFactor, AttentionFactor: attentionFactor,
		BetaFast: betaFast, BetaSlow: betaSlow, NormEpsilon: spec.RMSNormEpsilon,
	}
	attention := builder.DeepSeek4Attention(query, cacheKV, cachePositions, weights.AttentionSinks,
		compressorKV, compressorScore, weights.AttentionCompressorNorm,
		indexerQuery, indexerWeights, indexerKV, indexerScore, weights.IndexerCompressorNorm, attributes)
	groupDimension := uint64(spec.HeadCount/spec.AttentionOutputGroups) * headWidth
	attention = builder.Reshape(attention, groupDimension, uint64(spec.AttentionOutputGroups), tokens)
	outputA := builder.Reshape(weights.AttentionOutputA, groupDimension, uint64(spec.AttentionOutputRank), uint64(spec.AttentionOutputGroups))
	attention = builder.Reshape(builder.GroupedMulMat(outputA, attention), uint64(spec.AttentionOutputRank*spec.AttentionOutputGroups), tokens)
	attention = builder.MulMat(weights.AttentionOutput, attention)
	input = builder.DeepSeek4HCPost(attention, residual, weights.HyperAttentionFN, weights.HyperAttentionScale,
		weights.HyperAttentionBase, hc, spec.HyperSinkhornIters, spec.RMSNormEpsilon, spec.HyperConnectionEps)
	residual = input
	current = builder.DeepSeek4HCPre(input, weights.HyperFeedForwardFN, weights.HyperFeedForwardScale,
		weights.HyperFeedForwardBase, hc, spec.HyperSinkhornIters, spec.RMSNormEpsilon, spec.HyperConnectionEps)
	current = builder.WeightedRMSNorm(current, weights.FeedForwardNorm, spec.RMSNormEpsilon)
	var selected *tensor.Tensor
	if layerIndex < spec.HashLayerCount {
		selected = builder.GetRows(weights.FeedForwardHashExperts, tokenRows)
	}
	moePlan := spec.moeGraphPlan(layerIndex)
	moePlan.Routing = tensor.MoERoutingSqrtSoftplus
	moePlan.SelectionBias = true
	moePlan.SwiGLUClamp = spec.LayerExpertSwiGLUClamp(layerIndex)
	moe := moePlan.Build(builder, current, weights.FeedForwardRouter, MoEGraphInputs{
		Gate: weights.FeedForwardGateExperts, Up: weights.FeedForwardUpExperts,
		Down: weights.FeedForwardDownExperts, SelectionBias: weights.FeedForwardRouterBias,
		SelectedExperts: selected,
	})
	sharedGate := builder.MulMat(weights.FeedForwardSharedGate, current)
	sharedUp := builder.MulMat(weights.FeedForwardSharedUp, current)
	shared := builder.MulMat(weights.FeedForwardSharedDown,
		deepSeek4LimitedSwiGLU(builder, sharedGate, sharedUp, spec.LayerSharedSwiGLUClampLimit(layerIndex)))
	current = builder.Add(moe, shared)
	output := builder.DeepSeek4HCPost(current, residual, weights.HyperFeedForwardFN, weights.HyperFeedForwardScale,
		weights.HyperFeedForwardBase, hc, spec.HyperSinkhornIters, spec.RMSNormEpsilon, spec.HyperConnectionEps)
	if layerIndex+1 == spec.BlockCount {
		output = builder.DeepSeek4HCHead(output, weights.HyperHeadFN, weights.HyperHeadScale,
			weights.HyperHeadBase, hc, spec.RMSNormEpsilon, spec.HyperConnectionEps)
	}
	if err := builder.Err(); err != nil {
		return DenseBlockResult{}, err
	}
	return DenseBlockResult{Output: output, Key: cacheKV, Value: cacheKV, States: states}, nil
}

// BuildRWKV6Qwen2BlockCached: RMS/SwiGLU QRWKV recurrent block.
func BuildRWKV6Qwen2BlockCached(
	builder *tensor.Builder,
	input *tensor.Tensor,
	spec Spec,
	weights LayerGraphWeights,
	pastShift, pastState *tensor.Tensor,
	layerIndex uint32,
) (DenseBlockResult, error) {
	if spec.Profile().DenseGraph != DenseGraphRWKV6Qwen2 || builder == nil || input == nil || pastShift == nil || pastState == nil {
		return DenseBlockResult{}, errors.New("RWKV6-Qwen2 block input/state is invalid")
	}
	required := graphWeights{
		requireGraphWeight("attention norm", weights.AttentionNorm),
		requireGraphWeight("time-mix W1", weights.TimeMixW1),
		requireGraphWeight("time-mix W2", weights.TimeMixW2),
		requireGraphWeight("time-mix lerp X", weights.TimeMixLerpX),
		requireGraphWeight("time-mix fused lerp", weights.TimeMixLerpFused),
		requireGraphWeight("time decay", weights.TimeMixDecay),
		requireGraphWeight("time decay W1", weights.TimeMixDecayW1),
		requireGraphWeight("time decay W2", weights.TimeMixDecayW2),
		requireGraphWeight("time key", weights.TimeMixKey),
		requireGraphWeight("time value", weights.TimeMixValue),
		requireGraphWeight("time receptance", weights.TimeMixReceptance),
		requireGraphWeight("time gate", weights.TimeMixGate),
		requireGraphWeight("time output", weights.TimeMixOutput),
		requireGraphWeight("feed-forward norm", weights.FeedForwardNorm),
		requireGraphWeight("feed-forward gate", weights.FeedForwardGate),
		requireGraphWeight("feed-forward up", weights.FeedForwardUp),
		requireGraphWeight("feed-forward down", weights.FeedForwardDown),
	}
	if err := required.validate("RWKV6-Qwen2"); err != nil {
		return DenseBlockResult{}, err
	}
	if input.Shape.Rank != 2 || input.Shape.Dims[0] != uint64(spec.EmbeddingLength) || input.Shape.Dims[1] == 0 {
		return DenseBlockResult{}, errors.New("RWKV6-Qwen2 input shape is invalid")
	}
	embedding := uint64(spec.EmbeddingLength)
	width := uint64(spec.WKVHeadSize)
	heads := embedding / width
	tokens := input.Shape.Dims[1]
	if !pastShift.Shape.Equal(tensor.MustShape(embedding)) ||
		!pastState.Shape.Equal(tensor.MustShape(width, width, heads, 1)) {
		return DenseBlockResult{}, errors.New("RWKV6-Qwen2 cache shape is invalid")
	}
	normalized := builder.WeightedRMSNorm(input, weights.AttentionNorm, spec.RMSNormEpsilon)
	xPrev := builder.Reshape(pastShift, embedding, 1)
	if tokens > 1 {
		xPrev = builder.Concat(xPrev, builder.FlatSlice(normalized, 0, embedding, tokens-1), 1)
	}
	nextShift := builder.FlatSlice(normalized, embedding*(tokens-1), embedding)
	sx := builder.Add(xPrev, builder.Scale(normalized, -1))
	base := builder.Add(normalized, builder.Multiply(sx, builder.Reshape(weights.TimeMixLerpX, embedding, 1)))
	adjustments := builder.GroupedMulMat(
		weights.TimeMixW2,
		builder.Reshape(
			builder.Tanh(builder.MulMat(weights.TimeMixW1, base)),
			uint64(spec.TimeMixExtraDim), rwkv6TimeMixStreams, tokens,
		),
	)
	adjustments = builder.Reshape(adjustments, embedding*rwkv6TimeMixStreams, tokens)
	lerps := builder.Reshape(weights.TimeMixLerpFused, embedding*rwkv6TimeMixStreams, 1)
	mixed := make([]*tensor.Tensor, rwkv6TimeMixStreams)
	for index := uint64(0); index < rwkv6TimeMixStreams; index++ {
		adjustment := builder.Reshape(
			builder.GroupSlice(adjustments, index*embedding, embedding, 1, embedding*rwkv6TimeMixStreams),
			embedding, tokens,
		)
		lerp := builder.Reshape(
			builder.GroupSlice(lerps, index*embedding, embedding, 1, embedding*rwkv6TimeMixStreams),
			embedding,
		)
		mixed[index] = builder.Add(normalized, builder.Multiply(sx, builder.Add(adjustment, lerp)))
	}
	xw, xk, xv, xr, xg := mixed[0], mixed[1], mixed[2], mixed[3], mixed[4]
	key := builder.MulMat(weights.TimeMixKey, xk)
	value := builder.MulMat(weights.TimeMixValue, xv)
	receptance := builder.MulMat(weights.TimeMixReceptance, xr)
	if weights.AttentionKBias != nil {
		key = builder.Add(key, weights.AttentionKBias)
	}
	if weights.AttentionVBias != nil {
		value = builder.Add(value, weights.AttentionVBias)
	}
	if weights.AttentionQBias != nil {
		receptance = builder.Add(receptance, weights.AttentionQBias)
	}
	gate := builder.Sigmoid(builder.MulMat(weights.TimeMixGate, xg))
	decay := builder.MulMat(weights.TimeMixDecayW2, builder.Tanh(builder.MulMat(weights.TimeMixDecayW1, xw)))
	decay = builder.Exp(builder.Scale(builder.Exp(builder.Add(decay, weights.TimeMixDecay)), -1))
	kvHeads := uint64(spec.HeadCountKV)
	key = builder.Reshape(key, width, kvHeads, tokens, 1)
	value = builder.Reshape(value, width, kvHeads, tokens, 1)
	receptance = builder.Reshape(receptance, width, heads, tokens, 1)
	decay = builder.Reshape(decay, width, heads, tokens, 1)
	packed := builder.GatedLinearAttention(
		key, value, receptance, decay, pastState,
		1/float32(math.Sqrt(float64(width))),
	)
	attentionElements := embedding * tokens
	attention := builder.FlatSlice(packed, 0, embedding, tokens)
	nextState := builder.FlatSlice(packed, attentionElements, width, width, heads, 1)
	attention = builder.MulMat(weights.TimeMixOutput, builder.Multiply(attention, gate))
	residual := builder.Add(input, attention)
	ffnInput := builder.WeightedRMSNorm(residual, weights.FeedForwardNorm, spec.RMSNormEpsilon)
	feedForward := builder.MulMat(
		weights.FeedForwardDown,
		builder.SwiGLU(builder.MulMat(weights.FeedForwardGate, ffnInput), builder.MulMat(weights.FeedForwardUp, ffnInput)),
	)
	output := builder.Add(residual, feedForward)
	if spec.RescaleEvery > 0 && (layerIndex+1)%spec.RescaleEvery == 0 {
		output = builder.Scale(output, rwkvLayerRescale)
	}
	if err := builder.Err(); err != nil {
		return DenseBlockResult{}, err
	}
	return DenseBlockResult{Output: output, Key: nextShift, Value: nextState}, nil
}

// BuildRWKV6BlockCached: affine-LN WKV6 recurrent block.
func BuildRWKV6BlockCached(
	builder *tensor.Builder,
	input *tensor.Tensor,
	spec Spec,
	weights LayerGraphWeights,
	pastShift, pastState *tensor.Tensor,
	layerIndex uint32,
) (DenseBlockResult, error) {
	if spec.Profile().DenseGraph != DenseGraphRWKV6 || builder == nil || input == nil || pastShift == nil || pastState == nil {
		return DenseBlockResult{}, errors.New("RWKV6 block input/state is invalid")
	}
	required := graphWeights{
		requireGraphWeight("attention norm", weights.AttentionNorm),
		requireGraphWeight("attention norm bias", weights.AttentionNormBias),
		requireGraphWeight("channel norm", weights.AttentionNorm2),
		requireGraphWeight("channel norm bias", weights.AttentionNorm2Bias),
		requireGraphWeight("time-mix W1", weights.TimeMixW1),
		requireGraphWeight("time-mix W2", weights.TimeMixW2),
		requireGraphWeight("time-mix lerp X", weights.TimeMixLerpX),
		requireGraphWeight("time first", weights.TimeMixFirst),
		requireGraphWeight("time decay", weights.TimeMixDecay),
		requireGraphWeight("time decay W1", weights.TimeMixDecayW1),
		requireGraphWeight("time decay W2", weights.TimeMixDecayW2),
		requireGraphWeight("time key", weights.TimeMixKey),
		requireGraphWeight("time value", weights.TimeMixValue),
		requireGraphWeight("time receptance", weights.TimeMixReceptance),
		requireGraphWeight("time gate", weights.TimeMixGate),
		requireGraphWeight("time-mix norm", weights.TimeMixLN),
		requireGraphWeight("time-mix norm bias", weights.TimeMixLNBias),
		requireGraphWeight("time output", weights.TimeMixOutput),
		requireGraphWeight("channel lerp K", weights.ChannelMixLerpK),
		requireGraphWeight("channel lerp R", weights.ChannelMixLerpR),
		requireGraphWeight("channel key", weights.ChannelMixKey),
		requireGraphWeight("channel value", weights.ChannelMixValue),
		requireGraphWeight("channel receptance", weights.ChannelMixReceptance),
	}
	if err := required.validate("RWKV6"); err != nil {
		return DenseBlockResult{}, err
	}
	if weights.TimeMixLerpFused == nil &&
		(weights.TimeMixLerpW == nil || weights.TimeMixLerpK == nil || weights.TimeMixLerpV == nil ||
			weights.TimeMixLerpR == nil || weights.TimeMixLerpG == nil) {
		return DenseBlockResult{}, errors.New("RWKV6 time-mix lerp catalog is incomplete")
	}
	embedding := uint64(spec.EmbeddingLength)
	width := uint64(spec.WKVHeadSize)
	heads := uint64(spec.HeadCount)
	tokens := input.Shape.Dims[1]
	if input.Shape.Rank != 2 || input.Shape.Dims[0] != embedding || tokens == 0 ||
		!pastShift.Shape.Equal(tensor.MustShape(embedding, 2)) ||
		!pastState.Shape.Equal(tensor.MustShape(width, width, heads, 1)) {
		return DenseBlockResult{}, errors.New("RWKV6 input/cache shape is invalid")
	}
	attNorm := ApplyNormalization(builder, input, weights.AttentionNorm, weights.AttentionNormBias, spec)
	attPrev := builder.Reshape(builder.FlatSlice(pastShift, 0, embedding), embedding, 1)
	if tokens > 1 {
		attPrev = builder.Concat(attPrev, builder.FlatSlice(attNorm, 0, embedding, tokens-1), 1)
	}
	sx := builder.Add(attPrev, builder.Scale(attNorm, -1))
	base := builder.Add(attNorm, builder.Multiply(sx, builder.Reshape(weights.TimeMixLerpX, embedding, 1)))
	adjustments := builder.GroupedMulMat(
		weights.TimeMixW2,
		builder.Reshape(
			builder.Tanh(builder.MulMat(weights.TimeMixW1, base)),
			uint64(spec.TimeMixExtraDim), rwkv6TimeMixStreams, tokens,
		),
	)
	adjustments = builder.Reshape(adjustments, embedding*rwkv6TimeMixStreams, tokens)
	mixed := make([]*tensor.Tensor, rwkv6TimeMixStreams)
	separate := []*tensor.Tensor{
		weights.TimeMixLerpW, weights.TimeMixLerpK, weights.TimeMixLerpV,
		weights.TimeMixLerpR, weights.TimeMixLerpG,
	}
	for index := uint64(0); index < rwkv6TimeMixStreams; index++ {
		adjustment := builder.Reshape(
			builder.GroupSlice(
				adjustments, index*embedding, embedding, 1, embedding*rwkv6TimeMixStreams,
			), embedding, tokens,
		)
		var lerp *tensor.Tensor
		if weights.TimeMixLerpFused != nil {
			fused := builder.Reshape(weights.TimeMixLerpFused, embedding*rwkv6TimeMixStreams, 1)
			lerp = builder.Reshape(
				builder.GroupSlice(fused, index*embedding, embedding, 1, embedding*rwkv6TimeMixStreams),
				embedding,
			)
		} else {
			lerp = builder.Reshape(separate[index], embedding)
		}
		mixed[index] = builder.Add(attNorm, builder.Multiply(sx, builder.Add(adjustment, lerp)))
	}
	xw, xk, xv, xr, xg := mixed[0], mixed[1], mixed[2], mixed[3], mixed[4]
	key := builder.Reshape(builder.MulMat(weights.TimeMixKey, xk), width, heads, tokens, 1)
	value := builder.Reshape(builder.MulMat(weights.TimeMixValue, xv), width, heads, tokens, 1)
	receptance := builder.Reshape(builder.MulMat(weights.TimeMixReceptance, xr), width, heads, tokens, 1)
	gate := builder.SiLU(builder.MulMat(weights.TimeMixGate, xg))
	decay := builder.MulMat(weights.TimeMixDecayW2, builder.Tanh(builder.MulMat(weights.TimeMixDecayW1, xw)))
	decay = builder.Reshape(builder.Exp(builder.Scale(builder.Exp(builder.Add(decay, weights.TimeMixDecay)), -1)), width, heads, tokens, 1)
	packed := builder.RWKV6(key, value, receptance, weights.TimeMixFirst, decay, pastState)
	attentionElements := embedding * tokens
	attention := builder.FlatSlice(packed, 0, embedding, tokens)
	nextState := builder.FlatSlice(packed, attentionElements, width, width, heads, 1)
	attention = builder.Reshape(
		builder.LayerNorm(builder.Reshape(attention, width, heads, tokens), rwkvHeadNormEpsilon), embedding, tokens,
	)
	attention = builder.Add(builder.Multiply(attention, weights.TimeMixLN), weights.TimeMixLNBias)
	attention = builder.MulMat(weights.TimeMixOutput, builder.Multiply(attention, gate))
	ffnInput := builder.Add(input, attention)
	ffnNorm := ApplyNormalization(builder, ffnInput, weights.AttentionNorm2, weights.AttentionNorm2Bias, spec)
	ffnPrev := builder.Reshape(builder.FlatSlice(pastShift, embedding, embedding), embedding, 1)
	if tokens > 1 {
		ffnPrev = builder.Concat(ffnPrev, builder.FlatSlice(ffnNorm, 0, embedding, tokens-1), 1)
	}
	channelShift := builder.Add(ffnPrev, builder.Scale(ffnNorm, -1))
	channelKeyInput := builder.Add(ffnNorm, builder.Multiply(channelShift, builder.Reshape(weights.ChannelMixLerpK, embedding, 1)))
	channelReceptanceInput := builder.Add(ffnNorm, builder.Multiply(channelShift, builder.Reshape(weights.ChannelMixLerpR, embedding, 1)))
	channel := builder.MulMat(weights.ChannelMixValue, builder.ReLUSquared(builder.MulMat(weights.ChannelMixKey, channelKeyInput)))
	channel = builder.Multiply(builder.Sigmoid(builder.MulMat(weights.ChannelMixReceptance, channelReceptanceInput)), channel)
	output := builder.Add(ffnInput, channel)
	if spec.RescaleEvery > 0 && (layerIndex+1)%spec.RescaleEvery == 0 {
		output = builder.Scale(output, rwkvLayerRescale)
	}
	nextAttShift := builder.FlatSlice(attNorm, embedding*(tokens-1), embedding)
	nextFFNShift := builder.FlatSlice(ffnNorm, embedding*(tokens-1), embedding)
	nextShift := builder.Concat(builder.Reshape(nextAttShift, embedding, 1), builder.Reshape(nextFFNShift, embedding, 1), 1)
	if err := builder.Err(); err != nil {
		return DenseBlockResult{}, err
	}
	return DenseBlockResult{Output: output, Key: nextShift, Value: nextState}, nil
}

// BuildRWKV7BlockCached: RWKV7/ARWKV7 recurrent block.
func BuildRWKV7BlockCached(
	builder *tensor.Builder,
	input *tensor.Tensor,
	spec Spec,
	weights LayerGraphWeights,
	pastShift, pastState *tensor.Tensor,
	layerIndex uint32,
) (DenseBlockResult, error) {
	profile := spec.Profile()
	if profile.DenseGraph != DenseGraphRWKV7 ||
		builder == nil || input == nil || pastShift == nil || pastState == nil {
		return DenseBlockResult{}, errors.New("RWKV7 block input/state is invalid")
	}
	required := graphWeights{
		requireGraphWeight("attention norm", weights.AttentionNorm),
		requireGraphWeight("time W0", weights.TimeMixW0),
		requireGraphWeight("time W1", weights.TimeMixW1),
		requireGraphWeight("time W2", weights.TimeMixW2),
		requireGraphWeight("time A0", weights.TimeMixA0),
		requireGraphWeight("time A1", weights.TimeMixA1),
		requireGraphWeight("time A2", weights.TimeMixA2),
		requireGraphWeight("time V0", weights.TimeMixV0),
		requireGraphWeight("time V1", weights.TimeMixV1),
		requireGraphWeight("time V2", weights.TimeMixV2),
		requireGraphWeight("time lerp", weights.TimeMixLerpFused),
		requireGraphWeight("time KK", weights.TimeMixKK),
		requireGraphWeight("time KA", weights.TimeMixKA),
		requireGraphWeight("time RK", weights.TimeMixRK),
		requireGraphWeight("time key", weights.TimeMixKey),
		requireGraphWeight("time value", weights.TimeMixValue),
		requireGraphWeight("time receptance", weights.TimeMixReceptance),
		requireGraphWeight("time output", weights.TimeMixOutput),
	}
	channelMix := profile.Normalization == NormalizationLayer
	if channelMix {
		required.add("attention norm bias", weights.AttentionNormBias)
		required.add("channel norm", weights.AttentionNorm2)
		required.add("channel norm bias", weights.AttentionNorm2Bias)
		required.add("time norm", weights.TimeMixLN)
		required.add("time norm bias", weights.TimeMixLNBias)
		required.add("channel lerp", weights.ChannelMixLerpK)
		required.add("channel key", weights.ChannelMixKey)
		required.add("channel value", weights.ChannelMixValue)
	} else {
		required.add("feed-forward norm", weights.FeedForwardNorm)
		required.add("feed-forward gate", weights.FeedForwardGate)
		required.add("feed-forward up", weights.FeedForwardUp)
		required.add("feed-forward down", weights.FeedForwardDown)
	}
	if spec.GateLoRARank > 0 {
		required.add("time G1", weights.TimeMixG1)
		required.add("time G2", weights.TimeMixG2)
	}
	if err := required.validate("RWKV7"); err != nil {
		return DenseBlockResult{}, err
	}
	embedding := uint64(spec.EmbeddingLength)
	width := uint64(spec.WKVHeadSize)
	heads := uint64(spec.HeadCount)
	tokens := input.Shape.Dims[1]
	shiftCount := uint64(spec.TokenShiftCount)
	if input.Shape.Rank != 2 || input.Shape.Dims[0] != embedding || tokens == 0 ||
		!pastShift.Shape.Equal(tensor.MustShape(embedding, shiftCount)) ||
		!pastState.Shape.Equal(tensor.MustShape(width, width, heads, 1)) {
		return DenseBlockResult{}, errors.New("RWKV7 input/cache shape is invalid")
	}
	attNorm := ApplyNormalization(builder, input, weights.AttentionNorm, weights.AttentionNormBias, spec)
	attPrev := builder.Reshape(builder.FlatSlice(pastShift, 0, embedding), embedding, 1)
	if tokens > 1 {
		attPrev = builder.Concat(attPrev, builder.FlatSlice(attNorm, 0, embedding, tokens-1), 1)
	}
	sx := builder.Add(attPrev, builder.Scale(attNorm, -1))
	lerpCount := rwkv7GatedMixStreams
	if spec.GateLoRARank == 0 {
		lerpCount = rwkv7TimeMixStreams
	}
	lerps := builder.Reshape(weights.TimeMixLerpFused, embedding*lerpCount, 1)
	mixed := make([]*tensor.Tensor, lerpCount)
	for index := uint64(0); index < lerpCount; index++ {
		lerp := builder.Reshape(builder.GroupSlice(lerps, index*embedding, embedding, 1, embedding*lerpCount), embedding)
		mixed[index] = builder.Add(attNorm, builder.Multiply(sx, lerp))
	}
	xr, xw, xk, xv, xa := mixed[0], mixed[1], mixed[2], mixed[3], mixed[4]
	receptance := builder.MulMat(weights.TimeMixReceptance, xr)
	decay := builder.Add(
		builder.MulMat(weights.TimeMixW2, builder.Tanh(builder.MulMat(weights.TimeMixW1, xw))),
		weights.TimeMixW0,
	)
	decay = builder.Exp(builder.Scale(builder.Sigmoid(decay), rwkv7DecayScale))
	key := builder.MulMat(weights.TimeMixKey, xk)
	value := builder.MulMat(weights.TimeMixValue, xv)
	var auxiliary *tensor.Tensor
	if layerIndex == 0 {
		if weights.PerLayerInput != nil {
			return DenseBlockResult{}, errors.New("RWKV7 first layer received a value residual")
		}
		auxiliary = value
	} else {
		if weights.PerLayerInput == nil {
			return DenseBlockResult{}, errors.New("RWKV7 value residual is missing")
		}
		valueMix := builder.Sigmoid(builder.Add(
			builder.MulMat(weights.TimeMixV2, builder.MulMat(weights.TimeMixV1, xv)),
			weights.TimeMixV0,
		))
		value = builder.Add(value, builder.Multiply(builder.Add(weights.PerLayerInput, builder.Scale(value, -1)), valueMix))
	}
	a := builder.Sigmoid(builder.Add(
		builder.MulMat(weights.TimeMixA2, builder.MulMat(weights.TimeMixA1, xa)),
		weights.TimeMixA0,
	))
	kk := builder.L2Norm(
		builder.Reshape(builder.Multiply(key, weights.TimeMixKK), width, heads, tokens, 1),
		rwkvKeyNormEpsilon,
	)
	ka := builder.Multiply(key, weights.TimeMixKA)
	key = builder.Add(key, builder.Add(builder.Multiply(a, ka), builder.Scale(ka, -1)))
	receptance4 := builder.Reshape(receptance, width, heads, tokens, 1)
	decay4 := builder.Reshape(decay, width, heads, tokens, 1)
	key4 := builder.Reshape(key, width, heads, tokens, 1)
	value4 := builder.Reshape(value, width, heads, tokens, 1)
	a4 := builder.Reshape(a, width, heads, tokens, 1)
	packed := builder.RWKV7(receptance4, decay4, key4, value4, builder.Scale(kk, -1), builder.Multiply(kk, a4), pastState)
	attentionElements := embedding * tokens
	attention := builder.FlatSlice(packed, 0, embedding, tokens)
	nextState := builder.FlatSlice(packed, attentionElements, width, width, heads, 1)
	if weights.TimeMixLN != nil && weights.TimeMixLNBias != nil {
		attention = builder.Reshape(
			builder.LayerNorm(builder.Reshape(attention, width, heads, tokens), rwkvHeadNormEpsilon), embedding, tokens,
		)
		attention = builder.Add(builder.Multiply(attention, weights.TimeMixLN), weights.TimeMixLNBias)
	}
	rkWeight := builder.Reshape(weights.TimeMixRK, width, heads, 1, 1)
	rk := builder.SumRows(builder.Multiply(builder.Multiply(key4, receptance4), rkWeight))
	attention = builder.Add(attention, builder.Reshape(builder.Multiply(value4, rk), embedding, tokens))
	if spec.GateLoRARank > 0 {
		xg := mixed[5]
		gate := builder.MulMat(weights.TimeMixG2, builder.Sigmoid(builder.MulMat(weights.TimeMixG1, xg)))
		attention = builder.Multiply(attention, gate)
	}
	attention = builder.MulMat(weights.TimeMixOutput, attention)
	ffnInput := builder.Add(input, attention)
	var ffnNorm, output *tensor.Tensor
	if channelMix {
		ffnNorm = ApplyNormalization(builder, ffnInput, weights.AttentionNorm2, weights.AttentionNorm2Bias, spec)
		ffnPrev := builder.Reshape(builder.FlatSlice(pastShift, embedding, embedding), embedding, 1)
		if tokens > 1 {
			ffnPrev = builder.Concat(ffnPrev, builder.FlatSlice(ffnNorm, 0, embedding, tokens-1), 1)
		}
		channelShift := builder.Add(ffnPrev, builder.Scale(ffnNorm, -1))
		channelInput := builder.Add(ffnNorm, builder.Multiply(channelShift, builder.Reshape(weights.ChannelMixLerpK, embedding, 1)))
		channel := builder.MulMat(weights.ChannelMixValue, builder.ReLUSquared(builder.MulMat(weights.ChannelMixKey, channelInput)))
		output = builder.Add(ffnInput, channel)
	} else {
		ffnNorm = builder.WeightedRMSNorm(ffnInput, weights.FeedForwardNorm, spec.RMSNormEpsilon)
		ffn := builder.MulMat(weights.FeedForwardDown, builder.SwiGLU(
			builder.MulMat(weights.FeedForwardGate, ffnNorm), builder.MulMat(weights.FeedForwardUp, ffnNorm),
		))
		output = builder.Add(ffnInput, ffn)
	}
	nextAttShift := builder.FlatSlice(attNorm, embedding*(tokens-1), embedding)
	nextShift := builder.Reshape(nextAttShift, embedding, 1)
	if shiftCount == 2 {
		nextFFNShift := builder.FlatSlice(ffnNorm, embedding*(tokens-1), embedding)
		nextShift = builder.Concat(nextShift, builder.Reshape(nextFFNShift, embedding, 1), 1)
	}
	if err := builder.Err(); err != nil {
		return DenseBlockResult{}, err
	}
	return DenseBlockResult{Output: output, Key: nextShift, Value: nextState, Auxiliary: auxiliary}, nil
}

// BuildKimiLinearBlockCached: KDA or no-RoPE MLA block.
func BuildKimiLinearBlockCached(
	builder *tensor.Builder,
	input *tensor.Tensor,
	spec Spec,
	weights LayerGraphWeights,
	positions []uint32,
	recurrent bool,
	pastKey, pastValue *tensor.Tensor,
	layerIndex uint32,
) (DenseBlockResult, error) {
	if spec.Profile().Block != BlockKimiLinear {
		return DenseBlockResult{}, errors.New("Kimi Linear block requires kimi-linear architecture")
	}
	if !recurrent {
		return BuildMLABlockCachedForLayer(builder, input, spec, weights, positions, pastKey, pastValue, layerIndex)
	}
	if builder == nil || input == nil || pastKey == nil || pastValue == nil {
		return DenseBlockResult{}, errors.New("Kimi Linear KDA input/state is nil")
	}
	required := graphWeights{
		requireGraphWeight("attention norm", weights.AttentionNorm),
		requireGraphWeight("attention Q", weights.AttentionQ),
		requireGraphWeight("attention K", weights.AttentionK),
		requireGraphWeight("attention V", weights.AttentionV),
		requireGraphWeight("attention output", weights.AttentionOutput),
		requireGraphWeight("Q convolution", weights.SSMQueryConv),
		requireGraphWeight("K convolution", weights.SSMKeyConv),
		requireGraphWeight("V convolution", weights.SSMValueConv),
		requireGraphWeight("forget A", weights.SSMForgetA),
		requireGraphWeight("forget B", weights.SSMForgetB),
		requireGraphWeight("beta", weights.SSMBeta),
		requireGraphWeight("SSM A", weights.SSMA),
		requireGraphWeight("time-step bias", weights.SSMTimeStep),
		requireGraphWeight("output gate A", weights.SSMOutputGateA),
		requireGraphWeight("output gate B", weights.SSMOutputGateB),
		requireGraphWeight("SSM norm", weights.SSMNorm),
		requireGraphWeight("feed-forward norm", weights.FeedForwardNorm),
	}
	addKimiFeedForwardRequirements(&required, spec, weights, layerIndex)
	if err := required.validate("Kimi Linear KDA"); err != nil {
		return DenseBlockResult{}, err
	}
	if input.Shape.Rank != 2 || len(positions) == 0 || uint64(len(positions)) != input.Shape.Dims[1] {
		return DenseBlockResult{}, errors.New("Kimi Linear KDA input shape is invalid")
	}
	headDim := uint64(spec.KDAHeadDim)
	heads := uint64(spec.HeadCount)
	inner := headDim * heads
	tokens := uint64(len(positions))
	window := uint64(spec.SSMConvKernel - 1)
	if !pastKey.Shape.Equal(tensor.MustShape(window, 3*inner)) ||
		!pastValue.Shape.Equal(tensor.MustShape(headDim, headDim, heads, 1)) {
		return DenseBlockResult{}, errors.New("Kimi Linear KDA cache shape is invalid")
	}
	normalized := builder.WeightedRMSNorm(input, weights.AttentionNorm, spec.RMSNormEpsilon)
	convolve := func(projection, kernel *tensor.Tensor, stateIndex uint64) (*tensor.Tensor, *tensor.Tensor) {
		state := builder.FlatSlice(pastKey, stateIndex*window*inner, window, inner)
		projected := builder.MulMat(projection, normalized)
		mixed := builder.Concat(state, builder.Transpose2D(projected), 0)
		next := builder.Reshape(builder.GroupSlice(mixed, tokens, window, 1, window), window, inner)
		kernel = builder.Reshape(kernel, uint64(spec.SSMConvKernel), inner)
		value := builder.SiLU(builder.SSMConv(mixed, kernel))
		return builder.Reshape(value, headDim, heads, tokens, 1), next
	}
	query, nextQ := convolve(weights.AttentionQ, weights.SSMQueryConv, 0)
	key, nextK := convolve(weights.AttentionK, weights.SSMKeyConv, 1)
	value, nextV := convolve(weights.AttentionV, weights.SSMValueConv, 2)
	nextConv := builder.Transpose2D(builder.Concat(
		builder.Concat(builder.Transpose2D(nextQ), builder.Transpose2D(nextK), 0),
		builder.Transpose2D(nextV), 0,
	))
	gate := builder.MulMat(weights.SSMForgetB, builder.MulMat(weights.SSMForgetA, normalized))
	gate = builder.Softplus(builder.Add(gate, weights.SSMTimeStep))
	gate = builder.Reshape(gate, headDim, heads, tokens, 1)
	gate = builder.Multiply(gate, builder.Reshape(weights.SSMA, 1, heads, 1, 1))
	beta := builder.Reshape(
		builder.Sigmoid(builder.MulMat(weights.SSMBeta, normalized)),
		1, heads, tokens, 1,
	)
	query = builder.L2Norm(query, spec.RMSNormEpsilon)
	key = builder.L2Norm(key, spec.RMSNormEpsilon)
	packed := builder.GatedDeltaNet(query, key, value, gate, beta, pastValue)
	attentionElements := headDim * heads * tokens
	attention := builder.FlatSlice(packed, 0, headDim, heads, tokens, 1)
	nextState := builder.FlatSlice(packed, attentionElements, headDim, headDim, heads, 1)
	outputGate := builder.MulMat(weights.SSMOutputGateB, builder.MulMat(weights.SSMOutputGateA, normalized))
	outputGate = builder.Reshape(outputGate, headDim, heads, tokens, 1)
	attention = builder.Multiply(
		builder.WeightedRMSNorm(attention, weights.SSMNorm, spec.RMSNormEpsilon),
		builder.Sigmoid(outputGate),
	)
	attention = builder.MulMat(weights.AttentionOutput, builder.Reshape(attention, inner, tokens))
	residual := builder.Add(input, attention)
	output := buildKimiFeedForward(builder, residual, spec, weights, layerIndex)
	if err := builder.Err(); err != nil {
		return DenseBlockResult{}, err
	}
	return DenseBlockResult{Output: output, Key: nextConv, Value: nextState}, nil
}

func buildKimiFeedForward(
	builder *tensor.Builder,
	residual *tensor.Tensor,
	spec Spec,
	weights LayerGraphWeights,
	layerIndex uint32,
) *tensor.Tensor {
	normalized := builder.WeightedRMSNorm(residual, weights.FeedForwardNorm, spec.RMSNormEpsilon)
	if layerIndex < spec.LeadingDenseBlocks {
		gate := builder.MulMat(weights.FeedForwardGate, normalized)
		up := builder.MulMat(weights.FeedForwardUp, normalized)
		return builder.Add(residual, builder.MulMat(weights.FeedForwardDown, builder.SwiGLU(gate, up)))
	}
	plan := spec.moeGraphPlan(layerIndex)
	plan.NormalizeTopKProb = true
	plan.SelectionBias = true
	routed := plan.BuildLayer(builder, normalized, nil, weights)
	shared := builder.MulMat(
		weights.FeedForwardSharedDown,
		builder.SwiGLU(
			builder.MulMat(weights.FeedForwardSharedGate, normalized),
			builder.MulMat(weights.FeedForwardSharedUp, normalized),
		),
	)
	return builder.Add(residual, builder.Add(routed, shared))
}

func addKimiFeedForwardRequirements(
	required *graphWeights,
	spec Spec,
	weights LayerGraphWeights,
	layerIndex uint32,
) {
	if layerIndex < spec.LeadingDenseBlocks {
		required.add("feed-forward gate", weights.FeedForwardGate)
		required.add("feed-forward up", weights.FeedForwardUp)
		required.add("feed-forward down", weights.FeedForwardDown)
		return
	}
	required.add("feed-forward router", weights.FeedForwardRouter)
	required.add("feed-forward expert gate", weights.FeedForwardGateExperts)
	required.add("feed-forward expert up", weights.FeedForwardUpExperts)
	required.add("feed-forward expert down", weights.FeedForwardDownExperts)
	required.add("feed-forward correction bias", weights.FeedForwardExpertBias)
	required.add("shared expert gate", weights.FeedForwardSharedGate)
	required.add("shared expert up", weights.FeedForwardSharedUp)
	required.add("shared expert down", weights.FeedForwardSharedDown)
}

// BuildLFM2BlockCached: attention or gated short-convolution block
// Recurrent cache: Key convolution window; Value reserved ABI slot
func BuildLFM2BlockCached(
	builder *tensor.Builder,
	input *tensor.Tensor,
	spec Spec,
	weights LayerGraphWeights,
	positions []uint32,
	recurrent bool,
	pastKey, pastValue *tensor.Tensor,
	layerIndex uint32,
) (LFM2BlockResult, error) {
	if spec.Profile().Attention != AttentionLFM2 {
		return LFM2BlockResult{}, errors.New("LFM2 block requires lfm2 or lfm2moe architecture")
	}
	if !recurrent {
		result, err := BuildDenseBlockCachedForLayer(
			builder, input, spec, weights, positions, pastKey, pastValue, layerIndex,
		)
		return LFM2BlockResult{
			Output: result.Output, Key: result.Key, Value: result.Value,
		}, err
	}
	if builder == nil || input == nil || pastKey == nil || pastValue == nil {
		return LFM2BlockResult{}, errors.New("LFM2 recurrent block input or state is nil")
	}
	required := graphWeights{
		requireGraphWeight("operator norm", weights.AttentionNorm),
		requireGraphWeight("short-convolution input", weights.ShortConvInput),
		requireGraphWeight("short-convolution kernel", weights.ShortConvKernel),
		requireGraphWeight("short-convolution output", weights.ShortConvOutput),
		requireGraphWeight("feed-forward norm", weights.FeedForwardNorm),
	}
	usesExperts := weights.FeedForwardRouter != nil
	if usesExperts {
		required.add("feed-forward router", weights.FeedForwardRouter)
		required.add("feed-forward expert gate", weights.FeedForwardGateExperts)
		required.add("feed-forward expert up", weights.FeedForwardUpExperts)
		required.add("feed-forward expert down", weights.FeedForwardDownExperts)
		required.add("feed-forward expert correction bias", weights.FeedForwardExpertBias)
	} else {
		required.add("feed-forward gate", weights.FeedForwardGate)
		required.add("feed-forward up", weights.FeedForwardUp)
		required.add("feed-forward down", weights.FeedForwardDown)
	}
	if err := required.validate("LFM2 recurrent block"); err != nil {
		return LFM2BlockResult{}, err
	}
	if input.Shape.Rank != 2 || len(positions) == 0 ||
		uint64(len(positions)) != input.Shape.Dims[1] {
		return LFM2BlockResult{}, errors.New("LFM2 recurrent block input shape is invalid")
	}
	embedding := uint64(spec.EmbeddingLength)
	window := uint64(spec.ShortConvCacheLength - 1)
	if !pastKey.Shape.Equal(tensor.MustShape(window, embedding)) ||
		!pastValue.Shape.Equal(tensor.MustShape(1)) {
		return LFM2BlockResult{}, errors.New("LFM2 recurrent cache shape is invalid")
	}
	tokens := uint64(len(positions))
	normalized := builder.WeightedRMSNorm(input, weights.AttentionNorm, spec.RMSNormEpsilon)
	mixed := builder.MulMat(weights.ShortConvInput, normalized)
	b := builder.Reshape(builder.GroupSlice(mixed, 0, embedding, 1, 3*embedding), embedding, tokens)
	c := builder.Reshape(builder.GroupSlice(mixed, embedding, embedding, 1, 3*embedding), embedding, tokens)
	x := builder.Reshape(builder.GroupSlice(mixed, 2*embedding, embedding, 1, 3*embedding), embedding, tokens)
	projected := builder.Transpose2D(builder.Multiply(b, x))
	convInput := builder.Concat(pastKey, projected, 0)
	if spec.NonCausalAttention {
		leftPad := window / 2
		rightPad := window - leftPad
		convInput = projected
		if leftPad > 0 {
			left := builder.GroupSlice(pastKey, window-leftPad, leftPad, 1, window)
			left = builder.Reshape(left, leftPad, embedding)
			convInput = builder.Concat(left, convInput, 0)
		}
		if rightPad > 0 {
			right := builder.GroupSlice(pastKey, window-rightPad, rightPad, 1, window)
			right = builder.Scale(builder.Reshape(right, rightPad, embedding), 0)
			convInput = builder.Concat(convInput, right, 0)
		}
	}
	nextState := builder.GroupSlice(convInput, tokens, window, 1, window)
	nextState = builder.Reshape(nextState, window, embedding)
	convolved := builder.SSMConv(convInput, weights.ShortConvKernel)
	shortConv := builder.MulMat(weights.ShortConvOutput, builder.Multiply(c, convolved))
	residual := builder.Add(input, shortConv)
	normalized = builder.WeightedRMSNorm(residual, weights.FeedForwardNorm, spec.RMSNormEpsilon)
	var feedForward *tensor.Tensor
	if usesExperts {
		plan := spec.moeGraphPlan(0)
		plan.NormalizeTopKProb = true
		plan.SelectionBias = true
		feedForward = plan.BuildLayer(builder, normalized, nil, weights)
	} else {
		gate := builder.MulMat(weights.FeedForwardGate, normalized)
		up := builder.MulMat(weights.FeedForwardUp, normalized)
		feedForward = builder.MulMat(weights.FeedForwardDown, builder.SwiGLU(gate, up))
	}
	output := builder.Add(residual, feedForward)
	nextReserved := builder.Scale(pastValue, 1)
	if err := builder.Err(); err != nil {
		return LFM2BlockResult{}, err
	}
	return LFM2BlockResult{
		Output: output, Key: nextState, Value: nextReserved, Recurrent: true,
	}, nil
}
