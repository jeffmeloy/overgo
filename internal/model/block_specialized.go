package model

import (
	"errors"

	"overgo/internal/hostmath"
	"overgo/internal/tensor"
)

const (
	wkv6WeightStream uint64 = iota
	wkv6KeyStream
	wkv6ValueStream
	wkv6ReceptanceStream
	wkv6GateStream
	wkv6TimeMixStreams
)

const (
	wkv7ReceptanceStream uint64 = iota
	wkv7WeightStream
	wkv7KeyStream
	wkv7ValueStream
	wkv7AuxiliaryStream
	wkv7GateStream
	wkv7GatedMixStreams
	wkv7TimeMixStreams = wkv7GateStream
	wkv7DecayScale     = float32(-0.606531)
)

// buildHyperAttentionStage: hyper-connected compressed attention.
func buildHyperAttentionStage(
	builder *tensor.Builder,
	input *tensor.Tensor,
	spec Spec,
	weights LayerGraphWeights,
	positions []uint32,
	pastKV *tensor.Tensor,
	pastStates CacheStates[*tensor.Tensor],
	currentPositions *tensor.Tensor,
	plan LayerPlan,
) (DenseBlockResult, error) {
	layerIndex := plan.Layer
	if builder == nil || input == nil || layerIndex >= spec.BlockCount ||
		len(positions) == tensor.FirstOffset || currentPositions == nil {
		return DenseBlockResult{}, errors.New("hyper-attention input is invalid")
	}
	tokens := uint64(len(positions))
	if layerIndex == tensor.FirstOffset {
		inputTokens, validInput := tensor.MatrixRows(input.Shape, uint64(spec.EmbeddingLength))
		if !validInput || inputTokens != tokens {
			return DenseBlockResult{}, errors.New("initial hyper-connection input must be rank 2")
		}
		input = builder.HyperConnectionInit(input, spec.HyperConnectionCount)
	} else {
		if !tensor.HasDimensions(input.Shape, uint64(spec.EmbeddingLength), uint64(spec.HyperConnectionCount), tokens) {
			return DenseBlockResult{}, errors.New("hyper-connection input is invalid")
		}
	}
	if !tensor.HasDimensions(
		currentPositions.Shape, tensor.SingletonExtent, tensor.SingletonExtent, tokens,
	) {
		return DenseBlockResult{}, errors.New("hyper-attention positions shape is invalid")
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
	}
	ratio := tensor.CompressionRatio(spec.CompressRatios[layerIndex])
	if ratio.Enabled() {
		required.add("compressor KV", weights.AttentionCompressorKV)
		required.add("compressor gate", weights.AttentionCompressorGate)
		required.add("compressor APE", weights.AttentionCompressorAPE)
		required.add("compressor norm", weights.AttentionCompressorNorm)
	}
	if ratio.UsesIndexer() {
		required.add("indexer projection", weights.IndexerProjection)
		required.add("indexer Q-B", weights.IndexerAttentionQB)
		required.add("indexer compressor KV", weights.IndexerCompressorKV)
		required.add("indexer compressor gate", weights.IndexerCompressorGate)
		required.add("indexer compressor APE", weights.IndexerCompressorAPE)
		required.add("indexer compressor norm", weights.IndexerCompressorNorm)
	}
	if err := required.validate("compressed attention"); err != nil {
		return DenseBlockResult{}, err
	}
	heads := uint64(spec.HeadCount)
	headWidth := uint64(spec.KeyLength)
	hc := spec.HyperConnectionCount
	residual := input
	current := builder.HyperConnectionPre(input, weights.HyperAttentionFN, weights.HyperAttentionScale,
		weights.HyperAttentionBase, hc, spec.HyperSinkhornIters, spec.RMSNormEpsilon, spec.HyperConnectionEps)
	current = builder.WeightedRMSNorm(current, weights.AttentionNorm, spec.RMSNormEpsilon)
	queryRank := builder.WeightedRMSNorm(builder.MulMat(weights.AttentionQ, current), weights.AttentionQNorm, spec.RMSNormEpsilon)
	query := builder.RMSNorm(builder.Reshape(builder.MulMat(weights.AttentionQB, queryRank), headWidth, heads, tokens), spec.RMSNormEpsilon)
	kv := builder.WeightedRMSNorm(builder.MulMat(weights.AttentionK, current), weights.AttentionKNorm, spec.RMSNormEpsilon)
	kv = builder.Reshape(kv, headWidth, tensor.SingletonExtent, tokens)
	frequencyBase := spec.RopeFrequencyBase
	frequencyScale := tensor.UnitFrequencyScale
	extFactor := float32(tensor.FirstOffset)
	attentionFactor := tensor.UnitScale
	originalContext := uint32(tensor.FirstOffset)
	betaFast, betaSlow := float32(tensor.FirstOffset), float32(tensor.FirstOffset)
	if ratio.Enabled() {
		frequencyBase = spec.CompressRopeBase
		if positiveFinite(spec.RopeScalingFactor) {
			frequencyScale = tensor.UnitFrequencyScale / spec.RopeScalingFactor
		}
		extFactor = spec.YaRNExtFactor
		attentionFactor = spec.YaRNAttentionFactor
		if !positiveFinite(attentionFactor) {
			attentionFactor = tensor.UnitScale
		}
		originalContext = spec.OriginalContextLength
		betaFast, betaSlow = spec.YaRNBetaFast, spec.YaRNBetaSlow
	}
	cacheKV := kv
	if pastKV != nil {
		_, cacheAxis, validCache := tensor.TrailingExtent(
			pastKV.Shape, headWidth, tensor.SingletonExtent,
		)
		if !validCache {
			return DenseBlockResult{}, errors.New("compressed-attention raw cache shape is invalid")
		}
		cacheKV = builder.Concat(pastKV, kv, cacheAxis)
	}
	states := make(CacheStates[*tensor.Tensor])
	cachePositions := currentPositions
	if previous := pastStates[CacheStatePositions].Value; previous != nil {
		cachePositions = builder.Concat(previous, currentPositions, tensor.PairedExtent)
	}
	states[CacheStatePositions] = CacheState[*tensor.Tensor]{Mode: CacheStateToken, Value: cachePositions}
	appendState := func(name CacheStateName, current *tensor.Tensor) *tensor.Tensor {
		if current == nil {
			return nil
		}
		current = builder.Reshape(
			current, current.Shape.ContiguousExtent(), tensor.SingletonExtent, current.Shape.RowCount(),
		)
		if previous := pastStates[name].Value; previous != nil {
			current = builder.Concat(previous, current, tensor.PairedExtent)
		}
		states[name] = CacheState[*tensor.Tensor]{Mode: CacheStateToken, Value: current}
		return current
	}
	var compressorKV, compressorScore, indexerQuery, indexerWeights, indexerKV, indexerScore *tensor.Tensor
	if ratio.Enabled() {
		rows := make([]uint32, len(positions))
		for index, position := range positions {
			rows[index] = position % uint32(ratio)
		}
		compressorKV = appendState(CacheStateCompressorKV, builder.MulMat(weights.AttentionCompressorKV, current))
		compressorScore = appendState(CacheStateCompressorScore, builder.Add(
			builder.MulMat(weights.AttentionCompressorGate, current), builder.GetRows(weights.AttentionCompressorAPE, rows),
		))
	}
	if ratio.UsesIndexer() {
		indexerWidth := uint64(spec.IndexerKeyLength)
		indexerHeads := uint64(spec.IndexerHeadCount)
		indexerQuery = builder.Reshape(builder.MulMat(weights.IndexerAttentionQB, queryRank), indexerWidth, indexerHeads, tokens)
		indexerWeights = builder.Scale(
			builder.MulMat(weights.IndexerProjection, current),
			hostmath.InvSqrt32(uint64(spec.IndexerKeyLength*spec.IndexerHeadCount)),
		)
		rows := make([]uint32, len(positions))
		for index, position := range positions {
			rows[index] = position % uint32(ratio)
		}
		indexerKV = appendState(CacheStateIndexerCompressorKV, builder.MulMat(weights.IndexerCompressorKV, current))
		indexerScore = appendState(CacheStateIndexerCompressorScore, builder.Add(
			builder.MulMat(weights.IndexerCompressorGate, current), builder.GetRows(weights.IndexerCompressorAPE, rows),
		))
	}
	attributes := tensor.CompressedAttentionAttributes{
		Positions: positions, Ratio: ratio, Window: spec.SlidingWindow, Heads: spec.HeadCount,
		IndexerHeads: spec.IndexerHeadCount, IndexerTopK: spec.IndexerTopK,
		RotaryDimensions: spec.RopeDimensionCount, FrequencyBase: frequencyBase, FrequencyScale: frequencyScale,
		OriginalContext: originalContext, ExtFactor: extFactor, AttentionFactor: attentionFactor,
		BetaFast: betaFast, BetaSlow: betaSlow, NormEpsilon: spec.RMSNormEpsilon,
	}
	attention := builder.CompressedAttention(query, cacheKV, cachePositions, weights.AttentionSinks,
		compressorKV, compressorScore, weights.AttentionCompressorNorm,
		indexerQuery, indexerWeights, indexerKV, indexerScore, weights.IndexerCompressorNorm, attributes)
	groupDimension := uint64(spec.HeadCount/spec.AttentionOutputGroups) * headWidth
	attention = builder.Reshape(attention, groupDimension, uint64(spec.AttentionOutputGroups), tokens)
	outputA := builder.Reshape(weights.AttentionOutputA, groupDimension, uint64(spec.AttentionOutputRank), uint64(spec.AttentionOutputGroups))
	attention = builder.Reshape(builder.GroupedMulMat(outputA, attention), uint64(spec.AttentionOutputRank*spec.AttentionOutputGroups), tokens)
	attention = builder.MulMat(weights.AttentionOutput, attention)
	input = builder.HyperConnectionPost(attention, residual, weights.HyperAttentionFN, weights.HyperAttentionScale,
		weights.HyperAttentionBase, hc, spec.HyperSinkhornIters, spec.RMSNormEpsilon, spec.HyperConnectionEps)
	if err := builder.Err(); err != nil {
		return DenseBlockResult{}, err
	}
	// Unique cache-value node: retained outputs require distinct K/V identities.
	cacheValue := builder.Reshape(cacheKV, cacheKV.Shape.Dims[:cacheKV.Shape.Rank]...)
	if err := builder.Err(); err != nil {
		return DenseBlockResult{}, err
	}
	return DenseBlockResult{Output: input, Key: cacheKV, Value: cacheValue, States: states}, nil
}

// buildHyperFeedForwardStage: hyper-connected routed FFN and terminal head.
func buildHyperFeedForwardStage(
	builder *tensor.Builder,
	input *tensor.Tensor,
	spec Spec,
	weights LayerGraphWeights,
	tokenRows []uint32,
	plan LayerPlan,
) (*tensor.Tensor, error) {
	layerIndex := plan.Layer
	if builder == nil || input == nil || layerIndex >= spec.BlockCount {
		return nil, errors.New("hyper feed-forward input is invalid")
	}
	_, connections, tokens, validInput := tensor.Extents3(input.Shape)
	if !validInput || connections != uint64(spec.HyperConnectionCount) {
		return nil, errors.New("hyper feed-forward input is invalid")
	}
	required := graphWeights{
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
		if uint64(len(tokenRows)) != tokens {
			return nil, errors.New("compressed-hyper hash routing rows are missing")
		}
	} else {
		required.add("router bias", weights.FeedForwardRouterBias)
	}
	if layerIndex+1 == spec.BlockCount {
		required.add("output HC function", weights.HyperHeadFN)
		required.add("output HC base", weights.HyperHeadBase)
		required.add("output HC scale", weights.HyperHeadScale)
	}
	if err := required.validate("compressed-hyper feed-forward"); err != nil {
		return nil, err
	}
	hc := spec.HyperConnectionCount
	residual := input
	current := builder.HyperConnectionPre(input, weights.HyperFeedForwardFN, weights.HyperFeedForwardScale,
		weights.HyperFeedForwardBase, hc, spec.HyperSinkhornIters, spec.RMSNormEpsilon, spec.HyperConnectionEps)
	current = builder.WeightedRMSNorm(current, weights.FeedForwardNorm, spec.RMSNormEpsilon)
	var selected *tensor.Tensor
	if layerIndex < spec.HashLayerCount {
		selected = builder.GetRows(weights.FeedForwardHashExperts, tokenRows)
	}
	moePlan := plan.Experts
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
		inputLimitedSwiGLU(builder, sharedGate, sharedUp, spec.LayerSharedSwiGLUClampLimit(layerIndex)))
	current = builder.Add(moe, shared)
	output := builder.HyperConnectionPost(current, residual, weights.HyperFeedForwardFN, weights.HyperFeedForwardScale,
		weights.HyperFeedForwardBase, hc, spec.HyperSinkhornIters, spec.RMSNormEpsilon, spec.HyperConnectionEps)
	if layerIndex+1 == spec.BlockCount {
		output = builder.HyperConnectionHead(output, weights.HyperHeadFN, weights.HyperHeadScale,
			weights.HyperHeadBase, hc, spec.RMSNormEpsilon, spec.HyperConnectionEps)
	}
	if err := builder.Err(); err != nil {
		return nil, err
	}
	return output, nil
}

// buildDynamicWKV6MixCached: RMS-input dynamic WKV6 recurrence.
func buildDynamicWKV6MixCached(
	builder *tensor.Builder,
	normalized *tensor.Tensor,
	spec Spec,
	weights LayerGraphWeights,
	pastShift, pastState *tensor.Tensor,
) (DenseBlockResult, error) {
	if builder == nil || normalized == nil || pastShift == nil || pastState == nil {
		return DenseBlockResult{}, errors.New("dynamic WKV6 input/state is invalid")
	}
	required := graphWeights{
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
	}
	if err := required.validate("dynamic WKV6"); err != nil {
		return DenseBlockResult{}, err
	}
	embedding := uint64(spec.EmbeddingLength)
	tokens, validInput := tensor.MatrixRows(normalized.Shape, embedding)
	if !validInput {
		return DenseBlockResult{}, errors.New("dynamic WKV6 input shape is invalid")
	}
	width := uint64(spec.WKVHeadSize)
	heads := embedding / width
	if !tensor.IsVector(pastShift.Shape, embedding) ||
		!tensor.HasDimensions(pastState.Shape, width, width, heads, tensor.SingletonExtent) {
		return DenseBlockResult{}, errors.New("dynamic WKV6 cache shape is invalid")
	}
	xPrev := builder.Reshape(pastShift, embedding, tensor.SingletonExtent)
	nextShift := builder.FlatSlice(normalized, embedding*(tokens-tensor.SingletonExtent), embedding)
	mixed := buildWKV6MixedStreams(builder, normalized, xPrev, spec, weights)
	xw, xk, xv := mixed[wkv6WeightStream], mixed[wkv6KeyStream], mixed[wkv6ValueStream]
	xr, xg := mixed[wkv6ReceptanceStream], mixed[wkv6GateStream]
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
	decay = builder.Exp(builder.Scale(
		builder.Exp(builder.Add(decay, weights.TimeMixDecay)), tensor.NegativeUnitScale,
	))
	kvHeads := uint64(spec.HeadCountKV)
	key = builder.Reshape(key, width, kvHeads, tokens, tensor.SingletonExtent)
	value = builder.Reshape(value, width, kvHeads, tokens, tensor.SingletonExtent)
	receptance = builder.Reshape(receptance, width, heads, tokens, tensor.SingletonExtent)
	decay = builder.Reshape(decay, width, heads, tokens, tensor.SingletonExtent)
	packed := builder.GatedLinearAttention(
		key, value, receptance, decay, pastState,
		hostmath.InvSqrt32(width),
	)
	attentionElements := embedding * tokens
	attention := builder.FlatSlice(packed, tensor.FirstOffset, embedding, tokens)
	nextState := builder.FlatSlice(packed, attentionElements, width, width, heads, tensor.SingletonExtent)
	attention = builder.MulMat(weights.TimeMixOutput, builder.Multiply(attention, gate))
	if err := builder.Err(); err != nil {
		return DenseBlockResult{}, err
	}
	return DenseBlockResult{Output: attention, Key: nextShift, Value: nextState}, nil
}

// buildAffineWKV6MixCached: affine-normalized WKV6 recurrence.
func buildAffineWKV6MixCached(
	builder *tensor.Builder,
	normalized *tensor.Tensor,
	spec Spec,
	weights LayerGraphWeights,
	pastShift, pastState *tensor.Tensor,
	layerPlan LayerPlan,
) (DenseBlockResult, error) {
	if builder == nil || normalized == nil || pastShift == nil || pastState == nil {
		return DenseBlockResult{}, errors.New("affine WKV6 input/state is invalid")
	}
	required := graphWeights{
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
	}
	if err := required.validate("affine WKV6"); err != nil {
		return DenseBlockResult{}, err
	}
	if weights.TimeMixLerpFused == nil &&
		(weights.TimeMixLerpW == nil || weights.TimeMixLerpK == nil || weights.TimeMixLerpV == nil ||
			weights.TimeMixLerpR == nil || weights.TimeMixLerpG == nil) {
		return DenseBlockResult{}, errors.New("WKV6 time-mix lerp catalog is incomplete")
	}
	embedding := uint64(spec.EmbeddingLength)
	width := uint64(spec.WKVHeadSize)
	heads := uint64(spec.HeadCount)
	tokens, validInput := tensor.MatrixRows(normalized.Shape, embedding)
	if !validInput ||
		!tensor.HasDimensions(pastShift.Shape, embedding, uint64(layerPlan.RecurrentRuntime.TokenShiftCount)) ||
		!tensor.HasDimensions(pastState.Shape, width, width, heads, tensor.SingletonExtent) {
		return DenseBlockResult{}, errors.New("affine WKV6 input/cache shape is invalid")
	}
	attPrev := builder.Reshape(
		builder.FlatSlice(pastShift, tensor.FirstOffset, embedding), embedding, tensor.SingletonExtent,
	)
	mixed := buildWKV6MixedStreams(builder, normalized, attPrev, spec, weights)
	xw, xk, xv := mixed[wkv6WeightStream], mixed[wkv6KeyStream], mixed[wkv6ValueStream]
	xr, xg := mixed[wkv6ReceptanceStream], mixed[wkv6GateStream]
	key := builder.Reshape(builder.MulMat(weights.TimeMixKey, xk), width, heads, tokens, tensor.SingletonExtent)
	value := builder.Reshape(builder.MulMat(weights.TimeMixValue, xv), width, heads, tokens, tensor.SingletonExtent)
	receptance := builder.Reshape(builder.MulMat(weights.TimeMixReceptance, xr), width, heads, tokens, tensor.SingletonExtent)
	gate := builder.SiLU(builder.MulMat(weights.TimeMixGate, xg))
	decay := builder.MulMat(weights.TimeMixDecayW2, builder.Tanh(builder.MulMat(weights.TimeMixDecayW1, xw)))
	decay = builder.Reshape(
		builder.Exp(builder.Scale(builder.Exp(builder.Add(decay, weights.TimeMixDecay)), tensor.NegativeUnitScale)),
		width, heads, tokens, tensor.SingletonExtent,
	)
	packed := builder.WKV6(key, value, receptance, weights.TimeMixFirst, decay, pastState)
	attentionElements := embedding * tokens
	attention := builder.FlatSlice(packed, tensor.FirstOffset, embedding, tokens)
	nextState := builder.FlatSlice(packed, attentionElements, width, width, heads, tensor.SingletonExtent)
	attention = builder.Reshape(builder.LayerNorm(
		builder.Reshape(attention, width, heads, tokens),
		layerPlan.RecurrentRuntime.HeadNormEpsilon,
	), embedding, tokens)
	attention = builder.Add(builder.Multiply(attention, weights.TimeMixLN), weights.TimeMixLNBias)
	attention = builder.MulMat(weights.TimeMixOutput, builder.Multiply(attention, gate))
	nextShift := builder.Reshape(
		builder.FlatSlice(normalized, embedding*(tokens-tensor.SingletonExtent), embedding), embedding, tensor.SingletonExtent,
	)
	if err := builder.Err(); err != nil {
		return DenseBlockResult{}, err
	}
	return DenseBlockResult{Output: attention, Key: nextShift, Value: nextState}, nil
}

func buildWKV6MixedStreams(
	builder *tensor.Builder,
	normalized, previous *tensor.Tensor,
	spec Spec,
	weights LayerGraphWeights,
) [wkv6TimeMixStreams]*tensor.Tensor {
	embedding, tokens := uint64(spec.EmbeddingLength), normalized.Shape.RowCount()
	if tokens > tensor.SingletonExtent {
		previous = builder.Concat(
			previous,
			builder.FlatSlice(normalized, tensor.FirstOffset, embedding, tokens-tensor.SingletonExtent),
			tensor.SingletonExtent,
		)
	}
	delta := builder.Add(previous, builder.Scale(normalized, tensor.NegativeUnitScale))
	base := builder.Add(normalized, builder.Multiply(
		delta, builder.Reshape(weights.TimeMixLerpX, embedding, tensor.SingletonExtent),
	))
	adjustments := builder.Reshape(builder.GroupedMulMat(
		weights.TimeMixW2, builder.Reshape(
			builder.Tanh(builder.MulMat(weights.TimeMixW1, base)),
			uint64(spec.TimeMixExtraDim), wkv6TimeMixStreams, tokens,
		),
	), embedding*wkv6TimeMixStreams, tokens)
	separate := [wkv6TimeMixStreams]*tensor.Tensor{
		weights.TimeMixLerpW, weights.TimeMixLerpK, weights.TimeMixLerpV,
		weights.TimeMixLerpR, weights.TimeMixLerpG,
	}
	var mixed [wkv6TimeMixStreams]*tensor.Tensor
	for index := uint64(tensor.FirstOffset); index < wkv6TimeMixStreams; index++ {
		adjustment := builder.Reshape(builder.GroupSlice(
			adjustments, index*embedding, embedding, tensor.SingletonExtent, embedding*wkv6TimeMixStreams,
		), embedding, tokens)
		lerp := separate[index]
		if weights.TimeMixLerpFused != nil {
			lerp = builder.GroupSlice(
				builder.Reshape(weights.TimeMixLerpFused, embedding*wkv6TimeMixStreams, tensor.SingletonExtent),
				index*embedding, embedding, tensor.SingletonExtent, embedding*wkv6TimeMixStreams,
			)
		}
		mixed[index] = builder.Add(normalized, builder.Multiply(
			delta, builder.Add(adjustment, builder.Reshape(lerp, embedding)),
		))
	}
	return mixed
}

// buildDynamicWKV7MixCached: dynamic-decay WKV7 recurrence.
func buildDynamicWKV7MixCached(
	builder *tensor.Builder,
	normalized *tensor.Tensor,
	spec Spec,
	weights LayerGraphWeights,
	pastShift, pastState *tensor.Tensor,
	layerPlan LayerPlan,
) (DenseBlockResult, error) {
	if builder == nil || normalized == nil || pastShift == nil || pastState == nil {
		return DenseBlockResult{}, errors.New("dynamic WKV7 input/state is invalid")
	}
	required := graphWeights{
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
	if layerPlan.Normalization.Operation == NormalizationLayer {
		required.add("time norm", weights.TimeMixLN)
		required.add("time norm bias", weights.TimeMixLNBias)
	}
	if spec.GateLoRARank > tensor.FirstOffset {
		required.add("time G1", weights.TimeMixG1)
		required.add("time G2", weights.TimeMixG2)
	}
	if err := required.validate("dynamic WKV7"); err != nil {
		return DenseBlockResult{}, err
	}
	embedding := uint64(spec.EmbeddingLength)
	width := uint64(spec.WKVHeadSize)
	heads := uint64(spec.HeadCount)
	tokens, validInput := tensor.MatrixRows(normalized.Shape, embedding)
	shiftCount := uint64(layerPlan.RecurrentRuntime.TokenShiftCount)
	if !validInput || !tensor.HasDimensions(pastShift.Shape, embedding, shiftCount) ||
		!tensor.HasDimensions(pastState.Shape, width, width, heads, tensor.SingletonExtent) {
		return DenseBlockResult{}, errors.New("dynamic WKV7 input/cache shape is invalid")
	}
	attPrev := builder.Reshape(
		builder.FlatSlice(pastShift, tensor.FirstOffset, embedding), embedding, tensor.SingletonExtent,
	)
	if tokens > tensor.SingletonExtent {
		attPrev = builder.Concat(
			attPrev,
			builder.FlatSlice(normalized, tensor.FirstOffset, embedding, tokens-tensor.SingletonExtent),
			tensor.SingletonExtent,
		)
	}
	sx := builder.Add(attPrev, builder.Scale(normalized, tensor.NegativeUnitScale))
	lerpCount := wkv7GatedMixStreams
	if spec.GateLoRARank == tensor.FirstOffset {
		lerpCount = wkv7TimeMixStreams
	}
	lerps := builder.Reshape(weights.TimeMixLerpFused, embedding*lerpCount, tensor.SingletonExtent)
	mixed := make([]*tensor.Tensor, lerpCount)
	for index := uint64(tensor.FirstOffset); index < lerpCount; index++ {
		lerp := builder.Reshape(builder.GroupSlice(
			lerps, index*embedding, embedding, tensor.SingletonExtent, embedding*lerpCount,
		), embedding)
		mixed[index] = builder.Add(normalized, builder.Multiply(sx, lerp))
	}
	xr, xw := mixed[wkv7ReceptanceStream], mixed[wkv7WeightStream]
	xk, xv, xa := mixed[wkv7KeyStream], mixed[wkv7ValueStream], mixed[wkv7AuxiliaryStream]
	receptance := builder.MulMat(weights.TimeMixReceptance, xr)
	decay := builder.Add(
		builder.MulMat(weights.TimeMixW2, builder.Tanh(builder.MulMat(weights.TimeMixW1, xw))),
		weights.TimeMixW0,
	)
	decay = builder.Exp(builder.Scale(builder.Sigmoid(decay), wkv7DecayScale))
	key := builder.MulMat(weights.TimeMixKey, xk)
	value := builder.MulMat(weights.TimeMixValue, xv)
	var auxiliary *tensor.Tensor
	if layerPlan.Layer == tensor.FirstOffset {
		if weights.PerLayerInput != nil {
			return DenseBlockResult{}, errors.New("WKV7 first layer received a value residual")
		}
		auxiliary = value
	} else {
		if weights.PerLayerInput == nil {
			return DenseBlockResult{}, errors.New("WKV7 value residual is missing")
		}
		valueMix := builder.Sigmoid(builder.Add(
			builder.MulMat(weights.TimeMixV2, builder.MulMat(weights.TimeMixV1, xv)),
			weights.TimeMixV0,
		))
		value = builder.Add(value, builder.Multiply(
			builder.Add(weights.PerLayerInput, builder.Scale(value, tensor.NegativeUnitScale)), valueMix,
		))
	}
	a := builder.Sigmoid(builder.Add(
		builder.MulMat(weights.TimeMixA2, builder.MulMat(weights.TimeMixA1, xa)),
		weights.TimeMixA0,
	))
	kk := builder.L2Norm(
		builder.Reshape(
			builder.Multiply(key, weights.TimeMixKK), width, heads, tokens, tensor.SingletonExtent,
		),
		layerPlan.RecurrentRuntime.KeyNormEpsilon,
	)
	ka := builder.Multiply(key, weights.TimeMixKA)
	key = builder.Add(key, builder.Add(builder.Multiply(a, ka), builder.Scale(ka, tensor.NegativeUnitScale)))
	receptance4 := builder.Reshape(receptance, width, heads, tokens, tensor.SingletonExtent)
	decay4 := builder.Reshape(decay, width, heads, tokens, tensor.SingletonExtent)
	key4 := builder.Reshape(key, width, heads, tokens, tensor.SingletonExtent)
	value4 := builder.Reshape(value, width, heads, tokens, tensor.SingletonExtent)
	a4 := builder.Reshape(a, width, heads, tokens, tensor.SingletonExtent)
	packed := builder.WKV7(
		receptance4, decay4, key4, value4,
		builder.Scale(kk, tensor.NegativeUnitScale), builder.Multiply(kk, a4), pastState,
	)
	attentionElements := embedding * tokens
	attention := builder.FlatSlice(packed, tensor.FirstOffset, embedding, tokens)
	nextState := builder.FlatSlice(packed, attentionElements, width, width, heads, tensor.SingletonExtent)
	if weights.TimeMixLN != nil && weights.TimeMixLNBias != nil {
		attention = builder.Reshape(builder.LayerNorm(
			builder.Reshape(attention, width, heads, tokens),
			layerPlan.RecurrentRuntime.HeadNormEpsilon,
		), embedding, tokens)
		attention = builder.Add(builder.Multiply(attention, weights.TimeMixLN), weights.TimeMixLNBias)
	}
	rkWeight := builder.Reshape(weights.TimeMixRK, width, heads, tensor.SingletonExtent, tensor.SingletonExtent)
	rk := builder.SumRows(builder.Multiply(builder.Multiply(key4, receptance4), rkWeight))
	attention = builder.Add(attention, builder.Reshape(builder.Multiply(value4, rk), embedding, tokens))
	if spec.GateLoRARank > tensor.FirstOffset {
		xg := mixed[wkv7GateStream]
		gate := builder.MulMat(weights.TimeMixG2, builder.Sigmoid(builder.MulMat(weights.TimeMixG1, xg)))
		attention = builder.Multiply(attention, gate)
	}
	attention = builder.MulMat(weights.TimeMixOutput, attention)
	nextShift := builder.Reshape(
		builder.FlatSlice(normalized, embedding*(tokens-tensor.SingletonExtent), embedding), embedding, tensor.SingletonExtent,
	)
	if err := builder.Err(); err != nil {
		return DenseBlockResult{}, err
	}
	return DenseBlockResult{Output: attention, Key: nextShift, Value: nextState, Auxiliary: auxiliary}, nil
}

// buildTokenShiftFeedForwardMix: normalized shifted-channel projection.
func buildTokenShiftFeedForwardMix(
	builder *tensor.Builder,
	input *tensor.Tensor,
	spec Spec,
	weights LayerGraphWeights,
	pastShift, nextAttentionShift *tensor.Tensor,
	operator LayerOperator,
	normalization NormalizationPlan,
	tokenShiftCount uint32,
) (DenseBlockResult, error) {
	required := graphWeights{
		requireGraphWeight("channel norm", weights.AttentionNorm2),
		requireGraphWeight("channel norm bias", weights.AttentionNorm2Bias),
		requireGraphWeight("channel lerp K", weights.ChannelMixLerpK),
		requireGraphWeight("channel key", weights.ChannelMixKey),
		requireGraphWeight("channel value", weights.ChannelMixValue),
	}
	if operator == LayerOperatorGatedTokenShiftSquaredReLU {
		required.add("channel lerp R", weights.ChannelMixLerpR)
		required.add("channel receptance", weights.ChannelMixReceptance)
	} else if operator != LayerOperatorTokenShiftSquaredReLU {
		return DenseBlockResult{}, errors.New("token-shift feed-forward policy is invalid")
	}
	if err := required.validate("token-shift feed-forward"); err != nil {
		return DenseBlockResult{}, err
	}
	embedding := uint64(spec.EmbeddingLength)
	if input == nil || pastShift == nil || nextAttentionShift == nil {
		return DenseBlockResult{}, errors.New("token-shift feed-forward input/cache shape is invalid")
	}
	tokens, validInput := tensor.MatrixRows(input.Shape, embedding)
	if !validInput || !tensor.HasDimensions(pastShift.Shape, embedding, uint64(tokenShiftCount)) ||
		!tensor.HasDimensions(nextAttentionShift.Shape, embedding, tensor.SingletonExtent) {
		return DenseBlockResult{}, errors.New("token-shift feed-forward input/cache shape is invalid")
	}
	normalized := normalization.Apply(builder, input, weights.AttentionNorm2, weights.AttentionNorm2Bias)
	previous := builder.Reshape(
		builder.FlatSlice(pastShift, embedding, embedding), embedding, tensor.SingletonExtent,
	)
	if tokens > tensor.SingletonExtent {
		previous = builder.Concat(
			previous,
			builder.FlatSlice(normalized, tensor.FirstOffset, embedding, tokens-tensor.SingletonExtent),
			tensor.SingletonExtent,
		)
	}
	shift := builder.Add(previous, builder.Scale(normalized, tensor.NegativeUnitScale))
	keyInput := builder.Add(normalized, builder.Multiply(
		shift, builder.Reshape(weights.ChannelMixLerpK, embedding, tensor.SingletonExtent),
	))
	channel := builder.MulMat(
		weights.ChannelMixValue, builder.ReLUSquared(builder.MulMat(weights.ChannelMixKey, keyInput)),
	)
	if operator == LayerOperatorGatedTokenShiftSquaredReLU {
		receptanceInput := builder.Add(normalized, builder.Multiply(
			shift, builder.Reshape(weights.ChannelMixLerpR, embedding, tensor.SingletonExtent),
		))
		channel = builder.Multiply(
			builder.Sigmoid(builder.MulMat(weights.ChannelMixReceptance, receptanceInput)), channel,
		)
	}
	nextChannelShift := builder.Reshape(
		builder.FlatSlice(normalized, embedding*(tokens-tensor.SingletonExtent), embedding),
		embedding, tensor.SingletonExtent,
	)
	nextShift := builder.Concat(nextAttentionShift, nextChannelShift, tensor.SingletonExtent)
	if err := builder.Err(); err != nil {
		return DenseBlockResult{}, err
	}
	return DenseBlockResult{Output: channel, Key: nextShift}, nil
}

// buildKeyedDeltaAttentionMixCached: convolutional keyed-delta recurrence.
func buildKeyedDeltaAttentionMixCached(
	builder *tensor.Builder,
	normalized *tensor.Tensor,
	spec Spec,
	weights LayerGraphWeights,
	positions []uint32,
	pastKey, pastValue *tensor.Tensor,
) (DenseBlockResult, error) {
	if builder == nil || normalized == nil || pastKey == nil || pastValue == nil {
		return DenseBlockResult{}, errors.New("keyed-delta attention input/state is nil")
	}
	required := graphWeights{
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
	}
	if err := required.validate("keyed-delta attention"); err != nil {
		return DenseBlockResult{}, err
	}
	tokens, validInput := tensor.MatrixRows(normalized.Shape, uint64(spec.EmbeddingLength))
	if !validInput || uint64(len(positions)) != tokens {
		return DenseBlockResult{}, errors.New("keyed-delta attention input shape is invalid")
	}
	headDim := uint64(spec.KDAHeadDim)
	heads := uint64(spec.HeadCount)
	inner := headDim * heads
	window := spec.ssmConvolutionWindow()
	if !tensor.HasDimensions(pastKey.Shape, window, tensor.TripleExtent*inner) ||
		!tensor.HasDimensions(pastValue.Shape, headDim, headDim, heads, tensor.SingletonExtent) {
		return DenseBlockResult{}, errors.New("keyed-delta attention cache shape is invalid")
	}
	convolve := func(projection, kernel *tensor.Tensor, stateIndex uint64) (*tensor.Tensor, *tensor.Tensor) {
		state := builder.FlatSlice(pastKey, stateIndex*window*inner, window, inner)
		projected := builder.MulMat(projection, normalized)
		mixed := builder.Concat(state, builder.Transpose2D(projected), tensor.FirstOffset)
		next := builder.Reshape(builder.GroupSlice(
			mixed, tokens, window, tensor.SingletonExtent, window,
		), window, inner)
		kernel = builder.Reshape(kernel, uint64(spec.SSMConvKernel), inner)
		value := builder.SiLU(builder.SSMConv(mixed, kernel))
		return builder.Reshape(value, headDim, heads, tokens, tensor.SingletonExtent), next
	}
	query, nextQ := convolve(weights.AttentionQ, weights.SSMQueryConv, tensor.FirstOffset)
	key, nextK := convolve(weights.AttentionK, weights.SSMKeyConv, tensor.SingletonExtent)
	value, nextV := convolve(weights.AttentionV, weights.SSMValueConv, tensor.PairedExtent)
	nextConv := builder.Transpose2D(builder.Concat(
		builder.Concat(builder.Transpose2D(nextQ), builder.Transpose2D(nextK), tensor.FirstOffset),
		builder.Transpose2D(nextV), tensor.FirstOffset,
	))
	gate := builder.MulMat(weights.SSMForgetB, builder.MulMat(weights.SSMForgetA, normalized))
	gate = builder.Softplus(builder.Add(gate, weights.SSMTimeStep))
	gate = builder.Reshape(gate, headDim, heads, tokens, tensor.SingletonExtent)
	gate = builder.Multiply(gate, builder.Reshape(
		weights.SSMA, tensor.SingletonExtent, heads, tensor.SingletonExtent, tensor.SingletonExtent,
	))
	beta := builder.Reshape(
		builder.Sigmoid(builder.MulMat(weights.SSMBeta, normalized)),
		tensor.SingletonExtent, heads, tokens, tensor.SingletonExtent,
	)
	query = builder.L2Norm(query, spec.RMSNormEpsilon)
	key = builder.L2Norm(key, spec.RMSNormEpsilon)
	packed := builder.GatedDeltaNet(query, key, value, gate, beta, pastValue)
	attentionElements := headDim * heads * tokens
	attention := builder.FlatSlice(packed, tensor.FirstOffset, headDim, heads, tokens, tensor.SingletonExtent)
	nextState := builder.FlatSlice(packed, attentionElements, headDim, headDim, heads, tensor.SingletonExtent)
	outputGate := builder.MulMat(weights.SSMOutputGateB, builder.MulMat(weights.SSMOutputGateA, normalized))
	outputGate = builder.Reshape(outputGate, headDim, heads, tokens, tensor.SingletonExtent)
	attention = builder.Multiply(
		builder.WeightedRMSNorm(attention, weights.SSMNorm, spec.RMSNormEpsilon),
		builder.Sigmoid(outputGate),
	)
	attention = builder.MulMat(weights.AttentionOutput, builder.Reshape(attention, inner, tokens))
	if err := builder.Err(); err != nil {
		return DenseBlockResult{}, err
	}
	return DenseBlockResult{Output: attention, Key: nextConv, Value: nextState}, nil
}

// buildShortConvolutionMixCached: short-convolution recurrence.
func buildShortConvolutionMixCached(
	builder *tensor.Builder,
	normalized *tensor.Tensor,
	spec Spec,
	weights LayerGraphWeights,
	positions []uint32,
	pastKey, pastValue *tensor.Tensor,
) (DenseBlockResult, error) {
	if builder == nil || normalized == nil || pastKey == nil || pastValue == nil {
		return DenseBlockResult{}, errors.New("short-convolution input/state is nil")
	}
	required := graphWeights{
		requireGraphWeight("short-convolution input", weights.ShortConvInput),
		requireGraphWeight("short-convolution kernel", weights.ShortConvKernel),
		requireGraphWeight("short-convolution output", weights.ShortConvOutput),
	}
	if err := required.validate("short-convolution recurrent mixer"); err != nil {
		return DenseBlockResult{}, err
	}
	embedding := uint64(spec.EmbeddingLength)
	tokens, validInput := tensor.MatrixRows(normalized.Shape, embedding)
	if !validInput || uint64(len(positions)) != tokens {
		return DenseBlockResult{}, errors.New("short-convolution recurrent mixer input shape is invalid")
	}
	window := uint64(spec.ShortConvCacheLength - tensor.SingletonExtent)
	if !tensor.HasDimensions(pastKey.Shape, window, embedding) ||
		!tensor.IsVector(pastValue.Shape, tensor.SingletonExtent) {
		return DenseBlockResult{}, errors.New("short-convolution recurrent cache shape is invalid")
	}
	mixed := builder.MulMat(weights.ShortConvInput, normalized)
	stride := tensor.TripleExtent * embedding
	b := builder.Reshape(builder.GroupSlice(
		mixed, tensor.FirstOffset, embedding, tensor.SingletonExtent, stride,
	), embedding, tokens)
	c := builder.Reshape(builder.GroupSlice(
		mixed, embedding, embedding, tensor.SingletonExtent, stride,
	), embedding, tokens)
	x := builder.Reshape(builder.GroupSlice(
		mixed, tensor.PairedExtent*embedding, embedding, tensor.SingletonExtent, stride,
	), embedding, tokens)
	projected := builder.Transpose2D(builder.Multiply(b, x))
	convInput := builder.Concat(pastKey, projected, tensor.FirstOffset)
	if spec.NonCausalAttention {
		leftPad := window / tensor.PairedExtent
		rightPad := window - leftPad
		convInput = projected
		if leftPad > tensor.FirstOffset {
			left := builder.GroupSlice(pastKey, window-leftPad, leftPad, tensor.SingletonExtent, window)
			left = builder.Reshape(left, leftPad, embedding)
			convInput = builder.Concat(left, convInput, tensor.FirstOffset)
		}
		if rightPad > tensor.FirstOffset {
			right := builder.GroupSlice(pastKey, window-rightPad, rightPad, tensor.SingletonExtent, window)
			right = builder.Scale(builder.Reshape(right, rightPad, embedding), tensor.FirstOffset)
			convInput = builder.Concat(convInput, right, tensor.FirstOffset)
		}
	}
	nextState := builder.GroupSlice(convInput, tokens, window, tensor.SingletonExtent, window)
	nextState = builder.Reshape(nextState, window, embedding)
	convolved := builder.SSMConv(convInput, weights.ShortConvKernel)
	shortConv := builder.MulMat(weights.ShortConvOutput, builder.Multiply(c, convolved))
	nextReserved := builder.Scale(pastValue, tensor.UnitScale)
	if err := builder.Err(); err != nil {
		return DenseBlockResult{}, err
	}
	return DenseBlockResult{
		Output: shortConv, Key: nextState, Value: nextReserved,
	}, nil
}
